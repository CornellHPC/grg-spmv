"""Lean BOLT-LMM-inf benchmark over chromosome GRG artifacts."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import re
from time import perf_counter
from typing import Any

import numpy as np

from pygrgl_spmv.grg import RuntimeRequirements, convert
from pygrgl_spmv.grg.artifact import artifact_path_for_grg, scan_grg_spmv
from scripts.bench.cusparse import DEFAULT_CUSPARSE_PLAN_NAME, parse_cusparse_plan


_DTYPE = np.dtype(np.float64)
_DEFAULT_GRG_DIR = Path("/global/cfs/projectdirs/m4341/grg/sim/500k/grg")
_CHR_RE = re.compile(r"^chr([0-9]+)(?:\D.*)?\.(?:grg|grg_spmv)$")
_LOGGER = logging.getLogger(__name__)
_P_HIST_BINS = 20
_CHI2_MEDIAN_DF1 = 0.454936423119572
_REML_MIN_H2 = 1e-8
_REML_MAX_H2 = 0.99


@dataclass
class TimingBucket:
    calls: int = 0
    total_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        return 0.0 if self.calls == 0 else self.total_ms / float(self.calls)

    def add(self, elapsed_ms: float) -> None:
        self.calls += 1
        self.total_ms += float(elapsed_ms)


@dataclass
class CgBucket:
    solves: int = 0
    iterations: int = 0
    max_iterations: int = 0
    max_rel_resid: float = 0.0

    def add(self, iterations: int, rel_resid: float) -> None:
        it = int(iterations)
        self.solves += 1
        self.iterations += it
        self.max_iterations = max(self.max_iterations, it)
        self.max_rel_resid = max(self.max_rel_resid, float(rel_resid))


@dataclass
class ChromState:
    label: int
    grg: Any
    up_op: Any
    down_op: Any
    views: dict[str, Any]
    used_mask: Any | None = None
    used_local_indices: Any | None = None
    num_used: int = 0
    num_monomorphic_ref: int = 0
    num_monomorphic_alt: int = 0
    mu: Any | None = None
    sigma: Any | None = None
    inv_sigma: Any | None = None
    mu_over_sigma: Any | None = None


@dataclass
class EffectCheckSnp:
    global_idx: int
    label: int
    local_idx: int
    true_beta: float


@dataclass
class PhenotypeSimulation:
    y: Any
    metrics: OrderedDict[str, Any]
    effect_snps: tuple[EffectCheckSnp, ...] = ()


def _array_module(value):
    if type(value).__module__.split(".", 1)[0] == "cupy":
        import cupy as cp

        return cp
    return np


def _to_float(value: Any) -> float:
    if hasattr(value, "get"):
        return float(value.get())
    return float(value)


def _dot(left, right) -> float:
    xp = _array_module(left)
    return _to_float(xp.sum(left * right))


def _variance(x) -> float:
    return _dot(x, x) / float(int(x.size))


def _next_cupy_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.int64).max))


def centered(x):
    xp = _array_module(x)
    arr = xp.asarray(x, dtype=_DTYPE)
    return arr - arr.mean()


def center_into(x, out=None):
    xp = _array_module(out if out is not None else x)
    if out is None:
        arr = xp.asarray(x, dtype=_DTYPE)
    else:
        arr = out
        arr[...] = x
    arr -= arr.mean()
    return arr


def _parse_chromosomes(value: str, available: Iterable[int] | None = None) -> tuple[int, ...] | None:
    token = str(value).strip().lower()
    available_tuple = None if available is None else tuple(int(label) for label in available)
    if token == "all":
        return available_tuple
    if token == "":
        raise ValueError("--chromosomes must not be empty")
    labels: list[int] = []
    seen: set[int] = set()
    for raw_part in token.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"invalid chromosome list {value!r}")
        try:
            label = int(part)
        except ValueError as exc:
            raise ValueError(f"chromosome labels must be numeric or 'all', got {part!r}") from exc
        if label < 1:
            raise ValueError(f"chromosome labels must be positive, got {label}")
        if label in seen:
            raise ValueError(f"duplicate chromosome label {label}")
        seen.add(label)
        labels.append(label)
    if available_tuple is not None:
        available_set = set(available_tuple)
        missing = [label for label in labels if label not in available_set]
        if missing:
            raise ValueError(f"requested chromosome(s) not found: {','.join(str(v) for v in missing)}")
    return tuple(labels)


def _chrom_label_from_path(path: Path) -> int | None:
    for part in reversed(Path(path).parts):
        match = _CHR_RE.match(part)
        if match is not None:
            return int(match.group(1))
    return None


def _discover_grgs(grg_dir: Path) -> dict[int, Path]:
    root = Path(grg_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)
    discovered: dict[int, Path] = {}
    for path in sorted(root.glob("chr*.grg")):
        label = _chrom_label_from_path(path)
        if label is None:
            continue
        if label in discovered:
            raise ValueError(f"multiple GRGs found for chromosome {label}: {discovered[label]} and {path}")
        discovered[label] = path
    if not discovered:
        raise FileNotFoundError(f"no chr*.grg files found in {root}")
    return dict(sorted(discovered.items()))


def _cached_artifact_for_grg(raw: Path, cache: Path) -> Path:
    artifact = artifact_path_for_grg(Path(raw), Path(cache).expanduser())
    if artifact.exists():
        try:
            scan_grg_spmv(artifact)
        except Exception as exc:
            raise ValueError(
                f"cached artifact {artifact} is unsupported by this repository and was not modified; "
                "choose a fresh --artifact-cache or manually remove and regenerate it"
            ) from exc
        return artifact
    _LOGGER.info("Converting %s to cached artifact %s", raw, artifact)
    convert(Path(raw), Path(cache).expanduser(), dtype=_DTYPE)
    return artifact


def _resolve_artifacts(
    *,
    grg_dir: Path | None,
    artifact_cache: Path | None,
    artifacts: Iterable[Path] | None,
    chromosomes: str,
) -> tuple[tuple[int, Path], ...]:
    artifact_paths = tuple(Path(path).expanduser() for path in artifacts or ())
    if artifact_paths:
        if grg_dir is not None:
            raise ValueError("--artifacts is mutually exclusive with --grg-dir")
        labels = tuple(_chrom_label_from_path(path) for path in artifact_paths)
        num_labeled = sum(label is not None for label in labels)
        if num_labeled == len(labels):
            label_to_path = dict(sorted((int(label), path) for label, path in zip(labels, artifact_paths, strict=True)))
            if len(label_to_path) != len(artifact_paths):
                raise ValueError("duplicate chromosome labels in --artifacts")
            selected = _parse_chromosomes(chromosomes, available=label_to_path)
            selected_labels = tuple(label_to_path) if selected is None else selected
            return tuple((label, label_to_path[label]) for label in selected_labels)
        if 0 < num_labeled < len(labels):
            raise ValueError(
                "--artifacts paths must be either all chr<num>-labeled or all unlabeled; mixed labels are ambiguous"
            )
        requested = _parse_chromosomes(chromosomes)
        if requested is None or len(requested) != len(artifact_paths):
            raise ValueError(
                "--artifacts paths must contain chr<num> labels, or --chromosomes must provide one label per artifact"
            )
        return tuple((label, path) for label, path in zip(requested, artifact_paths, strict=True))

    raw_dir = _DEFAULT_GRG_DIR if grg_dir is None else Path(grg_dir)
    if artifact_cache is None:
        raise ValueError("--artifact-cache is required when using --grg-dir")
    discovered = _discover_grgs(raw_dir)
    selected = _parse_chromosomes(chromosomes, available=discovered)
    selected_labels = tuple(discovered) if selected is None else selected
    return tuple((label, _cached_artifact_for_grg(discovered[label], Path(artifact_cache))) for label in selected_labels)


def _free_memory_bytes(device: int) -> int:
    import cupy as cp

    with cp.cuda.Device(int(device)):
        free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
    return int(free_bytes)


def _bolt_device_reservation_bytes(scans, *, phenotype_mode, scan_top_k: int) -> int:
    scan_tuple = tuple(scans)
    if not scan_tuple:
        raise ValueError("at least one chromosome artifact is required")
    mode = str(phenotype_mode)
    if mode not in {"null", "infinitesimal"}:
        raise ValueError(f"unknown phenotype mode {phenotype_mode!r}")

    raw_m = int(sum(int(scan.num_mutations) for scan in scan_tuple))
    max_m = int(max(int(scan.num_mutations) for scan in scan_tuple))
    n = int(scan_tuple[0].num_individuals)
    chrom_count = int(len(scan_tuple))
    trials = int(mc_trial_count(n))
    f = int(_DTYPE.itemsize)
    b = int(np.dtype(np.bool_).itemsize)
    i64 = int(np.dtype(np.int64).itemsize)

    base = int(raw_m * (4 * f + b + i64) + (max_m + n) * f)
    frequency_extra = int(3 * max_m * f + 2 * max_m * b)
    phenotype_extra = int(2 * n * f if mode == "null" else (4 * n + max_m) * f)
    reml_extra = int(((8 + 2 * trials) * n + max_m) * f)
    loco_calibration_effect_extra = int((chrom_count + 8) * n * f)
    scan_extra = int((chrom_count + 1) * n * f + 3 * max_m * f + (max_m * i64 if int(scan_top_k) > 0 else 0))
    return int(base + max(frequency_extra, phenotype_extra, reml_extra, loco_calibration_effect_extra, scan_extra))


def _cuda_event_ms(device, stream, fn):
    import cupy as cp

    with device:
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        with stream:
            start.record(stream)
            result = fn()
            end.record(stream)
        end.synchronize()
        return result, float(cp.cuda.get_elapsed_time(start, end))


def cg_solve(matvec_into, b, *, rel_tol: float, max_iter: int, bucket: CgBucket | None = None):
    xp = _array_module(b)
    b0 = center_into(xp.asarray(b, dtype=_DTYPE).copy())
    x = xp.zeros_like(b0)
    ax = xp.empty_like(b0)
    matvec_into(x, ax)
    r = b0 - ax
    center_into(r)
    p = r.copy()
    rr = _dot(r, r)
    b2 = max(_dot(b0, b0), 1e-300)
    threshold = (float(rel_tol) ** 2) * b2
    if rr <= threshold:
        rel = math.sqrt(rr / b2)
        if bucket is not None:
            bucket.add(0, rel)
        return x

    rel = math.sqrt(rr / b2)
    it_done = 0
    for it in range(1, int(max_iter) + 1):
        matvec_into(p, ax)
        denom = _dot(p, ax)
        if denom <= 0.0 or not math.isfinite(denom):
            raise RuntimeError("CG operator is not numerically SPD")
        alpha = rr / denom
        x += alpha * p
        r -= alpha * ax
        center_into(r)
        rr_new = _dot(r, r)
        rel = math.sqrt(rr_new / b2)
        it_done = it
        if rr_new <= threshold:
            if bucket is not None:
                bucket.add(it_done, rel)
            return center_into(x)
        beta = rr_new / rr
        p *= beta
        p += r
        rr = rr_new

    if bucket is not None:
        bucket.add(it_done, rel)
    raise RuntimeError(f"CG did not converge after {max_iter} iterations; relative residual={rel:g}")


def mc_trial_count(n: int) -> int:
    n_int = int(n)
    if n_int < 1:
        raise ValueError(f"n must be positive, got {n}")
    return int(np.clip(round(4e9 / float(n_int * n_int)), 3, 15))


def _log_delta_from_h2(h2: float) -> float:
    return math.log((1.0 - float(h2)) / float(h2))


def _reml_log_delta_bounds() -> tuple[float, float]:
    lo = _log_delta_from_h2(_REML_MAX_H2)
    hi = _log_delta_from_h2(_REML_MIN_H2)
    return lo, hi


class GrgBoltOps:
    def __init__(self, runtime, chrom_labels: Iterable[int], timing: Mapping[str, TimingBucket] | None = None) -> None:
        import cupy as cp

        self.cp = cp
        self.runtime = runtime
        self.device = runtime.device
        self.stream = runtime.stream
        self.chrom_labels = tuple(int(label) for label in chrom_labels)
        self.timing = dict(timing or {})
        self.timing.setdefault("up", TimingBucket())
        self.timing.setdefault("down", TimingBucket())
        self.states: dict[int, ChromState] = {}
        self._managers: list[Any] = []
        self._entered = False
        self.n = 0
        self.num_samples = 0
        self.raw_m = 0
        self.used_m = 0
        self._used_offsets: dict[int, int] = {}
        self.weights_work = None
        self.sample_work = None

    def __enter__(self) -> "GrgBoltOps":
        grgs = self.runtime.grgs
        if len(grgs) != len(self.chrom_labels):
            raise ValueError(f"got {len(self.chrom_labels)} chromosome labels for {len(grgs)} runtime GRGs")
        if not grgs:
            raise ValueError("at least one chromosome artifact is required")
        self.n = int(grgs[0].num_individuals)
        self.num_samples = int(grgs[0].num_samples)
        self.raw_m = 0
        self.used_m = 0
        self._used_offsets = {}
        try:
            with self.device:
                max_m = max(int(grg.num_mutations) for grg in grgs)
                self.weights_work = self.cp.empty((max_m,), dtype=_DTYPE)
                self.sample_work = self.cp.empty((self.n,), dtype=_DTYPE)
                for label, grg in zip(self.chrom_labels, grgs, strict=True):
                    _LOGGER.info(
                        "Preparing BOLT operators for chromosome %s: samples=%d mutations=%d",
                        label,
                        int(grg.num_individuals),
                        int(grg.num_mutations),
                    )
                    up_manager = grg.prepare_matmul_cuda(direction="up", k=1, by_individual=True)
                    up_op = up_manager.__enter__()
                    self._managers.append(up_manager)
                    down_manager = grg.prepare_matmul_cuda(direction="down", k=1, by_individual=True)
                    down_op = down_manager.__enter__()
                    self._managers.append(down_manager)
                    views = {
                        "up_input": self.cp.from_dlpack(up_op.input),
                        "up_output": self.cp.from_dlpack(up_op.output),
                        "down_input": self.cp.from_dlpack(down_op.input),
                        "down_output": self.cp.from_dlpack(down_op.output),
                    }
                    self.states[int(label)] = ChromState(int(label), grg, up_op, down_op, views)
                    self.raw_m += int(grg.num_mutations)
                    _LOGGER.info("Prepared BOLT operators for chromosome %s", label)
            self._entered = True
            return self
        except Exception:
            self.close()
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        with self.device:
            for manager in reversed(self._managers):
                manager.__exit__(None, None, None)
        self._managers = []
        self.weights_work = None
        self.sample_work = None
        self._entered = False

    def _require_initialized(self, state: ChromState) -> None:
        if (
            state.used_mask is None
            or state.used_local_indices is None
            or state.mu is None
            or state.sigma is None
            or state.inv_sigma is None
            or state.mu_over_sigma is None
        ):
            raise RuntimeError("initialize_frequencies() must be called before genotype operations")

    def _timed_call(self, state: ChromState, direction: str) -> None:
        op = state.up_op if direction == "up" else state.down_op
        _result, elapsed_ms = _cuda_event_ms(self.device, self.stream, op)
        self.timing[direction].add(elapsed_ms)

    def initialize_frequencies(self) -> None:
        used_total = 0
        self._used_offsets = {}
        with self.device:
            for state in self.states.values():
                _LOGGER.info("Initializing allele frequency vectors for chromosome %s", state.label)
                views = state.views
                with self.stream:
                    views["up_input"][0].fill(1.0)
                self._timed_call(state, "up")
                with self.stream:
                    counts = views["up_output"][0]
                    sample_count = float(state.grg.num_samples)
                    used_mask = (counts > 0.0) & (counts < sample_count)
                    freq = counts / sample_count
                    mu = float(state.grg.ploidy) * freq
                    sigma = self.cp.sqrt(float(state.grg.ploidy) * freq * (1.0 - freq))
                    num_monomorphic_ref = self.cp.count_nonzero(counts == 0.0)
                    num_monomorphic_alt = self.cp.count_nonzero(counts == sample_count)
                    num_used = self.cp.count_nonzero(used_mask)
                    state.used_mask = used_mask
                    state.used_local_indices = self.cp.flatnonzero(used_mask)
                    state.mu = mu
                    state.sigma = self.cp.where(used_mask, sigma, 1.0)
                    state.inv_sigma = self.cp.where(used_mask, 1.0 / state.sigma, 0.0)
                    state.mu_over_sigma = state.mu * state.inv_sigma
                    self.stream.synchronize()
                state.num_used = int(num_used.get())
                state.num_monomorphic_ref = int(num_monomorphic_ref.get())
                state.num_monomorphic_alt = int(num_monomorphic_alt.get())
                self._used_offsets[int(state.label)] = used_total
                used_total += int(state.num_used)
                skipped = int(state.grg.num_mutations) - int(state.num_used)
                _LOGGER.info(
                    "Initialized allele frequency vectors for chromosome %s: raw=%d used=%d skipped=%d "
                    "monomorphic_ref=%d monomorphic_alt=%d",
                    state.label,
                    int(state.grg.num_mutations),
                    int(state.num_used),
                    skipped,
                    int(state.num_monomorphic_ref),
                    int(state.num_monomorphic_alt),
                )
            self.used_m = used_total
        if self.used_m == 0:
            raise ValueError("all selected SNPs are monomorphic")

    def apply_x(self, chrom: int, weights, out):
        with self.device:
            state = self.states[int(chrom)]
            self._require_initialized(state)
            weights_arr = self.cp.asarray(weights, dtype=_DTYPE)
            if int(weights_arr.size) != int(state.grg.num_mutations):
                raise ValueError(
                    f"weights for chromosome {chrom} have length {weights_arr.size}, expected {state.grg.num_mutations}"
                )
            views = state.views
            with self.stream:
                constant = self.cp.dot(state.mu_over_sigma, weights_arr)
                views["down_input"][0, :] = weights_arr
                views["down_input"][0, :] *= state.inv_sigma
            self._timed_call(state, "down")
            with self.stream:
                out[...] = views["down_output"][0]
                out -= constant
                center_into(out)
            return out

    def column(self, chrom: int, local_idx: int):
        with self.device:
            state = self.states[int(chrom)]
            self._require_initialized(state)
            idx = int(local_idx)
            if idx < 0 or idx >= int(state.grg.num_mutations):
                raise IndexError(f"local SNP index out of range for chromosome {chrom}: {local_idx}")
            if not bool(state.used_mask[idx].get()):
                raise ValueError(f"chromosome {chrom} SNP {idx} is monomorphic and was skipped")
            views = state.views
            with self.stream:
                views["down_input"][0].fill(0.0)
                views["down_input"][0, idx] = state.inv_sigma[idx]
                constant = state.mu_over_sigma[idx]
            self._timed_call(state, "down")
            with self.stream:
                out = views["down_output"][0].copy()
                out -= constant
                center_into(out)
            return out

    def scores_view(self, chrom: int, v):
        with self.device:
            state = self.states[int(chrom)]
            self._require_initialized(state)
            views = state.views
            with self.stream:
                center_into(v, out=views["up_input"][0])
                input_sum = views["up_input"][0].sum()
            self._timed_call(state, "up")
            with self.stream:
                scores = views["up_output"][0]
                scores -= state.mu * input_sum
                scores *= state.inv_sigma
            return scores

    def apply_k(self, v, *, exclude_label: int | None, out):
        with self.device:
            with self.stream:
                out.fill(0.0)
            total_used = 0
            for label, state in self.states.items():
                if exclude_label is not None and int(label) == int(exclude_label):
                    continue
                scores = self.scores_view(label, v)
                weights_work = self.weights_work[: int(state.grg.num_mutations)]
                with self.stream:
                    weights_work[...] = scores
                self.apply_x(label, weights_work, self.sample_work)
                with self.stream:
                    out += self.sample_work
                total_used += int(state.num_used)
            if total_used == 0:
                with self.stream:
                    out.fill(0.0)
                return out
            with self.stream:
                out /= float(total_used)
                center_into(out)
            return out

    def score_norm2(self, v, *, exclude_label: int | None = None) -> float:
        with self.device:
            total = self.cp.zeros((), dtype=_DTYPE)
            for label in self.states:
                if exclude_label is not None and int(label) == int(exclude_label):
                    continue
                scores = self.scores_view(label, v)
                with self.stream:
                    total += self.cp.sum(scores * scores)
            self.stream.synchronize()
            return float(total.get())

    def used_global_to_local(self, global_idx: int) -> tuple[int, int]:
        idx = int(global_idx)
        if idx < 0 or idx >= self.used_m:
            raise IndexError(f"global SNP index out of range: {global_idx}")
        for label in self.chrom_labels:
            state = self.states[label]
            self._require_initialized(state)
            offset = self._used_offsets[label]
            m = int(state.num_used)
            if offset <= idx < offset + m:
                compact_local = idx - offset
                local_idx = int(state.used_local_indices[compact_local].get())
                return label, local_idx
        raise RuntimeError(f"failed to map global SNP index {global_idx}")


def _simulate_phenotype(
    ops: GrgBoltOps,
    *,
    mode: str,
    sim_h2: float,
    n_effect_check: int,
    phenotype_rng: np.random.Generator,
    validation_rng: np.random.Generator,
) -> PhenotypeSimulation:
    metrics: OrderedDict[str, Any] = OrderedDict()
    mode_name = str(mode)
    metrics["phenotype.mode"] = mode_name

    with ops.device, ops.stream:
        if mode_name == "null":
            cp_rng = ops.cp.random.default_rng(_next_cupy_seed(phenotype_rng))
            y = cp_rng.standard_normal(ops.n, dtype=_DTYPE)
            center_into(y)
            y_var = _variance(y)
            if y_var <= 0.0 or not math.isfinite(y_var):
                raise RuntimeError("null phenotype has nonpositive empirical variance")
            with ops.stream:
                y /= math.sqrt(y_var)
                center_into(y)
            metrics["phenotype.var"] = _variance(y)
            metrics["phenotype.true_h2"] = 0.0
            return PhenotypeSimulation(y=y.copy(), metrics=metrics)

        if mode_name != "infinitesimal":
            raise ValueError(f"unknown phenotype mode {mode!r}")

        check_count = min(max(int(n_effect_check), 0), int(ops.used_m))
        chosen = (
            validation_rng.choice(int(ops.used_m), size=check_count, replace=False)
            if check_count
            else np.empty((0,), dtype=np.int64)
        )
        check_global_by_local: dict[tuple[int, int], int] = {}
        for global_idx in chosen:
            label, local_idx = ops.used_global_to_local(int(global_idx))
            check_global_by_local[(int(label), int(local_idx))] = int(global_idx)

        base_effects: dict[tuple[int, int], float] = {}
        g = ops.cp.zeros((ops.n,), dtype=_DTYPE)
        scale = 1.0 / math.sqrt(float(ops.used_m))
        for label, state in ops.states.items():
            cp_rng = ops.cp.random.default_rng(_next_cupy_seed(phenotype_rng))
            weights = cp_rng.standard_normal(int(state.grg.num_mutations), dtype=_DTYPE)
            with ops.stream:
                ops.cp.multiply(weights, state.used_mask, out=weights)
                weights *= scale
                for (check_label, local_idx), _global_idx in check_global_by_local.items():
                    if int(check_label) == int(label):
                        base_effects[(int(label), int(local_idx))] = float(weights[int(local_idx)].get())
            ops.apply_x(label, weights, ops.sample_work)
            with ops.stream:
                g += ops.sample_work

        with ops.stream:
            center_into(g)
        g_var = _variance(g)
        if g_var <= 0.0 or not math.isfinite(g_var):
            raise RuntimeError("infinitesimal genetic component has nonpositive empirical variance")

        cp_rng = ops.cp.random.default_rng(_next_cupy_seed(phenotype_rng))
        e = cp_rng.standard_normal(ops.n, dtype=_DTYPE)
        with ops.stream:
            center_into(e)
            e -= (_dot(e, g) / _dot(g, g)) * g
            center_into(e)
        noise_var = _variance(e)
        if noise_var <= 0.0 or not math.isfinite(noise_var):
            raise RuntimeError("infinitesimal noise component has nonpositive empirical variance")

        genetic_scale = math.sqrt(float(sim_h2) / g_var)
        noise_scale = math.sqrt((1.0 - float(sim_h2)) / noise_var)
        with ops.stream:
            g *= genetic_scale
            e *= noise_scale
            y = g + e
            center_into(y)

        genetic_var = _variance(g)
        scaled_noise_var = _variance(e)
        y_var = _variance(y)
        genetic_noise_dot = _dot(g, e)
        metrics["phenotype.var"] = y_var
        metrics["phenotype.true_h2"] = float(sim_h2)
        metrics["phenotype.requested_h2"] = float(sim_h2)
        metrics["phenotype.empirical_h2"] = 0.0 if y_var <= 0.0 else genetic_var / y_var
        metrics["phenotype.genetic_var"] = genetic_var
        metrics["phenotype.noise_var"] = scaled_noise_var
        metrics["phenotype.genetic_noise_dot"] = genetic_noise_dot

        effect_snps = tuple(
            EffectCheckSnp(
                global_idx=int(global_idx),
                label=int(label),
                local_idx=int(local_idx),
                true_beta=float(base_effects[(int(label), int(local_idx))] * genetic_scale),
            )
            for (label, local_idx), global_idx in sorted(check_global_by_local.items(), key=lambda item: item[1])
        )
        return PhenotypeSimulation(y=y.copy(), metrics=metrics, effect_snps=effect_snps)


def _make_random_components(ops: GrgBoltOps, rng: np.random.Generator, trials: int):
    seed = int(rng.integers(0, np.iinfo(np.int64).max))
    with ops.device:
        cp_rng = ops.cp.random.default_rng(seed)
        g_rand = []
        e_rand = []
        scale = 1.0 / math.sqrt(float(ops.used_m))
        for trial in range(int(trials)):
            _LOGGER.info("Preparing REML Monte Carlo component %d/%d", trial + 1, int(trials))
            g = ops.cp.zeros((ops.n,), dtype=_DTYPE)
            for label, state in ops.states.items():
                weights = cp_rng.standard_normal(int(state.grg.num_mutations), dtype=_DTYPE) * scale
                ops.apply_x(label, weights, ops.sample_work)
                with ops.stream:
                    g += ops.sample_work
                with ops.stream:
                    center_into(g)
                    e = cp_rng.standard_normal(ops.n, dtype=_DTYPE)
                    center_into(e)
            g_rand.append(g)
            e_rand.append(e)
        return tuple(g_rand), tuple(e_rand)


def _estimate_variance_components(
    ops: GrgBoltOps,
    y,
    *,
    rng: np.random.Generator,
    rel_tol: float,
    max_iter: int,
    bucket: CgBucket,
) -> tuple[float, float, float, float, int]:
    trials = mc_trial_count(ops.n)
    _LOGGER.info("Starting REML variance component estimation: n=%d m=%d mc_trials=%d", ops.n, ops.used_m, trials)
    with ops.device, ops.stream:
        g_rand, e_rand = _make_random_components(ops, rng, trials)
        work = ops.cp.empty((ops.n,), dtype=_DTYPE)
        objective_evals = 0

        def beta2_e2(y0, delta: float) -> tuple[float, float]:
            def h_into(src, dst) -> None:
                ops.apply_k(src, exclude_label=None, out=dst)
                with ops.stream:
                    dst += float(delta) * centered(src)

            z = cg_solve(h_into, y0, rel_tol=rel_tol, max_iter=max_iter, bucket=bucket)
            beta2 = ops.score_norm2(z) / float(ops.used_m * ops.used_m)
            e2 = (float(delta) ** 2) * _dot(z, z)
            return beta2, e2

        def objective(log_delta: float) -> float:
            nonlocal objective_evals
            objective_evals += 1
            delta = math.exp(float(log_delta))
            _LOGGER.info("REML secant evaluation %d: log_delta=%g delta=%g", objective_evals, log_delta, delta)
            b2_data, e2_data = beta2_e2(y, delta)
            b2_rand_sum = 0.0
            e2_rand_sum = 0.0
            sqrt_delta = math.sqrt(delta)
            for g_t, e_t in zip(g_rand, e_rand, strict=True):
                with ops.stream:
                    work[...] = e_t
                    work *= sqrt_delta
                    work += g_t
                    center_into(work)
                b2_t, e2_t = beta2_e2(work, delta)
                b2_rand_sum += b2_t
                e2_rand_sum += e2_t
            if min(b2_data, e2_data, b2_rand_sum, e2_rand_sum) <= 0.0:
                raise RuntimeError("invalid REML objective component")
            value = math.log((b2_data / e2_data) / (b2_rand_sum / e2_rand_sum))
            _LOGGER.info("REML secant evaluation %d complete: objective=%g", objective_evals, value)
            return value

        x0 = _log_delta_from_h2(0.25)
        f0 = objective(x0)
        x1 = _log_delta_from_h2(0.125 if f0 < 0.0 else 0.5)
        f1 = objective(x1)
        lo, hi = _reml_log_delta_bounds()
        best_x, best_abs_f = (x0, abs(f0)) if abs(f0) <= abs(f1) else (x1, abs(f1))
        for _ in range(5):
            if abs(f1 - f0) < 1e-12:
                break
            x2 = (x0 * f1 - x1 * f0) / (f1 - f0)
            x2 = float(np.clip(x2, lo, hi))
            f2 = objective(x2)
            if abs(f2) < best_abs_f:
                best_x, best_abs_f = x2, abs(f2)
            if abs(x2 - x1) < 0.01:
                best_x = x2
                break
            x0, f0, x1, f1 = x1, f1, x2, f2

        delta = math.exp(float(best_x))

        def h_final_into(src, dst) -> None:
            ops.apply_k(src, exclude_label=None, out=dst)
            with ops.stream:
                dst += delta * centered(src)

        z = cg_solve(h_final_into, y, rel_tol=rel_tol, max_iter=max_iter, bucket=bucket)
        sigma_g2 = _dot(y, z) / float(ops.n - 1)
        sigma_e2 = delta * sigma_g2
        h2 = 1.0 / (1.0 + delta)
        if sigma_g2 <= 0.0 or sigma_e2 <= 0.0:
            raise RuntimeError("invalid variance component estimate")
        _LOGGER.info(
            "REML variance components: sigma_g2=%g sigma_e2=%g h2=%g delta=%g",
            sigma_g2,
            sigma_e2,
            h2,
            delta,
        )
        return sigma_g2, sigma_e2, h2, delta, trials


def _solve_loco_residuals(
    ops: GrgBoltOps,
    y,
    *,
    sigma_g2: float,
    sigma_e2: float,
    rel_tol: float,
    max_iter: int,
    bucket: CgBucket,
) -> dict[int, Any]:
    residuals: dict[int, Any] = {}
    _LOGGER.info("Starting LOCO residual solves for %d chromosome(s)", len(ops.chrom_labels))
    with ops.device, ops.stream:
        for label in ops.chrom_labels:
            _LOGGER.info("Solving LOCO residuals for chromosome %s", label)
            def v_into(src, dst, left_out=label) -> None:
                ops.apply_k(src, exclude_label=int(left_out), out=dst)
                with ops.stream:
                    dst *= float(sigma_g2)
                    dst += float(sigma_e2) * centered(src)

            residuals[int(label)] = cg_solve(v_into, y, rel_tol=rel_tol, max_iter=max_iter, bucket=bucket)
            _LOGGER.info("Finished LOCO residuals for chromosome %s", label)
    return residuals


def _calibrate_cinf(
    ops: GrgBoltOps,
    residuals: Mapping[int, Any],
    *,
    sigma_g2: float,
    sigma_e2: float,
    rng: np.random.Generator,
    n_calib: int,
    rel_tol: float,
    max_iter: int,
    bucket: CgBucket,
) -> float:
    count = min(int(n_calib), int(ops.used_m))
    if count < 1:
        raise ValueError("at least one SNP is required for calibration")
    chosen = rng.choice(int(ops.used_m), size=count, replace=False)
    _LOGGER.info("Starting BOLT-LMM-inf calibration with %d SNP(s)", count)
    sum_score2 = 0.0
    sum_prospective = 0.0
    with ops.device, ops.stream:
        for ordinal, global_idx in enumerate(chosen, start=1):
            label, local_idx = ops.used_global_to_local(int(global_idx))
            _LOGGER.info(
                "Calibration SNP %d/%d start: global=%d chr=%s local=%d",
                ordinal,
                count,
                int(global_idx),
                label,
                local_idx,
            )
            x = ops.column(label, local_idx)

            def v_into(src, dst, left_out=label) -> None:
                ops.apply_k(src, exclude_label=int(left_out), out=dst)
                with ops.stream:
                    dst *= float(sigma_g2)
                    dst += float(sigma_e2) * centered(src)

            q = cg_solve(v_into, x, rel_tol=rel_tol, max_iter=max_iter, bucket=bucket)
            dot = _dot(x, residuals[int(label)])
            score2 = dot * dot
            denom = _dot(x, q)
            if denom <= 0.0 or not math.isfinite(denom):
                raise RuntimeError("nonpositive prospective denominator during calibration")
            sum_score2 += score2
            sum_prospective += score2 / denom
            _LOGGER.info("Calibration SNP %d/%d complete: denominator=%g", ordinal, count, denom)
    c_inf = sum_score2 / sum_prospective
    if not math.isfinite(c_inf) or c_inf <= 0.0:
        raise RuntimeError("invalid BOLT-LMM-inf calibration constant")
    _LOGGER.info("Finished BOLT-LMM-inf calibration: c_inf=%g", c_inf)
    return c_inf


def _histogram_ks_approx(counts: np.ndarray) -> float:
    total = int(np.sum(counts))
    if total == 0:
        return 0.0
    cdf_right = np.cumsum(counts, dtype=np.float64) / float(total)
    edges_right = np.arange(1, int(counts.size) + 1, dtype=np.float64) / float(counts.size)
    cdf_left = np.concatenate(([0.0], cdf_right[:-1]))
    edges_left = np.arange(0, int(counts.size), dtype=np.float64) / float(counts.size)
    return float(max(np.max(np.abs(cdf_right - edges_right)), np.max(np.abs(cdf_left - edges_left))))


def _histogram_quantile(counts: np.ndarray, q: float) -> float:
    total = int(np.sum(counts))
    if total == 0:
        return float("nan")
    target = float(q) * float(total)
    cumulative = 0.0
    bin_count = int(counts.size)
    for idx, raw_count in enumerate(counts):
        count = float(raw_count)
        if count <= 0.0:
            continue
        if cumulative + count >= target:
            within = 0.0 if count == 0.0 else (target - cumulative) / count
            return float((float(idx) + float(np.clip(within, 0.0, 1.0))) / float(bin_count))
        cumulative += count
    return 1.0


def _scan_metrics(
    ops: GrgBoltOps,
    residuals: Mapping[int, Any],
    c_inf: float,
    *,
    top_k: int,
) -> OrderedDict[str, Any]:
    from cupyx.scipy.special import erfc as cp_erfc
    from scipy.stats import chi2 as scipy_chi2

    metrics: OrderedDict[str, Any] = OrderedDict()
    genome_hist = np.zeros((_P_HIST_BINS,), dtype=np.int64)
    top_candidates: list[tuple[float, int, int, float]] = []
    genome_sum = 0.0
    genome_count = 0
    genome_max = 0.0
    requested_top_k = max(int(top_k), 0)
    _LOGGER.info("Starting chromosome scan")
    with ops.device, ops.stream:
        hist_edges = ops.cp.linspace(0.0, 1.0, _P_HIST_BINS + 1, dtype=_DTYPE)
        for label in ops.chrom_labels:
            _LOGGER.info("Scanning chromosome %s", label)
            state = ops.states[int(label)]
            scores = ops.scores_view(label, residuals[int(label)])
            used_chi2 = scores[state.used_mask]
            used_chi2 *= used_chi2
            used_chi2 /= float(c_inf)
            count = int(state.num_used)
            if count:
                p_values = cp_erfc(ops.cp.sqrt(ops.cp.maximum(used_chi2, 0.0) / 2.0))
                sum_chi2 = _to_float(used_chi2.sum())
                max_chi2 = _to_float(used_chi2.max())
                min_p = _to_float(p_values.min())
                hist = ops.cp.histogram(p_values, bins=hist_edges)[0]
                genome_hist += ops.cp.asnumpy(hist).astype(np.int64, copy=False)
                if requested_top_k:
                    k = min(requested_top_k, count)
                    if k == count:
                        compact_top = ops.cp.argsort(-used_chi2)[:k]
                    else:
                        partition = ops.cp.argpartition(used_chi2, count - k)[count - k :]
                        compact_top = partition[ops.cp.argsort(-used_chi2[partition])]
                    raw_local = state.used_local_indices[compact_top]
                    for chi2_value, raw_idx, p_value in zip(
                        ops.cp.asnumpy(used_chi2[compact_top]),
                        ops.cp.asnumpy(raw_local),
                        ops.cp.asnumpy(p_values[compact_top]),
                        strict=True,
                    ):
                        top_candidates.append((float(chi2_value), int(label), int(raw_idx), float(p_value)))
            else:
                sum_chi2 = 0.0
                max_chi2 = 0.0
                min_p = 1.0
            mean_chi2 = 0.0 if count == 0 else sum_chi2 / float(count)
            metrics[f"scan.chr{label}.num_snps"] = count
            metrics[f"scan.chr{label}.sum_chi2"] = sum_chi2
            metrics[f"scan.chr{label}.mean_chi2"] = mean_chi2
            metrics[f"scan.chr{label}.max_chi2"] = max_chi2
            metrics[f"scan.chr{label}.min_p"] = min_p
            genome_sum += sum_chi2
            genome_count += count
            genome_max = max(genome_max, max_chi2)
            _LOGGER.info("Scanned chromosome %s: snps=%d mean_chi2=%g max_chi2=%g", label, count, mean_chi2, max_chi2)
    metrics["scan.genome.num_snps"] = genome_count
    metrics["scan.genome.sum_chi2"] = genome_sum
    metrics["scan.genome.mean_chi2"] = 0.0 if genome_count == 0 else genome_sum / float(genome_count)
    metrics["scan.genome.max_chi2"] = genome_max
    metrics["scan.genome.min_p"] = math.erfc(math.sqrt(max(genome_max, 0.0) / 2.0))
    metrics["scan.p_hist.bin_count"] = _P_HIST_BINS
    for idx, count in enumerate(genome_hist):
        metrics[f"scan.p_hist.bin{idx}.count"] = int(count)
    metrics["scan.p_hist.ks_approx"] = _histogram_ks_approx(genome_hist)
    median_p = _histogram_quantile(genome_hist, 0.5)
    if math.isnan(median_p):
        lambda_gc = float("nan")
    else:
        median_p = float(np.clip(median_p, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)))
        lambda_gc = float(scipy_chi2.isf(median_p, df=1) / _CHI2_MEDIAN_DF1)
    metrics["scan.lambda_gc_approx"] = lambda_gc
    if requested_top_k:
        top_candidates.sort(key=lambda row: row[0], reverse=True)
        for rank, (chi2_value, label, local_idx, p_value) in enumerate(top_candidates[:requested_top_k], start=1):
            metrics[f"scan.top{rank}.chr"] = int(label)
            metrics[f"scan.top{rank}.local_idx"] = int(local_idx)
            metrics[f"scan.top{rank}.chi2"] = float(chi2_value)
            metrics[f"scan.top{rank}.p"] = float(p_value)
    return metrics


def _effect_check_metrics(
    ops: GrgBoltOps,
    residuals: Mapping[int, Any],
    effect_snps: Iterable[EffectCheckSnp],
    *,
    sigma_g2: float,
    sigma_e2: float,
    rel_tol: float,
    max_iter: int,
    bucket: CgBucket,
) -> OrderedDict[str, Any]:
    metrics: OrderedDict[str, Any] = OrderedDict()
    snps = tuple(effect_snps)
    metrics["effect_check.count"] = len(snps)
    if not snps:
        return metrics

    beta_hat: list[float] = []
    beta_true: list[float] = []
    _LOGGER.info("Starting sampled effect check with %d SNP(s)", len(snps))
    with ops.device, ops.stream:
        for ordinal, snp in enumerate(snps, start=1):
            _LOGGER.info(
                "Effect-check SNP %d/%d start: global=%d chr=%s local=%d",
                ordinal,
                len(snps),
                int(snp.global_idx),
                int(snp.label),
                int(snp.local_idx),
            )
            x = ops.column(int(snp.label), int(snp.local_idx))

            def v_into(src, dst, left_out=snp.label) -> None:
                ops.apply_k(src, exclude_label=int(left_out), out=dst)
                with ops.stream:
                    dst *= float(sigma_g2)
                    dst += float(sigma_e2) * centered(src)

            q = cg_solve(v_into, x, rel_tol=rel_tol, max_iter=max_iter, bucket=bucket)
            denom = _dot(x, q)
            if denom <= 0.0 or not math.isfinite(denom):
                raise RuntimeError("nonpositive prospective denominator during effect check")
            beta_hat.append(_dot(x, residuals[int(snp.label)]) / denom)
            beta_true.append(float(snp.true_beta))
            _LOGGER.info("Effect-check SNP %d/%d complete: denominator=%g", ordinal, len(snps), denom)

    estimated = np.asarray(beta_hat, dtype=np.float64)
    truth = np.asarray(beta_true, dtype=np.float64)
    diff = estimated - truth
    truth_norm2 = float(np.dot(truth, truth))
    estimated_std = float(np.std(estimated))
    truth_std = float(np.std(truth))
    metrics["effect_check.beta_corr"] = (
        float(np.corrcoef(truth, estimated)[0, 1]) if len(snps) >= 2 and truth_std > 0.0 and estimated_std > 0.0 else float("nan")
    )
    metrics["effect_check.beta_slope"] = float(np.dot(truth, estimated) / truth_norm2) if truth_norm2 > 0.0 else float("nan")
    metrics["effect_check.beta_rmse"] = float(math.sqrt(float(np.mean(diff * diff))))
    metrics["effect_check.beta_mae"] = float(np.mean(np.abs(diff)))
    metrics["effect_check.sign_concordance"] = float(np.mean(np.sign(estimated) == np.sign(truth)))
    return metrics


def _write_summary(path: Path, metrics: Mapping[str, Any]) -> None:
    out = Path(path)
    _LOGGER.info("Writing summary metrics to %s", out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["metric\tvalue"]
    for key, value in metrics.items():
        lines.append(f"{key}\t{value}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _LOGGER.info("Wrote %d summary metrics to %s", len(metrics), out)


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="BOLT-LMM-inf GRG cuSPARSE benchmark")
    parser.add_argument(
        "--summary-file",
        required=True,
        type=Path,
        help="Required key/value TSV summary output path; parent directories are created and no per-SNP association file is written.",
    )
    parser.add_argument(
        "--grg-dir",
        type=Path,
        default=None,
        help=f"Raw chr*.grg discovery root used with --artifact-cache. Default: {_DEFAULT_GRG_DIR}",
    )
    parser.add_argument(
        "--artifact-cache",
        type=Path,
        default=None,
        help="Required cache root when converting or discovering raw GRGs.",
    )
    parser.add_argument(
        "--artifacts",
        nargs="+",
        type=Path,
        default=None,
        help="Direct .grg_spmv inputs; mutually exclusive with --grg-dir.",
    )
    parser.add_argument(
        "--chromosomes",
        default="all",
        help="'all' or comma-separated numeric labels; unlabeled artifacts require explicit labels. Default: all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base seed split into phenotype, analysis/calibration, and validation RNGs.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device ordinal.")
    parser.add_argument("--stream", type=int, default=0, help="CUDA stream handle.")
    parser.add_argument(
        "--vram-budget-bytes",
        type=int,
        default=None,
        help=(
            "Owned device-memory budget in bytes. Omit to probe free VRAM and reserve BOLT-side device arrays; "
            "0 requests planner full residency without probing."
        ),
    )
    parser.add_argument(
        "--ring-buffer-size",
        type=int,
        default=0,
        help="Streamed sparse block ring slots; resident layouts may allocate no ring slots.",
    )
    parser.add_argument(
        "--cg-tol",
        type=float,
        default=5e-4,
        help="Relative CG residual tolerance for REML, LOCO, calibration, and effect-check solves.",
    )
    parser.add_argument(
        "--cg-max-iter",
        type=int,
        default=10_000,
        help="Maximum CG iterations for REML, LOCO, calibration, and effect-check solves.",
    )
    parser.add_argument(
        "--n-calib",
        type=int,
        default=30,
        help="Random SNP count used to estimate c_inf, capped by the polymorphic SNP count.",
    )
    parser.add_argument(
        "--phenotype-mode",
        choices=("null", "infinitesimal"),
        default="null",
        help="Simulate either null noise or an infinitesimal genetic signal.",
    )
    parser.add_argument(
        "--sim-h2",
        type=float,
        default=0.3,
        help="Requested infinitesimal heritability; only meaningful with --phenotype-mode infinitesimal.",
    )
    parser.add_argument(
        "--n-effect-check",
        type=int,
        default=30,
        help="Infinitesimal-only sampled true-effect diagnostic count; 0 disables it.",
    )
    parser.add_argument(
        "--scan-top-k",
        type=int,
        default=20,
        help="Number of top scan hits emitted in the summary; 0 disables top-hit fields.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
        help="Package logging threshold. Default: INFO.",
    )
    args = parser.parse_args(argv)
    if args.artifacts is not None and args.grg_dir is not None:
        parser.error("--artifacts is mutually exclusive with --grg-dir")
    if args.cg_tol <= 0.0:
        parser.error("--cg-tol must be positive")
    if args.cg_max_iter < 1:
        parser.error("--cg-max-iter must be >= 1")
    if args.n_calib < 1:
        parser.error("--n-calib must be >= 1")
    if not (0.0 < args.sim_h2 < 1.0):
        parser.error("--sim-h2 must satisfy 0 < sim_h2 < 1")
    if args.n_effect_check < 0:
        parser.error("--n-effect-check must be >= 0")
    if args.scan_top_k < 0:
        parser.error("--scan-top-k must be >= 0")
    if args.ring_buffer_size < 0:
        parser.error("--ring-buffer-size must be >= 0")
    if args.vram_budget_bytes is not None and int(args.vram_budget_bytes) < 0:
        parser.error("--vram-budget-bytes must be >= 0")
    if args.artifacts is None and args.artifact_cache is None:
        parser.error("--artifact-cache is required when using --grg-dir")
    return args


def _validate_scans(selected: tuple[tuple[int, Path], ...]):
    scans = tuple(scan_grg_spmv(path) for _label, path in selected)
    first = scans[0]
    for (label, path), scan in zip(selected, scans, strict=True):
        if int(scan.num_individuals) != int(first.num_individuals):
            raise ValueError(f"chromosome {label} artifact {path} has mismatched num_individuals")
        if int(scan.num_samples) != int(first.num_samples):
            raise ValueError(f"chromosome {label} artifact {path} has mismatched num_samples")
        if int(scan.ploidy) != 2:
            raise ValueError(f"chromosome {label} artifact {path} has ploidy={scan.ploidy}; expected 2")
        if bool(scan.has_missing_data):
            raise ValueError(f"chromosome {label} artifact {path} has missing data; complete hard calls are required")
    return scans


def _add_cg_metrics(metrics: OrderedDict[str, Any], prefix: str, bucket: CgBucket) -> None:
    metrics[f"{prefix}.solves"] = int(bucket.solves)
    metrics[f"{prefix}.iterations"] = int(bucket.iterations)
    metrics[f"{prefix}.max_iterations"] = int(bucket.max_iterations)
    metrics[f"{prefix}.max_rel_resid"] = float(bucket.max_rel_resid)


def main(argv: list[str] | None = None) -> None:
    from pygrgl_spmv.backends.cusparse import CusparseRuntime, plan_cusparse_layout

    total_start = perf_counter()
    args = _parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
    _LOGGER.setLevel(getattr(logging, str(args.log_level)))
    setup_seconds: OrderedDict[str, float] = OrderedDict()

    start = perf_counter()
    _LOGGER.info("Resolving chromosome artifacts")
    selected = _resolve_artifacts(
        grg_dir=args.grg_dir,
        artifact_cache=args.artifact_cache,
        artifacts=args.artifacts,
        chromosomes=str(args.chromosomes),
    )
    setup_seconds["artifact_resolution_conversion"] = perf_counter() - start
    if not selected:
        raise ValueError("no chromosome artifacts selected")
    labels = tuple(label for label, _path in selected)
    artifacts = tuple(path for _label, path in selected)
    _LOGGER.info("Resolved %d chromosome artifact(s): %s", len(selected), ",".join(str(label) for label in labels))
    _LOGGER.info("Validating artifact metadata")
    scans = _validate_scans(selected)
    _LOGGER.info(
        "Validated artifact metadata: individuals=%d samples=%d total_mutations=%d",
        int(scans[0].num_individuals),
        int(scans[0].num_samples),
        int(sum(int(scan.num_mutations) for scan in scans)),
    )

    bolt_reservation_bytes = _bolt_device_reservation_bytes(
        scans,
        phenotype_mode=str(args.phenotype_mode),
        scan_top_k=int(args.scan_top_k),
    )
    if args.vram_budget_bytes is None:
        start = perf_counter()
        _LOGGER.info("Probing free memory on CUDA device %d", int(args.device))
        probed_free_vram_bytes = _free_memory_bytes(int(args.device))
        setup_seconds["free_memory_probe"] = perf_counter() - start
        if probed_free_vram_bytes <= bolt_reservation_bytes:
            raise RuntimeError(
                f"CUDA device {args.device} has {probed_free_vram_bytes} free bytes, "
                f"but BOLT-side device arrays reserve {bolt_reservation_bytes} bytes"
            )
        layout_budget_request = int(probed_free_vram_bytes - bolt_reservation_bytes)
        _LOGGER.info("Probed free VRAM: %d bytes", int(probed_free_vram_bytes))
    else:
        layout_budget_request = int(args.vram_budget_bytes)
        _LOGGER.info("Skipping free VRAM probe because --vram-budget-bytes was provided")
    _LOGGER.info("BOLT-side device reservation: %d bytes", int(bolt_reservation_bytes))
    _LOGGER.info("Layout budget request: %d bytes", int(layout_budget_request))

    requirements = RuntimeRequirements(
        max_k_up=1,
        max_k_down=1,
        need_down_miss_input=False,
        need_up_miss_output=False,
        need_init_vector=False,
        need_init_matrix=False,
        need_init_xtx=False,
    )
    start = perf_counter()
    _LOGGER.info("Planning cuSPARSE layout with ring_buffer_size=%d", int(args.ring_buffer_size))
    layout = plan_cusparse_layout(
        artifacts=artifacts,
        pair=parse_cusparse_plan(DEFAULT_CUSPARSE_PLAN_NAME),
        dtype=_DTYPE,
        requirements=requirements,
        vram_budget_bytes=int(layout_budget_request),
        ring_buffer_size=int(args.ring_buffer_size),
        allow_residency=True,
        device=int(args.device),
        stream=args.stream,
    )
    setup_seconds["layout_planning"] = perf_counter() - start
    _LOGGER.info(
        "Planned cuSPARSE layout: bytes_total=%d resident_sparse=%d ring_slots=%d "
        "allocated_ring_buffer_size=%d effective_vram_budget_bytes=%d",
        int(layout.bytes_total),
        int(layout.bytes_by_category.get("resident_sparse", 0)),
        int(layout.bytes_by_category.get("ring_slots", 0)),
        int(layout.allocated_ring_buffer_size),
        int(layout.vram_budget_bytes),
    )

    timing = {"up": TimingBucket(), "down": TimingBucket()}
    cg_buckets = {"reml": CgBucket(), "loco": CgBucket(), "calibration": CgBucket(), "effect_check": CgBucket()}
    phenotype_seed, analysis_seed, validation_seed = np.random.SeedSequence(int(args.seed)).spawn(3)
    phenotype_rng = np.random.default_rng(phenotype_seed)
    analysis_rng = np.random.default_rng(analysis_seed)
    validation_rng = np.random.default_rng(validation_seed)
    phenotype_metrics: OrderedDict[str, Any] = OrderedDict()
    scan_metrics: OrderedDict[str, Any] = OrderedDict()
    effect_check_metrics: OrderedDict[str, Any] = OrderedDict()

    start = perf_counter()
    _LOGGER.info("Initializing cuSPARSE runtime on CUDA device %d", int(args.device))
    runtime_cm = CusparseRuntime(layout)
    with runtime_cm as runtime:
        setup_seconds["runtime_initialization"] = perf_counter() - start
        _LOGGER.info("Initialized cuSPARSE runtime")
        with GrgBoltOps(runtime, labels, timing) as ops:
            start = perf_counter()
            ops.initialize_frequencies()
            setup_seconds["frequency_setup"] = perf_counter() - start
            start = perf_counter()
            simulation = _simulate_phenotype(
                ops,
                mode=str(args.phenotype_mode),
                sim_h2=float(args.sim_h2),
                n_effect_check=int(args.n_effect_check),
                phenotype_rng=phenotype_rng,
                validation_rng=validation_rng,
            )
            setup_seconds["phenotype_simulation"] = perf_counter() - start
            phenotype_metrics = simulation.metrics
            sigma_g2, sigma_e2, h2, delta, mc_trials = _estimate_variance_components(
                ops,
                simulation.y,
                rng=analysis_rng,
                rel_tol=float(args.cg_tol),
                max_iter=int(args.cg_max_iter),
                bucket=cg_buckets["reml"],
            )
            residuals = _solve_loco_residuals(
                ops,
                simulation.y,
                sigma_g2=sigma_g2,
                sigma_e2=sigma_e2,
                rel_tol=float(args.cg_tol),
                max_iter=int(args.cg_max_iter),
                bucket=cg_buckets["loco"],
            )
            c_inf = _calibrate_cinf(
                ops,
                residuals,
                sigma_g2=sigma_g2,
                sigma_e2=sigma_e2,
                rng=analysis_rng,
                n_calib=int(args.n_calib),
                rel_tol=float(args.cg_tol),
                max_iter=int(args.cg_max_iter),
                bucket=cg_buckets["calibration"],
            )
            scan_metrics = _scan_metrics(ops, residuals, c_inf, top_k=int(args.scan_top_k))
            if simulation.effect_snps:
                effect_check_metrics = _effect_check_metrics(
                    ops,
                    residuals,
                    simulation.effect_snps,
                    sigma_g2=sigma_g2,
                    sigma_e2=sigma_e2,
                    rel_tol=float(args.cg_tol),
                    max_iter=int(args.cg_max_iter),
                    bucket=cg_buckets["effect_check"],
                )

    total_elapsed = perf_counter() - total_start

    metrics: OrderedDict[str, Any] = OrderedDict()
    metrics["backend"] = "cusparse"
    metrics["chromosomes"] = ",".join(str(label) for label in labels)
    metrics["num_individuals"] = int(scans[0].num_individuals)
    metrics["num_samples"] = int(scans[0].num_samples)
    metrics["num_mutations.raw"] = int(sum(int(scan.num_mutations) for scan in scans))
    metrics["num_mutations.used"] = int(ops.used_m)
    metrics["num_mutations.monomorphic_skipped"] = int(ops.raw_m - ops.used_m)
    metrics["artifact_count"] = len(artifacts)
    metrics["seed"] = int(args.seed)
    metrics["cg_tol"] = float(args.cg_tol)
    metrics["cg_max_iter"] = int(args.cg_max_iter)
    metrics["n_effect_check"] = int(args.n_effect_check)
    metrics["scan_top_k"] = int(args.scan_top_k)
    metrics.update(phenotype_metrics)
    metrics["reml.sigma_g2"] = sigma_g2
    metrics["reml.sigma_e2"] = sigma_e2
    metrics["reml.h2"] = h2
    metrics["reml.delta"] = delta
    metrics["reml.mc_trials"] = int(mc_trials)
    metrics["calibration.c_inf"] = c_inf
    metrics["calibration.num_snps_requested"] = int(args.n_calib)
    metrics["calibration.num_snps_effective"] = min(int(args.n_calib), int(ops.used_m))
    for label in labels:
        state = ops.states[int(label)]
        metrics[f"chr{label}.num_mutations.raw"] = int(state.grg.num_mutations)
        metrics[f"chr{label}.num_mutations.used"] = int(state.num_used)
        metrics[f"chr{label}.num_mutations.monomorphic_skipped"] = int(state.grg.num_mutations) - int(state.num_used)
        metrics[f"chr{label}.num_mutations.monomorphic_ref"] = int(state.num_monomorphic_ref)
        metrics[f"chr{label}.num_mutations.monomorphic_alt"] = int(state.num_monomorphic_alt)
    for key, seconds in setup_seconds.items():
        metrics[f"setup.{key}.seconds"] = float(seconds)
    metrics["setup.total_elapsed.seconds"] = float(total_elapsed)
    metrics["layout.bytes.total"] = int(layout.bytes_total)
    for key, value in layout.bytes_by_category.items():
        metrics[f"layout.bytes.{key}"] = int(value)
    _add_cg_metrics(metrics, "cg.reml", cg_buckets["reml"])
    _add_cg_metrics(metrics, "cg.loco", cg_buckets["loco"])
    _add_cg_metrics(metrics, "cg.calibration", cg_buckets["calibration"])
    _add_cg_metrics(metrics, "cg.effect_check", cg_buckets["effect_check"])
    metrics["matmul.up.k1.calls"] = int(timing["up"].calls)
    metrics["matmul.up.k1.total_ms"] = float(timing["up"].total_ms)
    metrics["matmul.up.k1.avg_ms"] = float(timing["up"].avg_ms)
    metrics["matmul.down.k1.calls"] = int(timing["down"].calls)
    metrics["matmul.down.k1.total_ms"] = float(timing["down"].total_ms)
    metrics["matmul.down.k1.avg_ms"] = float(timing["down"].avg_ms)
    metrics.update(scan_metrics)
    metrics.update(effect_check_metrics)
    _write_summary(args.summary_file, metrics)


if __name__ == "__main__":
    main()
