"""Shared benchmark helpers for backend-specific benchmark scripts."""

from __future__ import annotations

import argparse
import gc
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter

import numpy as np

DTYPE = np.float64
INDEX_DTYPE = np.int64
DEFAULT_MATMUL_OPTIONS = (
    "baseline",
    "by_individual",
    "init_xtx",
    "init_vector",
    "init_matrix",
    "miss",
)
LOGGER = logging.getLogger("scripts.bench")
OUTPUT_ATOL = 1e-8
OUTPUT_RTOL = 1e-5
OUTPUT_ATOL_FLOAT32 = 1e-2
OUTPUT_RTOL_FLOAT32 = 1e-1
DEFAULT_GRG_PATH = "/pscratch/sd/q/qys/grg/simulation-mutation-200m.trees.v4.igd.final.grg"
LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR")
_PLAN_KEYS = ("k_hint", "store", "fmt", "opA", "opB", "orderB", "orderC", "algo")
_PAIR_LITERAL_RE = re.compile(r"^\[(.*?)\]\[(.*?)\]$")

PlanSpec = dict[str, str]
PlanPairSpec = tuple[PlanSpec | None, PlanSpec | None]


@dataclass(frozen=True)
class CommonBenchArgs:
    grg: str
    ks: list[int]
    plan_pair_specs: list[PlanPairSpec]
    options: list[str]
    n_trials: int
    n_warmup: int
    dtype: np.dtype
    index_dtype: np.dtype
    output_atol: float
    output_rtol: float
    log_level: str
    dry_run: bool
    skip_note: bool


def add_common_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--grg", default=DEFAULT_GRG_PATH)
    parser.add_argument(
        "--ks",
        type=str,
        default="32",
        help="Comma-separated runtime-k values (input rows); e.g. 1,4,16",
    )
    parser.add_argument("--trials", type=int, default=10, help="Number of timed trials")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument(
        "--plan-up-down",
        action="append",
        type=parse_plan_pair_literal,
        default=None,
        help="Repeated explicit plan-pair literal [..][..]",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=list(LOG_LEVEL_CHOICES),
        help="Log level passed to SpmvGRG and backend",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=np.dtype(DTYPE).name,
        choices=["float32", "float64"],
        help="Input/output floating dtype",
    )
    parser.add_argument(
        "--index-dtype",
        type=str,
        default=np.dtype(INDEX_DTYPE).name,
        choices=["int32", "int64"],
        help="Index dtype used by SpmvGRG",
    )
    parser.add_argument(
        "--matmul-options",
        type=str,
        default="all",
        help="Comma-separated: baseline,by_individual,init_xtx,init_vector,init_matrix,miss,all",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print expanded backend configs and exit")
    parser.add_argument("--skip-note", action="store_true", help="Hide the Note column in the summary table")


def parse_common_bench_args(args: argparse.Namespace) -> CommonBenchArgs:
    ks = parse_csv_ints(args.ks, "--ks")
    plan_pair_specs = list(args.plan_up_down or [])
    if not plan_pair_specs:
        raise ValueError("--plan-up-down must be provided at least once")
    options = parse_matmul_options(args.matmul_options)
    dtype = parse_dtype(args.dtype)
    index_dtype = parse_index_dtype(args.index_dtype)
    output_atol, output_rtol = tolerances_for_dtype(dtype)
    return CommonBenchArgs(
        grg=str(args.grg),
        ks=ks,
        plan_pair_specs=plan_pair_specs,
        options=options,
        n_trials=int(args.trials),
        n_warmup=int(args.warmup),
        dtype=dtype,
        index_dtype=index_dtype,
        output_atol=output_atol,
        output_rtol=output_rtol,
        log_level=str(args.log_level),
        dry_run=bool(args.dry_run),
        skip_note=bool(args.skip_note),
    )


def configure_logging(log_level: str) -> None:
    level_name = str(log_level).upper()
    level = getattr(logging, level_name, logging.WARNING)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    else:
        root.setLevel(level)
    LOGGER.setLevel(level)


def progress(msg: str) -> None:
    LOGGER.info("[bench] %s", msg)


def _parse_plan_group(body: str, *, raw: str) -> PlanSpec | None:
    if body == "":
        return None
    mapping: PlanSpec = {}
    for part in (chunk.strip() for chunk in body.split(",") if chunk.strip()):
        if "=" not in part:
            raise ValueError(f"Invalid plan field {part!r} in {raw!r}")
        key, value = (token.strip() for token in part.split("=", 1))
        if not key or not value:
            raise ValueError(f"Invalid plan field {part!r} in {raw!r}")
        if key in mapping:
            raise ValueError(f"Duplicate plan field {key!r} in {raw!r}")
        mapping[key] = value
    return mapping


def parse_plan_pair_literal(raw: str) -> PlanPairSpec:
    value = "".join(str(raw).split())
    match = _PAIR_LITERAL_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid --plan-up-down literal {raw!r}; expected [..][..]")
    pair = (_parse_plan_group(match.group(1), raw=raw), _parse_plan_group(match.group(2), raw=raw))
    if pair == (None, None):
        raise ValueError("Invalid --plan-up-down literal; both sides cannot be empty")
    return pair


def parse_csv_ints(raw: str, field_name: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        tok = token.strip()
        if not tok:
            continue
        value = int(tok)
        if value < 0:
            raise ValueError(f"{field_name} must contain non-negative integers, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{field_name} must contain at least one value")
    return values


def parse_dtype(raw: str) -> np.dtype:
    token = str(raw).strip().lower()
    if token == "float32":
        return np.dtype(np.float32)
    if token == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"--dtype must be one of float32,float64; got {raw!r}")


def parse_index_dtype(raw: str) -> np.dtype:
    token = str(raw).strip().lower()
    if token == "int32":
        return np.dtype(np.int32)
    if token == "int64":
        return np.dtype(np.int64)
    raise ValueError(f"--index-dtype must be one of int32,int64; got {raw!r}")


def tolerances_for_dtype(dtype: np.dtype) -> tuple[float, float]:
    dt = np.dtype(dtype)
    if dt == np.float32:
        return OUTPUT_ATOL_FLOAT32, OUTPUT_RTOL_FLOAT32
    if dt == np.float64:
        return OUTPUT_ATOL, OUTPUT_RTOL
    raise ValueError(f"Unsupported benchmark dtype for tolerance: {dt}")


def parse_matmul_options(raw: str) -> list[str]:
    tokens = [tok.strip().lower() for tok in raw.split(",") if tok.strip()]
    if not tokens or "all" in tokens:
        return list(DEFAULT_MATMUL_OPTIONS)
    valid = set(DEFAULT_MATMUL_OPTIONS)
    unknown = sorted(tok for tok in tokens if tok not in valid)
    if unknown:
        raise ValueError(f"Unknown matmul option(s): {', '.join(unknown)}")
    return tokens


def _spec_to_literal(spec: PlanSpec) -> str:
    return "[" + ",".join(f"{key}={spec[key]}" for key in spec) + "]"


def _spec_has_pattern(spec: PlanSpec | None) -> bool:
    return spec is not None and any(value == "*" or str(value).startswith("!") for value in spec.values())


def _render_plan(plan) -> str:
    if plan is None:
        return "<unspecified>"
    return str(plan)


def _config_label(backend: str, plan_up, plan_down) -> str:
    up = _render_plan(plan_up)
    down = _render_plan(plan_down)
    return f"{backend}-up={up}-down={down}"


def _config_entry(backend: str, plan_up, plan_down, *, log_level: str) -> dict[str, object]:
    return {
        "label": _config_label(backend, plan_up, plan_down),
        "config": {
            "type": backend,
            "plan_up": plan_up,
            "plan_down": plan_down,
            "log_level": str(log_level).upper(),
        },
    }


def expand_mkl_configs(
    plan_pair_specs: list[PlanPairSpec],
    log_level: str,
) -> list[dict[str, object]]:
    from pygrgl_spmv.backends.mkl import MklPlan

    configs: list[dict[str, object]] = []
    for up_spec, down_spec in plan_pair_specs:
        if _spec_has_pattern(up_spec) or _spec_has_pattern(down_spec):
            raise ValueError("MKL benchmark plans must be fully concrete; wildcard/negation expansion is cuSPARSE-only for now")
        configs.append(
            _config_entry(
                "mkl",
                None if up_spec is None else MklPlan.from_any(up_spec),
                None if down_spec is None else MklPlan.from_any(down_spec),
                log_level=log_level,
            )
        )
    return configs


def _cusparse_runtime_supported(plan) -> bool:
    return plan.supported and not (plan.fmt == plan.fmt.CSC and plan.algo == plan.algo.CSR_ALG3)


def _expand_cusparse_side(spec: PlanSpec | None, *, want_up: bool):
    import pygrgl
    from pygrgl_spmv.backends.cusparse import CusparsePlan

    if spec is None:
        return [None]
    direction = pygrgl.TraversalDirection.UP if want_up else pygrgl.TraversalDirection.DOWN
    return [
        plan
        for plan in CusparsePlan.expand_literal(_spec_to_literal(spec))
        if _cusparse_runtime_supported(plan) and plan.direction == direction
    ]


def expand_cusparse_configs(
    plan_pair_specs: list[PlanPairSpec],
    log_level: str,
) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for up_spec, down_spec in plan_pair_specs:
        for plan_up in _expand_cusparse_side(up_spec, want_up=True):
            for plan_down in _expand_cusparse_side(down_spec, want_up=False):
                configs.append(
                    _config_entry(
                        "cusparse",
                        plan_up,
                        plan_down,
                        log_level=log_level,
                    )
                )
    return configs


def format_dry_run_line(entry, ks, options, *, dtype, index_dtype):
    cfg = entry["config"]
    backend = str(cfg.get("type", ""))
    parts = [
        entry["label"],
        f"ks={','.join(str(k) for k in ks)}",
        f"options={','.join(options)}",
        f"dtype={np.dtype(dtype).name}",
        f"index_dtype={np.dtype(index_dtype).name}",
    ]
    if "plan_up" in cfg:
        parts.append(f"plan_up={_render_plan(cfg['plan_up'])}")
    if "plan_down" in cfg:
        parts.append(f"plan_down={_render_plan(cfg['plan_down'])}")
    return " ".join(parts)



def _bytes_to_gib(value: int | float) -> float:
    return float(value) / float(1024 ** 3)


def _format_static_note(prefix: str, values) -> str:
    parts = []
    for f in fields(type(values)):
        value = int(getattr(values, f.name))
        if value > 0:
            parts.append((f.name, value))
    if not parts:
        return f"{prefix}: none"
    body = ", ".join(f"{name}={_bytes_to_gib(value):.6f}GiB" for name, value in parts)
    return f"{prefix}: {body}"


def _skip_summary_row(*, label: str, scenario: str, direction: str, k: int, reason: str) -> dict[str, object]:
    return {
        "config": label,
        "scenario": scenario,
        "direction": direction,
        "k": int(k),
        "skip": reason,
        "call_ms_mean": None,
        "call_ms_std": None,
        "host_gib": None,
        "device_gib": None,
        "note": f"skip: {reason}",
    }


def _slug_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")
    return token or "value"


def _compare_outputs(
    ref: np.ndarray,
    arr: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, bool, float | None, float | None]:
    if ref.shape != arr.shape:
        return False, False, None, None
    abs_diff = np.abs(arr - ref)
    max_abs = float(np.max(abs_diff))
    denom = np.maximum(np.abs(ref), 1e-30)
    max_rel = float(np.max(abs_diff / denom))
    return bool(np.allclose(arr, ref, atol=atol, rtol=rtol)), True, max_abs, max_rel


def _validate_and_extract_runtime_memory(
    *,
    call_slice,
    direction: str,
    k: int,
    n_warmup: int,
    n_trials: int,
) -> tuple[int, int, str]:
    expected = n_warmup + n_trials
    if len(call_slice) != expected:
        raise AssertionError(
            f"Expected {expected} memory records for direction={direction}, k={k}, got {len(call_slice)}"
        )

    expected_stage = "run_up" if direction == "up" else "run_down"
    for idx, record in enumerate(call_slice):
        if record.stage != expected_stage:
            raise AssertionError(
                f"Unexpected stage in memory record {idx}: expected {expected_stage}, got {record.stage}"
            )
        if int(record.runtime_k) != int(k):
            raise AssertionError(
                f"Unexpected runtime_k in memory record {idx}: expected {k}, got {record.runtime_k}"
            )
        rec_direction = str(record.meta.get("direction", ""))
        if rec_direction != direction:
            raise AssertionError(
                f"Unexpected direction in memory record {idx}: expected {direction}, got {rec_direction}"
            )

    timed_records = call_slice[n_warmup:]
    if len(timed_records) != n_trials:
        raise AssertionError(
            f"Expected {n_trials} timed memory records for direction={direction}, k={k}, got {len(timed_records)}"
        )

    host_values = [int(record.host.total()) for record in timed_records]
    device_values = [int(record.device.total()) for record in timed_records]
    modes = [str(record.meta.get("mode", "n/a")) for record in timed_records]

    if len(set(host_values)) != 1 or len(set(device_values)) != 1:
        raise AssertionError(
            f"Memory totals vary across timed trials for direction={direction}, k={k}: "
            f"host={host_values}, device={device_values}"
        )
    if len(set(modes)) != 1:
        raise AssertionError(f"Execution mode varies across timed trials for direction={direction}, k={k}: {modes}")

    return host_values[0], device_values[0], modes[0]


def _configured_direction_names(backend) -> list[str]:
    if hasattr(backend, "_configured_directions"):
        return [str(direction.value) for direction in backend._configured_directions()]

    directions = []
    _missing = object()
    backend_plan_up = getattr(backend, "_plan_up", _missing)
    backend_plan_down = getattr(backend, "_plan_down", _missing)
    if backend_plan_up is _missing and backend_plan_down is _missing:
        return ["up", "down"]
    if backend_plan_up is not None and backend_plan_up is not _missing:
        directions.append("up")
    if backend_plan_down is not None and backend_plan_down is not _missing:
        directions.append("down")
    return directions


def benchmark_config(
    *,
    op,
    label: str,
    ks: list[int],
    options: list[str],
    n_trials: int,
    n_warmup: int,
    seed_base: int,
    output_dir: Path,
    dtype: np.dtype,
    output_atol: float,
    output_rtol: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs_by_k: dict[int, dict[str, np.ndarray | None]] = {}
    for k in ks:
        rng = np.random.default_rng(seed_base + int(k))
        inputs: dict[str, np.ndarray | None] = {
            "up_sample": rng.standard_normal((k, op.n), dtype=dtype),
            "down": rng.standard_normal((k, op.m), dtype=dtype),
            "init_vec": rng.standard_normal(k, dtype=dtype),
            "init_mat": rng.standard_normal((k, op.K), dtype=dtype),
            "miss_down": rng.standard_normal((k, op.m), dtype=dtype),
            "miss_up": np.zeros((k, op.m), dtype=dtype),
            "up_indiv": None,
        }
        if op.num_individuals != op.n:
            inputs["up_indiv"] = rng.standard_normal((k, op.num_individuals), dtype=dtype)
        inputs_by_k[int(k)] = inputs

    summary_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []

    host_static = op._backend.mem_usage.host_static
    device_static = op._backend.mem_usage.device_static
    summary_rows.append(
        {
            "config": label,
            "scenario": "static",
            "direction": "-",
            "k": None,
            "call_ms_mean": None,
            "call_ms_std": None,
            "host_gib": _bytes_to_gib(host_static.total()),
            "device_gib": _bytes_to_gib(device_static.total()),
            "note": _format_static_note("actual_host", host_static)
            + " | "
            + _format_static_note("actual_device", device_static),
        }
    )

    est_host, est_device = op._backend.estimate_static_bytes()
    summary_rows.append(
        {
            "config": label,
            "scenario": "static_est",
            "direction": "-",
            "k": None,
            "call_ms_mean": None,
            "call_ms_std": None,
            "host_gib": _bytes_to_gib(est_host.total()),
            "device_gib": _bytes_to_gib(est_device.total()),
            "note": _format_static_note("est_host", est_host) + " | " + _format_static_note("est_device", est_device),
        }
    )

    directions = _configured_direction_names(op._backend)

    for scenario in options:
        progress(f"{label}: scenario={scenario} start")
        for direction in directions:
            for k in ks:
                inputs = inputs_by_k[int(k)]
                matrix: np.ndarray | None = None
                skip_reason: str | None = None

                if scenario == "baseline":
                    matrix = inputs["up_sample"] if direction == "up" else inputs["down"]
                    kwargs_factory = lambda: {}
                elif scenario == "by_individual":
                    up_indiv = inputs["up_indiv"]
                    if up_indiv is None:
                        skip_reason = "num_individuals == num_samples"
                        kwargs_factory = lambda: {}
                    else:
                        matrix = up_indiv if direction == "up" else inputs["down"]
                        kwargs_factory = lambda: {"by_individual": True}
                elif scenario == "init_xtx":
                    matrix = inputs["up_sample"] if direction == "up" else inputs["down"]
                    kwargs_factory = lambda: {"init": "xtx"}
                elif scenario == "init_vector":
                    init_vec = inputs["init_vec"]
                    matrix = inputs["up_sample"] if direction == "up" else inputs["down"]
                    kwargs_factory = lambda init_vec=init_vec: {"init": init_vec}
                elif scenario == "init_matrix":
                    init_mat = inputs["init_mat"]
                    matrix = inputs["up_sample"] if direction == "up" else inputs["down"]
                    kwargs_factory = lambda init_mat=init_mat: {"init": init_mat}
                elif scenario == "miss":
                    if op.sel_miss.nnz == 0:
                        skip_reason = "GRG has no missingness selector entries"
                        kwargs_factory = lambda: {}
                    elif direction == "up":
                        miss_up = inputs["miss_up"]

                        def kwargs_factory(miss_up=miss_up):
                            assert isinstance(miss_up, np.ndarray)
                            miss_up.fill(0.0)
                            return {"miss": miss_up}

                        matrix = inputs["up_sample"]
                    else:
                        miss_down = inputs["miss_down"]
                        matrix = inputs["down"]
                        kwargs_factory = lambda miss_down=miss_down: {"miss": miss_down}
                else:
                    raise ValueError(f"Unhandled matmul scenario: {scenario}")

                if skip_reason is not None:
                    summary_rows.append(_skip_summary_row(label=label, scenario=scenario, direction=direction, k=int(k), reason=skip_reason))
                    continue

                if direction not in {"up", "down"}:
                    raise ValueError(f"Unknown direction {direction!r}")
                assert isinstance(matrix, np.ndarray)

                progress(f"{label}: scenario={scenario} direction={direction} k={k} start")

                calls = op._backend.mem_usage.calls
                start_idx = len(calls)
                ref_output: np.ndarray | None = None
                intra_errors = 0
                intra_trials = 0
                intra_fail_indices: list[str] = []
                intra_abs_sum = 0.0
                intra_rel_sum = 0.0
                intra_abs_max = 0.0
                intra_rel_max = 0.0
                intra_numeric_count = 0

                for warm_idx in range(n_warmup):
                    warm = np.asarray(op.matmul(matrix, direction, **kwargs_factory()))
                    if ref_output is None:
                        ref_output = warm
                    else:
                        intra_trials += 1
                        ok, shape_ok, max_abs, max_rel = _compare_outputs(
                            ref_output,
                            warm,
                            atol=output_atol,
                            rtol=output_rtol,
                        )
                        if shape_ok:
                            assert max_abs is not None and max_rel is not None
                            intra_numeric_count += 1
                            intra_abs_sum += max_abs
                            intra_rel_sum += max_rel
                            intra_abs_max = max(intra_abs_max, max_abs)
                            intra_rel_max = max(intra_rel_max, max_rel)
                        if not ok:
                            intra_errors += 1
                            intra_fail_indices.append(f"w{warm_idx + 1}")

                times: list[float] = []
                for trial_idx in range(n_trials):
                    t0 = perf_counter()
                    result = np.asarray(op.matmul(matrix, direction, **kwargs_factory()))
                    times.append(perf_counter() - t0)
                    if ref_output is None:
                        ref_output = result
                    else:
                        intra_trials += 1
                        ok, shape_ok, max_abs, max_rel = _compare_outputs(
                            ref_output,
                            result,
                            atol=output_atol,
                            rtol=output_rtol,
                        )
                        if shape_ok:
                            assert max_abs is not None and max_rel is not None
                            intra_numeric_count += 1
                            intra_abs_sum += max_abs
                            intra_rel_sum += max_rel
                            intra_abs_max = max(intra_abs_max, max_abs)
                            intra_rel_max = max(intra_rel_max, max_rel)
                        if not ok:
                            intra_errors += 1
                            intra_fail_indices.append(f"b{trial_idx + 1}")

                end_idx = len(calls)
                host_bytes, device_bytes, mode = _validate_and_extract_runtime_memory(
                    call_slice=calls[start_idx:end_idx],
                    direction=direction,
                    k=int(k),
                    n_warmup=n_warmup,
                    n_trials=n_trials,
                )

                assert ref_output is not None
                ms = np.asarray(times, dtype=np.float64) * 1000.0
                mean_ms = float(np.mean(ms))
                std_ms = float(np.std(ms))
                abs_err_avg = None if intra_numeric_count == 0 else float(intra_abs_sum / intra_numeric_count)
                rel_err_avg = None if intra_numeric_count == 0 else float(intra_rel_sum / intra_numeric_count)
                abs_err_max = None if intra_numeric_count == 0 else float(intra_abs_max)
                rel_err_max = None if intra_numeric_count == 0 else float(intra_rel_max)

                config_token = _slug_token(label)
                scenario_token = _slug_token(scenario)
                direction_token = _slug_token(direction)
                output_path = output_dir / f"{len(output_rows):06d}_{config_token}_{scenario_token}_{direction_token}_k{int(k)}.npy"
                np.save(output_path, ref_output, allow_pickle=False)

                summary_rows.append(
                    {
                        "config": label,
                        "scenario": scenario,
                        "direction": direction,
                        "k": int(k),
                        "path": str(output_path),
                        "call_ms_mean": mean_ms,
                        "call_ms_std": std_ms,
                        "host_gib": _bytes_to_gib(host_bytes),
                        "device_gib": _bytes_to_gib(device_bytes),
                        "note": f"mode={mode}",
                        "intra_errors": int(intra_errors),
                        "intra_trials": int(intra_trials),
                        "intra_fail_indices": intra_fail_indices,
                        "abs_err_avg": abs_err_avg,
                        "abs_err_max": abs_err_max,
                        "rel_err_avg": rel_err_avg,
                        "rel_err_max": rel_err_max,
                    }
                )
                output_rows.append(
                    {
                        "config": label,
                        "scenario": scenario,
                        "direction": direction,
                        "k": int(k),
                        "path": str(output_path),
                    }
                )

    return summary_rows, output_rows


def print_summary_table(rows: list[dict[str, object]], *, skip_note: bool = False) -> None:
    if not rows:
        print("No benchmark results.")
        return

    title = "BENCHMARK SUMMARY (time + memory + correctness diagnostics)"
    rendered: list[tuple[str, str, str, str, str, str, str, str, str, str]] = []
    for row in rows:
        config = str(row["config"])
        scenario = str(row["scenario"])
        direction = str(row["direction"])
        k_value = row.get("k")
        k_cell = "-" if k_value is None else str(int(k_value))
        if scenario in {"static", "static_est"}:
            call_cell = "-"
            err_cell = "-"
            abs_err_cell = "-"
            rel_err_cell = "-"
        elif "skip" in row:
            call_cell = f"SKIP ({row['skip']})"
            err_cell = "-"
            abs_err_cell = "-"
            rel_err_cell = "-"
        else:
            call_cell = f"{float(row['call_ms_mean']):.4f}+/-{float(row['call_ms_std']):.4f}"
            intra_errors = int(row.get("intra_errors", 0))
            intra_trials = int(row.get("intra_trials", 0))
            err_cell = f"{intra_errors}/{intra_trials}"
            abs_avg = row.get("abs_err_avg")
            abs_max = row.get("abs_err_max")
            rel_avg = row.get("rel_err_avg")
            rel_max = row.get("rel_err_max")
            abs_err_cell = "-" if abs_avg is None or abs_max is None else f"{float(abs_avg):.3e}/{float(abs_max):.3e}"
            rel_err_cell = "-" if rel_avg is None or rel_max is None else f"{float(rel_avg):.3e}/{float(rel_max):.3e}"
        host = row.get("host_gib")
        device = row.get("device_gib")
        host_cell = "SKIP" if host is None else f"{float(host):.6f}"
        device_cell = "SKIP" if device is None else f"{float(device):.6f}"
        note = str(row.get("note", ""))
        intra_fail = row.get("intra_fail_indices", [])
        tags: list[str] = []
        if isinstance(intra_fail, list) and intra_fail:
            tags.append(f"intra_fail={','.join(str(v) for v in intra_fail)}")
        if tags:
            note = f"{note} | {' | '.join(tags)}" if note else " | ".join(tags)
        rendered.append(
            (config, scenario, direction, k_cell, call_cell, err_cell, abs_err_cell, rel_err_cell, host_cell, device_cell, note)
        )

    config_w = max(len("Config"), max(len(item[0]) for item in rendered))
    scenario_w = max(len("Scenario"), max(len(item[1]) for item in rendered))
    direction_w = max(len("Direction"), max(len(item[2]) for item in rendered))
    k_w = max(len("k"), max(len(item[3]) for item in rendered))
    call_w = max(len("Call ms"), max(len(item[4]) for item in rendered))
    err_w = max(len("Err/Trials"), max(len(item[5]) for item in rendered))
    abs_w = max(len("Abs Err (avg/max)"), max(len(item[6]) for item in rendered))
    rel_w = max(len("Rel Err (avg/max)"), max(len(item[7]) for item in rendered))
    host_w = max(len("Host GiB"), max(len(item[8]) for item in rendered))
    device_w = max(len("Device GiB"), max(len(item[9]) for item in rendered))

    header = f"{'Config':<{config_w}} {'Scenario':<{scenario_w}} "
    header += f"{'Direction':<{direction_w}} {'k':>{k_w}} {'Call ms':>{call_w}} "
    header += f"{'Err/Trials':>{err_w}} {'Abs Err (avg/max)':>{abs_w}} {'Rel Err (avg/max)':>{rel_w}} "
    header += f"{'Host GiB':>{host_w}} {'Device GiB':>{device_w}}"
    if not skip_note:
        header += " Note"
    width = max(len(title), len(header))

    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(header)
    print("-" * width)

    for config, scenario, direction, k_cell, call_cell, err_cell, abs_err_cell, rel_err_cell, host_cell, device_cell, note in rendered:
        row = (
            f"{config:<{config_w}} {scenario:<{scenario_w}} "
            f"{direction:<{direction_w}} {k_cell:>{k_w}} {call_cell:>{call_w}} "
            f"{err_cell:>{err_w}} {abs_err_cell:>{abs_w}} {rel_err_cell:>{rel_w}} "
            f"{host_cell:>{host_w}} {device_cell:>{device_w}}"
        )
        if not skip_note:
            row += f" {note}"
        print(row)


def evaluate_output_equivalence(
    rows: list[dict[str, object]],
    *,
    atol: float = OUTPUT_ATOL,
    rtol: float = OUTPUT_RTOL,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    classes: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        scenario = str(row["scenario"])
        direction = str(row["direction"])
        k = int(row["k"])
        key = ("baseline" if scenario == "miss" and direction == "up" else scenario, direction, k)
        classes.setdefault(key, []).append(row)

    compared = 0
    checked_classes = 0
    errors = 0
    per_config: dict[str, dict[str, int]] = {}
    for key in sorted(classes):
        points = classes[key]
        if len(points) < 2:
            continue
        checked_classes += 1
        arr_cache: dict[str, np.ndarray] = {}
        for i in range(len(points) - 1):
            row_i = points[i]
            cfg_i = str(row_i["config"])
            path_i = str(row_i["path"])
            if cfg_i not in per_config:
                per_config[cfg_i] = {"failures": 0, "trials": 0}
            arr_i = arr_cache.get(path_i)
            if arr_i is None:
                arr_i = np.load(Path(path_i), allow_pickle=False)
                arr_cache[path_i] = arr_i
            for j in range(i + 1, len(points)):
                row_j = points[j]
                cfg_j = str(row_j["config"])
                path_j = str(row_j["path"])
                if cfg_j not in per_config:
                    per_config[cfg_j] = {"failures": 0, "trials": 0}
                arr_j = arr_cache.get(path_j)
                if arr_j is None:
                    arr_j = np.load(Path(path_j), allow_pickle=False)
                    arr_cache[path_j] = arr_j
                compared += 1
                per_config[cfg_i]["trials"] += 1
                per_config[cfg_j]["trials"] += 1
                ok_ab, _, _, _ = _compare_outputs(
                    arr_i,
                    arr_j,
                    atol=atol,
                    rtol=rtol,
                )
                ok_ba, _, _, _ = _compare_outputs(
                    arr_j,
                    arr_i,
                    atol=atol,
                    rtol=rtol,
                )
                if not (ok_ab and ok_ba):
                    errors += 1
                    per_config[cfg_i]["failures"] += 1
                    per_config[cfg_j]["failures"] += 1

    print(
        "\nOutput equivalence diagnostics: "
        f"classes={checked_classes}, comparisons={compared}, errors={errors}, atol={atol}, rtol={rtol}"
    )
    return {"classes": checked_classes, "comparisons": compared, "errors": errors}, per_config


def summarize_intra_diagnostics(rows: list[dict[str, object]]) -> tuple[int, int]:
    errors = 0
    trials = 0
    for row in rows:
        scenario = str(row.get("scenario", ""))
        if scenario in {"static", "static_est"} or "skip" in row:
            continue
        errors += int(row.get("intra_errors", 0))
        trials += int(row.get("intra_trials", 0))
    return errors, trials


def run_benchmark_suite(
    *,
    grg_path: str,
    configs: list[dict[str, object]],
    ks: list[int],
    options: list[str],
    n_trials: int,
    n_warmup: int,
    dtype: np.dtype,
    index_dtype: np.dtype,
    output_atol: float,
    output_rtol: float,
    skip_note: bool,
    seed_base: int = 2026,
) -> None:
    from pygrgl_spmv import SpmvGRG

    progress(f"GRG file: {grg_path}")
    all_summary_rows: list[dict[str, object]] = []
    all_outputs: list[dict[str, object]] = []
    output_dir = Path(tempfile.mkdtemp(prefix="pygrgl_spmv_bench_refs_"))
    keep_output_dir = False
    try:
        for entry in configs:
            label = entry["label"]
            cfg = entry["config"]
            assert isinstance(label, str)
            assert isinstance(cfg, dict)

            progress(f"{'=' * 72}")
            progress(f"loading operator {label}")
            t_load = perf_counter()
            op = SpmvGRG(grg_path, cfg, dtype, index_dtype)
            progress(f"{label}: operator ready in {(perf_counter() - t_load) * 1000.0:.2f} ms")

            cfg_rows, cfg_outputs = benchmark_config(
                op=op,
                label=label,
                ks=ks,
                options=options,
                n_trials=n_trials,
                n_warmup=n_warmup,
                seed_base=seed_base,
                output_dir=output_dir,
                dtype=dtype,
                output_atol=output_atol,
                output_rtol=output_rtol,
            )
            all_summary_rows.extend(cfg_rows)
            all_outputs.extend(cfg_outputs)
            del op
            gc.collect()

        cross_stats, per_config_cross = evaluate_output_equivalence(all_outputs, atol=output_atol, rtol=output_rtol)
        intra_errors, intra_trials = summarize_intra_diagnostics(all_summary_rows)
        print_summary_table(all_summary_rows, skip_note=skip_note)
        print(
            "\nCorrectness diagnostics: "
            f"intra_errors={intra_errors}/{intra_trials}, "
            f"cross_errors={cross_stats['errors']}/{cross_stats['comparisons']}"
        )
        config_order: dict[str, int] = {}
        for idx, entry in enumerate(configs):
            label = str(entry["label"])
            if label not in config_order:
                config_order[label] = idx
        print("Cross failures by config:")
        ordered_cfgs = sorted(
            config_order.keys(),
            key=lambda cfg: (-int(per_config_cross.get(cfg, {}).get("failures", 0)), config_order[cfg]),
        )
        for cfg in ordered_cfgs:
            stats = per_config_cross.get(cfg, {"failures": 0, "trials": 0})
            failures = int(stats.get("failures", 0))
            trials = int(stats.get("trials", 0))
            rate = 0.0 if trials == 0 else 100.0 * float(failures) / float(trials)
            print(f"  {cfg}: {failures}/{trials} ({rate:.1f}%)")
    except Exception:
        keep_output_dir = True
        print(f"\nSaved benchmark reference outputs to: {output_dir}")
        raise
    finally:
        if not keep_output_dir:
            shutil.rmtree(output_dir, ignore_errors=True)


__all__ = [
    "DTYPE",
    "INDEX_DTYPE",
    "DEFAULT_GRG_PATH",
    "LOG_LEVEL_CHOICES",
    "DEFAULT_MATMUL_OPTIONS",
    "OUTPUT_ATOL",
    "OUTPUT_RTOL",
    "OUTPUT_ATOL_FLOAT32",
    "OUTPUT_RTOL_FLOAT32",
    "CommonBenchArgs",
    "add_common_bench_args",
    "configure_logging",
    "progress",
    "parse_common_bench_args",
    "parse_csv_ints",
    "parse_plan_pair_literal",
    "parse_dtype",
    "parse_index_dtype",
    "tolerances_for_dtype",
    "parse_matmul_options",
    "expand_mkl_configs",
    "expand_cusparse_configs",
    "format_dry_run_line",
    "_validate_and_extract_runtime_memory",
    "benchmark_config",
    "run_benchmark_suite",
    "print_summary_table",
    "evaluate_output_equivalence",
    "summarize_intra_diagnostics",
]
