"""pygrgl_spmv - sparse matmul for GRG-based genotype matrices."""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
import os
from pathlib import Path

import numpy as np

from pygrgl_spmv.grg import SpmvGRG, convert

_CONFIG_ENV_VAR = "PYGRGL_SPMV_CONFIG"
_LOGGER = logging.getLogger(__name__)
_KNOWN_ROOT_KEYS = ("backend", "reference", "mkl", "cusparse", "triton")
_BACKEND_SPECS = {
    "reference": {
        "section_keys": ("log_level", "up", "down"),
        "plan_keys": ("store", "fmt", "k_hint"),
    },
    "mkl": {
        "section_keys": ("log_level", "up", "down"),
        "plan_keys": ("store", "fmt", "n_threads", "k_hint"),
    },
    "cusparse": {
        "section_keys": ("device", "stream", "ring_buffer_size", "log_level", "up", "down"),
        "plan_keys": ("k_hint", "store", "fmt", "opA", "opB", "orderB", "orderC", "algo", "scratch"),
    },
    "triton": {
        "section_keys": ("device", "stream", "ring_buffer_size", "log_level", "up", "down"),
        "plan_keys": ("k_hint", "store", "fmt", "scratch"),
    },
}


def load(
    source,
    dtype=np.float64,
    *,
    artifact_dir: str | Path = "pygrgl_spmv_artifacts",
    shared_slot_pool=None,
) -> SpmvGRG:
    """Load a SpmvGRG using an explicit backend JSON config.

    ``PYGRGL_SPMV_CONFIG`` must point to a JSON file. Bundled sample configs in
    ``pygrgl_spmv.configs`` show the exact schema; each backend section mirrors
    the corresponding backend plan dictionaries and requires all fields
    explicitly.
    """
    config_path = os.environ.get(_CONFIG_ENV_VAR)
    if config_path is None or not str(config_path).strip():
        raise ValueError(
            f"{_CONFIG_ENV_VAR} must point to a JSON config file. "
            "Use one of the bundled samples: reference-default.json, "
            "mkl-default.json, cusparse-default.json, or triton-default.json."
        )
    config_path = str(config_path).strip()
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    backend_name, section, plan_up, plan_down = _parse_backend_config(config)
    backend = _build_backend(
        backend_name, section, plan_up, plan_down, shared_slot_pool=shared_slot_pool
    )
    _LOGGER.info("Selected %s backend from %s", backend_name, config_path)
    return SpmvGRG(source, backend, dtype, artifact_dir=artifact_dir)


def load_shared_slot_pool(
    sources,
    dtype=np.float64,
    *,
    ring_buffer_size=None,
    artifact_dir: str | Path = "pygrgl_spmv_artifacts",
):
    """Create a SharedSlotPool sized for multiple sources using the env config.

    Reads ``PYGRGL_SPMV_CONFIG`` to determine the cuSPARSE plan pair, device,
    and default ring buffer size.  Loads each source's compiled operator state,
    computes the maximum slot buffer dimensions across all sources, and
    pre-allocates the pool on the GPU.

    The returned pool is passed to :func:`load` via ``shared_slot_pool=``.
    All :func:`load` calls that share this pool must use the same backend
    configuration and the same ``ring_buffer_size``.  The pool must outlive all
    :class:`SpmvGRG` instances that reference it.

    **Sequential use only.** Backends sharing a pool must call ``matmul``
    sequentially — never from concurrent threads.  Concurrent access corrupts
    the shared slot buffers and causes ``CUDA_ERROR_ILLEGAL_ADDRESS``.  For
    concurrent workloads, create one pool per concurrent worker.

    Args:
        sources: Iterable of paths to ``.grg`` or ``.grg_spmv`` files, or
            loaded ``pygrgl.ImmutableGRG`` objects.
        dtype: Floating-point dtype. Default: ``float64``.
        ring_buffer_size: Number of reusable sparse-structure slots.
            ``None`` reads the value from the env config (default ``2``).
        artifact_dir: Directory for caching compiled ``.grg_spmv`` artifacts.

    Returns:
        A :class:`~pygrgl_spmv.backends.cusparse.SharedSlotPool` instance.

    Raises:
        RuntimeError: If ``PYGRGL_SPMV_CONFIG`` is not set.
        RuntimeError: If the configured backend is not ``"cusparse"``.
    """
    from pygrgl_spmv.backends.cusparse import cusparse_shared_slot_pool

    config_path = os.environ.get(_CONFIG_ENV_VAR)
    if config_path is None:
        raise RuntimeError(
            f"{_CONFIG_ENV_VAR} is not set. load_shared_slot_pool requires an explicit "
            "cuSPARSE configuration file."
        )
    with open(config_path) as f:
        config = json.load(f)
    backend_name = str(config.get("backend", "")).lower()
    if backend_name != "cusparse":
        raise RuntimeError(
            f"load_shared_slot_pool requires backend='cusparse' in the {_CONFIG_ENV_VAR} "
            f"config, got {backend_name!r}."
        )

    pair, device, cfg_rbs = _cusparse_params_from_config(config)
    effective_rbs = int(ring_buffer_size) if ring_buffer_size is not None else cfg_rbs

    dtype = np.dtype(dtype)
    compiled = [_load_compiled_state(s, dtype, artifact_dir) for s in sources]
    setups = [c.to_backend_setup(dtype) for c in compiled]

    return cusparse_shared_slot_pool(
        setups, pair=pair, ring_buffer_size=effective_rbs, device=device,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _require_keys(
    mapping: Mapping[str, object],
    *,
    label: str,
    required_keys: tuple[str, ...],
    allowed_keys: tuple[str, ...] | None = None,
) -> None:
    present = set(mapping)
    required = set(required_keys)
    allowed = required if allowed_keys is None else set(allowed_keys)
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"{label} is missing required field(s): {missing}")
    extra = sorted(present - allowed)
    if extra:
        raise ValueError(f"{label} has unknown field(s): {extra}")


def _require_optional_plan(
    value: object,
    *,
    label: str,
    required_keys: tuple[str, ...],
) -> dict[str, object] | None:
    if value is None:
        return None
    plan = _require_mapping(value, label=label)
    _require_keys(plan, label=label, required_keys=required_keys)
    return plan


def _parse_backend_config(
    config: object,
) -> tuple[str, dict[str, object], dict[str, object] | None, dict[str, object] | None]:
    root = _require_mapping(config, label="config")
    _require_keys(root, label="config", required_keys=("backend",), allowed_keys=_KNOWN_ROOT_KEYS)

    backend_name = str(root["backend"]).strip().lower()
    if backend_name not in _BACKEND_SPECS:
        raise ValueError(
            f"config backend must be one of {sorted(_BACKEND_SPECS)}, got {root['backend']!r}"
        )
    if backend_name not in root:
        raise ValueError(f"config is missing required '{backend_name}' section")

    spec = _BACKEND_SPECS[backend_name]
    section = _require_mapping(root[backend_name], label=f"{backend_name} config")
    _require_keys(section, label=f"{backend_name} config", required_keys=spec["section_keys"])

    plan_up = _require_optional_plan(
        section["up"],
        label=f"{backend_name}.up",
        required_keys=spec["plan_keys"],
    )
    plan_down = _require_optional_plan(
        section["down"],
        label=f"{backend_name}.down",
        required_keys=spec["plan_keys"],
    )
    if plan_up is None and plan_down is None:
        raise ValueError(f"{backend_name} config must enable at least one of up/down")
    return backend_name, section, plan_up, plan_down


def _build_backend(
    backend_name: str,
    section: dict[str, object],
    plan_up: dict[str, object] | None,
    plan_down: dict[str, object] | None,
    shared_slot_pool=None,
):
    match backend_name:
        case "reference":
            return _build_reference_backend(section, plan_up, plan_down)
        case "mkl":
            return _build_mkl_backend(section, plan_up, plan_down)
        case "cusparse":
            return _build_cusparse_backend(section, plan_up, plan_down, shared_slot_pool=shared_slot_pool)
        case "triton":
            return _build_triton_backend(section, plan_up, plan_down)
        case _:
            raise ValueError(f"Unsupported backend {backend_name!r}")


def _build_reference_backend(
    section: dict[str, object],
    plan_up: dict[str, object] | None,
    plan_down: dict[str, object] | None,
):
    from pygrgl_spmv.backends import ReferenceBackend, ReferencePlan, ReferencePlanPair

    return ReferenceBackend(
        pair=ReferencePlanPair(
            plan_up=None if plan_up is None else ReferencePlan.from_dict(plan_up),
            plan_down=None if plan_down is None else ReferencePlan.from_dict(plan_down),
        ),
        log_level=str(section["log_level"]),
    )


def _build_mkl_backend(
    section: dict[str, object],
    plan_up: dict[str, object] | None,
    plan_down: dict[str, object] | None,
):
    from pygrgl_spmv.backends.mkl import MklBackend, MklPlanPair

    return MklBackend(
        pair=MklPlanPair.from_dicts(plan_up, plan_down),
        log_level=str(section["log_level"]),
    )


def _build_cusparse_backend(
    section: dict[str, object],
    plan_up: dict[str, object] | None,
    plan_down: dict[str, object] | None,
    shared_slot_pool=None,
):
    from pygrgl_spmv.backends.cusparse import CusparseBackend, CusparsePlanPair

    return CusparseBackend(
        device=int(section["device"]),
        stream=section["stream"],
        pair=CusparsePlanPair.from_dicts(plan_up, plan_down),
        ring_buffer_size=int(section["ring_buffer_size"]),
        log_level=str(section["log_level"]),
        shared_slot_pool=shared_slot_pool,
    )


def _build_triton_backend(
    section: dict[str, object],
    plan_up: dict[str, object] | None,
    plan_down: dict[str, object] | None,
):
    from pygrgl_spmv.backends.triton import TritonBackend, TritonPlanPair

    return TritonBackend(
        device=int(section["device"]),
        stream=section["stream"],
        pair=TritonPlanPair.from_dicts(plan_up, plan_down),
        ring_buffer_size=int(section["ring_buffer_size"]),
        log_level=str(section["log_level"]),
    )


def _cusparse_params_from_config(config: dict):
    """Extract cuSPARSE plan pair, device, and ring_buffer_size from a config dict."""
    from pygrgl_spmv.backends.cusparse import CusparsePlanPair

    c = config.get("cusparse", {})
    up = c.get("up", {})
    down = c.get("down", {})
    plan_up = {
        "k_hint": up.get("k_hint"), "store": "N",
        "fmt": str(up.get("fmt", "csr")).upper(),
        "opA": "N", "opB": "N", "orderB": "ROW", "orderC": "ROW",
        "algo": str(up.get("algo", "default")).upper(),
        "scratch": up.get("scratch", "none"),
    }
    plan_down = {
        "k_hint": down.get("k_hint"), "store": "T",
        "fmt": str(down.get("fmt", "csc")).upper(),
        "opA": "N", "opB": "N", "orderB": "ROW", "orderC": "ROW",
        "algo": str(down.get("algo", "default")).upper(),
        "scratch": down.get("scratch", "none"),
    }
    pair = CusparsePlanPair.from_dicts(plan_up, plan_down)
    device = int(c.get("device", 0))
    ring_buffer_size = int(c.get("ring_buffer_size", 2))
    return pair, device, ring_buffer_size


def _load_compiled_state(source, dtype, artifact_dir):
    """Load a single source into a CompiledOperatorState."""
    from pygrgl_spmv.grg.artifact import load_grg_spmv, artifact_path_for_grg
    import logging

    logger = logging.getLogger(__name__)
    dtype = np.dtype(dtype)
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if source_path.suffix == ".grg":
            artifact_root = Path(artifact_dir).expanduser()
            artifact_path = artifact_path_for_grg(source_path, artifact_root)
            if artifact_path.exists():
                logger.info("Loading SpmvGRG artifact from %s", artifact_path)
                try:
                    return load_grg_spmv(artifact_path, dtype)
                except (KeyError, ValueError) as exc:
                    logger.warning(
                        "SpmvGRG artifact at %s is invalid (%s); rebuilding from %s",
                        artifact_path, exc, source_path,
                    )
            logger.info("Building SpmvGRG from %s", source_path)
            return convert(source_path, artifact_path.parent, dtype=dtype,
                           name=artifact_path.stem)
        elif source_path.suffix == ".grg_spmv":
            return load_grg_spmv(source_path, dtype)
        else:
            raise ValueError(
                f"Unsupported source {source_path}; expected .grg or .grg_spmv"
            )
    else:
        return convert(source, dtype=dtype)


def _fmt_mib(nbytes: int) -> str:
    mib = nbytes / (1024 * 1024)
    if mib >= 1000:
        return f"{mib / 1024:,.1f} GiB"
    if mib >= 1:
        return f"{mib:,.1f} MiB"
    return f"{nbytes / 1024:,.1f} KiB"


def print_gpu_memory(
    spmv: SpmvGRG,
    *,
    k: int | None = None,
    shared_slot_pool=None,
    file=None,
) -> None:
    """Print a GPU VRAM summary for a loaded SpmvGRG.

    Prints three sections:
    1. **Persistent GPU VRAM** — always loaded after :func:`load`.
    2. **Runtime GPU VRAM** — additional memory needed per matmul call (scales with k).
       If a ``k_hint`` was used, actual captured sizes are shown; otherwise an estimate
       is printed.
    3. **Shared Ring Buffer Pool** — only if ``shared_slot_pool`` is provided; shown
       with a distinct prefix to indicate it is shared across backends.

    Args:
        spmv: A loaded :class:`SpmvGRG` instance.
        k: Optional column count for concrete runtime estimates.
        shared_slot_pool: A :class:`~pygrgl_spmv.backends.cusparse.SharedSlotPool`
            passed at load time.  When provided the ring-buffer bytes are excluded
            from the persistent total and shown in their own section.
        file: Output stream.  Defaults to ``sys.stdout``.
    """
    import sys

    out = file if file is not None else sys.stdout

    snapshot = spmv.memory.retained
    if snapshot is None:
        print("(no memory snapshot available — backend may not have been set up yet)", file=out)
        return

    # --- Partition allocations -----------------------------------------------
    ring_buffer = []
    persistent_gpu = []
    workspace_gpu = []   # CUDA graph captured
    staging_gpu = []     # pre-allocated staging (output_main, input_miss, …)
    cpu_pinned = []

    _HOST_BLOCK_LABELS = {"host_blocks_up", "host_blocks_down"}
    for alloc in snapshot.allocations:
        if alloc.space == "cuda" and "slot_buffers" in alloc.labels:
            ring_buffer.append(alloc)
        elif alloc.space == "cuda" and "persistent" in alloc.retentions:
            persistent_gpu.append(alloc)
        elif alloc.space == "cuda" and "captured" in alloc.retentions:
            workspace_gpu.append(alloc)
        elif alloc.space == "cuda" and "staging" in alloc.retentions:
            staging_gpu.append(alloc)
        elif alloc.space == "cpu" and alloc.labels & _HOST_BLOCK_LABELS:
            cpu_pinned.append(alloc)

    # --- Helper: sub-group persistent allocations by label -------------------
    _SELECTOR_LABELS = {"selector_mut", "selector_miss"}
    _SCALAR_LABELS = {"alpha", "beta_zero", "beta_one"}

    def _group_persistent(allocs):
        groups = {
            "shared_values": [],
            "selectors": [],
            "scalars": [],
            "other": [],
        }
        for a in allocs:
            if "shared_values" in a.labels:
                groups["shared_values"].append(a)
            elif a.labels & _SELECTOR_LABELS:
                groups["selectors"].append(a)
            elif a.labels & _SCALAR_LABELS:
                groups["scalars"].append(a)
            else:
                groups["other"].append(a)
        return groups

    # --- Source name for header -----------------------------------------------
    try:
        source_name = Path(spmv._compiled.source_path).name
    except Exception:
        source_name = repr(spmv)

    _W = 57
    _SEP = "\u2500" * _W
    _INNER_SEP = "  " + "\u2500" * (_W - 4)

    def _row(label, nbytes, indent=2):
        pad = " " * indent
        return f"{pad}{label:<42}{_fmt_mib(nbytes):>12}"

    print(f"GPU Memory: {source_name}", file=out)
    print(_SEP, file=out)

    # =========================================================================
    # Section 1 — Persistent GPU VRAM
    # =========================================================================
    groups = _group_persistent(persistent_gpu)

    rows = []
    if groups["shared_values"]:
        sv_bytes = sum(a.nbytes for a in groups["shared_values"])
        sv_vmm = any(a.storage == "cuda_vmm" for a in groups["shared_values"])
        sv_label = "Ones array (VMM-compressed, physical)" if sv_vmm else "Ones array (shared_values)"
        rows.append((sv_label, sv_bytes))
    if groups["selectors"]:
        rows.append(("Selectors (mut + miss)", sum(a.nbytes for a in groups["selectors"])))
    if groups["scalars"]:
        rows.append(("Scalar constants (\u03b1, \u03b2\u2080, \u03b2\u2081)", sum(a.nbytes for a in groups["scalars"])))
    if groups["other"]:
        rows.append(("Other GPU allocations", sum(a.nbytes for a in groups["other"])))
    # When no shared pool, the backend owns its ring-buffer slots — count them here
    if shared_slot_pool is None and ring_buffer:
        # Each slot has struct0 + struct1 → 2 allocs per slot
        n_slots = len(ring_buffer) // 2
        rows.append((f"Ring buffer slots ({n_slots} slots, private)", sum(a.nbytes for a in ring_buffer)))

    total_persistent = sum(nbytes for _, nbytes in rows)

    print("Persistent GPU VRAM (always loaded):", file=out)
    for label, nbytes in rows:
        print(_row(label, nbytes), file=out)
    print(_INNER_SEP, file=out)
    print(_row("Total", total_persistent), file=out)
    print(file=out)

    # =========================================================================
    # Section 2 — Runtime GPU VRAM per matmul
    # =========================================================================
    if workspace_gpu or staging_gpu:
        # k_hint / graph-capture mode: show actual captured + staging sizes
        print("Runtime GPU VRAM per matmul (graph-capture, pre-allocated):", file=out)
        _WORKSPACE_ORDER = ["dense_state", "source_state", "input_primary", "spmm_ext", "scratch_views"]
        _WORKSPACE_LABELS = {
            "dense_state": "Dense state buffers",
            "source_state": "Source state buffers",
            "input_primary": "Input buffer",
            "spmm_ext": "SpMM extension buffer",
            "scratch_views": "Scratch views",
        }
        listed = set()
        for key in _WORKSPACE_ORDER:
            matched = [a for a in workspace_gpu if key in a.labels]
            if matched:
                nbytes = sum(a.nbytes for a in matched)
                print(_row(_WORKSPACE_LABELS.get(key, key), nbytes), file=out)
                listed.update(id(a) for a in matched)
        other_ws = [a for a in workspace_gpu if id(a) not in listed]
        if other_ws:
            print(_row("Other graph workspace", sum(a.nbytes for a in other_ws)), file=out)
        # Staging buffers (output_main, input_miss, init_*, xtx_bias)
        _STAGING_ORDER = ["output_main", "output_miss", "input_miss", "init_vector", "init_matrix", "xtx_bias"]
        _STAGING_LABELS = {
            "output_main": "Output matrix",
            "output_miss": "Output (missing)",
            "input_miss": "Input (missing)",
            "init_vector": "Init vector",
            "init_matrix": "Init matrix",
            "xtx_bias": "XtX bias",
        }
        listed_s = set()
        for key in _STAGING_ORDER:
            matched = [a for a in staging_gpu if key in a.labels]
            if matched:
                nbytes = sum(a.nbytes for a in matched)
                print(_row(_STAGING_LABELS.get(key, key), nbytes), file=out)
                listed_s.update(id(a) for a in matched)
        other_s = [a for a in staging_gpu if id(a) not in listed_s]
        if other_s:
            print(_row("Other staging", sum(a.nbytes for a in other_s)), file=out)
        print(_INNER_SEP, file=out)
        total_rt = sum(a.nbytes for a in workspace_gpu) + sum(a.nbytes for a in staging_gpu)
        print(_row("Total (pre-allocated)", total_rt), file=out)
    else:
        # Dynamic mode: estimate from public properties
        dtype = spmv.dtype
        num_nodes = spmv.num_nodes
        num_samples = spmv.num_samples
        num_mutations = spmv.num_mutations
        dense_per_k = num_nodes * dtype.itemsize
        io_per_k = (num_samples + num_mutations) * dtype.itemsize

        print("Runtime GPU VRAM per matmul (dynamic, not pre-allocated):", file=out)
        if k is not None:
            ds_label = f"Dense state ({num_nodes:,} nodes \u00d7 {dtype})"
            io_label = f"I/O buffers ({num_samples:,} samples + {num_mutations:,} mutations)"
            print(_row(ds_label, dense_per_k * k), file=out)
            print(_row(io_label, io_per_k * k), file=out)
            print(_INNER_SEP, file=out)
            print(_row(f"Total @ k={k}  (estimate)", (dense_per_k + io_per_k) * k), file=out)
        else:
            ds_label = f"Dense state  ~{_fmt_mib(dense_per_k)}/col   ({num_nodes:,} nodes \u00d7 {dtype})"
            io_label = f"I/O buffers   ~{_fmt_mib(io_per_k)}/col   ({num_samples:,} samples + {num_mutations:,} mutations)"
            print(f"  {ds_label}", file=out)
            print(f"  {io_label}", file=out)
            print(_INNER_SEP, file=out)
            total_per_k = dense_per_k + io_per_k
            print(f"  Total  ~{_fmt_mib(total_per_k)}/col   (pass k= for a concrete value)", file=out)
    print(file=out)

    # =========================================================================
    # Section 3 — CPU Pinned
    # =========================================================================
    if cpu_pinned:
        print("CPU Pinned (host blocks, not GPU VRAM):", file=out)
        print(_row("Sparse structure", sum(a.nbytes for a in cpu_pinned)), file=out)
        print(file=out)

    # =========================================================================
    # Section 4 — Shared Ring Buffer Pool
    # =========================================================================
    if shared_slot_pool is not None:
        pool = shared_slot_pool
        try:
            pool_bytes = pool._ring_buffer_size * (
                pool._max_struct0_len * pool._struct0_dtype.itemsize
                + pool._max_struct1_len * pool._struct1_dtype.itemsize
            )
            n_slots = pool._ring_buffer_size
        except AttributeError:
            pool_bytes = sum(a.nbytes for a in ring_buffer)
            n_slots = "?"
        slot_label = f"Slot pool ({n_slots} slots \u00d7 2 struct arrays)"
        print(f"\u25b6 Shared Ring Buffer Pool (not owned by this backend):", file=out)
        print(_row(slot_label, pool_bytes, indent=2), file=out)
        print(file=out)

    print(_SEP, file=out)


__all__ = ["SpmvGRG", "convert", "load", "load_shared_slot_pool", "print_gpu_memory"]
