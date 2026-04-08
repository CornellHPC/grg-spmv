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


__all__ = ["SpmvGRG", "convert", "load", "load_shared_slot_pool"]
