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
_KNOWN_ROOT_KEYS = ("backend", "reference", "mkl", "cusparse", "triton", "hybrid")
_BACKEND_SPECS = {
    "reference": {
        "section_keys": ("log_level", "up", "down"),
        "plan_keys": ("store", "fmt", "k_hint"),
    },
    "mkl": {
        "section_keys": ("log_level", "up", "down"),
        "plan_keys": ("store", "fmt", "n_threads", "k_hint", "optimize"),
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
    source_name = Path(source).stem if isinstance(source, (str, os.PathLike)) else None
    backend_name, section, plan_up, plan_down = _parse_backend_config(config, source_name=source_name)
    backend = _build_backend(backend_name, section, plan_up, plan_down)
    _LOGGER.info("Selected %s backend from %s", backend_name, config_path)
    return SpmvGRG(source, backend, dtype, artifact_dir=artifact_dir)


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


def _parse_hybrid_config(
    root: dict[str, object],
    source_name: str | None,
) -> tuple[str, dict[str, object], dict[str, object] | None, dict[str, object] | None]:
    if "hybrid" not in root:
        raise ValueError("config is missing required 'hybrid' section")
    hybrid_map = _require_mapping(root["hybrid"], label="hybrid config")
    if source_name is None:
        raise ValueError(
            "hybrid backend requires a file-path source so the GRG stem can be looked up; "
            "in-memory ImmutableGRG sources are not supported with hybrid configs"
        )
    if source_name not in hybrid_map:
        available = sorted(hybrid_map)
        raise ValueError(
            f"hybrid config has no entry for {source_name!r}; "
            f"available keys: {available}"
        )
    sub_config = _require_mapping(hybrid_map[source_name], label=f"hybrid[{source_name!r}]")
    return _parse_backend_config(sub_config)


def _parse_backend_config(
    config: object,
    *,
    source_name: str | None = None,
) -> tuple[str, dict[str, object], dict[str, object] | None, dict[str, object] | None]:
    root = _require_mapping(config, label="config")
    _require_keys(root, label="config", required_keys=("backend",), allowed_keys=_KNOWN_ROOT_KEYS)

    backend_name = str(root["backend"]).strip().lower()
    if backend_name == "hybrid":
        return _parse_hybrid_config(root, source_name)
    if backend_name not in _BACKEND_SPECS:
        raise ValueError(
            f"config backend must be one of {sorted([*_BACKEND_SPECS, 'hybrid'])}, got {root['backend']!r}"
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
):
    match backend_name:
        case "reference":
            return _build_reference_backend(section, plan_up, plan_down)
        case "mkl":
            return _build_mkl_backend(section, plan_up, plan_down)
        case "cusparse":
            return _build_cusparse_backend(section, plan_up, plan_down)
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
):
    from pygrgl_spmv.backends.cusparse import CusparseBackend, CusparsePlanPair

    return CusparseBackend(
        device=int(section["device"]),
        stream=section["stream"],
        pair=CusparsePlanPair.from_dicts(plan_up, plan_down),
        ring_buffer_size=int(section["ring_buffer_size"]),
        log_level=str(section["log_level"]),
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


__all__ = ["SpmvGRG", "convert", "load"]
