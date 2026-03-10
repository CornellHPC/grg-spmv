"""Backend factory and configuration validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_COMMON_BACKEND_KEYS = {"type", "log_level", "plan_up", "plan_down"}
_MKL_BACKEND_KEYS = _COMMON_BACKEND_KEYS
_CUSPARSE_BACKEND_KEYS = _COMMON_BACKEND_KEYS


def _validate_backend_config_keys(config: Mapping[str, Any], allowed_keys: set[str], backend_type: str) -> None:
    unknown = sorted(set(config) - allowed_keys)
    if unknown:
        raise ValueError(
            f"Unknown backend_config key(s) for backend={backend_type!r}: {unknown}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )


def create_backend(config: Mapping[str, Any]):
    """Create a backend instance from the public config dictionary."""
    backend_type = str(config.get("type", "mkl"))
    log_level = str(config.get("log_level", "WARNING"))
    match backend_type:
        case "mkl":
            from pygrgl_spmv.backends.mkl import MklBackend, MklPlan

            _validate_backend_config_keys(config, _MKL_BACKEND_KEYS, backend_type)
            return MklBackend(
                plan_up=MklPlan.from_any(config.get("plan_up")),
                plan_down=MklPlan.from_any(config.get("plan_down")),
                log_level=log_level,
            )
        case "cusparse":
            from pygrgl_spmv.backends.cusparse import CusparseBackend, CusparsePlan

            _validate_backend_config_keys(config, _CUSPARSE_BACKEND_KEYS, backend_type)
            return CusparseBackend(
                plan_up=CusparsePlan.from_any(config.get("plan_up")),
                plan_down=CusparsePlan.from_any(config.get("plan_down")),
                log_level=log_level,
            )
        case _:
            raise ValueError(f"No config handler for backend: {backend_type!r}")


__all__ = ["create_backend"]
