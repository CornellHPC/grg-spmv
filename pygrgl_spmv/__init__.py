"""pygrgl_spmv - sparse matmul for GRG-based genotype matrices."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np

from pygrgl_spmv.grg import SpmvGRG, convert

_ENV_VAR = "PYGRGL_SPMV_CONFIG"


def load(
    source,
    dtype=np.float64,
    *,
    artifact_dir: str | Path = "pygrgl_spmv_artifacts",
) -> SpmvGRG:
    """Load a SpmvGRG using backend configuration from the environment.

    Reads ``PYGRGL_SPMV_CONFIG`` (path to a JSON config file) to determine
    the backend and its plan parameters.  If the env var is not set, a default
    backend is selected automatically (MKL preferred, cuSPARSE as fallback) and
    a ``UserWarning`` is emitted.

    The JSON file format is::

        {
          "backend": "mkl",
          "mkl": {
            "n_threads": 0,
            "log_level": "WARNING",
            "up":   {"fmt": "csr", "k_hint": null},
            "down": {"fmt": "csc", "k_hint": null}
          },
          "cusparse": {
            "device": 0, "stream": 0, "ring_buffer_size": 2,
            "log_level": "WARNING",
            "up":   {"fmt": "csr", "k_hint": null, "algo": "default", "scratch": "none"},
            "down": {"fmt": "csc", "k_hint": null, "algo": "default", "scratch": "none"}
          }
        }

    Both backend sections may be present; ``"backend"`` selects which is used.

    Args:
        source: Path to a ``.grg`` or ``.grg_spmv`` file, or a loaded
            ``pygrgl.ImmutableGRG`` object.
        dtype: Floating-point dtype for the operator. Default: ``float64``.
        artifact_dir: Directory for caching compiled ``.grg_spmv`` artifacts.

    Returns:
        A fully initialised :class:`SpmvGRG` instance.
    """
    backend = _backend_from_env()
    return SpmvGRG(source, backend, dtype, artifact_dir=artifact_dir)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _backend_from_env():
    config_path = os.environ.get(_ENV_VAR)
    if config_path is None:
        warnings.warn(
            f"{_ENV_VAR} is not set; using default backend configuration. "
            "Set this env var to a JSON config file to silence this warning.",
            UserWarning,
            stacklevel=3,
        )
        return _default_backend()
    with open(config_path) as f:
        config = json.load(f)
    return _backend_from_config(config)


def _default_backend():
    # 1. Try MKL
    try:
        from pygrgl_spmv.backends.mkl import MklBackend, MklPlanPair
        from pygrgl_spmv.backends.mkl import ffi as mkl_ffi

        mkl_ffi._ensure_loaded()
        plan_up = {"store": "N", "fmt": "CSR", "k_hint": None, "n_threads": 0}
        plan_down = {"store": "T", "fmt": "CSC", "k_hint": None, "n_threads": 0}
        warnings.warn("Using default MKL backend configuration.")
        return MklBackend(pair=MklPlanPair.from_dicts(plan_up, plan_down), log_level="WARNING")
    except Exception:
        pass

    # 2. Try cuSPARSE
    try:
        import cupy  # noqa: F401

        from pygrgl_spmv.backends.cusparse import CusparseBackend, CusparsePlanPair

        plan_up = {"k_hint": None, "store": "N", "fmt": "CSR",
                   "opA": "N", "opB": "N", "orderB": "ROW", "orderC": "ROW",
                   "algo": "DEFAULT", "scratch": "none"}
        plan_down = {"k_hint": None, "store": "T", "fmt": "CSC",
                     "opA": "N", "opB": "N", "orderB": "ROW", "orderC": "ROW",
                     "algo": "DEFAULT", "scratch": "none"}
        warnings.warn("Using default cuSPARSE backend configuration.")
        return CusparseBackend(
            device=0, stream=0,
            pair=CusparsePlanPair.from_dicts(plan_up, plan_down),
            ring_buffer_size=2, log_level="WARNING",
        )
    except Exception:
        pass

    raise RuntimeError(
        "No backend is available. Install MKL (libmkl_rt.so) or cuSPARSE (cupy) "
        f"and set {_ENV_VAR} to a JSON config file."
    )


def _backend_from_config(config: dict):
    backend_name = str(config.get("backend", "")).lower()
    if backend_name == "mkl":
        warnings.warn("Using MKL backend configuration from JSON.")
        return _mkl_backend_from_config(config)
    if backend_name == "cusparse":
        warnings.warn("Using cuSPARSE backend configuration from JSON.")
        return _cusparse_backend_from_config(config)
    raise ValueError(
        f"Unknown backend {backend_name!r} in {_ENV_VAR} config; expected 'mkl' or 'cusparse'."
    )


def _mkl_backend_from_config(config: dict):
    from pygrgl_spmv.backends.mkl import MklBackend, MklPlanPair

    c = config.get("mkl", {})
    up = c.get("up", {})
    down = c.get("down", {})
    n_threads = c.get("n_threads", 0)
    log_level = c.get("log_level", "WARNING")
    plan_up = {"store": "N", "fmt": str(up.get("fmt", "csr")).upper(),
               "k_hint": up.get("k_hint"), "n_threads": n_threads}
    plan_down = {"store": "T", "fmt": str(down.get("fmt", "csc")).upper(),
                 "k_hint": down.get("k_hint"), "n_threads": n_threads}
    return MklBackend(pair=MklPlanPair.from_dicts(plan_up, plan_down), log_level=log_level)


def _cusparse_backend_from_config(config: dict):
    from pygrgl_spmv.backends.cusparse import CusparseBackend, CusparsePlanPair

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
    return CusparseBackend(
        device=c.get("device", 0),
        stream=c.get("stream", 0),
        pair=CusparsePlanPair.from_dicts(plan_up, plan_down),
        ring_buffer_size=c.get("ring_buffer_size", 2),
        log_level=c.get("log_level", "WARNING"),
    )


__all__ = ["SpmvGRG", "convert", "load"]
