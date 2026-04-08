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
    shared_slot_pool=None,
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
        shared_slot_pool: Optional :class:`~pygrgl_spmv.backends.cusparse.SharedSlotPool`
            pre-allocated via :func:`load_shared_slot_pool`. Only valid with the
            cuSPARSE backend; raises ``ValueError`` for other backends.

    Returns:
        A fully initialised :class:`SpmvGRG` instance.
    """
    backend = _backend_from_env(shared_slot_pool=shared_slot_pool)
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

    config_path = os.environ.get(_ENV_VAR)
    if config_path is None:
        raise RuntimeError(
            f"{_ENV_VAR} is not set. load_shared_slot_pool requires an explicit "
            "cuSPARSE configuration file."
        )
    with open(config_path) as f:
        config = json.load(f)
    backend_name = str(config.get("backend", "")).lower()
    if backend_name != "cusparse":
        raise RuntimeError(
            f"load_shared_slot_pool requires backend='cusparse' in the {_ENV_VAR} "
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


def _backend_from_env(shared_slot_pool=None):
    config_path = os.environ.get(_ENV_VAR)
    if config_path is None:
        warnings.warn(
            f"{_ENV_VAR} is not set; using default backend configuration. "
            "Set this env var to a JSON config file to silence this warning.",
            UserWarning,
            stacklevel=3,
        )
        return _default_backend(shared_slot_pool=shared_slot_pool)
    with open(config_path) as f:
        config = json.load(f)
    return _backend_from_config(config, shared_slot_pool=shared_slot_pool)


def _default_backend(shared_slot_pool=None):
    # 1. Try MKL
    try:
        from pygrgl_spmv.backends.mkl import MklBackend, MklPlanPair
        from pygrgl_spmv.backends.mkl import ffi as mkl_ffi

        mkl_ffi._ensure_loaded()
        if shared_slot_pool is not None:
            raise ValueError(
                "shared_slot_pool requires the cuSPARSE backend; "
                "the default selected backend is MKL."
            )
        plan_up = {"store": "N", "fmt": "CSR", "k_hint": None, "n_threads": 0}
        plan_down = {"store": "T", "fmt": "CSC", "k_hint": None, "n_threads": 0}
        warnings.warn("Using default MKL backend configuration.")
        return MklBackend(pair=MklPlanPair.from_dicts(plan_up, plan_down), log_level="WARNING")
    except Exception as exc:
        if shared_slot_pool is not None and "shared_slot_pool" in str(exc):
            raise

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
            shared_slot_pool=shared_slot_pool,
        )
    except Exception:
        pass

    raise RuntimeError(
        "No backend is available. Install MKL (libmkl_rt.so) or cuSPARSE (cupy) "
        f"and set {_ENV_VAR} to a JSON config file."
    )


def _backend_from_config(config: dict, shared_slot_pool=None):
    backend_name = str(config.get("backend", "")).lower()
    if backend_name == "mkl":
        if shared_slot_pool is not None:
            raise ValueError(
                "shared_slot_pool requires the cuSPARSE backend; "
                f"the configured backend is 'mkl'."
            )
        warnings.warn("Using MKL backend configuration from JSON.")
        return _mkl_backend_from_config(config)
    if backend_name == "cusparse":
        warnings.warn("Using cuSPARSE backend configuration from JSON.")
        return _cusparse_backend_from_config(config, shared_slot_pool=shared_slot_pool)
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


def _cusparse_backend_from_config(config: dict, shared_slot_pool=None):
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
        shared_slot_pool=shared_slot_pool,
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
