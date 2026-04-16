"""Shared fixtures and pytest controls for the runtime-era test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygrgl
import pytest
from pygrgl_spmv import convert

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIMARY_GRG = str(REPO_ROOT / "pygrgl_spmv" / "tests" / "data" / "msprime.example.igd.final.grg")
DEFAULT_MISSING_GRG = str(REPO_ROOT / "pygrgl_spmv" / "tests" / "data" / "test-200-samples.miss.final.grg")

DATA_DTYPE = np.float64


def _has_mkl_runtime() -> bool:
    try:
        from pygrgl_spmv.backends.mkl import ffi as mkl_ffi

        mkl_ffi._ensure_loaded()
        return True
    except Exception:
        return False


def _has_cusparse_runtime() -> bool:
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def _has_triton_runtime() -> bool:
    try:
        import torch
        import triton  # noqa: F401

        return bool(torch.cuda.is_available())
    except Exception:
        return False


HAS_MKL_RUNTIME = _has_mkl_runtime()
HAS_CUSPARSE_RUNTIME = _has_cusparse_runtime()
HAS_TRITON_RUNTIME = _has_triton_runtime()


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        default="all",
        choices=["mkl", "cusparse", "triton", "all"],
        help="Which backend-specific tests to include; shared/reference tests always run.",
    )
    parser.addoption(
        "--stress",
        action="store_true",
        default=False,
        help="Run long streamed GPU stress tests.",
    )
    parser.addoption(
        "--grg",
        default=DEFAULT_PRIMARY_GRG,
        help="Primary GRG file used by tests.",
    )
    parser.addoption(
        "--missing-grg",
        default=DEFAULT_MISSING_GRG,
        help="Missingness GRG file used by tests.",
    )


def pytest_collection_modifyitems(config, items):
    backend = str(config.getoption("--backend"))
    stress = bool(config.getoption("--stress"))

    match backend:
        case "all":
            pass
        case "mkl":
            skip = pytest.mark.skip(reason="--backend=mkl")
            for item in items:
                if "cusparse" in item.keywords or "triton" in item.keywords or "gpu" in item.keywords:
                    item.add_marker(skip)
        case "cusparse":
            skip = pytest.mark.skip(reason="--backend=cusparse")
            for item in items:
                if "mkl" in item.keywords or "triton" in item.keywords:
                    item.add_marker(skip)
        case "triton":
            skip = pytest.mark.skip(reason="--backend=triton")
            for item in items:
                if "mkl" in item.keywords or "cusparse" in item.keywords:
                    item.add_marker(skip)
        case _:
            raise ValueError(f"unexpected --backend value {backend!r}")

    if not HAS_MKL_RUNTIME:
        skip = pytest.mark.skip(reason="MKL runtime unavailable (libmkl_rt.so not found)")
        for item in items:
            if "mkl" in item.keywords:
                item.add_marker(skip)

    if not HAS_CUSPARSE_RUNTIME:
        skip = pytest.mark.skip(reason="cuSPARSE runtime unavailable (CuPy + CUDA not found)")
        for item in items:
            if "cusparse" in item.keywords:
                item.add_marker(skip)

    if not HAS_TRITON_RUNTIME:
        skip = pytest.mark.skip(reason="Triton runtime unavailable (torch + triton CUDA not found)")
        for item in items:
            if "triton" in item.keywords:
                item.add_marker(skip)

    if not stress:
        skip = pytest.mark.skip(reason="stress tests require --stress")
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip)


def tol(dtype) -> tuple[float, float]:
    return (1e-3, 1e-3) if np.dtype(dtype) == np.float32 else (1e-5, 1e-5)


def binary_pm1(rng: np.random.Generator, shape: tuple[int, ...], dtype) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0], dtype=np.dtype(dtype)), size=shape)


@pytest.fixture(scope="session")
def backend_filter(request) -> str:
    return str(request.config.getoption("--backend"))


@pytest.fixture(scope="session")
def primary_grg_path(request) -> str:
    path = Path(str(request.config.getoption("--grg")))
    if not path.exists():
        pytest.skip(f"primary GRG file not found: {path}")
    return str(path)


@pytest.fixture(scope="session")
def missing_grg_path(request) -> str:
    path = Path(str(request.config.getoption("--missing-grg")))
    if not path.exists():
        pytest.skip(f"missingness GRG file not found: {path}")
    return str(path)


@pytest.fixture(scope="session")
def primary_grg(primary_grg_path):
    return pygrgl.load_immutable_grg(primary_grg_path, load_up_edges=True)


@pytest.fixture(scope="session")
def missing_grg(missing_grg_path):
    return pygrgl.load_immutable_grg(missing_grg_path, load_up_edges=True)


@pytest.fixture(scope="session")
def artifact_cache_dir() -> Path:
    path = REPO_ROOT / ".pytest_cache" / "runtime_artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def primary_artifact(primary_grg_path, artifact_cache_dir) -> Path:
    return convert(primary_grg_path, artifact_cache_dir)


@pytest.fixture(scope="session")
def missing_artifact(missing_grg_path, artifact_cache_dir) -> Path:
    return convert(missing_grg_path, artifact_cache_dir)


@pytest.fixture(autouse=True)
def _disable_triton_autotune_for_tests(request, monkeypatch):
    if "triton" not in request.keywords:
        return
    from pygrgl_spmv.backends.triton import TritonRuntime

    monkeypatch.setattr(
        TritonRuntime,
        "_tune_direction",
        lambda self, direction: self._candidate_configs(direction)[0],
    )
