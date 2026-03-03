"""Shared fixtures for operator-level tests."""

from __future__ import annotations

import pygrgl
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.tests.conftest import DATA_DTYPE, INDEX_DTYPE


def _backend_configs():
    params = [
        pytest.param({"type": "mkl", "n_threads": 0, "log_level": "INFO"}, id="mkl", marks=pytest.mark.mkl),
    ]
    try:
        import cupy  # noqa: F401
    except ImportError:
        return params

    params.extend(
        [
            pytest.param(
                {
                    "type": "cusparse",
                    "fmt_up": "csr",
                    "algo_up": "default",
                    "algo_down": "default",
                    "k_hint": None,
                    "log_level": "INFO",
                },
                id="cusparse-dyn",
                marks=pytest.mark.gpu,
            ),
            pytest.param(
                {
                    "type": "cusparse",
                    "fmt_up": "csr",
                    "algo_up": "default",
                    "algo_down": "default",
                    "k_hint": 4,
                    "log_level": "INFO",
                },
                id="cusparse-graph-k4",
                marks=pytest.mark.gpu,
            ),
        ]
    )
    return params


@pytest.fixture(params=_backend_configs())
def backend_config(request, backend_filter):
    cfg = request.param
    btype = str(cfg["type"])
    if backend_filter == "mkl" and btype != "mkl":
        pytest.skip("filtered to mkl backend")
    if backend_filter == "cusparse" and btype != "cusparse":
        pytest.skip("filtered to cusparse backend")
    return cfg


@pytest.fixture(scope="session")
def grg_ref(primary_grg_path):
    return pygrgl.load_immutable_grg(primary_grg_path)


@pytest.fixture(scope="session")
def missing_grg_ref(missing_grg_path):
    return pygrgl.load_immutable_grg(missing_grg_path)


@pytest.fixture
def op(backend_config, primary_grg_path, spmv_cache_dir):
    return SpmvGRG(primary_grg_path, backend_config, DATA_DTYPE, INDEX_DTYPE, cache_dir=spmv_cache_dir)


@pytest.fixture
def op_missing(backend_config, missing_grg_path, spmv_cache_dir):
    return SpmvGRG(missing_grg_path, backend_config, DATA_DTYPE, INDEX_DTYPE, cache_dir=spmv_cache_dir)
