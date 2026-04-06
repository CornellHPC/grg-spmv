"""Shared fixtures for operator-level tests."""

from __future__ import annotations

import pygrgl
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.tests.conftest import DATA_DTYPE, default_backend_builders


@pytest.fixture(params=default_backend_builders(log_level="INFO"))
def backend_builder(request, backend_filter):
    backend_name, builder = request.param
    btype = str(backend_name)
    if backend_filter == "mkl" and btype != "mkl":
        pytest.skip("filtered to mkl backend")
    if backend_filter == "cusparse" and btype != "cusparse":
        pytest.skip("filtered to cusparse backend")
    if backend_filter == "triton" and btype != "triton":
        pytest.skip("filtered to triton backend")
    return builder


@pytest.fixture
def backend_config(backend_builder):
    return backend_builder()


@pytest.fixture(scope="session")
def grg_ref(primary_grg_path):
    return pygrgl.load_immutable_grg(primary_grg_path)


@pytest.fixture(scope="session")
def missing_grg_ref(missing_grg_path):
    return pygrgl.load_immutable_grg(missing_grg_path)


@pytest.fixture
def op(backend_config, primary_grg_path, spmv_cache_dir):
    return SpmvGRG(primary_grg_path, backend_config, DATA_DTYPE, artifact_dir=spmv_cache_dir)


@pytest.fixture
def op_missing(backend_config, missing_grg_path, spmv_cache_dir):
    return SpmvGRG(missing_grg_path, backend_config, DATA_DTYPE, artifact_dir=spmv_cache_dir)
