"""Static memory accounting alignment tests across backends."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.backends.memory import StaticBytes
from pygrgl_spmv.tests.conftest import DATA_DTYPE, INDEX_DTYPE


def _assert_static_bytes_equal(actual: StaticBytes, estimated: StaticBytes) -> None:
    for f in fields(StaticBytes):
        name = f.name
        assert int(getattr(actual, name)) == int(getattr(estimated, name)), (
            f"Static memory mismatch for {name}: actual={getattr(actual, name)} "
            f"estimated={getattr(estimated, name)}"
        )


@pytest.mark.mkl
@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param({"type": "mkl", "n_threads": 1, "fmt_up": "csr", "fmt_down": None}, id="mkl-csr-none", marks=pytest.mark.smoke),
        pytest.param({"type": "mkl", "n_threads": 1, "fmt_up": "csr", "fmt_down": "csr"}, id="mkl-csr-csr"),
        pytest.param({"type": "mkl", "n_threads": 1, "fmt_up": "coo", "fmt_down": "coo"}, id="mkl-coo-coo"),
    ],
)
def test_mkl_static_estimate_matches_recorded(primary_grg_path, spmv_cache_dir, cfg):
    op = SpmvGRG(primary_grg_path, cfg, DATA_DTYPE, INDEX_DTYPE, cache_dir=spmv_cache_dir)
    est_host, est_device = op._backend.estimate_static_bytes()
    _assert_static_bytes_equal(op._backend.mem_usage.host_static, est_host)
    _assert_static_bytes_equal(op._backend.mem_usage.device_static, est_device)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param(
            {"type": "cusparse", "fmt_up": "csr", "fmt_down": None, "k_hint": None, "algo_up": "default", "algo_down": "default"},
            id="cusparse-csr-none",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            {"type": "cusparse", "fmt_up": None, "fmt_down": "csc", "k_hint": None, "algo_up": "default", "algo_down": "default"},
            id="cusparse-none-csc",
        ),
        pytest.param(
            {"type": "cusparse", "fmt_up": "coo", "fmt_down": "coo", "k_hint": None, "algo_up": "coo_alg1", "algo_down": "coo_alg2"},
            id="cusparse-coo-coo",
        ),
    ],
)
def test_cusparse_static_estimate_matches_recorded(primary_grg_path, spmv_cache_dir, cfg):
    pytest.importorskip("cupy")
    op = SpmvGRG(primary_grg_path, cfg, DATA_DTYPE, INDEX_DTYPE, cache_dir=spmv_cache_dir)
    est_host, est_device = op._backend.estimate_static_bytes()
    _assert_static_bytes_equal(op._backend.mem_usage.host_static, est_host)
    _assert_static_bytes_equal(op._backend.mem_usage.device_static, est_device)
