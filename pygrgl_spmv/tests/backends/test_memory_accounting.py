"""Static memory accounting alignment tests across backends."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.backends.memory import StaticBytes
from pygrgl_spmv.tests.conftest import (
    DATA_DTYPE,
    HAS_TRITON_RUNTIME,
    INDEX_DTYPE,
    make_cusparse_backend,
    make_mkl_backend,
    make_triton_backend,
)


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
        pytest.param(make_mkl_backend(fmt_up="csr", fmt_down=None, n_threads=1), id="mkl-csr-none", marks=pytest.mark.smoke),
        pytest.param(make_mkl_backend(fmt_up="csr", fmt_down="csr", n_threads=1), id="mkl-csr-csr"),
        pytest.param(make_mkl_backend(fmt_up="coo", fmt_down="coo", n_threads=1), id="mkl-coo-coo"),
    ],
)
def test_mkl_static_estimate_matches_recorded(primary_grg_path, spmv_cache_dir, cfg):
    op = SpmvGRG(primary_grg_path, cfg, DATA_DTYPE, INDEX_DTYPE, artifact_dir=spmv_cache_dir)
    est_host, est_device = op._backend.estimate_static_bytes()
    _assert_static_bytes_equal(op._backend.mem_usage.host_static, est_host)
    _assert_static_bytes_equal(op._backend.mem_usage.device_static, est_device)


@pytest.mark.gpu
@pytest.mark.cusparse
@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param(
            make_cusparse_backend(fmt_up="csr", fmt_down=None, k_hint=None, algo_up="default", algo_down="default"),
            id="cusparse-csr-none",
            marks=[pytest.mark.smoke, pytest.mark.cusparse],
        ),
        pytest.param(
            make_cusparse_backend(fmt_up=None, fmt_down="csc", k_hint=None, algo_up="default", algo_down="default"),
            id="cusparse-none-csc",
            marks=pytest.mark.cusparse,
        ),
        pytest.param(
            make_cusparse_backend(fmt_up="coo", fmt_down="coo", k_hint=None, algo_up="coo_alg1", algo_down="coo_alg2"),
            id="cusparse-coo-coo",
            marks=pytest.mark.cusparse,
        ),
    ],
)
def test_cusparse_static_estimate_matches_recorded(primary_grg_path, spmv_cache_dir, cfg):
    pytest.importorskip("cupy")
    op = SpmvGRG(primary_grg_path, cfg, DATA_DTYPE, INDEX_DTYPE, artifact_dir=spmv_cache_dir)
    est_host, est_device = op._backend.estimate_static_bytes()
    _assert_static_bytes_equal(op._backend.mem_usage.host_static, est_host)
    _assert_static_bytes_equal(op._backend.mem_usage.device_static, est_device)


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param(make_triton_backend(fmt_up="csr", fmt_down=None, k_hint=1, infer_missing=False), id="triton-csr-none", marks=pytest.mark.smoke),
        pytest.param(make_triton_backend(fmt_up="csr", fmt_down="csc", k_hint=1), id="triton-csr-csc-shared"),
        pytest.param(make_triton_backend(fmt_up="csc", fmt_down="csr", k_hint=1), id="triton-csc-csr-shared"),
        pytest.param(make_triton_backend(fmt_up="csc", fmt_down="csc", k_hint=1), id="triton-csc-csc-unshared"),
        pytest.param(make_triton_backend(fmt_up="csr", fmt_down="csc", k_hint=1, scratch_up="1", scratch_down="0"), id="triton-scratch"),
        pytest.param(
            make_triton_backend(fmt_up="csr", fmt_down="csc", k_hint=1, instrumentation=True),
            id="triton-instrumented",
        ),
    ],
)
def test_triton_static_estimate_matches_recorded(primary_grg_path, spmv_cache_dir, cfg):
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    op = SpmvGRG(primary_grg_path, cfg, DATA_DTYPE, INDEX_DTYPE, artifact_dir=spmv_cache_dir)
    est_host, est_device = op._backend.estimate_static_bytes()
    _assert_static_bytes_equal(op._backend.mem_usage.host_static, est_host)
    _assert_static_bytes_equal(op._backend.mem_usage.device_static, est_device)
