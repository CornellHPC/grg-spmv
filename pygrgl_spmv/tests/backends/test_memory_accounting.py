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
        pytest.param(
            make_cusparse_backend(fmt_up="csr", fmt_down="csc", k_hint=None, scratch_up="1", scratch_down="0"),
            id="cusparse-scratch-dynamic",
            marks=pytest.mark.cusparse,
        ),
        pytest.param(
            make_cusparse_backend(fmt_up="csr", fmt_down="csc", k_hint=4, scratch_up="1", scratch_down="0"),
            id="cusparse-scratch-graph",
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
@pytest.mark.cusparse
def test_cusparse_xtx_runtime_memory_is_dynamic(primary_grg_path, spmv_cache_dir):
    pytest.importorskip("cupy")
    op = SpmvGRG(
        primary_grg_path,
        make_cusparse_backend(fmt_up="csr", fmt_down="csc", k_hint=1, algo_up="default", algo_down="default"),
        DATA_DTYPE,
        INDEX_DTYPE,
        artifact_dir=spmv_cache_dir,
    )
    before_host, before_device = op._backend.estimate_static_bytes()
    x = np.ones((1, op.num_samples), dtype=DATA_DTYPE)
    _ = op.matmul(x, "up", emit_all_nodes=True, init="xtx")
    after_host, after_device = op._backend.estimate_static_bytes()
    _assert_static_bytes_equal(before_host, after_host)
    _assert_static_bytes_equal(before_device, after_device)
    _assert_static_bytes_equal(op._backend.mem_usage.host_static, before_host)
    _assert_static_bytes_equal(op._backend.mem_usage.device_static, before_device)
    assert int(before_host.xtx_init) == 0
    assert int(before_device.xtx_init) == 0
    assert op._backend._staging_up_by_k[1].xtx_bias is not None
    assert int(op._backend.mem_usage.calls[-1].device.outputs) == 0
    assert int(op._backend.mem_usage.calls[-1].device.aux) >= int(op._backend._staging_up_by_k[1].xtx_bias.nbytes)


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


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_xtx_runtime_memory_is_dynamic(primary_grg_path, spmv_cache_dir):
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    op = SpmvGRG(
        primary_grg_path,
        make_triton_backend(fmt_up="csr", fmt_down="csc", k_hint=1),
        DATA_DTYPE,
        INDEX_DTYPE,
        artifact_dir=spmv_cache_dir,
    )
    before_host, before_device = op._backend.estimate_static_bytes()
    x = np.ones((1, op.num_samples), dtype=DATA_DTYPE)
    _ = op.matmul(x, "up", emit_all_nodes=True, init="xtx")
    after_host, after_device = op._backend.estimate_static_bytes()
    _assert_static_bytes_equal(before_host, after_host)
    _assert_static_bytes_equal(before_device, after_device)
    _assert_static_bytes_equal(op._backend.mem_usage.host_static, before_host)
    _assert_static_bytes_equal(op._backend.mem_usage.device_static, before_device)
    assert int(before_host.xtx_init) == 0
    assert int(before_device.xtx_init) == 0
    assert op._backend._staging_up is not None
    assert op._backend._staging_up.xtx_bias is not None
    xtx_nbytes = int(op._backend._staging_up.xtx_bias.numel() * op._backend._staging_up.xtx_bias.element_size())
    assert int(op._backend.mem_usage.calls[-1].device.outputs) == 0
    assert int(op._backend.mem_usage.calls[-1].device.aux) >= xtx_nbytes
