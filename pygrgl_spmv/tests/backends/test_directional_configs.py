"""Shared directional-plan behavior across MKL and cuSPARSE backends."""

from __future__ import annotations

import numpy as np
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.tests.conftest import (
    DATA_DTYPE,
    INDEX_DTYPE,
    make_cusparse_backend,
    make_mkl_backend,
    make_triton_backend,
    tol,
)


BACKEND_CONFIG_BUILDERS = [
    pytest.param("mkl", make_mkl_backend, id="mkl", marks=pytest.mark.mkl),
    pytest.param("cusparse", make_cusparse_backend, id="cusparse", marks=[pytest.mark.gpu, pytest.mark.cusparse]),
    pytest.param("triton", make_triton_backend, id="triton", marks=[pytest.mark.gpu, pytest.mark.triton]),
]


def _make_directional_op(grg_path, *, artifact_dir, make_config, **kwargs):
    if make_config is make_cusparse_backend:
        pytest.importorskip("cupy")
    if make_config is make_triton_backend:
        pytest.importorskip("torch")
        pytest.importorskip("triton")
    return SpmvGRG(
        grg_path,
        make_config(**kwargs),
        DATA_DTYPE,
        INDEX_DTYPE,
        artifact_dir=artifact_dir,
    )


@pytest.mark.parametrize(("backend_name", "make_config"), BACKEND_CONFIG_BUILDERS)
def test_forward_one_sided_plan(primary_grg_path, gt_small, spmv_cache_dir, backend_name, make_config):
    op = _make_directional_op(
        primary_grg_path,
        artifact_dir=spmv_cache_dir,
        make_config=make_config,
        fmt_up="csr",
        fmt_down=None,
        k_hint=1 if make_config is make_triton_backend else None,
        infer_missing=False,
    )
    x, y_expected = gt_small.get("forward", 4, seed=841, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(op.matmul(x.T, "up").T, y_expected, atol=atol, rtol=rtol)
    assert op._backend._plan_up is not None
    assert op._backend._plan_down is None


@pytest.mark.parametrize(("backend_name", "make_config"), BACKEND_CONFIG_BUILDERS)
def test_backward_one_sided_plan(primary_grg_path, gt_small, spmv_cache_dir, backend_name, make_config):
    op = _make_directional_op(
        primary_grg_path,
        artifact_dir=spmv_cache_dir,
        make_config=make_config,
        fmt_up=None,
        fmt_down="csc",
        k_hint=1 if make_config is make_triton_backend else None,
        infer_missing=False,
    )
    x, y_expected = gt_small.get("backward", 4, seed=842, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(op.matmul(x.T, "down").T, y_expected, atol=atol, rtol=rtol)
    assert op._backend._plan_up is None
    assert op._backend._plan_down is not None


@pytest.mark.parametrize(("backend_name", "make_config"), BACKEND_CONFIG_BUILDERS)
def test_one_sided_plan_drops_unused_static_block_storage(
    primary_grg_path,
    spmv_cache_dir,
    backend_name,
    make_config,
):
    baseline = _make_directional_op(
        primary_grg_path,
        artifact_dir=spmv_cache_dir,
        make_config=make_config,
        fmt_up="csr",
        fmt_down="csr",
        k_hint=1 if make_config is make_triton_backend else None,
    )
    optimized = _make_directional_op(
        primary_grg_path,
        artifact_dir=spmv_cache_dir,
        make_config=make_config,
        fmt_up="csr",
        fmt_down=None,
        k_hint=1 if make_config is make_triton_backend else None,
        infer_missing=False,
    )

    space = "cpu" if backend_name == "mkl" else "cuda"

    def _node_bytes(op, node: str) -> int:
        assert op.memory.retained is not None
        return int(
            sum(
                row.nbytes
                for row in op.memory.retained.allocations
                if row.space == space and row.retention == "persistent" and node in row.labels
            )
        )

    def _root_bytes(op) -> int:
        assert op.memory.retained is not None
        return int(sum(row.nbytes for row in op.memory.retained.allocations if row.space == space))

    assert _node_bytes(baseline, "blocks_down") > 0
    assert _node_bytes(optimized, "blocks_down") == 0
    assert _node_bytes(optimized, "blocks_up") == _node_bytes(baseline, "blocks_up")
    assert _root_bytes(optimized) < _root_bytes(baseline)
