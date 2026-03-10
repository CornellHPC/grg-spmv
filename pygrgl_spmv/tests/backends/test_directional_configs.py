"""Shared directional-plan behavior across MKL and cuSPARSE backends."""

from __future__ import annotations

import numpy as np
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.tests.conftest import (
    DATA_DTYPE,
    INDEX_DTYPE,
    make_cusparse_config,
    make_mkl_config,
    tol,
)


BACKEND_CONFIG_BUILDERS = [
    pytest.param("mkl", make_mkl_config, id="mkl", marks=pytest.mark.mkl),
    pytest.param("cusparse", make_cusparse_config, id="cusparse", marks=pytest.mark.gpu),
]


def _make_directional_op(grg_path, *, cache_dir, make_config, **kwargs):
    if make_config is make_cusparse_config:
        pytest.importorskip("cupy")
    return SpmvGRG(
        grg_path,
        make_config(**kwargs),
        DATA_DTYPE,
        INDEX_DTYPE,
        cache_dir=cache_dir,
    )


@pytest.mark.parametrize(("backend_name", "make_config"), BACKEND_CONFIG_BUILDERS)
def test_forward_one_sided_plan(primary_grg_path, gt_small, spmv_cache_dir, backend_name, make_config):
    op = _make_directional_op(
        primary_grg_path,
        cache_dir=spmv_cache_dir,
        make_config=make_config,
        fmt_up="csr",
        fmt_down=None,
        k_hint=None,
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
        cache_dir=spmv_cache_dir,
        make_config=make_config,
        fmt_up=None,
        fmt_down="csc",
        k_hint=None,
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
        cache_dir=spmv_cache_dir,
        make_config=make_config,
        fmt_up="csr",
        fmt_down="csr",
        k_hint=None,
    )
    optimized = _make_directional_op(
        primary_grg_path,
        cache_dir=spmv_cache_dir,
        make_config=make_config,
        fmt_up="csr",
        fmt_down=None,
        k_hint=None,
        infer_missing=False,
    )

    static_attr = "host_static" if backend_name == "mkl" else "device_static"
    base_static = getattr(baseline._backend.mem_usage, static_attr)
    opt_static = getattr(optimized._backend.mem_usage, static_attr)
    assert int(base_static.blocks_down) > 0
    assert int(opt_static.blocks_down) == 0
    assert int(opt_static.blocks_up) == int(base_static.blocks_up)
    assert int(opt_static.total()) < int(base_static.total())
