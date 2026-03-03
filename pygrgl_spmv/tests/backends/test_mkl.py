"""MKL backend tests."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.tests.conftest import DATA_DTYPE, INDEX_DTYPE, K_CORE, K_MATRIX, tol

pytestmark = pytest.mark.mkl

MKL_FMTS = ["csr", "csc", "coo"]


def _make_mkl_op(
    grg_path,
    *,
    cache_dir,
    n_threads=0,
    fmt_up="csr",
    fmt_down=None,
    k_hint=None,
    dtype=DATA_DTYPE,
):
    return SpmvGRG(
        grg_path,
        {
            "type": "mkl",
            "n_threads": n_threads,
            "fmt_up": fmt_up,
            "fmt_down": fmt_down,
            "k_hint": k_hint,
        },
        dtype,
        INDEX_DTYPE,
        cache_dir=cache_dir,
    )


def _run_up(op, X_col_major):
    return op.matmul(X_col_major.T, "up").T


def _run_down(op, X_col_major):
    return op.matmul(X_col_major.T, "down").T


class TestSmokeCoreConfigMatrix:
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "cfg,k",
        [
            pytest.param({"fmt_up": "csr", "fmt_down": None, "k_hint": 1}, 1, id="up-csr-down-auto-k1"),
            pytest.param({"fmt_up": None, "fmt_down": "csc", "k_hint": 4}, 4, id="up-auto-down-csc-k4"),
            pytest.param({"fmt_up": "coo", "fmt_down": "coo", "k_hint": None}, 2, id="up-coo-down-coo-dyn"),
        ],
    )
    def test_forward(self, primary_grg_path, gt_small, spmv_cache_dir, cfg, k):
        op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, **cfg)
        X, Y_expected = gt_small.get("forward", k, seed=6201, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "cfg,k",
        [
            pytest.param({"fmt_up": "csr", "fmt_down": None, "k_hint": 1}, 1, id="up-csr-down-auto-k1"),
            pytest.param({"fmt_up": None, "fmt_down": "csc", "k_hint": 4}, 4, id="up-auto-down-csc-k4"),
            pytest.param({"fmt_up": "coo", "fmt_down": "coo", "k_hint": None}, 2, id="up-coo-down-coo-dyn"),
        ],
    )
    def test_backward(self, primary_grg_path, gt_small, spmv_cache_dir, cfg, k):
        op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, **cfg)
        X, Y_expected = gt_small.get("backward", k, seed=6202, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("n_threads", [0, 1, 2, 4, 8])
@pytest.mark.parametrize("k", K_CORE)
def test_thread_counts_forward(primary_grg_path, gt_small, spmv_cache_dir, n_threads, k):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, n_threads=n_threads)
    X, Y_expected = gt_small.get("forward", k, seed=123, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("n_threads", [0, 1, 2, 4, 8])
@pytest.mark.parametrize("k", K_CORE)
def test_thread_counts_backward(primary_grg_path, gt_small, spmv_cache_dir, n_threads, k):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, n_threads=n_threads)
    X, Y_expected = gt_small.get("backward", k, seed=123, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt", MKL_FMTS)
@pytest.mark.parametrize("k", K_CORE)
def test_formats_forward(primary_grg_path, gt_small, spmv_cache_dir, fmt, k):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up=fmt)
    X, Y_expected = gt_small.get("forward", k, seed=223, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt", MKL_FMTS)
@pytest.mark.parametrize("k", K_CORE)
def test_formats_backward(primary_grg_path, gt_small, spmv_cache_dir, fmt, k):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up=fmt)
    X, Y_expected = gt_small.get("backward", k, seed=223, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("k_hint", [None, 1, 4])
@pytest.mark.parametrize("k", K_CORE)
def test_k_hint_invariance_forward(primary_grg_path, gt_small, spmv_cache_dir, k_hint, k):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, k_hint=k_hint)
    X, Y_expected = gt_small.get("forward", k, seed=323, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("k_hint", [None, 1, 4])
@pytest.mark.parametrize("k", K_CORE)
def test_k_hint_invariance_backward(primary_grg_path, gt_small, spmv_cache_dir, k_hint, k):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, k_hint=k_hint)
    X, Y_expected = gt_small.get("backward", k, seed=323, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


def test_none_hint_emits_no_warning(primary_grg_path, gt_small, spmv_cache_dir):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, k_hint=None)
    X, _ = gt_small.get("forward", 3, seed=4101, dtype=DATA_DTYPE)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _run_up(op, X)
    assert len(rec) == 0


def test_warn_every_mismatch_call(primary_grg_path, gt_small, spmv_cache_dir):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, k_hint=4)
    X1, _ = gt_small.get("forward", 1, seed=4102, dtype=DATA_DTYPE)
    X2, _ = gt_small.get("forward", 2, seed=4103, dtype=DATA_DTYPE)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _run_up(op, X1)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _run_up(op, X1)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _run_up(op, X2)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
@pytest.mark.parametrize("k", K_MATRIX)
def test_primary_grg_forward(primary_grg_path, gt_primary, spmv_cache_dir, fmt, k):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up=fmt, n_threads=1)
    X, Y_expected = gt_primary.get("forward", k, seed=5123, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
@pytest.mark.parametrize("k", K_MATRIX)
def test_primary_grg_backward(primary_grg_path, gt_primary, spmv_cache_dir, fmt, k):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up=fmt, n_threads=1)
    X, Y_expected = gt_primary.get("backward", k, seed=5123, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


def test_mkl_bsr_not_supported(primary_grg_path, spmv_cache_dir):
    with pytest.raises(ValueError, match="fmt_up"):
        _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="bsr")


def test_mem_usage_static_dedupe_components(primary_grg_path, spmv_cache_dir):
    baseline = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="csr", fmt_down="csr", k_hint=None)
    optimized = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="csr", fmt_down=None, k_hint=None)

    base_static = baseline._backend.mem_usage.host_static
    opt_static = optimized._backend.mem_usage.host_static
    assert int(base_static.blocks_down) > 0
    assert int(opt_static.blocks_down) == 0
    assert int(opt_static.blocks_up) == int(base_static.blocks_up)
    assert int(opt_static.total()) < int(base_static.total())
