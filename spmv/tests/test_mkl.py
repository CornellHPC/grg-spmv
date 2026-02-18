"""MKL-specific tests for MklBackend."""

import numpy as np
import pytest

from spmv import SpMVOperator
from spmv.tests.conftest import DATA_DTYPE, INDEX_DTYPE, K_SMOKE, K_FULL, tol

pytestmark = pytest.mark.mkl

MKL_FMTS = ['csr', 'csc', 'coo']


def _make_mkl_op(grg_path, n_threads=0, fmt='csr', k_hint=1,
                 dtype=DATA_DTYPE):
    return SpMVOperator(
        grg_path,
        {'type': 'mkl', 'n_threads': n_threads, 'fmt': fmt, 'k_hint': k_hint},
        dtype, INDEX_DTYPE,
    )


class TestThreadCounts:
    """Verify correctness across different MKL thread counts."""

    @pytest.mark.parametrize("n_threads", [0, 1, 2, 4, 8])
    @pytest.mark.parametrize("k", K_SMOKE)
    def test_forward(self, small_grg_path, gt_small, n_threads, k):
        op = _make_mkl_op(small_grg_path, n_threads=n_threads)
        X, Y_expected = gt_small.get('forward', k, seed=123, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op.H @ X, Y_expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("n_threads", [0, 1, 2, 4, 8])
    @pytest.mark.parametrize("k", K_SMOKE)
    def test_backward(self, small_grg_path, gt_small, n_threads, k):
        op = _make_mkl_op(small_grg_path, n_threads=n_threads)
        X, Y_expected = gt_small.get('backward', k, seed=123, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op @ X, Y_expected, atol=atol, rtol=rtol)


class TestFormats:
    """Verify correctness across sparse formats."""

    @pytest.mark.parametrize("fmt", MKL_FMTS)
    @pytest.mark.parametrize("k", K_SMOKE)
    def test_forward(self, small_grg_path, gt_small, fmt, k):
        op = _make_mkl_op(small_grg_path, fmt=fmt)
        X, Y_expected = gt_small.get('forward', k, seed=123, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op.H @ X, Y_expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("fmt", MKL_FMTS)
    @pytest.mark.parametrize("k", K_SMOKE)
    def test_backward(self, small_grg_path, gt_small, fmt, k):
        op = _make_mkl_op(small_grg_path, fmt=fmt)
        X, Y_expected = gt_small.get('backward', k, seed=123, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op @ X, Y_expected, atol=atol, rtol=rtol)


class TestKHintInvariance:
    """Results must be identical regardless of k_hint (only perf differs)."""

    @pytest.mark.parametrize("k_hint", [1, 4])
    @pytest.mark.parametrize("k", K_SMOKE)
    def test_forward(self, small_grg_path, gt_small, k_hint, k):
        op = _make_mkl_op(small_grg_path, k_hint=k_hint)
        X, Y_expected = gt_small.get('forward', k, seed=123, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op.H @ X, Y_expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("k_hint", [1, 4])
    @pytest.mark.parametrize("k", K_SMOKE)
    def test_backward(self, small_grg_path, gt_small, k_hint, k):
        op = _make_mkl_op(small_grg_path, k_hint=k_hint)
        X, Y_expected = gt_small.get('backward', k, seed=123, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op @ X, Y_expected, atol=atol, rtol=rtol)


class TestLargeGRG:
    """Heavy-weight tests on the large GRG."""

    @pytest.mark.parametrize("n_threads", [0, 1, 4])
    @pytest.mark.parametrize("fmt", ['csr', 'csc'])
    @pytest.mark.parametrize("k", K_SMOKE)
    def test_forward(self, large_grg_path, gt_large, n_threads, fmt, k):
        op = _make_mkl_op(large_grg_path, n_threads=n_threads, fmt=fmt)
        X, Y_expected = gt_large.get('forward', k, seed=123, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op.H @ X, Y_expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("n_threads", [0, 1, 4])
    @pytest.mark.parametrize("fmt", ['csr', 'csc'])
    @pytest.mark.parametrize("k", K_SMOKE)
    def test_backward(self, large_grg_path, gt_large, n_threads, fmt, k):
        op = _make_mkl_op(large_grg_path, n_threads=n_threads, fmt=fmt)
        X, Y_expected = gt_large.get('backward', k, seed=123, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op @ X, Y_expected, atol=atol, rtol=rtol)
