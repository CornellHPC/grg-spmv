"""MKL-specific tests for MklBackend."""

import numpy as np
import pytest

from spmv import SpMVOperator
from spmv.tests.conftest import DATA_DTYPE, INDEX_DTYPE, tol

pytestmark = pytest.mark.mkl


def _make_mkl_op(grg_path, n_threads=0, dtype=DATA_DTYPE):
    return SpMVOperator(
        grg_path,
        {'type': 'mkl', 'n_threads': n_threads},
        dtype, INDEX_DTYPE,
    )


class TestThreadCounts:
    """Verify correctness across different MKL thread counts."""

    @pytest.mark.parametrize("n_threads", [0, 1, 2, 4, 8])
    def test_forward(self, small_grg_path, gt_small, n_threads):
        op = _make_mkl_op(small_grg_path, n_threads=n_threads)
        X, Y_expected = gt_small.get('forward', 4, seed=123, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op.H @ X, Y_expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("n_threads", [0, 1, 2, 4, 8])
    def test_backward(self, small_grg_path, gt_small, n_threads):
        op = _make_mkl_op(small_grg_path, n_threads=n_threads)
        X, Y_expected = gt_small.get('backward', 4, seed=123, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op @ X, Y_expected, atol=atol, rtol=rtol)


class TestLargeGRG:
    """Heavy-weight thread scaling tests on the large GRG."""

    @pytest.mark.parametrize("n_threads", [0, 1, 4])
    def test_forward(self, large_grg_path, gt_large, n_threads):
        op = _make_mkl_op(large_grg_path, n_threads=n_threads)
        X, Y_expected = gt_large.get('forward', 4, seed=123, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op.H @ X, Y_expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("n_threads", [0, 1, 4])
    def test_backward(self, large_grg_path, gt_large, n_threads):
        op = _make_mkl_op(large_grg_path, n_threads=n_threads)
        X, Y_expected = gt_large.get('backward', 4, seed=123, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op @ X, Y_expected, atol=atol, rtol=rtol)
