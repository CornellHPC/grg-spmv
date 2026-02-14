"""
GPU-specific tests for CusparseBackend.

All tests are skipped if CuPy is not available.
Run on a GPU node: uv run python -u -m pytest spmv/tests/test_cusparse.py -v
"""

import numpy as np
import pytest

from spmv import SpMVOperator
from spmv.tests.conftest import (
    DATA_DTYPE, INDEX_DTYPE, K_FULL,
    make_fmt_alg_params, valid_fmt_alg_params, binary_pm1, tol,
)

# Skip entire module if no GPU
cp = pytest.importorskip('cupy')

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_op(grg_path, fmt='csr', k=None, algorithm='default', verbose=False,
             dtype=DATA_DTYPE):
    return SpMVOperator(grg_path, {
        'type': 'cusparse', 'fmt': fmt, 'k': k,
        'algorithm': algorithm, 'verbose': verbose,
    }, dtype, INDEX_DTYPE)


# ---------------------------------------------------------------------------
# TestFormatAlgorithmValidity — construct operator for all combos
# ---------------------------------------------------------------------------

class TestFormatAlgorithmValidity:
    """Verify that valid (fmt, alg) combos construct, invalid ones raise ValueError."""

    @pytest.mark.parametrize("fmt,alg", make_fmt_alg_params())
    def test_construct(self, small_grg_path, fmt, alg):
        _make_op(small_grg_path, fmt=fmt, k=4, algorithm=alg)


# ---------------------------------------------------------------------------
# TestFmtAlgCorrectness — forward+backward for all valid combos
# ---------------------------------------------------------------------------

class TestFmtAlgCorrectness:
    """Correctness for every valid (fmt, alg) combo in dynamic and graph modes."""

    @pytest.mark.parametrize("fmt,alg", make_fmt_alg_params())
    def test_forward_dynamic(self, small_grg_path, gt_small, fmt, alg):
        op = _make_op(small_grg_path, fmt=fmt, k=None, algorithm=alg)
        X, Y_expected = gt_small.get('forward', 4, seed=42, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op.H @ X, Y_expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("fmt,alg", make_fmt_alg_params())
    def test_backward_dynamic(self, small_grg_path, gt_small, fmt, alg):
        op = _make_op(small_grg_path, fmt=fmt, k=None, algorithm=alg)
        X, Y_expected = gt_small.get('backward', 4, seed=42, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op @ X, Y_expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("fmt,alg", make_fmt_alg_params())
    def test_forward_graph(self, small_grg_path, gt_small, fmt, alg):
        op = _make_op(small_grg_path, fmt=fmt, k=4, algorithm=alg)
        X, Y_expected = gt_small.get('forward', 4, seed=42, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op.H @ X, Y_expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("fmt,alg", make_fmt_alg_params())
    def test_backward_graph(self, small_grg_path, gt_small, fmt, alg):
        op = _make_op(small_grg_path, fmt=fmt, k=4, algorithm=alg)
        X, Y_expected = gt_small.get('backward', 4, seed=42, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op @ X, Y_expected, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# TestKSweepGraph — CUDA graph capture at every k in K_FULL
# ---------------------------------------------------------------------------

class TestKSweepGraph:
    """Test CUDA graph capture and replay at every k value across fmt/alg combos."""

    @pytest.mark.parametrize("fmt,alg", valid_fmt_alg_params())
    @pytest.mark.parametrize("k", K_FULL)
    def test_forward(self, small_grg_path, gt_small, fmt, alg, k):
        op = _make_op(small_grg_path, fmt=fmt, k=k, algorithm=alg)
        X, Y_expected = gt_small.get('forward', k, seed=42, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op.H @ X, Y_expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("fmt,alg", valid_fmt_alg_params())
    @pytest.mark.parametrize("k", K_FULL)
    def test_backward(self, small_grg_path, gt_small, fmt, alg, k):
        op = _make_op(small_grg_path, fmt=fmt, k=k, algorithm=alg)
        X, Y_expected = gt_small.get('backward', k, seed=42, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op @ X, Y_expected, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# TestGraphBehavior — graph vs dynamic, fallback, first-call, interleaved
# ---------------------------------------------------------------------------

class TestGraphBehavior:
    """Graph-specific behavior tests."""

    @pytest.mark.parametrize("fmt,alg", valid_fmt_alg_params())
    def test_graph_vs_dynamic(self, small_grg_path, fmt, alg):
        """Graph and dynamic modes produce identical results."""
        op_graph = _make_op(small_grg_path, fmt=fmt, k=4, algorithm=alg)
        op_dynamic = _make_op(small_grg_path, fmt=fmt, k=None, algorithm=alg)
        rng = np.random.default_rng(7001)
        V = rng.standard_normal((op_graph.n, 4), dtype=DATA_DTYPE)
        W = rng.standard_normal((op_graph.m, 4), dtype=DATA_DTYPE)
        np.testing.assert_allclose(op_graph.H @ V, op_dynamic.H @ V, atol=1e-5)
        np.testing.assert_allclose(op_graph @ W, op_dynamic @ W, atol=1e-5)

    def test_graph_fallback(self, small_grg_path, gt_small):
        """Graph k=4 operator falls back to dynamic for k=1, then resumes graph for k=4."""
        op = _make_op(small_grg_path, fmt='csr', k=4)
        # k=4 via graph
        X4, Y4_exp = gt_small.get('backward', 4, seed=5001, dtype=DATA_DTYPE)
        Y4 = op @ X4
        np.testing.assert_allclose(Y4, Y4_exp, atol=1e-5, rtol=1e-5)
        # k=1 via dynamic fallback
        X1, Y1_exp = gt_small.get('backward', 1, seed=5002, dtype=DATA_DTYPE)
        Y1 = op @ X1
        np.testing.assert_allclose(Y1, Y1_exp, atol=1e-5, rtol=1e-5)
        # k=4 again via graph — must not be corrupted
        Y4_again = op @ X4
        np.testing.assert_allclose(Y4_again, Y4, atol=1e-5, rtol=1e-5)

    def test_graph_first_call(self, small_grg_path, gt_small):
        """Fresh graph operator: first call must be correct (stale buffer regression)."""
        op = _make_op(small_grg_path, fmt='csr', k=4)
        X, Y_expected = gt_small.get('forward', 4, seed=5003, dtype=DATA_DTYPE)
        Y_actual = op.H @ X
        np.testing.assert_allclose(Y_actual, Y_expected, atol=1e-5, rtol=1e-5)

    def test_interleaved_fwd_bwd(self, small_grg_path, gt_small):
        """Interleaved fwd -> bwd -> fwd: verify consistency."""
        op = _make_op(small_grg_path, fmt='csr', k=4)
        Xf, Yf_exp = gt_small.get('forward', 4, seed=5004, dtype=DATA_DTYPE)
        Xb, Yb_exp = gt_small.get('backward', 4, seed=5005, dtype=DATA_DTYPE)
        Y_fwd1 = op.H @ Xf
        Y_bwd = op @ Xb
        Y_fwd2 = op.H @ Xf
        np.testing.assert_allclose(Y_fwd1, Yf_exp, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(Y_bwd, Yb_exp, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(Y_fwd2, Y_fwd1, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# TestLargeGRG — binary exact tests on the large GRG
# ---------------------------------------------------------------------------

class TestLargeGRG:
    """Exact binary and allele count tests on the large GRG."""

    @pytest.mark.parametrize("fmt", ['csr', 'csc', 'coo'])
    def test_exact_forward(self, large_grg_path, gt_large, fmt):
        op = _make_op(large_grg_path, fmt=fmt, k=None)
        X, Y_expected = gt_large.get('forward', 4, seed=8001, dtype=DATA_DTYPE,
                                     input_fn=binary_pm1)
        np.testing.assert_array_equal(op.H @ X, Y_expected)

    @pytest.mark.parametrize("fmt", ['csr', 'csc', 'coo'])
    def test_exact_backward(self, large_grg_path, gt_large, fmt):
        op = _make_op(large_grg_path, fmt=fmt, k=None)
        X, Y_expected = gt_large.get('backward', 4, seed=8001, dtype=DATA_DTYPE,
                                     input_fn=binary_pm1)
        np.testing.assert_array_equal(op @ X, Y_expected)

    def test_allele_counts(self, large_grg_path, gt_large):
        """op.H @ ones on large GRG: exact integer equality."""
        def _ones_input(rng, shape, dtype):
            return np.ones(shape, dtype=dtype)
        _ones_input.__name__ = "ones"

        op = _make_op(large_grg_path, fmt='csr', k=None)
        X, Y_expected = gt_large.get('forward', 1, seed=0, dtype=DATA_DTYPE,
                                     input_fn=_ones_input)
        np.testing.assert_array_equal(op.H @ X, Y_expected)

    def test_dynamic_csr(self, large_grg_path, gt_large):
        """Dynamic CSR on large GRG, approximate."""
        op = _make_op(large_grg_path, fmt='csr', k=None)
        X, Y_expected = gt_large.get('forward', 4, seed=8002, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op.H @ X, Y_expected, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# TestFailureModes — boundary conditions and regressions
# ---------------------------------------------------------------------------

class TestFailureModes:
    """Boundary conditions: k=1 graph, k=2, odd k, F-order, stale buffer."""

    def test_k1_graph(self, small_grg_path, gt_small):
        """k=1 graph: cuSPARSE may dispatch to SpMV internally."""
        op = _make_op(small_grg_path, fmt='csr', k=1)
        X, Y_expected = gt_small.get('forward', 1, seed=9001, dtype=DATA_DTYPE)
        np.testing.assert_allclose(op.H @ X, Y_expected, atol=1e-5, rtol=1e-5)

    def test_k2_graph(self, small_grg_path, gt_small):
        """k=2 graph: old k_spmm = max(k, 2) hack regression."""
        op = _make_op(small_grg_path, fmt='csr', k=2)
        X, Y_expected = gt_small.get('backward', 2, seed=9002, dtype=DATA_DTYPE)
        np.testing.assert_allclose(op @ X, Y_expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("k", [3, 7, 9])
    def test_odd_k(self, small_grg_path, gt_small, k):
        """Non-power-of-2 k: CUDA tile fallback paths."""
        op = _make_op(small_grg_path, fmt='csr', k=k)
        X, Y_expected = gt_small.get('forward', k, seed=9003, dtype=DATA_DTYPE)
        np.testing.assert_allclose(op.H @ X, Y_expected, atol=1e-5, rtol=1e-5)

    def test_f_order_input(self, small_grg_path, gt_small):
        """Fortran-order input array handling."""
        op = _make_op(small_grg_path, fmt='csr', k=None)
        X, Y_expected = gt_small.get('forward', 4, seed=9004, dtype=DATA_DTYPE)
        X_f = np.asfortranarray(X)
        np.testing.assert_allclose(op.H @ X_f, Y_expected, atol=1e-5, rtol=1e-5)

    def test_graph_first_call_fresh(self, small_grg_path, gt_small):
        """Fresh operator: first call correct (stale buffer after graph capture)."""
        op = _make_op(small_grg_path, fmt='csr', k=4)
        X, Y_expected = gt_small.get('backward', 4, seed=9005, dtype=DATA_DTYPE)
        Y_actual = op @ X
        np.testing.assert_allclose(Y_actual, Y_expected, atol=1e-5, rtol=1e-5)
