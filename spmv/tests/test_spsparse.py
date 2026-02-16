"""
Spsparse-specific tests for SpsparseBackend.

Tests specific to the SciPy sparse + multithread backend: worker count scaling,
sequential vs parallel consistency.

Operator-level tests (structure, RCM, correctness) are in test_correctness.py.
"""

import numpy as np
import pytest

from spmv import SpMVOperator
from spmv.tests.conftest import DATA_DTYPE, INDEX_DTYPE, tol

pytestmark = pytest.mark.spsparse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spmv_seq(small_grg_path):
    """Sequential SpMVOperator (n_workers=1, no RCM)."""
    from pathlib import Path
    path = Path(small_grg_path)
    for f in path.parent.glob(f"{path.stem}*.npz"):
        f.unlink()
    return SpMVOperator(
        small_grg_path,
        {'type': 'spsparse', 'n_workers': 1, 'chunk_size': 4096, 'verbose': True},
        DATA_DTYPE, INDEX_DTYPE, use_rcm=False,
    )


@pytest.fixture(scope="module")
def spmv_par(small_grg_path):
    """Parallel SpMVOperator (n_workers=4, no RCM)."""
    return SpMVOperator(
        small_grg_path,
        {'type': 'spsparse', 'n_workers': 4, 'chunk_size': 4096},
        DATA_DTYPE, INDEX_DTYPE, use_rcm=False,
    )


@pytest.fixture
def vecs(spmv_seq):
    """Test vectors."""
    rng = np.random.default_rng(42)
    n, m = spmv_seq.shape
    return dict(
        v=rng.standard_normal(n, dtype=DATA_DTYPE),
        w=rng.standard_normal(m, dtype=DATA_DTYPE),
        V=rng.standard_normal((n, 5), dtype=DATA_DTYPE),
        W=rng.standard_normal((m, 5), dtype=DATA_DTYPE),
    )


# ---------------------------------------------------------------------------
class TestSequentialVsParallel:
    """Test that sequential and parallel produce identical results."""

    def test_matvec_G(self, spmv_seq, spmv_par, vecs):
        result_seq = spmv_seq @ vecs['w']
        result_par = spmv_par @ vecs['w']
        np.testing.assert_allclose(result_seq, result_par)

    def test_matvec_GT(self, spmv_seq, spmv_par, vecs):
        result_seq = spmv_seq.H @ vecs['v']
        result_par = spmv_par.H @ vecs['v']
        np.testing.assert_allclose(result_seq, result_par)

    def test_matmat_G(self, spmv_seq, spmv_par, vecs):
        result_seq = spmv_seq @ vecs['W']
        result_par = spmv_par @ vecs['W']
        np.testing.assert_allclose(result_seq, result_par)

    def test_matmat_GT(self, spmv_seq, spmv_par, vecs):
        result_seq = spmv_seq.H @ vecs['V']
        result_par = spmv_par.H @ vecs['V']
        np.testing.assert_allclose(result_seq, result_par)


# ---------------------------------------------------------------------------
class TestWorkerCounts:
    """Test correctness with various worker counts vs ground truth."""

    @pytest.mark.parametrize("n_workers", [1, 2, 4, 8])
    def test_forward(self, small_grg_path, gt_small, n_workers):
        op = SpMVOperator(
            small_grg_path,
            {'type': 'spsparse', 'n_workers': n_workers, 'chunk_size': 4096},
            DATA_DTYPE, INDEX_DTYPE, use_rcm=False,
        )
        X, Y_expected = gt_small.get('forward', 4, seed=123, dtype=DATA_DTYPE)
        np.testing.assert_allclose(op.H @ X, Y_expected)

    @pytest.mark.parametrize("n_workers", [1, 2, 4, 8])
    def test_backward(self, small_grg_path, gt_small, n_workers):
        op = SpMVOperator(
            small_grg_path,
            {'type': 'spsparse', 'n_workers': n_workers, 'chunk_size': 4096},
            DATA_DTYPE, INDEX_DTYPE, use_rcm=False,
        )
        X, Y_expected = gt_small.get('backward', 4, seed=123, dtype=DATA_DTYPE)
        np.testing.assert_allclose(op @ X, Y_expected)


# ---------------------------------------------------------------------------
class TestLargeGRG:
    """Heavy-weight worker scaling tests on the large GRG."""

    @pytest.mark.parametrize("n_workers", [1, 4])
    def test_forward(self, large_grg_path, gt_large, n_workers):
        op = SpMVOperator(
            large_grg_path,
            {'type': 'spsparse', 'n_workers': n_workers, 'chunk_size': 4096},
            DATA_DTYPE, INDEX_DTYPE,
        )
        X, Y_expected = gt_large.get('forward', 4, seed=123, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op.H @ X, Y_expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("n_workers", [1, 4])
    def test_backward(self, large_grg_path, gt_large, n_workers):
        op = SpMVOperator(
            large_grg_path,
            {'type': 'spsparse', 'n_workers': n_workers, 'chunk_size': 4096},
            DATA_DTYPE, INDEX_DTYPE,
        )
        X, Y_expected = gt_large.get('backward', 4, seed=123, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(op @ X, Y_expected, atol=atol, rtol=rtol)
