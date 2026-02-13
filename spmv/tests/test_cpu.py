"""
CPU-specific tests for the multithread backend.

Tests that are specific to the CPU multithread backend: structure validation,
worker count scaling, sequential vs parallel consistency, and RCM reordering.

Correctness vs ground truth is covered by test_correctness.py.
"""

import numpy as np
import pytest

from spmv import SpMVOperator
from spmv.tests.conftest import DATA_DTYPE, INDEX_DTYPE

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def example_grg(small_grg_path):
    import pygrgl
    return pygrgl.load_immutable_grg(small_grg_path)


@pytest.fixture(scope="module")
def spmv_seq(small_grg_path):
    """Sequential SpMVOperator (n_workers=1, no RCM)."""
    from pathlib import Path
    path = Path(small_grg_path)
    for f in path.parent.glob(f"{path.stem}*.npz"):
        f.unlink()
    return SpMVOperator(
        small_grg_path,
        {'type': 'multithread', 'n_workers': 1, 'chunk_size': 4096, 'verbose': True},
        DATA_DTYPE, INDEX_DTYPE, use_rcm=False,
    )


@pytest.fixture(scope="module")
def spmv_par(small_grg_path):
    """Parallel SpMVOperator (n_workers=4, no RCM)."""
    return SpMVOperator(
        small_grg_path,
        {'type': 'multithread', 'n_workers': 4, 'chunk_size': 4096},
        DATA_DTYPE, INDEX_DTYPE, use_rcm=False,
    )


@pytest.fixture(scope="module")
def spmv_rcm(small_grg_path):
    """SpMVOperator with RCM reordering."""
    return SpMVOperator(
        small_grg_path,
        {'type': 'multithread', 'n_workers': 1, 'chunk_size': 4096},
        DATA_DTYPE, INDEX_DTYPE, use_rcm=True,
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
class TestStructure:
    """Tests for operator structure."""

    def test_shape(self, example_grg, spmv_seq):
        assert spmv_seq.shape == (example_grg.num_samples, example_grg.num_mutations)

    def test_level_offsets(self, spmv_seq):
        off = spmv_seq.level_offsets
        assert off[0] == 0 and off[-1] == spmv_seq.K
        assert np.all(np.diff(off) > 0)

    def test_slices_structure(self, spmv_seq):
        """Forward/backward blocks cover all edges with matching nnz."""
        off = spmv_seq.level_offsets
        num_levels = len(off) - 1
        backend = spmv_seq._backend
        assert len(backend._A_blocks) == num_levels
        assert len(backend._AT_blocks) == num_levels
        total_fwd = sum(blk.nnz for blocks in backend._A_blocks for blk in blocks)
        total_bwd = sum(blk.nnz for blocks in backend._AT_blocks for blk in blocks)
        assert total_fwd == total_bwd
        assert total_fwd > 0

    def test_selector_shape(self, spmv_seq):
        n, m = spmv_seq.shape
        assert spmv_seq.sel.shape == (m, spmv_seq.K)
        assert spmv_seq.sel_T.shape == (spmv_seq.K, m)


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
class TestDifferentWorkerCounts:
    """Test correctness with various worker counts vs ground truth."""

    @pytest.mark.parametrize("n_workers", [1, 2, 4, 8])
    def test_forward(self, small_grg_path, gt_small, n_workers):
        op = SpMVOperator(
            small_grg_path,
            {'type': 'multithread', 'n_workers': n_workers, 'chunk_size': 4096},
            DATA_DTYPE, INDEX_DTYPE, use_rcm=False,
        )
        X, Y_expected = gt_small.get('forward', 4, seed=123, dtype=DATA_DTYPE)
        np.testing.assert_allclose(op.H @ X, Y_expected)

    @pytest.mark.parametrize("n_workers", [1, 2, 4, 8])
    def test_backward(self, small_grg_path, gt_small, n_workers):
        op = SpMVOperator(
            small_grg_path,
            {'type': 'multithread', 'n_workers': n_workers, 'chunk_size': 4096},
            DATA_DTYPE, INDEX_DTYPE, use_rcm=False,
        )
        X, Y_expected = gt_small.get('backward', 4, seed=123, dtype=DATA_DTYPE)
        np.testing.assert_allclose(op @ X, Y_expected)


# ---------------------------------------------------------------------------
class TestRCMReordering:
    """Test RCM reordering produces correct results vs ground truth."""

    def test_rcm_forward(self, spmv_rcm, gt_small):
        X, Y_expected = gt_small.get('forward', 4, seed=42, dtype=DATA_DTYPE)
        np.testing.assert_allclose(spmv_rcm.H @ X, Y_expected)

    def test_rcm_backward(self, spmv_rcm, gt_small):
        X, Y_expected = gt_small.get('backward', 4, seed=42, dtype=DATA_DTYPE)
        np.testing.assert_allclose(spmv_rcm @ X, Y_expected)
