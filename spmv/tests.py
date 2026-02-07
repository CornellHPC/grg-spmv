"""
Tests for SpMVOperator.

Tests correctness for both sequential and parallel execution against
the golden standard (grapp.linalg.ops_scipy.SciPyXOperator).
"""

import numpy as np
import pytest
from pathlib import Path

from spmv import SpMVOperator


# Use small file for faster tests
SMALL_GRG = "/pscratch/sd/q/qys/grg/msprime.example.igd.final.grg"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def grg_path():
    return SMALL_GRG


@pytest.fixture(scope="session")
def example_grg(grg_path):
    import pygrgl
    return pygrgl.load_immutable_grg(grg_path)


@pytest.fixture(scope="session")
def ref_op(example_grg):
    """Golden standard: grapp SciPyXOperator."""
    import pygrgl
    from grapp.linalg.ops_scipy import SciPyXOperator
    return SciPyXOperator(example_grg, pygrgl.TraversalDirection.UP, haploid=True)


@pytest.fixture(scope="session")
def spmv_seq(grg_path):
    """Sequential SpMVOperator (n_workers=1)."""
    path = Path(grg_path)
    for f in path.parent.glob(f"{path.stem}*.npz"):
        f.unlink()
    backend_config = {'type': 'multithread', 'n_workers': 1, 'chunk_size': 4096, 'verbose': True}
    return SpMVOperator(grg_path, backend_config=backend_config, use_rcm=False)


@pytest.fixture(scope="session")
def spmv_par(grg_path):
    """Parallel SpMVOperator (n_workers=4)."""
    backend_config = {'type': 'multithread', 'n_workers': 4, 'chunk_size': 4096, 'verbose': False}
    return SpMVOperator(grg_path, backend_config=backend_config, use_rcm=False)


@pytest.fixture(scope="session")
def spmv_rcm(grg_path):
    """SpMVOperator with RCM reordering."""
    backend_config = {'type': 'multithread', 'n_workers': 1, 'chunk_size': 4096, 'verbose': False}
    return SpMVOperator(grg_path, backend_config=backend_config, use_rcm=True)


@pytest.fixture
def vecs(spmv_seq):
    """Test vectors."""
    rng = np.random.default_rng(42)
    n, m = spmv_seq.shape
    return dict(
        v=rng.standard_normal(n, dtype=np.float32),
        w=rng.standard_normal(m, dtype=np.float32),
        V=rng.standard_normal((n, 5), dtype=np.float32),
        W=rng.standard_normal((m, 5), dtype=np.float32),
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
        """Forward/backward slices cover all edges with matching nnz."""
        off = spmv_seq.level_offsets
        num_levels = len(off) - 1
        # A_fwd and AT_bwd are now owned by the backend
        backend = spmv_seq._backend
        assert len(backend._A_fwd) == num_levels
        assert len(backend._AT_bwd) == num_levels
        total_fwd = sum(s.nnz for s in backend._A_fwd)
        total_bwd = sum(s.nnz for s in backend._AT_bwd)
        assert total_fwd == total_bwd
        assert total_fwd > 0

    def test_selector_shape(self, spmv_seq):
        n, m = spmv_seq.shape
        assert spmv_seq.sel.shape == (m, spmv_seq.K)
        assert spmv_seq.sel_T.shape == (spmv_seq.K, m)


# ---------------------------------------------------------------------------
class TestSequentialVsReference:
    """Test sequential SpMVOperator matches golden standard."""
    
    def test_matvec_G(self, spmv_seq, ref_op, vecs):
        """G @ w matches reference."""
        result = spmv_seq @ vecs['w']
        expected = ref_op @ vecs['w']
        assert np.allclose(result, expected), f"max diff: {np.max(np.abs(result - expected))}"

    def test_matvec_GT(self, spmv_seq, ref_op, vecs):
        """G^T @ v matches reference."""
        result = spmv_seq.H @ vecs['v']
        expected = ref_op.H @ vecs['v']
        assert np.allclose(result, expected), f"max diff: {np.max(np.abs(result - expected))}"

    def test_matmat_G(self, spmv_seq, ref_op, vecs):
        """G @ W matches reference."""
        result = spmv_seq @ vecs['W']
        expected = ref_op @ vecs['W']
        assert np.allclose(result, expected), f"max diff: {np.max(np.abs(result - expected))}"

    def test_matmat_GT(self, spmv_seq, ref_op, vecs):
        """G^T @ V matches reference."""
        result = spmv_seq.H @ vecs['V']
        expected = ref_op.H @ vecs['V']
        assert np.allclose(result, expected), f"max diff: {np.max(np.abs(result - expected))}"


# ---------------------------------------------------------------------------
class TestParallelVsReference:
    """Test parallel SpMVOperator matches golden standard."""
    
    def test_matvec_G(self, spmv_par, ref_op, vecs):
        """Parallel G @ w matches reference."""
        result = spmv_par @ vecs['w']
        expected = ref_op @ vecs['w']
        assert np.allclose(result, expected), f"max diff: {np.max(np.abs(result - expected))}"

    def test_matvec_GT(self, spmv_par, ref_op, vecs):
        """Parallel G^T @ v matches reference."""
        result = spmv_par.H @ vecs['v']
        expected = ref_op.H @ vecs['v']
        assert np.allclose(result, expected), f"max diff: {np.max(np.abs(result - expected))}"

    def test_matmat_G(self, spmv_par, ref_op, vecs):
        """Parallel G @ W matches reference."""
        result = spmv_par @ vecs['W']
        expected = ref_op @ vecs['W']
        assert np.allclose(result, expected), f"max diff: {np.max(np.abs(result - expected))}"

    def test_matmat_GT(self, spmv_par, ref_op, vecs):
        """Parallel G^T @ V matches reference."""
        result = spmv_par.H @ vecs['V']
        expected = ref_op.H @ vecs['V']
        assert np.allclose(result, expected), f"max diff: {np.max(np.abs(result - expected))}"


# ---------------------------------------------------------------------------
class TestSequentialVsParallel:
    """Test that sequential and parallel produce identical results."""
    
    def test_matvec_G(self, spmv_seq, spmv_par, vecs):
        """Sequential and parallel G @ w are identical."""
        result_seq = spmv_seq @ vecs['w']
        result_par = spmv_par @ vecs['w']
        assert np.allclose(result_seq, result_par), f"max diff: {np.max(np.abs(result_seq - result_par))}"

    def test_matvec_GT(self, spmv_seq, spmv_par, vecs):
        """Sequential and parallel G^T @ v are identical."""
        result_seq = spmv_seq.H @ vecs['v']
        result_par = spmv_par.H @ vecs['v']
        assert np.allclose(result_seq, result_par), f"max diff: {np.max(np.abs(result_seq - result_par))}"

    def test_matmat_G(self, spmv_seq, spmv_par, vecs):
        """Sequential and parallel G @ W are identical."""
        result_seq = spmv_seq @ vecs['W']
        result_par = spmv_par @ vecs['W']
        assert np.allclose(result_seq, result_par), f"max diff: {np.max(np.abs(result_seq - result_par))}"

    def test_matmat_GT(self, spmv_seq, spmv_par, vecs):
        """Sequential and parallel G^T @ V are identical."""
        result_seq = spmv_seq.H @ vecs['V']
        result_par = spmv_par.H @ vecs['V']
        assert np.allclose(result_seq, result_par), f"max diff: {np.max(np.abs(result_seq - result_par))}"


# ---------------------------------------------------------------------------
class TestAlgebraic:
    """Algebraic property tests."""
    
    def test_adjoint(self, spmv_seq, vecs):
        """<Gw, v> = <w, G^T v>."""
        lhs = np.dot(spmv_seq @ vecs['w'], vecs['v'])
        rhs = np.dot(vecs['w'], spmv_seq.H @ vecs['v'])
        assert np.isclose(lhs, rhs)

    def test_matmat_equals_matvec(self, spmv_seq, vecs):
        """G @ W column-by-column equals G @ w_i."""
        W = vecs['W']
        result_matmat = spmv_seq @ W
        result_matvec = np.column_stack([spmv_seq @ W[:, i] for i in range(W.shape[1])])
        assert np.allclose(result_matmat, result_matvec)

    def test_allele_counts(self, spmv_seq):
        """G^T @ 1 gives non-negative integer counts."""
        counts = spmv_seq.H @ np.ones(spmv_seq.shape[0], dtype=np.float32)
        assert np.all(counts >= 0)
        assert np.allclose(counts, np.round(counts))


# ---------------------------------------------------------------------------
class TestDifferentWorkerCounts:
    """Test correctness with various worker counts."""
    
    @pytest.mark.parametrize("n_workers", [1, 2, 4, 8])
    def test_matvec_G_workers(self, grg_path, ref_op, n_workers):
        """G @ w is correct for different worker counts."""
        backend_config = {'type': 'multithread', 'n_workers': n_workers, 'chunk_size': 4096, 'verbose': False}
        op = SpMVOperator(grg_path, backend_config=backend_config, use_rcm=False)
        rng = np.random.default_rng(123)
        w = rng.standard_normal(op.m, dtype=np.float32)
        result = op @ w
        expected = ref_op @ w
        assert np.allclose(result, expected), f"n_workers={n_workers}, max diff: {np.max(np.abs(result - expected))}"

    @pytest.mark.parametrize("n_workers", [1, 2, 4, 8])
    def test_matvec_GT_workers(self, grg_path, ref_op, n_workers):
        """G^T @ v is correct for different worker counts."""
        backend_config = {'type': 'multithread', 'n_workers': n_workers, 'chunk_size': 4096, 'verbose': False}
        op = SpMVOperator(grg_path, backend_config=backend_config, use_rcm=False)
        rng = np.random.default_rng(123)
        v = rng.standard_normal(op.n, dtype=np.float32)
        result = op.H @ v
        expected = ref_op.H @ v
        assert np.allclose(result, expected), f"n_workers={n_workers}, max diff: {np.max(np.abs(result - expected))}"


# ---------------------------------------------------------------------------
class TestRCMReordering:
    """Test RCM reordering produces correct results."""
    
    def test_rcm_matvec_G(self, spmv_rcm, ref_op, vecs):
        """RCM G @ w matches reference."""
        result = spmv_rcm @ vecs['w']
        expected = ref_op @ vecs['w']
        assert np.allclose(result, expected), f"max diff: {np.max(np.abs(result - expected))}"

    def test_rcm_matvec_GT(self, spmv_rcm, ref_op, vecs):
        """RCM G^T @ v matches reference."""
        result = spmv_rcm.H @ vecs['v']
        expected = ref_op.H @ vecs['v']
        assert np.allclose(result, expected), f"max diff: {np.max(np.abs(result - expected))}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
