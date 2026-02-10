"""
Tests for CusparseBackend (GPU).

All tests are skipped if CuPy is not available.
Run on a GPU node: uv run python -u -m pytest spmv/tests/test_cusparse.py -v
"""

import pytest
import numpy as np
from spmv import SpMVOperator, DATA_DTYPE

# Skip entire module if no GPU
cp = pytest.importorskip('cupy')

GRG_PATH = "/pscratch/sd/q/qys/grg/msprime.example.igd.final.grg"


@pytest.fixture(scope="session")
def grg_path():
    return GRG_PATH


@pytest.fixture(scope="session")
def example_grg(grg_path):
    import pygrgl
    return pygrgl.load_immutable_grg(grg_path)


@pytest.fixture(scope="session")
def scipy_ref_op(example_grg):
    """Golden standard: grapp SciPyXOperator."""
    import pygrgl
    from grapp.linalg.ops_scipy import SciPyXOperator
    return SciPyXOperator(example_grg, pygrgl.TraversalDirection.UP, haploid=True)


@pytest.fixture(scope="session")
def ref_op(grg_path):
    """CPU reference operator (MultithreadBackend, 1 worker)."""
    return SpMVOperator(grg_path, {'type': 'multithread', 'n_workers': 1, 'chunk_size': 4096}, dtype=DATA_DTYPE)


# --- Parameterized fixtures for all GPU configurations ---

@pytest.fixture(params=['csr', 'csc'], ids=['CSR', 'CSC'])
def fmt(request):
    return request.param


@pytest.fixture(params=[None, 1, 4], ids=['dynamic', 'graph-k1', 'graph-k4'])
def k_param(request):
    return request.param


@pytest.fixture
def gpu_op(grg_path, fmt, k_param):
    """GPU operator for a given (fmt, k) combination."""
    return SpMVOperator(grg_path, {
        'type': 'cusparse', 'fmt': fmt, 'k': k_param, 'verbose': False
    }, dtype=DATA_DTYPE)


@pytest.fixture(scope="session")
def vecs(ref_op):
    rng = np.random.default_rng(42)
    n, m = ref_op.n, ref_op.m
    return {
        'w1': rng.standard_normal((m, 1), dtype=DATA_DTYPE),
        'w4': rng.standard_normal((m, 4), dtype=DATA_DTYPE),
        'v1': rng.standard_normal((n, 1), dtype=DATA_DTYPE),
        'v4': rng.standard_normal((n, 4), dtype=DATA_DTYPE),
    }


class TestCorrectnessVsCPU:
    """GPU results must match CPU MultithreadBackend within tolerance."""

    def test_matvec_G_k1(self, ref_op, gpu_op, vecs):
        """G @ w (single vector)."""
        expected = ref_op @ vecs['w1']
        actual = gpu_op @ vecs['w1']
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_matvec_GT_k1(self, ref_op, gpu_op, vecs):
        """G^T @ v (single vector)."""
        expected = ref_op.T @ vecs['v1']
        actual = gpu_op.T @ vecs['v1']
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_matmat_G_k4(self, ref_op, gpu_op, vecs):
        """G @ W (4 vectors)."""
        expected = ref_op @ vecs['w4']
        actual = gpu_op @ vecs['w4']
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_matmat_GT_k4(self, ref_op, gpu_op, vecs):
        """G^T @ V (4 vectors)."""
        expected = ref_op.T @ vecs['v4']
        actual = gpu_op.T @ vecs['v4']
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


class TestCorrectnessVsSciPy:
    """GPU results must match grapp SciPyXOperator golden standard."""

    @pytest.fixture
    def gpu_csr(self, grg_path):
        return SpMVOperator(grg_path, {'type': 'cusparse', 'fmt': 'csr', 'k': 4}, dtype=DATA_DTYPE)

    def test_matvec_G(self, gpu_csr, scipy_ref_op, vecs):
        """GPU G @ w matches SciPyXOperator."""
        expected = scipy_ref_op @ vecs['w1']
        actual = gpu_csr @ vecs['w1']
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_matvec_GT(self, gpu_csr, scipy_ref_op, vecs):
        """GPU G^T @ v matches SciPyXOperator."""
        expected = scipy_ref_op.H @ vecs['v1']
        actual = gpu_csr.H @ vecs['v1']
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_matmat_G(self, gpu_csr, scipy_ref_op, vecs):
        """GPU G @ W matches SciPyXOperator."""
        expected = scipy_ref_op @ vecs['w4']
        actual = gpu_csr @ vecs['w4']
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_matmat_GT(self, gpu_csr, scipy_ref_op, vecs):
        """GPU G^T @ V matches SciPyXOperator."""
        expected = scipy_ref_op.H @ vecs['v4']
        actual = gpu_csr.H @ vecs['v4']
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


class TestGraphVsDynamic:
    """Graph mode and dynamic mode must produce identical results."""

    @pytest.fixture
    def gpu_graph(self, grg_path):
        return SpMVOperator(grg_path, {'type': 'cusparse', 'k': 4}, dtype=DATA_DTYPE)

    @pytest.fixture
    def gpu_dynamic(self, grg_path):
        return SpMVOperator(grg_path, {'type': 'cusparse', 'k': None}, dtype=DATA_DTYPE)

    def test_forward_identical(self, gpu_graph, gpu_dynamic, vecs):
        r_graph = gpu_graph.T @ vecs['v4']
        r_dynamic = gpu_dynamic.T @ vecs['v4']
        np.testing.assert_allclose(r_graph, r_dynamic, atol=1e-5)

    def test_backward_identical(self, gpu_graph, gpu_dynamic, vecs):
        r_graph = gpu_graph @ vecs['w4']
        r_dynamic = gpu_dynamic @ vecs['w4']
        np.testing.assert_allclose(r_graph, r_dynamic, atol=1e-5)


class TestGraphFallback:
    """When graph was captured for k=4 but called with k=1, falls back to dynamic."""

    @pytest.fixture
    def gpu_k4(self, grg_path):
        return SpMVOperator(grg_path, {'type': 'cusparse', 'k': 4}, dtype=DATA_DTYPE)

    def test_fallback_k1_forward(self, gpu_k4, ref_op, vecs):
        """Call with k=1 on an operator whose graph was captured for k=4."""
        expected = ref_op.T @ vecs['v1']
        actual = gpu_k4.T @ vecs['v1']
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_fallback_k1_backward(self, gpu_k4, ref_op, vecs):
        expected = ref_op @ vecs['w1']
        actual = gpu_k4 @ vecs['w1']
        np.testing.assert_allclose(actual, expected, atol=1e-5)


class TestRepeatedCalls:
    """Graph replay must be deterministic across repeated calls."""

    @pytest.fixture
    def gpu(self, grg_path):
        return SpMVOperator(grg_path, {'type': 'cusparse', 'k': 4}, dtype=DATA_DTYPE)

    def test_repeated_forward(self, gpu, vecs):
        results = [gpu.T @ vecs['v4'] for _ in range(5)]
        for r in results[1:]:
            np.testing.assert_allclose(r, results[0], atol=1e-5, rtol=1e-5)

    def test_repeated_backward(self, gpu, vecs):
        results = [gpu @ vecs['w4'] for _ in range(5)]
        for r in results[1:]:
            np.testing.assert_allclose(r, results[0], atol=1e-5, rtol=1e-5)


class TestDifferentInputs:
    """Different inputs produce different (correct) outputs — not stale cached results."""

    @pytest.fixture
    def gpu(self, grg_path):
        return SpMVOperator(grg_path, {'type': 'cusparse', 'k': 4}, dtype=DATA_DTYPE)

    def test_different_inputs_forward(self, gpu, ref_op):
        rng = np.random.default_rng(123)
        for _ in range(3):
            v = rng.standard_normal((ref_op.n, 4), dtype=DATA_DTYPE)
            np.testing.assert_allclose(gpu.T @ v, ref_op.T @ v, atol=5e-5, rtol=1e-5)

    def test_different_inputs_backward(self, gpu, ref_op):
        rng = np.random.default_rng(456)
        for _ in range(3):
            w = rng.standard_normal((ref_op.m, 4), dtype=DATA_DTYPE)
            np.testing.assert_allclose(gpu @ w, ref_op @ w, atol=5e-5, rtol=1e-5)


class TestAlgebraic:
    """Algebraic properties must hold for GPU backend."""

    @pytest.fixture
    def gpu(self, grg_path):
        return SpMVOperator(grg_path, {'type': 'cusparse', 'k': 4}, dtype=DATA_DTYPE)

    def test_adjoint_identity(self, gpu):
        """<Gw, v> == <w, G^T v>."""
        rng = np.random.default_rng(789)
        w = rng.standard_normal((gpu.m, 1), dtype=DATA_DTYPE)
        v = rng.standard_normal((gpu.n, 1), dtype=DATA_DTYPE)
        lhs = (gpu @ w).T @ v
        rhs = w.T @ (gpu.T @ v)
        np.testing.assert_allclose(lhs, rhs, atol=1e-3, rtol=1e-3)


class TestCSRvsCSC:
    """CSR and CSC formats must produce identical results."""

    @pytest.fixture
    def gpu_csr(self, grg_path):
        return SpMVOperator(grg_path, {'type': 'cusparse', 'fmt': 'csr', 'k': 4}, dtype=DATA_DTYPE)

    @pytest.fixture
    def gpu_csc(self, grg_path):
        return SpMVOperator(grg_path, {'type': 'cusparse', 'fmt': 'csc', 'k': 4}, dtype=DATA_DTYPE)

    def test_forward_csr_vs_csc(self, gpu_csr, gpu_csc, vecs):
        np.testing.assert_allclose(
            gpu_csr.T @ vecs['v4'], gpu_csc.T @ vecs['v4'], atol=1e-4, rtol=1e-5)

    def test_backward_csr_vs_csc(self, gpu_csr, gpu_csc, vecs):
        np.testing.assert_allclose(
            gpu_csr @ vecs['w4'], gpu_csc @ vecs['w4'], atol=1e-4, rtol=1e-5)


class TestVerboseOutput:
    """Verbose mode should not crash and should produce output."""

    def test_verbose_setup(self, grg_path, capsys):
        SpMVOperator(grg_path, {'type': 'cusparse', 'verbose': True, 'k': 1}, dtype=DATA_DTYPE)
        captured = capsys.readouterr()
        assert 'nnz' in captured.out.lower() or 'level' in captured.out.lower()

    def test_verbose_forward(self, grg_path, capsys):
        op = SpMVOperator(grg_path, {'type': 'cusparse', 'verbose': True, 'k': 1}, dtype=DATA_DTYPE)
        v = np.ones((op.n, 1), dtype=DATA_DTYPE)
        op.T @ v
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_k16(self, grg_path, ref_op):
        """Large k value."""
        gpu = SpMVOperator(grg_path, {'type': 'cusparse', 'k': 16}, dtype=DATA_DTYPE)
        rng = np.random.default_rng(999)
        w = rng.standard_normal((ref_op.m, 16), dtype=DATA_DTYPE)
        np.testing.assert_allclose(gpu @ w, ref_op @ w, atol=1e-4)

    def test_allele_counts(self, grg_path):
        """G^T @ 1 should produce non-negative integers (allele counts)."""
        gpu = SpMVOperator(grg_path, {'type': 'cusparse', 'k': 1}, dtype=DATA_DTYPE)
        ones = np.ones((gpu.n, 1), dtype=DATA_DTYPE)
        ac = gpu.T @ ones
        assert np.all(ac >= 0)
        np.testing.assert_allclose(ac, np.round(ac), atol=1e-4)

    def test_zero_input_forward(self, grg_path):
        """Zero vector input to forward => zero output."""
        gpu = SpMVOperator(grg_path, {'type': 'cusparse', 'k': 4}, dtype=DATA_DTYPE)
        v = np.zeros((gpu.n, 4), dtype=DATA_DTYPE)
        result = gpu.T @ v
        np.testing.assert_array_equal(result, 0.0)

    def test_zero_input_backward(self, grg_path):
        """Zero vector input to backward => zero output."""
        gpu = SpMVOperator(grg_path, {'type': 'cusparse', 'k': 4}, dtype=DATA_DTYPE)
        w = np.zeros((gpu.m, 4), dtype=DATA_DTYPE)
        result = gpu @ w
        np.testing.assert_array_equal(result, 0.0)


class TestStressLargeK:
    """Stress tests with large k values (wide dense matrices)."""

    def test_k32_forward(self, grg_path, ref_op):
        """k=32 forward: graph captured for 32 columns."""
        gpu = SpMVOperator(grg_path, {'type': 'cusparse', 'k': 32}, dtype=DATA_DTYPE)
        rng = np.random.default_rng(2001)
        v = rng.standard_normal((ref_op.n, 32), dtype=DATA_DTYPE)
        np.testing.assert_allclose(gpu.T @ v, ref_op.T @ v, atol=1e-4, rtol=1e-5)

    def test_k32_backward(self, grg_path, ref_op):
        """k=32 backward."""
        gpu = SpMVOperator(grg_path, {'type': 'cusparse', 'k': 32}, dtype=DATA_DTYPE)
        rng = np.random.default_rng(2002)
        w = rng.standard_normal((ref_op.m, 32), dtype=DATA_DTYPE)
        np.testing.assert_allclose(gpu @ w, ref_op @ w, atol=1e-4, rtol=1e-5)

    def test_k64_forward(self, grg_path, ref_op):
        """k=64 forward: stress test for large dense blocks."""
        gpu = SpMVOperator(grg_path, {'type': 'cusparse', 'k': 64}, dtype=DATA_DTYPE)
        rng = np.random.default_rng(2003)
        v = rng.standard_normal((ref_op.n, 64), dtype=DATA_DTYPE)
        np.testing.assert_allclose(gpu.T @ v, ref_op.T @ v, atol=1e-4, rtol=1e-5)

    def test_k64_backward(self, grg_path, ref_op):
        """k=64 backward: stress test for large dense blocks."""
        gpu = SpMVOperator(grg_path, {'type': 'cusparse', 'k': 64}, dtype=DATA_DTYPE)
        rng = np.random.default_rng(2004)
        w = rng.standard_normal((ref_op.m, 64), dtype=DATA_DTYPE)
        np.testing.assert_allclose(gpu @ w, ref_op @ w, atol=1e-4, rtol=1e-5)


class TestInterleaved:
    """Interleaved forward and backward calls must not corrupt state."""

    @pytest.fixture
    def gpu(self, grg_path):
        return SpMVOperator(grg_path, {'type': 'cusparse', 'k': 4}, dtype=DATA_DTYPE)

    def test_fwd_bwd_fwd_consistency(self, gpu, ref_op):
        """Forward, backward, then forward again with same input."""
        rng = np.random.default_rng(3001)
        v = rng.standard_normal((ref_op.n, 4), dtype=DATA_DTYPE)
        w = rng.standard_normal((ref_op.m, 4), dtype=DATA_DTYPE)

        r1_fwd = gpu.T @ v
        r_bwd = gpu @ w
        r2_fwd = gpu.T @ v

        np.testing.assert_allclose(r1_fwd, ref_op.T @ v, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(r_bwd, ref_op @ w, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(r2_fwd, r1_fwd, atol=1e-5, rtol=1e-5)

    def test_bwd_fwd_bwd_consistency(self, gpu, ref_op):
        """Backward, forward, then backward again."""
        rng = np.random.default_rng(3002)
        v = rng.standard_normal((ref_op.n, 4), dtype=DATA_DTYPE)
        w = rng.standard_normal((ref_op.m, 4), dtype=DATA_DTYPE)

        r1_bwd = gpu @ w
        r_fwd = gpu.T @ v
        r2_bwd = gpu @ w

        np.testing.assert_allclose(r1_bwd, ref_op @ w, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(r_fwd, ref_op.T @ v, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(r2_bwd, r1_bwd, atol=1e-5, rtol=1e-5)

    def test_rapid_alternation(self, gpu, ref_op):
        """Rapidly alternate between fwd and bwd with different inputs."""
        rng = np.random.default_rng(3003)
        for _ in range(10):
            v = rng.standard_normal((ref_op.n, 4), dtype=DATA_DTYPE)
            w = rng.standard_normal((ref_op.m, 4), dtype=DATA_DTYPE)
            np.testing.assert_allclose(gpu.T @ v, ref_op.T @ v, atol=5e-5, rtol=1e-5)
            np.testing.assert_allclose(gpu @ w, ref_op @ w, atol=5e-5, rtol=1e-5)


class TestLinearity:
    """Linearity: G(ax + by) = a*G(x) + b*G(y) for both directions."""

    @pytest.fixture
    def gpu(self, grg_path):
        return SpMVOperator(grg_path, {'type': 'cusparse', 'k': 4}, dtype=DATA_DTYPE)

    def test_forward_linearity(self, gpu):
        """G^T(a*v1 + b*v2) == a * G^T(v1) + b * G^T(v2)."""
        rng = np.random.default_rng(4001)
        v1 = rng.standard_normal((gpu.n, 4), dtype=DATA_DTYPE)
        v2 = rng.standard_normal((gpu.n, 4), dtype=DATA_DTYPE)
        a, b = 2.5, -1.3

        lhs = gpu.T @ (a * v1 + b * v2)
        rhs = a * (gpu.T @ v1) + b * (gpu.T @ v2)
        np.testing.assert_allclose(lhs, rhs, atol=1e-4, rtol=1e-5)

    def test_backward_linearity(self, gpu):
        """G(a*w1 + b*w2) == a * G(w1) + b * G(w2)."""
        rng = np.random.default_rng(4002)
        w1 = rng.standard_normal((gpu.m, 4), dtype=DATA_DTYPE)
        w2 = rng.standard_normal((gpu.m, 4), dtype=DATA_DTYPE)
        a, b = 0.7, 3.1

        lhs = gpu @ (a * w1 + b * w2)
        rhs = a * (gpu @ w1) + b * (gpu @ w2)
        np.testing.assert_allclose(lhs, rhs, atol=1e-4, rtol=1e-5)


class TestMixedGraphDynamic:
    """Graph operators must correctly fall back to dynamic, then resume graph."""

    @pytest.fixture
    def gpu_k4(self, grg_path):
        return SpMVOperator(grg_path, {'type': 'cusparse', 'k': 4}, dtype=DATA_DTYPE)

    def test_graph_then_dynamic_then_graph(self, gpu_k4, ref_op):
        """Use graph (k=4), then dynamic (k=2), then graph (k=4) again."""
        rng = np.random.default_rng(5001)

        # Graph path (k=4)
        w4 = rng.standard_normal((ref_op.m, 4), dtype=DATA_DTYPE)
        r1 = gpu_k4 @ w4
        np.testing.assert_allclose(r1, ref_op @ w4, atol=1e-5, rtol=1e-5)

        # Dynamic fallback (k=2, mismatches graph k=4)
        w2 = rng.standard_normal((ref_op.m, 2), dtype=DATA_DTYPE)
        r2 = gpu_k4 @ w2
        np.testing.assert_allclose(r2, ref_op @ w2, atol=1e-5, rtol=1e-5)

        # Back to graph (k=4) — must not be corrupted by dynamic call
        r3 = gpu_k4 @ w4
        np.testing.assert_allclose(r3, r1, atol=1e-5, rtol=1e-5)

    def test_forward_graph_dynamic_graph(self, gpu_k4, ref_op):
        """Forward direction: graph -> dynamic -> graph."""
        rng = np.random.default_rng(5002)

        v4 = rng.standard_normal((ref_op.n, 4), dtype=DATA_DTYPE)
        r1 = gpu_k4.T @ v4
        np.testing.assert_allclose(r1, ref_op.T @ v4, atol=1e-5, rtol=1e-5)

        v3 = rng.standard_normal((ref_op.n, 3), dtype=DATA_DTYPE)
        r2 = gpu_k4.T @ v3
        np.testing.assert_allclose(r2, ref_op.T @ v3, atol=1e-5, rtol=1e-5)

        r3 = gpu_k4.T @ v4
        np.testing.assert_allclose(r3, r1, atol=1e-5, rtol=1e-5)


class TestVerboseDynamic:
    """Verbose timing in dynamic mode (no graph) should not crash."""

    def test_verbose_dynamic_forward(self, grg_path, capsys):
        op = SpMVOperator(grg_path, {'type': 'cusparse', 'verbose': True, 'k': None}, dtype=DATA_DTYPE)
        v = np.ones((op.n, 4), dtype=DATA_DTYPE)
        op.T @ v
        captured = capsys.readouterr()
        assert 'forward (dynamic)' in captured.out

    def test_verbose_dynamic_backward(self, grg_path, capsys):
        op = SpMVOperator(grg_path, {'type': 'cusparse', 'verbose': True, 'k': None}, dtype=DATA_DTYPE)
        w = np.ones((op.m, 4), dtype=DATA_DTYPE)
        op @ w
        captured = capsys.readouterr()
        assert 'backward (dynamic)' in captured.out

    def test_verbose_graph_forward(self, grg_path, capsys):
        op = SpMVOperator(grg_path, {'type': 'cusparse', 'verbose': True, 'k': 4}, dtype=DATA_DTYPE)
        v = np.ones((op.n, 4), dtype=DATA_DTYPE)
        op.T @ v
        captured = capsys.readouterr()
        assert 'forward (graph)' in captured.out

    def test_verbose_graph_backward(self, grg_path, capsys):
        op = SpMVOperator(grg_path, {'type': 'cusparse', 'verbose': True, 'k': 4}, dtype=DATA_DTYPE)
        w = np.ones((op.m, 4), dtype=DATA_DTYPE)
        op @ w
        captured = capsys.readouterr()
        assert 'backward (graph)' in captured.out

    def test_verbose_timing_breakdown(self, grg_path, capsys):
        """Timing output must include H2D, wavefront/kernel, and D2H phases."""
        op = SpMVOperator(grg_path, {'type': 'cusparse', 'verbose': True, 'k': 4}, dtype=DATA_DTYPE)
        v = np.ones((op.n, 4), dtype=DATA_DTYPE)
        op.T @ v
        captured = capsys.readouterr()
        assert 'H2D=' in captured.out
        assert 'wavefront=' in captured.out
        assert 'D2H=' in captured.out
        assert 'total=' in captured.out


class TestCOOFormat:
    """COO sparse format must work correctly."""

    @pytest.fixture
    def gpu_coo(self, grg_path):
        return SpMVOperator(grg_path, {'type': 'cusparse', 'fmt': 'coo', 'k': 4}, dtype=DATA_DTYPE)

    def test_coo_forward(self, gpu_coo, ref_op, vecs):
        np.testing.assert_allclose(
            gpu_coo.T @ vecs['v4'], ref_op.T @ vecs['v4'], atol=1e-4, rtol=1e-5)

    def test_coo_backward(self, gpu_coo, ref_op, vecs):
        np.testing.assert_allclose(
            gpu_coo @ vecs['w4'], ref_op @ vecs['w4'], atol=1e-4, rtol=1e-5)

    def test_coo_dynamic(self, grg_path, ref_op, vecs):
        gpu = SpMVOperator(grg_path, {'type': 'cusparse', 'fmt': 'coo', 'k': None}, dtype=DATA_DTYPE)
        np.testing.assert_allclose(
            gpu.T @ vecs['v4'], ref_op.T @ vecs['v4'], atol=1e-4, rtol=1e-5)


class TestAlgorithmSelection:
    """Different SpMM algorithms must produce correct results."""

    @pytest.fixture(params=['default', 'csr_alg1', 'csr_alg2'],
                    ids=['alg-default', 'alg-csr1', 'alg-csr2'])
    def gpu_alg(self, grg_path, request):
        return SpMVOperator(grg_path, {
            'type': 'cusparse', 'fmt': 'csr', 'algorithm': request.param, 'k': 4,
        }, dtype=DATA_DTYPE)

    def test_forward_algorithm(self, gpu_alg, ref_op, vecs):
        np.testing.assert_allclose(
            gpu_alg.T @ vecs['v4'], ref_op.T @ vecs['v4'], atol=1e-4, rtol=1e-5)

    def test_backward_algorithm(self, gpu_alg, ref_op, vecs):
        np.testing.assert_allclose(
            gpu_alg @ vecs['w4'], ref_op @ vecs['w4'], atol=1e-4, rtol=1e-5)


class TestRepeatedStress:
    """Extended repeated calls for stress testing graph replay stability."""

    @pytest.fixture
    def gpu(self, grg_path):
        return SpMVOperator(grg_path, {'type': 'cusparse', 'k': 4}, dtype=DATA_DTYPE)

    def test_20_repeated_forward(self, gpu, ref_op):
        """20 repeated forward calls must all match CPU."""
        rng = np.random.default_rng(6001)
        v = rng.standard_normal((ref_op.n, 4), dtype=DATA_DTYPE)
        expected = ref_op.T @ v
        for i in range(20):
            actual = gpu.T @ v
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5,
                                       err_msg=f"Diverged at forward call #{i}")

    def test_20_repeated_backward(self, gpu, ref_op):
        """20 repeated backward calls must all match CPU."""
        rng = np.random.default_rng(6002)
        w = rng.standard_normal((ref_op.m, 4), dtype=DATA_DTYPE)
        expected = ref_op @ w
        for i in range(20):
            actual = gpu @ w
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5,
                                       err_msg=f"Diverged at backward call #{i}")

    def test_many_different_inputs(self, gpu, ref_op):
        """20 forward calls with different random inputs."""
        rng = np.random.default_rng(6003)
        for i in range(20):
            v = rng.standard_normal((ref_op.n, 4), dtype=DATA_DTYPE)
            np.testing.assert_allclose(gpu.T @ v, ref_op.T @ v, atol=5e-5, rtol=1e-5,
                                       err_msg=f"Diverged at call #{i}")
