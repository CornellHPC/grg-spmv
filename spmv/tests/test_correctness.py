"""
Backend-agnostic correctness tests.

Runs for all available backends (filtered by --backend).
Each test compares against SciPyXOperator ground truth via disk cache.
"""

import numpy as np
import pytest

from spmv import SpMVOperator
from spmv.tests.conftest import (
    DATA_DTYPE, INDEX_DTYPE, K_SMOKE, K_FULL,
    binary_pm1, tol,
)


# ---------------------------------------------------------------------------
# Backend config generation
# ---------------------------------------------------------------------------

def _make_backend_configs(backend_filter):
    configs = []
    if backend_filter in ("cpu", "all"):
        configs += [
            pytest.param({'type': 'multithread', 'n_workers': 1}, id='cpu-1w'),
            pytest.param({'type': 'multithread', 'n_workers': 4}, id='cpu-4w'),
        ]
    if backend_filter in ("cusparse", "all"):
        try:
            import cupy  # noqa: F401
        except ImportError:
            pass
        else:
            from spmv.backends.cusparse import VALID_FMT_ALG_COMBOS
            # Dynamic mode for all valid combos
            for fmt, alg in sorted(VALID_FMT_ALG_COMBOS):
                configs.append(pytest.param(
                    {'type': 'cusparse', 'fmt': fmt, 'algorithm': alg, 'k': None},
                    id=f'gpu-dyn-{fmt}-{alg}',
                ))
            # Graph mode for all valid combos (k=4)
            for fmt, alg in sorted(VALID_FMT_ALG_COMBOS):
                configs.append(pytest.param(
                    {'type': 'cusparse', 'fmt': fmt, 'algorithm': alg, 'k': 4},
                    id=f'gpu-graph-{fmt}-{alg}',
                ))
    return configs


# Module-level operator cache (keyed by config string) so each config is created once
_OP_CACHE = {}


def _get_op(config, grg_path, dtype=DATA_DTYPE):
    key = f"{config}_{grg_path}_{dtype}"
    if key not in _OP_CACHE:
        _OP_CACHE[key] = SpMVOperator(grg_path, config, dtype, INDEX_DTYPE)
    return _OP_CACHE[key]


# ---------------------------------------------------------------------------
# Dynamic parametrization via pytest_generate_tests
# ---------------------------------------------------------------------------

def pytest_generate_tests(metafunc):
    """Dynamically parametrize 'backend_config' and 'k' based on CLI options."""
    backend_filter = metafunc.config.getoption("--backend", default="all")
    smoke = metafunc.config.getoption("--smoke", default=False)

    if "backend_config" in metafunc.fixturenames:
        configs = _make_backend_configs(backend_filter)
        if not configs:
            pytest.skip("No backends available for the selected filter")
        metafunc.parametrize("backend_config", configs)

    if "k" in metafunc.fixturenames:
        k_vals = K_SMOKE if smoke else K_FULL
        metafunc.parametrize("k", k_vals)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def op(backend_config, small_grg_path):
    return _get_op(backend_config, small_grg_path)


# ---------------------------------------------------------------------------
# TestCorrectness — forward & backward vs ground truth, parametrized by k
# ---------------------------------------------------------------------------

class TestCorrectness:
    """Forward and backward vs ground truth at various k values."""

    def test_forward(self, op, k, gt_small):
        X, Y_expected = gt_small.get('forward', k, seed=42, dtype=DATA_DTYPE)
        Y_actual = op.H @ X
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(Y_actual, Y_expected, atol=atol, rtol=rtol)

    def test_backward(self, op, k, gt_small):
        X, Y_expected = gt_small.get('backward', k, seed=42, dtype=DATA_DTYPE)
        Y_actual = op @ X
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(Y_actual, Y_expected, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# TestExactBinary — +-1 vectors, exact integer results
# ---------------------------------------------------------------------------

class TestExactBinary:
    """+-1 input vectors produce exact integer results in float64."""

    def test_exact_forward(self, op, gt_small):
        X, Y_expected = gt_small.get('forward', 4, seed=100, dtype=DATA_DTYPE,
                                     input_fn=binary_pm1)
        Y_actual = op.H @ X
        np.testing.assert_array_equal(Y_actual, Y_expected)

    def test_exact_backward(self, op, gt_small):
        X, Y_expected = gt_small.get('backward', 4, seed=100, dtype=DATA_DTYPE,
                                     input_fn=binary_pm1)
        Y_actual = op @ X
        np.testing.assert_array_equal(Y_actual, Y_expected)


# ---------------------------------------------------------------------------
# TestAlleleCounts — op.H @ ones, exact equality
# ---------------------------------------------------------------------------

def _ones_input(rng, shape, dtype):
    """Return all-ones matrix (ignores rng)."""
    return np.ones(shape, dtype=dtype)


_ones_input.__name__ = "ones"


class TestAlleleCounts:
    """op.H @ ones produces exact integer allele counts."""

    def test_allele_counts(self, op, gt_small):
        X, Y_expected = gt_small.get('forward', 1, seed=0, dtype=DATA_DTYPE,
                                     input_fn=_ones_input)
        Y_actual = op.H @ X
        np.testing.assert_array_equal(Y_actual, Y_expected)


# ---------------------------------------------------------------------------
# TestDtypeSweep — float32 and float64
# ---------------------------------------------------------------------------

class TestDtypeSweep:
    """Test with different data types."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64],
                             ids=["f32", "f64"])
    def test_forward_dtype(self, backend_config, small_grg_path, gt_small, dtype):
        op = SpMVOperator(small_grg_path, backend_config, dtype, INDEX_DTYPE)
        X, Y_expected = gt_small.get('forward', 4, seed=200, dtype=dtype)
        Y_actual = op.H @ X
        atol, rtol = tol(dtype)
        np.testing.assert_allclose(Y_actual, Y_expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64],
                             ids=["f32", "f64"])
    def test_backward_dtype(self, backend_config, small_grg_path, gt_small, dtype):
        op = SpMVOperator(small_grg_path, backend_config, dtype, INDEX_DTYPE)
        X, Y_expected = gt_small.get('backward', 4, seed=200, dtype=dtype)
        Y_actual = op @ X
        atol, rtol = tol(dtype)
        np.testing.assert_allclose(Y_actual, Y_expected, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# TestRepeatedCalls — 5x same input, verify consistency
# ---------------------------------------------------------------------------

class TestRepeatedCalls:
    """Repeated calls with the same input produce consistent results."""

    def test_repeated_forward(self, op, gt_small):
        X, _ = gt_small.get('forward', 4, seed=300, dtype=DATA_DTYPE)
        results = [op.H @ X for _ in range(5)]
        for r in results[1:]:
            np.testing.assert_allclose(r, results[0], atol=1e-10)

    def test_repeated_backward(self, op, gt_small):
        X, _ = gt_small.get('backward', 4, seed=300, dtype=DATA_DTYPE)
        results = [op @ X for _ in range(5)]
        for r in results[1:]:
            np.testing.assert_allclose(r, results[0], atol=1e-10)


# ---------------------------------------------------------------------------
# TestZeroInput — zero in, zero out
# ---------------------------------------------------------------------------

class TestZeroInput:
    """Zero input produces zero output."""

    def test_zero_forward(self, op):
        n, m = op.shape
        X = np.zeros((n, 4), dtype=DATA_DTYPE)
        Y = op.H @ X
        np.testing.assert_array_equal(Y, 0.0)

    def test_zero_backward(self, op):
        n, m = op.shape
        X = np.zeros((m, 4), dtype=DATA_DTYPE)
        Y = op @ X
        np.testing.assert_array_equal(Y, 0.0)
