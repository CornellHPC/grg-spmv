"""
Shared fixtures, ground truth cache, and CLI options for the test suite.
"""

from pathlib import Path

import numpy as np
import pytest

from spmv import SpMVOperator
from spmv.backends.cusparse import is_valid_combo

# ---------------------------------------------------------------------------
# Configuration constants (single source of truth)
# ---------------------------------------------------------------------------

GRG_SMALL = "/pscratch/sd/q/qys/grg/msprime.example.igd.final.grg"
GRG_LARGE = "/pscratch/sd/q/qys/grg/simulation-mutation-200m.trees.v4.igd.final.grg"
CACHE_DIR = "/pscratch/sd/q/qys/grg/test_cache"
INDEX_DTYPE = np.uintp
DATA_DTYPE = np.float64
K_SMOKE = [1, 2, 16]
K_FULL = [1, 2, 3, 7, 8, 9, 16, 20]

# All (format, algorithm) combos with validity
_ALL_FMTS = ['csr', 'csc', 'coo']
_ALL_ALGS = ['default', 'csr_alg1', 'csr_alg2', 'csr_alg3',
             'coo_alg1', 'coo_alg2', 'coo_alg3', 'coo_alg4']


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--backend", default="all",
                     choices=["spsparse", "mkl", "cusparse", "cpu", "all"],
                     help="Which backend(s) to test")
    parser.addoption("--smoke", action="store_true", default=False,
                     help="Run smoke tests only (subset of k values)")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on --backend flag and markers."""
    backend = config.getoption("--backend")
    match backend:
        case "all":
            return
        case "cpu":
            skip = pytest.mark.skip(reason="--backend=cpu excludes GPU tests")
            for item in items:
                if "gpu" in item.keywords:
                    item.add_marker(skip)
        case "spsparse":
            skip_mkl = pytest.mark.skip(reason="--backend=spsparse")
            skip_gpu = pytest.mark.skip(reason="--backend=spsparse")
            for item in items:
                if "mkl" in item.keywords:
                    item.add_marker(skip_mkl)
                elif "gpu" in item.keywords:
                    item.add_marker(skip_gpu)
        case "mkl":
            skip_sp = pytest.mark.skip(reason="--backend=mkl")
            skip_gpu = pytest.mark.skip(reason="--backend=mkl")
            for item in items:
                if "spsparse" in item.keywords:
                    item.add_marker(skip_sp)
                elif "gpu" in item.keywords:
                    item.add_marker(skip_gpu)
        case "cusparse":
            skip_sp = pytest.mark.skip(reason="--backend=cusparse")
            skip_mkl = pytest.mark.skip(reason="--backend=cusparse")
            for item in items:
                if "spsparse" in item.keywords:
                    item.add_marker(skip_sp)
                elif "mkl" in item.keywords:
                    item.add_marker(skip_mkl)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fmt_alg_params():
    """Build pytest.param list for all (fmt, alg) combos: valid pass, invalid xfail."""
    params = []
    for fmt in _ALL_FMTS:
        for alg in _ALL_ALGS:
            pid = f"{fmt}-{alg}"
            if is_valid_combo(fmt, alg):
                params.append(pytest.param(fmt, alg, id=pid))
            else:
                params.append(pytest.param(
                    fmt, alg, id=pid,
                    marks=pytest.mark.xfail(raises=ValueError, strict=True),
                ))
    return params


def valid_fmt_alg_params():
    """Build pytest.param list for only valid (fmt, alg) combos (no xfail)."""
    params = []
    for fmt in _ALL_FMTS:
        for alg in _ALL_ALGS:
            if is_valid_combo(fmt, alg):
                params.append(pytest.param(fmt, alg, id=f"{fmt}-{alg}"))
    return params


def binary_pm1(rng, shape, dtype):
    """Return a random {-1, +1} matrix."""
    return rng.choice(np.array([-1.0, 1.0], dtype=dtype), size=shape)


def tol(dtype):
    """Return (atol, rtol) appropriate for the given dtype."""
    if np.dtype(dtype) == np.float32:
        return 1e-3, 1e-3
    return 1e-5, 1e-5


# ---------------------------------------------------------------------------
# Ground truth cache
# ---------------------------------------------------------------------------

class GroundTruthCache:
    """Disk-cached SciPyXOperator results."""

    def __init__(self, grg_path, cache_dir=CACHE_DIR):
        self._grg_path = grg_path
        self._dir = Path(cache_dir) / Path(grg_path).stem
        self._ref = None  # lazy-loaded

    def _ensure_ref(self):
        if self._ref is None:
            import pygrgl
            from grapp.linalg.ops_scipy import SciPyXOperator
            grg = pygrgl.load_immutable_grg(self._grg_path)
            self._ref = SciPyXOperator(grg, pygrgl.TraversalDirection.UP, haploid=True)

    @property
    def shape(self):
        self._ensure_ref()
        return self._ref.shape

    def get(self, direction, k, seed, dtype=np.float64, input_fn=None):
        """Return (X, Y_expected). Computes if not cached.

        Parameters
        ----------
        direction : str
            'forward' or 'backward'.
        k : int
            Number of dense columns.
        seed : int
            Random seed for input generation.
        dtype : numpy dtype
            Data type for computation.
        input_fn : callable or None
            If provided, called as input_fn(rng, shape, dtype) -> X.
        """
        tag = f"{direction}_k{k}_s{seed}_{np.dtype(dtype).name}"
        if input_fn is not None:
            tag += f"_{input_fn.__name__}"
        path = self._dir / f"{tag}.npz"
        if path.exists():
            d = np.load(path)
            return d['X'], d['Y']
        self._ensure_ref()
        rng = np.random.default_rng(seed)
        n, m = self._ref.shape
        shape = (n, k) if direction == 'forward' else (m, k)
        X = input_fn(rng, shape, dtype) if input_fn else rng.standard_normal(shape, dtype=dtype)
        Y = self._ref.H @ X if direction == 'forward' else self._ref @ X
        self._dir.mkdir(parents=True, exist_ok=True)
        np.savez(path, X=X, Y=Y)
        return X, Y


# ---------------------------------------------------------------------------
# Session fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def k_values(request):
    """Return K_SMOKE or K_FULL based on --smoke flag."""
    return K_SMOKE if request.config.getoption("--smoke") else K_FULL


@pytest.fixture(scope="session")
def backend_filter(request):
    """Return which backends to test based on --backend flag."""
    return request.config.getoption("--backend")


@pytest.fixture(scope="session")
def small_grg_path():
    if not Path(GRG_SMALL).exists():
        pytest.skip(f"GRG file not found: {GRG_SMALL}")
    return GRG_SMALL


@pytest.fixture(scope="session")
def large_grg_path():
    if not Path(GRG_LARGE).exists():
        pytest.skip(f"GRG file not found: {GRG_LARGE}")
    return GRG_LARGE


@pytest.fixture(scope="session")
def gt_small(small_grg_path):
    return GroundTruthCache(small_grg_path)


@pytest.fixture(scope="session")
def gt_large(large_grg_path):
    return GroundTruthCache(large_grg_path)
