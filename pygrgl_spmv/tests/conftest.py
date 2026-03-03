"""Shared fixtures, CLI options, and reference cache helpers for tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIMARY_GRG = str(REPO_ROOT / "pygrgl_spmv" / "tests" / "data" / "msprime.example.igd.final.grg")
DEFAULT_MISSING_GRG = str(REPO_ROOT / "pygrgl_spmv" / "tests" / "data" / "test-200-samples.miss.final.grg")

INDEX_DTYPE = np.uintp
DATA_DTYPE = np.float64
K_CORE = [1, 2, 4]
K_MATRIX = [1, 2, 3, 7, 8, 9, 16, 20]

_ALL_FMTS = ["csr", "csc", "coo"]
_ALL_ALGOS = ["default", "csr_alg1", "csr_alg2", "csr_alg3", "coo_alg1", "coo_alg2", "coo_alg3", "coo_alg4"]


def _has_mkl_runtime() -> bool:
    try:
        from pygrgl_spmv.backends import mkl_utils

        mkl_utils._ensure_loaded()
        return True
    except Exception:
        return False


HAS_MKL_RUNTIME = _has_mkl_runtime()


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        default="all",
        choices=["mkl", "cusparse", "all"],
        help="Which backend(s) to test",
    )
    parser.addoption(
        "--smoke",
        action="store_true",
        default=False,
        help="Run only tests marked 'smoke'",
    )
    parser.addoption(
        "--grg",
        default=DEFAULT_PRIMARY_GRG,
        help="Primary GRG file used by traversal/backends tests",
    )
    parser.addoption(
        "--missing-grg",
        default=DEFAULT_MISSING_GRG,
        help="Missingness GRG file used by missingness tests",
    )


def pytest_collection_modifyitems(config, items):
    backend = config.getoption("--backend")
    smoke = bool(config.getoption("--smoke"))

    match backend:
        case "all":
            pass
        case "mkl":
            skip_gpu = pytest.mark.skip(reason="--backend=mkl")
            for item in items:
                if "gpu" in item.keywords:
                    item.add_marker(skip_gpu)
        case "cusparse":
            skip_mkl = pytest.mark.skip(reason="--backend=cusparse")
            for item in items:
                if "mkl" in item.keywords:
                    item.add_marker(skip_mkl)

    if not HAS_MKL_RUNTIME:
        skip_no_mkl = pytest.mark.skip(reason="MKL runtime unavailable (libmkl_rt.so not found)")
        for item in items:
            if "mkl" in item.keywords:
                item.add_marker(skip_no_mkl)

    if smoke:
        keep: list[pytest.Item] = []
        deselected: list[pytest.Item] = []
        for item in items:
            if "smoke" in item.keywords:
                keep.append(item)
            else:
                deselected.append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = keep


def make_fmt_algo_params():
    """Build pytest.param list for all (fmt, algo) combos: valid pass, invalid xfail."""
    from pygrgl_spmv.backends.cusparse import is_valid_combo

    params = []
    for fmt in _ALL_FMTS:
        for algo in _ALL_ALGOS:
            pid = f"{fmt}-{algo}"
            if is_valid_combo(fmt, algo):
                params.append(pytest.param(fmt, algo, id=pid))
            else:
                params.append(pytest.param(fmt, algo, id=pid, marks=pytest.mark.xfail(raises=ValueError, strict=True)))
    return params


def valid_fmt_algo_params():
    """Build pytest.param list for valid (fmt, algo) combos only."""
    from pygrgl_spmv.backends.cusparse import is_valid_combo

    params = []
    for fmt in _ALL_FMTS:
        for algo in _ALL_ALGOS:
            if is_valid_combo(fmt, algo):
                params.append(pytest.param(fmt, algo, id=f"{fmt}-{algo}"))
    return params


def binary_pm1(rng, shape, dtype):
    return rng.choice(np.array([-1.0, 1.0], dtype=dtype), size=shape)


def tol(dtype):
    if np.dtype(dtype) == np.float32:
        return 1e-3, 1e-3
    return 1e-5, 1e-5


class GroundTruthCache:
    """Disk-cached SciPyXOperator results."""

    def __init__(self, grg_path: str, cache_dir: Path):
        self._grg_path = str(grg_path)
        digest = hashlib.sha1(self._grg_path.encode("utf-8")).hexdigest()[:16]
        self._dir = cache_dir / digest
        self._ref = None

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
        tag = f"{direction}_k{k}_s{seed}_{np.dtype(dtype).name}"
        if input_fn is not None:
            tag += f"_{input_fn.__name__}"
        path = self._dir / f"{tag}.npz"
        if path.exists():
            d = np.load(path)
            return d["X"], d["Y"]

        self._ensure_ref()
        rng = np.random.default_rng(seed)
        n, m = self._ref.shape
        shape = (n, k) if direction == "forward" else (m, k)
        X = input_fn(rng, shape, dtype) if input_fn else rng.standard_normal(shape, dtype=dtype)
        Y = self._ref.H @ X if direction == "forward" else self._ref @ X
        self._dir.mkdir(parents=True, exist_ok=True)
        np.savez(path, X=X, Y=Y)
        return X, Y


@pytest.fixture(scope="session")
def backend_filter(request):
    return request.config.getoption("--backend")


@pytest.fixture(scope="session")
def smoke_mode(request) -> bool:
    return bool(request.config.getoption("--smoke"))


@pytest.fixture(scope="session")
def k_values(smoke_mode):
    return K_CORE if smoke_mode else K_MATRIX


@pytest.fixture(scope="session")
def primary_grg_path(request):
    path = Path(str(request.config.getoption("--grg")))
    if not path.exists():
        pytest.skip(f"Primary GRG file not found: {path}")
    return str(path)


@pytest.fixture(scope="session")
def missing_grg_path(request):
    path = Path(str(request.config.getoption("--missing-grg")))
    if not path.exists():
        pytest.skip(f"Missingness GRG file not found: {path}")
    return str(path)


@pytest.fixture(scope="session")
def spmv_cache_dir(request):
    base = Path(request.config.rootpath) / ".pytest_cache" / "pygrgl_spmv_npz"
    base.mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture(scope="session")
def ground_truth_cache_dir(request):
    base = Path(request.config.rootpath) / ".pytest_cache" / "pygrgl_spmv_ground_truth"
    base.mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture(scope="session")
def gt_small(primary_grg_path, ground_truth_cache_dir):
    return GroundTruthCache(primary_grg_path, ground_truth_cache_dir)


@pytest.fixture(scope="session")
def gt_primary(primary_grg_path, ground_truth_cache_dir):
    return GroundTruthCache(primary_grg_path, ground_truth_cache_dir)
