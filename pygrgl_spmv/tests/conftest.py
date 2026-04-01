"""Shared fixtures, CLI options, and reference cache helpers for tests."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIMARY_GRG = str(REPO_ROOT / "pygrgl_spmv" / "tests" / "data" / "msprime.example.igd.final.grg")
DEFAULT_MISSING_GRG = str(REPO_ROOT / "pygrgl_spmv" / "tests" / "data" / "test-200-samples.miss.final.grg")

INDEX_DTYPE = np.int32
DATA_DTYPE = np.float64
K_CORE = [1, 2, 4]
K_MATRIX = [1, 2, 3, 7, 8, 9, 16, 20]

_ALL_FMTS = ["csr", "csc", "coo"]
_ALL_ALGOS = ["default", "csr_alg1", "csr_alg2", "csr_alg3", "coo_alg1", "coo_alg2", "coo_alg3", "coo_alg4"]
_TRANSPOSE_COMPATIBLE_FMT = {"csr": "CSC", "csc": "CSR", "coo": "COO"}


def make_cusparse_plan(
    *,
    k_hint=None,
    store="N",
    fmt="CSR",
    op_a="N",
    op_b="N",
    order_b="ROW",
    order_c="ROW",
    algo="DEFAULT",
    scratch="none",
):
    return {
        "k_hint": k_hint,
        "store": store,
        "fmt": fmt,
        "opA": op_a,
        "opB": op_b,
        "orderB": order_b,
        "orderC": order_c,
        "algo": algo,
        "scratch": scratch,
    }


def make_mkl_plan(*, k_hint=None, store="N", fmt="CSR", n_threads=0):
    return {
        "k_hint": k_hint,
        "store": store,
        "fmt": fmt,
        "n_threads": n_threads,
    }


def make_triton_plan(*, k_hint=1, store="N", fmt="CSR", scratch="none"):
    return {
        "k_hint": k_hint,
        "store": store,
        "fmt": fmt,
        "scratch": scratch,
    }


def transpose_compatible_fmt(fmt: str) -> str:
    token = str(fmt).lower()
    if token not in _TRANSPOSE_COMPATIBLE_FMT:
        raise ValueError(f"Unsupported sparse format {fmt!r}; expected CSR/CSC/COO")
    return _TRANSPOSE_COMPATIBLE_FMT[token]


def make_mkl_backend(
    *,
    fmt_up="csr",
    fmt_down=None,
    k_hint=None,
    n_threads=0,
    log_level="WARNING",
    instrumentation=False,
    infer_missing=True,
):
    from pygrgl_spmv.backends.mkl import MklBackend, MklPlanPair

    plan_up = None
    if fmt_up is not None:
        plan_up = make_mkl_plan(k_hint=k_hint, store="N", fmt=str(fmt_up).upper(), n_threads=n_threads)
    elif infer_missing:
        assert fmt_down is not None
        plan_up = make_mkl_plan(k_hint=k_hint, store="T", fmt=str(fmt_down).upper(), n_threads=n_threads)

    plan_down = None
    if fmt_down is not None:
        plan_down = make_mkl_plan(k_hint=k_hint, store="T", fmt=str(fmt_down).upper(), n_threads=n_threads)
    elif infer_missing:
        assert fmt_up is not None
        plan_down = make_mkl_plan(
            k_hint=k_hint,
            store="T",
            fmt=transpose_compatible_fmt(fmt_up),
            n_threads=n_threads,
        )

    return MklBackend(
        pair=MklPlanPair.from_dicts(plan_up, plan_down),
        log_level=log_level,
        instrumentation=instrumentation,
    )


def make_cusparse_backend(
    *,
    device,
    stream=0,
    fmt_up="csr",
    fmt_down=None,
    k_hint=None,
    algo_up="default",
    algo_down="default",
    scratch_up="none",
    scratch_down="none",
    log_level="WARNING",
    instrumentation=False,
    infer_missing=True,
):
    from pygrgl_spmv.backends.cusparse import CusparseBackend, CusparsePlanPair

    if fmt_up is None and fmt_down is None:
        raise ValueError("fmt_up/fmt_down")
    if not isinstance(algo_up, str):
        raise TypeError("algo_up")
    if not isinstance(algo_down, str):
        raise TypeError("algo_down")

    plan_up = None
    if fmt_up is not None:
        plan_up = make_cusparse_plan(
            k_hint=k_hint,
            store="N",
            fmt=str(fmt_up).upper(),
            op_a="N",
            op_b="N",
            order_b="ROW",
            order_c="ROW",
            algo=algo_up.upper(),
            scratch=scratch_up,
        )
    elif infer_missing:
        assert fmt_down is not None
        plan_up = make_cusparse_plan(
            k_hint=k_hint,
            store="T",
            fmt=str(fmt_down).upper(),
            op_a="T",
            op_b="N",
            order_b="ROW",
            order_c="ROW",
            algo=algo_up.upper(),
            scratch=scratch_up,
        )

    plan_down = None
    if fmt_down is not None:
        plan_down = make_cusparse_plan(
            k_hint=k_hint,
            store="T",
            fmt=str(fmt_down).upper(),
            op_a="N",
            op_b="N",
            order_b="ROW",
            order_c="ROW",
            algo=algo_down.upper(),
            scratch=scratch_down,
        )
    elif infer_missing:
        assert fmt_up is not None
        plan_down = make_cusparse_plan(
            k_hint=k_hint,
            store="T",
            fmt=transpose_compatible_fmt(fmt_up),
            op_a="N",
            op_b="N",
            order_b="ROW",
            order_c="ROW",
            algo=algo_down.upper(),
            scratch=scratch_down,
        )

    return CusparseBackend(
        device=device,
        stream=stream,
        pair=CusparsePlanPair.from_dicts(plan_up, plan_down),
        log_level=log_level,
        instrumentation=instrumentation,
    )


def make_triton_backend(
    *,
    device,
    stream=0,
    fmt_up="csr",
    fmt_down=None,
    k_hint=1,
    scratch_up="none",
    scratch_down="none",
    log_level="WARNING",
    instrumentation=False,
    infer_missing=True,
):
    from pygrgl_spmv.backends.triton import TritonBackend, TritonPlanPair

    if fmt_up is None and fmt_down is None:
        raise ValueError("fmt_up/fmt_down")
    if k_hint not in {None, 1}:
        raise ValueError("Triton backend requires k_hint=none or 1")

    plan_up = None
    if fmt_up is not None:
        plan_up = make_triton_plan(k_hint=k_hint, store="N", fmt=str(fmt_up).upper(), scratch=scratch_up)
    elif infer_missing:
        assert fmt_down is not None
        plan_up = make_triton_plan(k_hint=k_hint, store="N", fmt=transpose_compatible_fmt(fmt_down), scratch=scratch_up)

    plan_down = None
    if fmt_down is not None:
        plan_down = make_triton_plan(k_hint=k_hint, store="T", fmt=str(fmt_down).upper(), scratch=scratch_down)
    elif infer_missing:
        assert fmt_up is not None
        plan_down = make_triton_plan(k_hint=k_hint, store="T", fmt=transpose_compatible_fmt(fmt_up), scratch=scratch_down)

    return TritonBackend(
        device=device,
        stream=stream,
        pair=TritonPlanPair.from_dicts(plan_up, plan_down),
        log_level=log_level,
        instrumentation=instrumentation,
    )


def default_backend_builders(*, log_level="INFO", instrumentation=False):
    params = [
        pytest.param(
            ("mkl", lambda: make_mkl_backend(
                fmt_up="csr",
                fmt_down="csc",
                n_threads=0,
                k_hint=None,
                log_level=log_level,
                instrumentation=instrumentation,
            )),
            id="mkl",
            marks=pytest.mark.mkl,
        ),
    ]
    try:
        import cupy  # noqa: F401
    except ImportError:
        return params

    params.extend(
        [
            pytest.param(
                ("cusparse", lambda: make_cusparse_backend(
                    device=0,
                    fmt_up="csr",
                    fmt_down="csc",
                    k_hint=None,
                    algo_up="default",
                    algo_down="default",
                    log_level=log_level,
                    instrumentation=instrumentation,
                )),
                id="cusparse-dyn",
                marks=pytest.mark.gpu,
            ),
            pytest.param(
                ("cusparse", lambda: make_cusparse_backend(
                    device=0,
                    fmt_up="csr",
                    fmt_down="csc",
                    k_hint=4,
                    algo_up="default",
                    algo_down="default",
                    log_level=log_level,
                    instrumentation=instrumentation,
                )),
                id="cusparse-graph-k4",
                marks=pytest.mark.gpu,
            ),
        ]
    )
    if HAS_TRITON_RUNTIME:
        params.append(
            pytest.param(
                ("triton", lambda: make_triton_backend(
                    device=0,
                    fmt_up="csr",
                    fmt_down="csc",
                    k_hint=1,
                    scratch_up="none",
                    scratch_down="none",
                    log_level=log_level,
                    instrumentation=instrumentation,
                )),
                id="triton-csr-csc-k1",
                marks=[pytest.mark.gpu, pytest.mark.triton],
            )
        )
    return params

def _has_mkl_runtime() -> bool:
    try:
        from pygrgl_spmv.backends.mkl import ffi as mkl_ffi

        mkl_ffi._ensure_loaded()
        return True
    except Exception:
        return False


HAS_MKL_RUNTIME = _has_mkl_runtime()


def _has_triton_runtime() -> bool:
    try:
        import torch  # noqa: F401
        import triton  # noqa: F401

        return bool(torch.cuda.is_available())
    except Exception:
        return False


HAS_TRITON_RUNTIME = _has_triton_runtime()


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        default="all",
        choices=["mkl", "cusparse", "triton", "all"],
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
                if "mkl" in item.keywords or "triton" in item.keywords:
                    item.add_marker(skip_mkl)
        case "triton":
            skip_other = pytest.mark.skip(reason="--backend=triton")
            for item in items:
                if "mkl" in item.keywords or "cusparse" in item.keywords:
                    item.add_marker(skip_other)

    if not HAS_MKL_RUNTIME:
        skip_no_mkl = pytest.mark.skip(reason="MKL runtime unavailable (libmkl_rt.so not found)")
        for item in items:
            if "mkl" in item.keywords:
                item.add_marker(skip_no_mkl)

    if not HAS_TRITON_RUNTIME:
        skip_no_triton = pytest.mark.skip(reason="Triton runtime unavailable (torch+triton CUDA not found)")
        for item in items:
            if "triton" in item.keywords:
                item.add_marker(skip_no_triton)

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


def valid_fmt_algo_params(*, transpose_bool: bool = False):
    """Build pytest.param list for valid (fmt, algo) combos only for one transpose mode."""
    from pygrgl_spmv.backends.cusparse import is_valid_combo

    params = []
    for fmt in _ALL_FMTS:
        for algo in _ALL_ALGOS:
            if is_valid_combo(fmt, transpose_bool, algo):
                params.append(pytest.param(fmt, algo, id=f"{fmt}-t{int(bool(transpose_bool))}-{algo}"))
    return params


def invalid_fmt_algo_params(*, transpose_bool: bool = False):
    """Build pytest.param list for invalid (fmt, algo) combos only for one transpose mode."""
    from pygrgl_spmv.backends.cusparse import is_valid_combo

    params = []
    for fmt in _ALL_FMTS:
        for algo in _ALL_ALGOS:
            if not is_valid_combo(fmt, transpose_bool, algo):
                params.append(pytest.param(fmt, algo, id=f"{fmt}-t{int(bool(transpose_bool))}-{algo}"))
    return params


def matmul_expect_k_hint_warning(op, input_matrix, direction, /, **kwargs):
    """Run ``op.matmul`` and expect a k_hint warning only when runtime-k mismatches the plan hint."""
    import pygrgl

    arr = np.asarray(input_matrix)
    if arr.ndim != 2:
        raise ValueError(f"matmul_expect_k_hint_warning requires a 2D input, got shape {arr.shape}")

    if direction == pygrgl.TraversalDirection.UP or str(direction).lower() == "up":
        plan = op._backend._plan_up
    elif direction == pygrgl.TraversalDirection.DOWN or str(direction).lower() == "down":
        plan = op._backend._plan_down
    else:
        raise ValueError(f"Unsupported direction {direction!r}")

    backend_type = str(getattr(op._backend, "__class__", type(op._backend)).__module__).lower()
    hint = None if plan is None else getattr(plan, "k_hint", None)
    expect_warning = "cusparse" in backend_type and hint is not None and int(arr.shape[0]) != int(hint)
    with pytest.warns(RuntimeWarning, match="k_hint") if expect_warning else nullcontext():
        return op.matmul(input_matrix, direction, **kwargs)


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
