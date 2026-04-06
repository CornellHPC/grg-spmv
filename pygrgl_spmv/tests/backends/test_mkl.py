"""MKL backend tests."""

from __future__ import annotations

import ctypes
import warnings

import numpy as np
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.backends.mkl import MklBackend, MklPlan, MklPlanPair
from pygrgl_spmv.tests.conftest import (
    DATA_DTYPE,
    K_CORE,
    K_MATRIX,
    make_mkl_backend,
    make_mkl_plan,
    tol,
)

pytestmark = pytest.mark.mkl

MKL_FMTS = ["csr", "csc", "coo"]


class _FakeMklLib:
    def __init__(self):
        self.calls: list[tuple[str, int, int]] = []

    def mkl_sparse_d_create_csr(self, _handle, _base, m, n, _rows_start, _rows_end, _col_idx, _values):
        self.calls.append(("csr", int(getattr(m, "value", m)), int(getattr(n, "value", n))))
        return 0

    def mkl_sparse_destroy(self, _handle):
        return 0


class _FakeCsr:
    def __init__(
        self,
        *,
        shape: tuple[int, int] = (1, 1),
        indices_dtype=np.int32,
        indptr_dtype=np.int32,
        nnz: int = 1,
    ) -> None:
        self.shape = tuple(int(v) for v in shape)
        self.nnz = int(nnz)
        self.indices = np.zeros(self.nnz, dtype=indices_dtype)
        self.indptr = np.array([0, self.nnz], dtype=indptr_dtype)
        self.data = np.ones(self.nnz, dtype=np.float64)


def _make_mkl_op(
    grg_path,
    *,
    cache_dir,
    n_threads=0,
    fmt_up="csr",
    fmt_down=None,
    k_hint=None,
    infer_missing=True,
    dtype=DATA_DTYPE,
):
    return SpmvGRG(
        grg_path,
        make_mkl_backend(
            fmt_up=fmt_up,
            fmt_down=fmt_down,
            k_hint=k_hint,
            n_threads=n_threads,
            infer_missing=infer_missing,
            log_level="WARNING",
        ),
        dtype,
        artifact_dir=cache_dir,
    )


def _run_up(op, X_col_major):
    return op.matmul(X_col_major.T, "up").T


def _run_down(op, X_col_major):
    return op.matmul(X_col_major.T, "down").T


def _assert_sparse_equal(left, right):
    np.testing.assert_array_equal(left.toarray(), right.toarray())


def _first_present_handle(grid):
    for row_idx, row in enumerate(grid):
        for col_idx, handle in enumerate(row):
            if handle is not None:
                return row_idx, col_idx, handle
    raise AssertionError("expected at least one sparse handle")


def _find_op(ops_by_level, *, level: int, src_level: int, handle):
    for op in ops_by_level[level]:
        if op.src_level == src_level and op.handle is handle:
            return op
    raise AssertionError(
        f"expected op for level={level} src_level={src_level} handle_id={id(handle)}"
    )


def _make_backend(primary_grg_path, spmv_cache_dir, *, plan_up, plan_down):
    op = SpmvGRG(
        primary_grg_path,
        MklBackend(
            pair=MklPlanPair(
                plan_up=None if plan_up is None else MklPlan.from_dict(plan_up),
                plan_down=None if plan_down is None else MklPlan.from_dict(plan_down),
            ),
            log_level="WARNING",
        ),
        DATA_DTYPE,
        artifact_dir=spmv_cache_dir,
    )
    return op._backend


class TestSmokeCoreConfigMatrix:
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "cfg,k",
        [
            pytest.param({"fmt_up": "csr", "fmt_down": None, "k_hint": 1}, 1, id="up-csr-down-auto-k1"),
            pytest.param({"fmt_up": None, "fmt_down": "csc", "k_hint": 4}, 4, id="up-auto-down-csc-k4"),
            pytest.param({"fmt_up": "coo", "fmt_down": "coo", "k_hint": None}, 2, id="up-coo-down-coo-dyn"),
        ],
    )
    def test_forward(self, primary_grg_path, gt_small, spmv_cache_dir, cfg, k):
        op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, **cfg)
        X, Y_expected = gt_small.get("forward", k, seed=6201, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "cfg,k",
        [
            pytest.param({"fmt_up": "csr", "fmt_down": None, "k_hint": 1}, 1, id="up-csr-down-auto-k1"),
            pytest.param({"fmt_up": None, "fmt_down": "csc", "k_hint": 4}, 4, id="up-auto-down-csc-k4"),
            pytest.param({"fmt_up": "coo", "fmt_down": "coo", "k_hint": None}, 2, id="up-coo-down-coo-dyn"),
        ],
    )
    def test_backward(self, primary_grg_path, gt_small, spmv_cache_dir, cfg, k):
        op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, **cfg)
        X, Y_expected = gt_small.get("backward", k, seed=6202, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("n_threads", [0, 1, 2, 4, 8])
@pytest.mark.parametrize("k", K_CORE)
def test_thread_counts_forward(primary_grg_path, gt_small, spmv_cache_dir, n_threads, k):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, n_threads=n_threads)
    X, Y_expected = gt_small.get("forward", k, seed=123, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("n_threads", [0, 1, 2, 4, 8])
@pytest.mark.parametrize("k", K_CORE)
def test_thread_counts_backward(primary_grg_path, gt_small, spmv_cache_dir, n_threads, k):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, n_threads=n_threads)
    X, Y_expected = gt_small.get("backward", k, seed=123, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt", MKL_FMTS)
@pytest.mark.parametrize("k", K_CORE)
def test_formats_forward(primary_grg_path, gt_small, spmv_cache_dir, fmt, k):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up=fmt)
    X, Y_expected = gt_small.get("forward", k, seed=223, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt", MKL_FMTS)
@pytest.mark.parametrize("k", K_CORE)
def test_formats_backward(primary_grg_path, gt_small, spmv_cache_dir, fmt, k):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up=fmt)
    X, Y_expected = gt_small.get("backward", k, seed=223, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("k_hint", [None, 1, 4])
@pytest.mark.parametrize("k", K_CORE)
def test_k_hint_invariance_forward(primary_grg_path, gt_small, spmv_cache_dir, k_hint, k):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, k_hint=k_hint)
    X, Y_expected = gt_small.get("forward", k, seed=323, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("k_hint", [None, 1, 4])
@pytest.mark.parametrize("k", K_CORE)
def test_k_hint_invariance_backward(primary_grg_path, gt_small, spmv_cache_dir, k_hint, k):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, k_hint=k_hint)
    X, Y_expected = gt_small.get("backward", k, seed=323, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


def test_none_hint_emits_no_warning(primary_grg_path, gt_small, spmv_cache_dir):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, k_hint=None)
    X, _ = gt_small.get("forward", 3, seed=4101, dtype=DATA_DTYPE)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _run_up(op, X)
    assert len(rec) == 0


def test_warn_every_mismatch_call(primary_grg_path, gt_small, spmv_cache_dir):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, k_hint=4)
    X1, _ = gt_small.get("forward", 1, seed=4102, dtype=DATA_DTYPE)
    X2, _ = gt_small.get("forward", 2, seed=4103, dtype=DATA_DTYPE)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _run_up(op, X1)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _run_up(op, X1)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _run_up(op, X2)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
@pytest.mark.parametrize("k", K_MATRIX)
def test_primary_grg_forward(primary_grg_path, gt_primary, spmv_cache_dir, fmt, k):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up=fmt, n_threads=1)
    X, Y_expected = gt_primary.get("forward", k, seed=5123, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
@pytest.mark.parametrize("k", K_MATRIX)
def test_primary_grg_backward(primary_grg_path, gt_primary, spmv_cache_dir, fmt, k):
    op = _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up=fmt, n_threads=1)
    X, Y_expected = gt_primary.get("backward", k, seed=5123, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


def test_mkl_bsr_not_supported(primary_grg_path, spmv_cache_dir):
    with pytest.raises(ValueError, match="sparse format|CSR|CSC|COO"):
        _make_mkl_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="bsr")


def test_run_uses_per_direction_thread_counts(primary_grg_path, gt_small, spmv_cache_dir, monkeypatch):
    import pygrgl_spmv.backends.mkl.backend as mkl_backend

    calls: list[int] = []
    monkeypatch.setattr(mkl_backend, "mkl_set_num_threads", lambda n: calls.append(int(n)))
    op = SpmvGRG(
        primary_grg_path,
        MklBackend(
            pair=MklPlanPair.from_dicts(
                make_mkl_plan(store="N", fmt="CSR", n_threads=1, k_hint=None),
                make_mkl_plan(store="T", fmt="CSC", n_threads=4, k_hint=None),
            ),
            log_level="WARNING",
        ),
        DATA_DTYPE,
        artifact_dir=spmv_cache_dir,
    )

    calls.clear()
    x_up, _ = gt_small.get("forward", 2, seed=6401, dtype=DATA_DTYPE)
    x_down, _ = gt_small.get("backward", 2, seed=6402, dtype=DATA_DTYPE)
    _run_up(op, x_up)
    _run_down(op, x_down)
    assert calls == [1, 4]


def test_nonshared_handles_keep_distinct_k_hints(primary_grg_path, spmv_cache_dir, monkeypatch):
    from pygrgl_spmv.backends.mkl.ffi import MklSparseHandle

    calls: list[tuple[int, int, bool]] = []
    original = MklSparseHandle.set_mm_hint

    def record(self, k, *, transpose=False, expected_calls=1000):
        calls.append((id(self), int(k), bool(transpose)))
        return original(self, k, transpose=transpose, expected_calls=expected_calls)

    monkeypatch.setattr(MklSparseHandle, "set_mm_hint", record)
    _ = SpmvGRG(
        primary_grg_path,
        MklBackend(
            pair=MklPlanPair.from_dicts(
                make_mkl_plan(store="N", fmt="CSR", n_threads=1, k_hint=2),
                make_mkl_plan(store="T", fmt="CSR", n_threads=1, k_hint=16),
            ),
            log_level="WARNING",
        ),
        DATA_DTYPE,
        artifact_dir=spmv_cache_dir,
    )

    assert sorted(set(k for _, k, _ in calls)) == [2, 16]
    assert all(not transpose for _, _, transpose in calls)
    assert len({handle_id for handle_id, _, _ in calls}) >= 2


def test_lp64_rejects_int64_csr_indices_before_mkl_call(monkeypatch):
    import pygrgl_spmv.backends.mkl.ffi as mkl_ffi

    fake_lib = _FakeMklLib()
    fake_mat = _FakeCsr(indices_dtype=np.int64, indptr_dtype=np.int32)
    monkeypatch.setattr(mkl_ffi, "_ensure_loaded", lambda: (fake_lib, np.int32, ctypes.c_int))
    monkeypatch.setattr(mkl_ffi, "_scipy_to_fmt", lambda _mat, _fmt: fake_mat)

    with pytest.raises(ValueError, match="LP64 MKL requires CSR indices to use int32"):
        mkl_ffi.MklSparseHandle(object(), "csr")
    assert fake_lib.calls == []


def test_lp64_rejects_oversized_shape_before_mkl_call(monkeypatch):
    import pygrgl_spmv.backends.mkl.ffi as mkl_ffi

    fake_lib = _FakeMklLib()
    fake_mat = _FakeCsr(shape=(1, np.iinfo(np.int32).max + 1), nnz=0)
    monkeypatch.setattr(mkl_ffi, "_ensure_loaded", lambda: (fake_lib, np.int32, ctypes.c_int))
    monkeypatch.setattr(mkl_ffi, "_scipy_to_fmt", lambda _mat, _fmt: fake_mat)

    with pytest.raises(ValueError, match="LP64 MKL requires ncols <="):
        mkl_ffi.MklSparseHandle(object(), "csr")
    assert fake_lib.calls == []


def test_lp64_accepts_valid_int32_csr(monkeypatch):
    import pygrgl_spmv.backends.mkl.ffi as mkl_ffi

    fake_lib = _FakeMklLib()
    fake_mat = _FakeCsr(indices_dtype=np.int32, indptr_dtype=np.int32)
    monkeypatch.setattr(mkl_ffi, "_ensure_loaded", lambda: (fake_lib, np.int32, ctypes.c_int))
    monkeypatch.setattr(mkl_ffi, "_scipy_to_fmt", lambda _mat, _fmt: fake_mat)

    handle = mkl_ffi.MklSparseHandle(object(), "csr")
    assert fake_lib.calls == [("csr", 1, 1)]
    handle.destroy()


def test_up_handles_store_transposed_blocks_when_plan_requests_store_t(primary_grg_path, spmv_cache_dir):
    backend = _make_backend(
        primary_grg_path,
        spmv_cache_dir,
        plan_up=make_mkl_plan(store="T", fmt="CSR", n_threads=1, k_hint=None),
        plan_down=None,
    )
    reference = _make_backend(
        primary_grg_path,
        spmv_cache_dir,
        plan_up=make_mkl_plan(store="N", fmt="CSR", n_threads=1, k_hint=None),
        plan_down=None,
    )

    level, src_level, handle = _first_present_handle(backend._blocks_up)
    reference_handle = reference._blocks_up[level][src_level]
    assert reference_handle is not None
    _assert_sparse_equal(handle._mat, reference_handle._mat.T)

    block_op = _find_op(backend._ops_up, level=level, src_level=src_level, handle=handle)
    assert block_op.transpose


def test_down_handles_store_base_blocks_when_plan_requests_store_n(primary_grg_path, spmv_cache_dir):
    backend = _make_backend(
        primary_grg_path,
        spmv_cache_dir,
        plan_up=None,
        plan_down=make_mkl_plan(store="N", fmt="CSC", n_threads=1, k_hint=None),
    )
    reference = _make_backend(
        primary_grg_path,
        spmv_cache_dir,
        plan_up=None,
        plan_down=make_mkl_plan(store="T", fmt="CSC", n_threads=1, k_hint=None),
    )

    level, src_offset, handle = _first_present_handle(backend._blocks_down)
    src_level = level + src_offset + 1
    reference_handle = reference._blocks_down[level][src_offset]
    assert reference_handle is not None
    _assert_sparse_equal(handle._mat, reference_handle._mat.T)

    block_op = _find_op(backend._ops_down, level=level, src_level=src_level, handle=handle)
    assert block_op.transpose


def test_shared_down_ops_reuse_up_store_t_handles_without_extra_transpose(primary_grg_path, spmv_cache_dir):
    backend = _make_backend(
        primary_grg_path,
        spmv_cache_dir,
        plan_up=make_mkl_plan(store="T", fmt="CSC", n_threads=1, k_hint=None),
        plan_down=make_mkl_plan(store="N", fmt="CSR", n_threads=1, k_hint=None),
    )
    reference = _make_backend(
        primary_grg_path,
        spmv_cache_dir,
        plan_up=make_mkl_plan(store="N", fmt="CSC", n_threads=1, k_hint=None),
        plan_down=None,
    )

    assert not backend._store_blocks_down
    assert all(len(row) == 0 for row in backend._blocks_down)

    level, src_level, handle = _first_present_handle(backend._blocks_up)
    reference_handle = reference._blocks_up[level][src_level]
    assert reference_handle is not None
    _assert_sparse_equal(handle._mat, reference_handle._mat.T)

    shared_op = _find_op(backend._ops_down, level=src_level, src_level=level, handle=handle)
    assert not shared_op.transpose
