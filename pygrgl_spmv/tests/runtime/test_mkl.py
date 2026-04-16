from __future__ import annotations

import ctypes

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv import MklPlan, MklPlanPair, MklRuntime
from pygrgl_spmv.backends.mkl.ffi import MklSparseHandle
from pygrgl_spmv.backends.types import SparseFormat, transpose_compatible_format
from pygrgl_spmv.tests.conftest import DATA_DTYPE, tol
from pygrgl_spmv.tests.runtime._runtime_builders import build_mkl_layout, full_requirements, mkl_pair

pytestmark = pytest.mark.mkl


class _FakeMklLib:
    def __init__(self):
        self.create_calls: list[tuple[str, str, int, int]] = []
        self.hint_calls: list[tuple[str, int, int, int]] = []
        self.optimize_calls: list[int] = []
        self.created_handles: list[int] = []
        self.destroyed_handles: list[int] = []
        self.destroy_calls = 0
        self._next_handle = 1

    def _record_create(self, scalar: str, _handle, m, n):
        self.create_calls.append((scalar, "csr", int(getattr(m, "value", m)), int(getattr(n, "value", n))))
        handle = self._next_handle
        ctypes.cast(_handle, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(handle)
        self.created_handles.append(handle)
        self._next_handle += 1
        return 0

    def mkl_sparse_d_create_csr(self, _handle, _base, m, n, _rows_start, _rows_end, _col_idx, _values):
        return self._record_create("d", _handle, m, n)

    def mkl_sparse_s_create_csr(self, _handle, _base, m, n, _rows_start, _rows_end, _col_idx, _values):
        return self._record_create("s", _handle, m, n)

    def mkl_sparse_destroy(self, handle):
        handle_id = int(getattr(handle, "value", handle))
        assert handle_id in self.created_handles
        assert handle_id not in self.destroyed_handles
        self.destroyed_handles.append(handle_id)
        self.destroy_calls += 1
        return 0

    def mkl_sparse_set_mv_hint(self, handle, op, _descr, expected):
        self.hint_calls.append(("mv", int(getattr(handle, "value", handle)), int(getattr(op, "value", op)), int(getattr(expected, "value", expected))))
        return 0

    def mkl_sparse_set_mm_hint(self, handle, op, _descr, _layout, k, expected):
        self.hint_calls.append(("mm", int(getattr(handle, "value", handle)), int(getattr(op, "value", op)), int(getattr(k, "value", k))))
        assert int(getattr(expected, "value", expected)) == 1000
        return 0

    def mkl_sparse_optimize(self, handle):
        self.optimize_calls.append(int(getattr(handle, "value", handle)))
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


def _open_mkl_grg(artifact, *, pair=None, requirements=None):
    layout = build_mkl_layout(
        [artifact],
        pair=mkl_pair() if pair is None else pair,
        requirements=full_requirements(max_k_up=4, max_k_down=4) if requirements is None else requirements,
    )
    return MklRuntime(layout)


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
    raise AssertionError(f"expected op for level={level} src_level={src_level} handle_id={id(handle)}")


def _assert_sparse_equal(left, right):
    np.testing.assert_array_equal(left.toarray(), right.toarray())


def test_mkl_runtime_matches_pygrgl(primary_artifact, primary_grg):
    with _open_mkl_grg(primary_artifact) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(13)
        up = rng.standard_normal((2, grg.num_samples), dtype=np.float64)
        down = rng.standard_normal((2, grg.num_mutations), dtype=np.float64)
        np.testing.assert_allclose(
            grg.matmul(up, "up"),
            np.asarray(pygrgl.matmul(primary_grg, up, pygrgl.TraversalDirection.UP)),
        )
        np.testing.assert_allclose(
            grg.matmul(down, "down"),
            np.asarray(pygrgl.matmul(primary_grg, down, pygrgl.TraversalDirection.DOWN)),
        )


@pytest.mark.parametrize("n_threads", [0, 1, 2, 4])
def test_mkl_thread_counts_match_reference(primary_artifact, primary_grg, n_threads):
    pair = mkl_pair(
        plan_up=mkl_pair().plan_up.__class__(store="N", fmt="CSR", n_threads=n_threads),
        plan_down=mkl_pair().plan_down.__class__(store="T", fmt="CSC", n_threads=n_threads),
    )
    with _open_mkl_grg(primary_artifact, pair=pair) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(123 + int(n_threads))
        x_up = rng.standard_normal((2, primary_grg.num_samples), dtype=DATA_DTYPE)
        x_down = rng.standard_normal((2, primary_grg.num_mutations), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x_up, "up"), np.asarray(pygrgl.matmul(primary_grg, x_up, pygrgl.TraversalDirection.UP)), atol=atol, rtol=rtol)
        np.testing.assert_allclose(grg.matmul(x_down, "down"), np.asarray(pygrgl.matmul(primary_grg, x_down, pygrgl.TraversalDirection.DOWN)), atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt", [SparseFormat.CSR, SparseFormat.CSC, SparseFormat.COO], ids=["csr", "csc", "coo"])
def test_mkl_format_sweep_matches_reference(primary_artifact, primary_grg, fmt):
    pair = mkl_pair(
        plan_up=mkl_pair().plan_up.__class__(store="N", fmt=fmt, n_threads=1),
        plan_down=mkl_pair().plan_down.__class__(store="T", fmt=transpose_compatible_format(fmt), n_threads=1),
    )
    with _open_mkl_grg(primary_artifact, pair=pair) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(223)
        x_up = rng.standard_normal((3, primary_grg.num_samples), dtype=DATA_DTYPE)
        x_down = rng.standard_normal((3, primary_grg.num_mutations), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x_up, "up"), np.asarray(pygrgl.matmul(primary_grg, x_up, pygrgl.TraversalDirection.UP)), atol=atol, rtol=rtol)
        np.testing.assert_allclose(grg.matmul(x_down, "down"), np.asarray(pygrgl.matmul(primary_grg, x_down, pygrgl.TraversalDirection.DOWN)), atol=atol, rtol=rtol)


def test_run_uses_per_direction_thread_counts(primary_artifact, monkeypatch):
    import pygrgl_spmv.backends.mkl.backend as mkl_backend

    calls: list[int] = []
    monkeypatch.setattr(mkl_backend, "mkl_set_num_threads", lambda n: calls.append(int(n)))
    pair = mkl_pair(
        plan_up=mkl_pair().plan_up.__class__(store="N", fmt="CSR", n_threads=1),
        plan_down=mkl_pair().plan_down.__class__(store="T", fmt="CSC", n_threads=4),
    )
    layout = build_mkl_layout([primary_artifact], pair=pair, requirements=full_requirements(max_k_up=2, max_k_down=2))
    with MklRuntime(layout) as runtime:
        calls.clear()
        (grg,) = runtime.grgs
        rng = np.random.default_rng(6401)
        _ = grg.matmul(rng.standard_normal((2, grg.num_samples), dtype=DATA_DTYPE), "up")
        _ = grg.matmul(rng.standard_normal((2, grg.num_mutations), dtype=DATA_DTYPE), "down")
        assert calls == [1, 4]


def test_lp64_rejects_int64_csr_indices_before_mkl_call(monkeypatch):
    import pygrgl_spmv.backends.mkl.ffi as mkl_ffi

    fake_lib = _FakeMklLib()
    fake_mat = _FakeCsr(indices_dtype=np.int64, indptr_dtype=np.int32)
    monkeypatch.setattr(mkl_ffi, "_ensure_loaded", lambda: (fake_lib, np.int32, ctypes.c_int))
    monkeypatch.setattr(mkl_ffi, "_scipy_to_fmt", lambda _mat, _fmt: fake_mat)

    with pytest.raises(ValueError, match="LP64 MKL requires CSR indices to use int32"):
        mkl_ffi.MklSparseHandle(object(), "csr")
    assert fake_lib.create_calls == []


def test_lp64_rejects_oversized_shape_before_mkl_call(monkeypatch):
    import pygrgl_spmv.backends.mkl.ffi as mkl_ffi

    fake_lib = _FakeMklLib()
    fake_mat = _FakeCsr(shape=(1, np.iinfo(np.int32).max + 1), nnz=0)
    monkeypatch.setattr(mkl_ffi, "_ensure_loaded", lambda: (fake_lib, np.int32, ctypes.c_int))
    monkeypatch.setattr(mkl_ffi, "_scipy_to_fmt", lambda _mat, _fmt: fake_mat)

    with pytest.raises(ValueError, match="LP64 MKL requires ncols <="):
        mkl_ffi.MklSparseHandle(object(), "csr")
    assert fake_lib.create_calls == []


def test_lp64_accepts_valid_int32_csr(monkeypatch):
    import pygrgl_spmv.backends.mkl.ffi as mkl_ffi

    fake_lib = _FakeMklLib()
    fake_mat = _FakeCsr(indices_dtype=np.int32, indptr_dtype=np.int32)
    monkeypatch.setattr(mkl_ffi, "_ensure_loaded", lambda: (fake_lib, np.int32, ctypes.c_int))
    monkeypatch.setattr(mkl_ffi, "_scipy_to_fmt", lambda _mat, _fmt: fake_mat)

    handle = mkl_ffi.MklSparseHandle(object(), "csr")
    assert fake_lib.create_calls == [("d", "csr", 1, 1)]
    handle.destroy()
    assert fake_lib.created_handles == [1]
    assert fake_lib.destroyed_handles == [1]
    assert fake_lib.destroy_calls == 1


def test_lp64_accepts_valid_int32_csr_float32(monkeypatch):
    import pygrgl_spmv.backends.mkl.ffi as mkl_ffi

    fake_lib = _FakeMklLib()
    fake_mat = _FakeCsr(indices_dtype=np.int32, indptr_dtype=np.int32)
    monkeypatch.setattr(mkl_ffi, "_ensure_loaded", lambda: (fake_lib, np.int32, ctypes.c_int))
    monkeypatch.setattr(mkl_ffi, "_scipy_to_fmt", lambda _mat, _fmt: fake_mat)

    handle = mkl_ffi.MklSparseHandle(object(), "csr", dtype=np.float32)
    assert fake_lib.create_calls == [("s", "csr", 1, 1)]
    handle.destroy()
    assert fake_lib.created_handles == [1]
    assert fake_lib.destroyed_handles == [1]
    assert fake_lib.destroy_calls == 1


def test_mkl_runtime_matches_pygrgl_float32(primary_artifact, primary_grg):
    layout = build_mkl_layout([primary_artifact], dtype=np.float32, requirements=full_requirements(max_k_up=4, max_k_down=4))
    with MklRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(17)
        up = rng.standard_normal((3, grg.num_samples), dtype=np.float32)
        down = rng.standard_normal((3, grg.num_mutations), dtype=np.float32)
        atol, rtol = tol(np.float32)
        np.testing.assert_allclose(
            grg.matmul(up, "up"),
            np.asarray(pygrgl.matmul(primary_grg, up, pygrgl.TraversalDirection.UP)),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            grg.matmul(down, "down"),
            np.asarray(pygrgl.matmul(primary_grg, down, pygrgl.TraversalDirection.DOWN)),
            atol=atol,
            rtol=rtol,
        )


def test_mkl_layout_bytes_and_shared_values_are_exact(primary_artifact):
    requirements = full_requirements(max_k_up=4, max_k_down=3)
    layout = build_mkl_layout([primary_artifact], requirements=requirements)
    expected_sparse = sum(block.struct_bytes for artifact in layout.artifacts for block in (*artifact.blocks_up, *artifact.blocks_down))
    with MklRuntime(layout) as runtime:
        state = runtime._artifacts[0].state
        shared = runtime._shared_values
        assert shared is not None
        selector_bytes = int(state.sel_mut.indices.nbytes + state.sel_mut.indptr.nbytes + state.sel_mut.data.nbytes)
        selector_bytes += int(state.sel_miss.indices.nbytes + state.sel_miss.indptr.nbytes + state.sel_miss.data.nbytes)
        expected = {
            "resident_sparse": expected_sparse,
            "selectors": selector_bytes,
            "workspace_up": int(state.num_nodes * int(requirements.max_k_up) * np.dtype(layout.dtype).itemsize),
            "workspace_down": int(state.num_nodes * int(requirements.max_k_down) * np.dtype(layout.dtype).itemsize),
            "shared_values": int(layout.shared_values.physical_bytes),
        }
        assert layout.bytes_by_category == expected
        assert layout.bytes_total == int(sum(expected.values()))
        assert sum(item.nbytes for item in layout.budget_items) == layout.bytes_total
        assert layout.required_budget_for_full_residency == layout.bytes_total
        data_arrays = [
            handle._mat.data
            for artifact in runtime._artifacts
            for grid in (artifact.up_grid, artifact.down_grid)
            for row in grid
            for handle in row
            if handle is not None and handle._mat.nnz > 0
        ]
        assert data_arrays
        for data in data_arrays:
            assert np.shares_memory(data, shared.array)
            assert not data.flags.writeable


def test_mkl_shared_values_plan_alias(monkeypatch, primary_artifact):
    import pygrgl_spmv.backends.mkl.backend as mkl_backend

    monkeypatch.setattr(mkl_backend, "_page_size", lambda: 4096)
    monkeypatch.setattr(mkl_backend, "_vm_max_map_count", lambda: 10000)
    monkeypatch.setattr(mkl_backend, "_current_map_count", lambda: 100)
    monkeypatch.setattr(mkl_backend.os, "memfd_create", lambda *_args, **_kwargs: 0, raising=False)
    layout = build_mkl_layout([primary_artifact], requirements=full_requirements(max_k_up=1, max_k_down=1))
    assert layout.shared_values.mode == "alias"
    assert layout.shared_values.physical_bytes < layout.shared_values.logical_bytes


def test_mkl_shared_values_plan_materialized(monkeypatch, primary_artifact):
    import pygrgl_spmv.backends.mkl.backend as mkl_backend

    monkeypatch.setattr(mkl_backend, "_page_size", lambda: 4096)
    monkeypatch.setattr(mkl_backend, "_vm_max_map_count", lambda: mkl_backend._HEADROOM_MAPS + 101)
    monkeypatch.setattr(mkl_backend, "_current_map_count", lambda: 100)
    layout = build_mkl_layout([primary_artifact], requirements=full_requirements(max_k_up=1, max_k_down=1))
    assert layout.shared_values.mode == "materialized"
    assert layout.shared_values.physical_bytes == layout.shared_values.logical_bytes


def test_optimize_false_skips_handle_hints(primary_artifact, monkeypatch):
    calls: list[tuple[str, bool, int | None]] = []

    monkeypatch.setattr(MklSparseHandle, "set_mv_hint", lambda self, *, transpose=False, expected_calls=1000: calls.append(("mv", bool(transpose), None)))
    monkeypatch.setattr(MklSparseHandle, "set_mm_hint", lambda self, k, *, transpose=False, expected_calls=1000: calls.append(("mm", bool(transpose), int(k))))
    monkeypatch.setattr(MklSparseHandle, "optimize", lambda self: calls.append(("opt", False, None)))
    pair = MklPlanPair(
        plan_up=MklPlan(store="N", fmt="CSR", n_threads=1, optimize=False),
        plan_down=MklPlan(store="T", fmt="CSC", n_threads=1, optimize=False),
    )
    with MklRuntime(build_mkl_layout([primary_artifact], pair=pair, requirements=full_requirements(max_k_up=2, max_k_down=4))):
        pass
    assert calls == []


def test_optimize_true_uses_direction_max_k(primary_artifact, monkeypatch):
    calls: list[tuple[str, bool, int | None]] = []

    monkeypatch.setattr(MklSparseHandle, "set_mv_hint", lambda self, *, transpose=False, expected_calls=1000: calls.append(("mv", bool(transpose), None)))
    monkeypatch.setattr(MklSparseHandle, "set_mm_hint", lambda self, k, *, transpose=False, expected_calls=1000: calls.append(("mm", bool(transpose), int(k))))
    monkeypatch.setattr(MklSparseHandle, "optimize", lambda self: calls.append(("opt", False, None)))
    with MklRuntime(build_mkl_layout([primary_artifact], requirements=full_requirements(max_k_up=2, max_k_down=4))):
        pass
    mm_calls = sorted((transpose, k) for kind, transpose, k in calls if kind == "mm")
    assert mm_calls == [(False, 2), (True, 4)]
    assert sorted(kind for kind, _transpose, _k in calls).count("mv") == 2
    assert sorted(kind for kind, _transpose, _k in calls).count("opt") == 2


def test_mixed_optimize_uses_only_enabled_direction(primary_artifact, monkeypatch):
    calls: list[tuple[str, bool, int | None]] = []

    monkeypatch.setattr(MklSparseHandle, "set_mv_hint", lambda self, *, transpose=False, expected_calls=1000: calls.append(("mv", bool(transpose), None)))
    monkeypatch.setattr(MklSparseHandle, "set_mm_hint", lambda self, k, *, transpose=False, expected_calls=1000: calls.append(("mm", bool(transpose), int(k))))
    monkeypatch.setattr(MklSparseHandle, "optimize", lambda self: calls.append(("opt", False, None)))
    pair = MklPlanPair(
        plan_up=MklPlan(store="N", fmt="CSR", n_threads=1, optimize=False),
        plan_down=MklPlan(store="T", fmt="CSC", n_threads=1, optimize=True),
    )
    with MklRuntime(build_mkl_layout([primary_artifact], pair=pair, requirements=full_requirements(max_k_up=2, max_k_down=4))):
        pass
    assert [entry for entry in calls if entry[0] == "mv"] == [("mv", True, None)]
    assert [entry for entry in calls if entry[0] == "mm"] == [("mm", True, 4)]
    assert len([entry for entry in calls if entry[0] == "opt"]) == 1


def test_up_handles_store_transposed_blocks_when_plan_requests_store_t(primary_artifact):
    pair_store_t = MklPlanPair(plan_up=MklPlan(store="T", fmt="CSR", n_threads=1), plan_down=None)
    pair_reference = MklPlanPair(plan_up=MklPlan(store="N", fmt="CSR", n_threads=1), plan_down=None)
    with MklRuntime(build_mkl_layout([primary_artifact], pair=pair_store_t)) as backend, MklRuntime(build_mkl_layout([primary_artifact], pair=pair_reference)) as reference:
        artifact = backend._artifacts[0]
        reference_artifact = reference._artifacts[0]
        level, src_level, handle = _first_present_handle(artifact.up_grid)
        reference_handle = reference_artifact.up_grid[level][src_level]
        assert reference_handle is not None
        _assert_sparse_equal(handle._mat, reference_handle._mat.T)
        block_op = _find_op(artifact.up_ops, level=level, src_level=src_level, handle=handle)
        assert block_op.transpose


def test_down_handles_store_base_blocks_when_plan_requests_store_n(primary_artifact):
    pair_store_n = MklPlanPair(plan_up=None, plan_down=MklPlan(store="N", fmt="CSC", n_threads=1))
    pair_reference = MklPlanPair(plan_up=None, plan_down=MklPlan(store="T", fmt="CSC", n_threads=1))
    with MklRuntime(build_mkl_layout([primary_artifact], pair=pair_store_n)) as backend, MklRuntime(build_mkl_layout([primary_artifact], pair=pair_reference)) as reference:
        artifact = backend._artifacts[0]
        reference_artifact = reference._artifacts[0]
        level, src_offset, handle = _first_present_handle(artifact.down_grid)
        reference_handle = reference_artifact.down_grid[level][src_offset]
        assert reference_handle is not None
        _assert_sparse_equal(handle._mat, reference_handle._mat.T)
        block_op = _find_op(artifact.down_ops, level=src_offset, src_level=level, handle=handle)
        assert block_op.transpose


def test_shared_down_ops_reuse_up_store_t_handles_without_extra_transpose(primary_artifact):
    pair_shared = MklPlanPair(
        plan_up=MklPlan(store="T", fmt="CSC", n_threads=1),
        plan_down=MklPlan(store="N", fmt="CSR", n_threads=1),
    )
    pair_reference = MklPlanPair(
        plan_up=MklPlan(store="N", fmt="CSC", n_threads=1),
        plan_down=None,
    )
    with MklRuntime(build_mkl_layout([primary_artifact], pair=pair_shared)) as backend, MklRuntime(build_mkl_layout([primary_artifact], pair=pair_reference)) as reference:
        artifact = backend._artifacts[0]
        reference_artifact = reference._artifacts[0]
        assert all(all(block is None for block in row) for row in artifact.down_grid)
        level, src_level, handle = _first_present_handle(artifact.up_grid)
        reference_handle = reference_artifact.up_grid[level][src_level]
        assert reference_handle is not None
        _assert_sparse_equal(handle._mat, reference_handle._mat.T)
        shared_op = _find_op(artifact.down_ops, level=src_level, src_level=level, handle=handle)
        assert not shared_op.transpose
