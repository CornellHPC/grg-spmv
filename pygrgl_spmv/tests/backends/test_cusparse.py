"""cuSPARSE backend tests."""

from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass
import functools
import logging
import weakref
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.backends import BackendSetup, ReferenceBackend, ReferencePlanPair
from pygrgl_spmv.backends.cusparse import CusparseBackend, CusparsePlan, CusparsePlanPair, is_valid_combo
from pygrgl_spmv.backends.types import Direction, InitMode
from pygrgl_spmv.grg.sparse import binary_csr_from_parts
from pygrgl_spmv.memory import alloc_field, capture_snapshot, live_snapshot, tree_rows
from pygrgl_spmv.tests.backends._streaming_stress import (
    LargeBandCase,
    build_large_band_setup,
    build_overlap_band_setup,
    clear_gpu_state,
    expected_down,
    expected_up,
    prepare_cusparse_large_band_case,
)
from pygrgl_spmv.tests.conftest import (
    DATA_DTYPE,
    K_CORE,
    K_MATRIX,
    binary_pm1,
    invalid_fmt_algo_params,
    make_cusparse_backend,
    make_cusparse_plan,
    tol,
    valid_fmt_algo_params,
)

cp = pytest.importorskip("cupy")
pytestmark = [pytest.mark.gpu, pytest.mark.cusparse]

_CACHE_DIR = Path(".pytest_cache") / "pygrgl_spmv_npz"


def _make_op(
    grg_path,
    *,
    device=0,
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
    dtype=DATA_DTYPE,
    cache_dir=_CACHE_DIR,
):
    return SpmvGRG(
        grg_path,
        make_cusparse_backend(
            device=device,
            stream=stream,
            fmt_up=fmt_up,
            fmt_down=fmt_down,
            k_hint=k_hint,
            algo_up=algo_up,
            algo_down=algo_down,
            scratch_up=scratch_up,
            scratch_down=scratch_down,
            log_level=log_level,
            instrumentation=instrumentation,
            infer_missing=infer_missing,
        ),
        dtype,
        artifact_dir=cache_dir,
    )


def _run_up(op, X_col_major):
    return op.matmul(X_col_major.T, "up").T


def _run_down(op, X_col_major):
    return op.matmul(X_col_major.T, "down").T


def _synthetic_setup(
    *,
    n: int = 4,
    block_indices_dtype=np.int32,
    block_indptr_dtype=np.int32,
    selector_indices_dtype=np.int32,
    selector_indptr_dtype=np.int32,
    level_offsets_dtype=np.int32,
):
    I = binary_csr_from_parts(
        indices=np.arange(n, dtype=np.int32),
        indptr=np.arange(n + 1, dtype=np.int32),
        shape=(n, n),
        shared_data=True,
    )
    P = binary_csr_from_parts(
        indices=np.roll(np.arange(n, dtype=np.int32), 1),
        indptr=np.arange(n + 1, dtype=np.int32),
        shape=(n, n),
        shared_data=True,
    )
    sel_mut = binary_csr_from_parts(
        indices=np.arange(2 * n, 3 * n, dtype=np.int32),
        indptr=np.arange(n + 1, dtype=np.int32),
        shape=(n, 3 * n),
        shared_data=True,
    )
    sel_miss = binary_csr_from_parts(
        indices=np.empty(0, dtype=np.int32),
        indptr=np.zeros(n + 1, dtype=np.int32),
        shape=(n, 3 * n),
        shared_data=True,
    )

    for block in (I, P):
        block.indices = block.indices.astype(block_indices_dtype)
        block.indptr = block.indptr.astype(block_indptr_dtype)
    sel_mut.indices = sel_mut.indices.astype(selector_indices_dtype)
    sel_mut.indptr = sel_mut.indptr.astype(selector_indptr_dtype)
    sel_miss.indices = sel_miss.indices.astype(selector_indices_dtype)
    sel_miss.indptr = sel_miss.indptr.astype(selector_indptr_dtype)

    return BackendSetup(
        A_blocks=[[], [I], [P, I]],
        level_offsets=np.asarray([0, n, 2 * n, 3 * n], dtype=level_offsets_dtype),
        num_samples=n,
        num_mutations=n,
        num_nodes=3 * n,
        sel_mut=sel_mut,
        sel_miss=sel_miss,
        coalescence_counts=None,
        dtype=np.float64,
    )


def _materialized_csc_block_with_large_row_index() -> sp.csc_matrix:
    nrows = int(np.iinfo(np.int32).max) + 2
    block = sp.csc_matrix((1, 1), dtype=np.bool_)
    block.data = np.ones(1, dtype=np.bool_)
    block.indices = np.array([nrows - 1], dtype=np.int64)
    block.indptr = np.array([0, 1], dtype=np.int32)
    block._shape = (nrows, 1)
    return block


def _materialized_coo_block_with_large_row_index() -> sp.coo_matrix:
    nrows = int(np.iinfo(np.int32).max) + 2
    block = sp.coo_matrix((1, 1), dtype=np.bool_)
    block.data = np.ones(1, dtype=np.bool_)
    block.row = np.array([nrows - 1], dtype=np.int64)
    block.col = np.array([0], dtype=np.int32)
    block._shape = (nrows, 1)
    return block


@pytest.fixture(scope="module")
def large_band_case() -> LargeBandCase:
    return prepare_cusparse_large_band_case()


def _make_stream_backend(*, ring_buffer_size: int) -> object:
    return make_cusparse_backend(
        device=0,
        ring_buffer_size=int(ring_buffer_size),
        fmt_up="csr",
        fmt_down="csc",
        k_hint=None,
        infer_missing=False,
        log_level="WARNING",
    )


def _force_int64_slot_dtypes(monkeypatch, backend) -> None:
    # CUDA 12.9 cuSPARSE SpMM becomes unreliable when nnz approaches 2^31 - 1
    # for both int32 and int64 structure, so these large-stream tests keep nnz
    # below that boundary and force int64 slot families explicitly.
    monkeypatch.setattr(
        backend,
        "_scan_slot_struct_dtypes",
        lambda: (np.dtype(np.int64), np.dtype(np.int64)),
    )


def _make_reference_backend() -> ReferenceBackend:
    return ReferenceBackend(
        pair=ReferencePlanPair(
            plan_up=ReferenceBackend.plan(fmt="CSR", store="N", k_hint=None),
            plan_down=ReferenceBackend.plan(fmt="CSC", store="T", k_hint=None),
        ),
        log_level="WARNING",
    )


def _run_stream_direction(backend: object, direction: str, primary: np.ndarray) -> np.ndarray:
    if direction == "up":
        output, _ = backend.run_up(primary, init_mode=InitMode.NONE, init=None, need_miss_output=False)
        return output
    if direction == "down":
        return backend.run_down(primary, miss=None, init_mode=InitMode.NONE, init=None)
    raise ValueError(f"unknown direction {direction!r}")


def _expected_band_direction(
    direction: str,
    primary: np.ndarray,
    *,
    shifts: tuple[int, int, int],
    bandwidth: int,
) -> np.ndarray:
    if direction == "up":
        return expected_up(primary, shifts=shifts, bandwidth=bandwidth)
    if direction == "down":
        return expected_down(primary, shifts=shifts, bandwidth=bandwidth)
    raise ValueError(f"unknown direction {direction!r}")


@functools.cache
def _spin_kernel():
    module = cp.RawModule(
        code=r"""
        extern "C" __global__ void spin(unsigned long long iters) {
          unsigned long long start = clock64();
          while (clock64() - start < iters) {}
        }
        """
    )
    return module.get_function("spin")


def _warm_spin_kernel() -> None:
    _spin_kernel()((1,), (1,), (1_000_000,))
    cp.cuda.runtime.deviceSynchronize()


def _install_bridge_probe(monkeypatch, backend):
    counts = {"entered": 0, "exited": 0}
    original = type(backend)._caller_root_scope

    @contextmanager
    def _wrapped(self):
        counts["entered"] += 1
        with original(self):
            try:
                yield
            finally:
                counts["exited"] += 1

    monkeypatch.setattr(type(backend), "_caller_root_scope", _wrapped)
    return counts


def _install_scope_depth_probe(monkeypatch, backend):
    depth = {"value": 0}
    original = type(backend)._caller_root_scope

    @contextmanager
    def _wrapped(self):
        depth["value"] += 1
        try:
            with original(self):
                yield
        finally:
            depth["value"] -= 1

    monkeypatch.setattr(type(backend), "_caller_root_scope", _wrapped)
    return depth


_CSR_ALG3_GRAPH_XFAIL = pytest.mark.xfail(
    reason="known cuSPARSE CSR_ALG3 CUDA graph capture bug",
    raises=RuntimeError,
    strict=True,
)


def _graph_fmt_algo_params():
    params = []
    for param in valid_fmt_algo_params():
        if param.values[1] == "csr_alg3":
            params.append(pytest.param(*param.values, marks=(*param.marks, _CSR_ALG3_GRAPH_XFAIL), id=param.id))
        else:
            params.append(param)
    return params


def _graph_fmt_algo_k_params():
    params = []
    for k in K_CORE:
        for param in valid_fmt_algo_params():
            marks = param.marks
            if param.values[1] == "csr_alg3" and k > 1:
                marks = (*marks, _CSR_ALG3_GRAPH_XFAIL)
            params.append(pytest.param(*param.values, k, marks=marks, id=f"{k}-{param.id}"))
    return params


def test_cupy_memory_accounting_uses_logical_bytes():
    @dataclass
    class _CupyRoot:
        payload: object | None = alloc_field(
            label="payload",
            kind="temporary",
            owner="backend",
            retention="persistent",
            activity="always",
            default=None,
        )

    arr = cp.zeros((3,), dtype=cp.float64)
    snapshot = capture_snapshot(_CupyRoot(payload=arr), stage="retained", runtime_k=None)
    assert len(snapshot.allocations) == 1
    assert snapshot.allocations[0].nbytes == int(arr.nbytes)


def test_cusparse_host_pin_uses_separate_struct_dtypes_for_csc_blocks():
    backend = make_cusparse_backend(device=0, fmt_up="csc", fmt_down=None, k_hint=None, infer_missing=False)
    backend._apply_setup_state(_synthetic_setup())
    backend._H = len(backend._level_offsets) - 1
    stored = _materialized_csc_block_with_large_row_index()

    backend._slot_struct0_dtype = np.dtype(np.int32)
    backend._slot_struct1_dtype = np.dtype(np.int64)
    host_block = backend._pin_host_block(stored)
    assert host_block is not None
    assert host_block.struct_buffers[0].dtype == np.int32
    assert host_block.struct_buffers[1].dtype == np.int64


def test_cusparse_coo_slots_use_common_coordinate_dtype(monkeypatch):
    backend = make_cusparse_backend(device=0, fmt_up="coo", fmt_down=None, k_hint=None, algo_up="coo_alg1", infer_missing=False)
    backend._apply_setup_state(_synthetic_setup())
    backend._H = len(backend._level_offsets) - 1
    stored = _materialized_coo_block_with_large_row_index()
    original = backend._stored_matrix

    def _wrapped(direction: Direction, *, dst_level: int, src_level: int):
        if direction == Direction.UP and dst_level == 1 and src_level == 0:
            return stored
        return original(direction, dst_level=dst_level, src_level=src_level)

    monkeypatch.setattr(backend, "_stored_matrix", _wrapped)
    struct0_dtype, struct1_dtype = backend._scan_slot_struct_dtypes()
    assert struct0_dtype == np.dtype(np.int64)
    assert struct1_dtype == np.dtype(np.int64)


def test_cusparse_store_t_coo_materialization_is_row_sorted():
    block = binary_csr_from_parts(
        indices=np.array([1, 2, 0, 2], dtype=np.int32),
        indptr=np.array([0, 2, 4], dtype=np.int32),
        shape=(2, 3),
        shared_data=True,
    )
    direct = block.T.tocoo()
    assert not np.all(np.asarray(direct.row)[1:] >= np.asarray(direct.row)[:-1])

    sel_mut = binary_csr_from_parts(
        indices=np.empty(0, dtype=np.int32),
        indptr=np.zeros(2, dtype=np.int32),
        shape=(1, 5),
        shared_data=True,
    )
    sel_miss = binary_csr_from_parts(
        indices=np.empty(0, dtype=np.int32),
        indptr=np.zeros(2, dtype=np.int32),
        shape=(1, 5),
        shared_data=True,
    )
    backend = make_cusparse_backend(
        device=0,
        fmt_up=None,
        fmt_down="coo",
        k_hint=None,
        algo_down="coo_alg3",
        infer_missing=False,
    )
    backend._apply_setup_state(
        BackendSetup(
            A_blocks=[[], [block]],
            level_offsets=np.array([0, 3, 5], dtype=np.int32),
            num_samples=3,
            num_mutations=1,
            num_nodes=5,
            sel_mut=sel_mut,
            sel_miss=sel_miss,
            coalescence_counts=None,
            dtype=np.float64,
        )
    )
    backend._H = len(backend._level_offsets) - 1

    stored = backend._materialize_stored_block(Direction.DOWN, dst_level=0, src_level=1)
    assert stored is not None
    rows = np.asarray(stored.row)
    cols = np.asarray(stored.col)
    assert np.all(rows[1:] >= rows[:-1])
    assert np.all((rows[1:] > rows[:-1]) | ((rows[1:] == rows[:-1]) & (cols[1:] >= cols[:-1])))
    if hasattr(stored, "has_canonical_format"):
        assert bool(stored.has_canonical_format)


def test_cusparse_slot_pool_compacts_safe_int64_indices_and_offsets():
    backend = make_cusparse_backend(device=0, fmt_up="csr", fmt_down="csc", k_hint=1)
    backend.setup(
        _synthetic_setup(
            block_indices_dtype=np.int64,
            block_indptr_dtype=np.int64,
            selector_indices_dtype=np.int64,
            selector_indptr_dtype=np.int64,
            level_offsets_dtype=np.int64,
        )
    )
    assert backend._slot_struct0_dtype == np.dtype(np.int32)
    assert backend._slot_struct1_dtype == np.dtype(np.int32)
    assert backend._slot_pool is not None
    assert backend._slot_pool.slots[0].struct0.dtype == np.int32
    assert backend._slot_pool.slots[0].struct1.dtype == np.int32
    y, _ = backend.run_up(np.arange(1, 5, dtype=np.float64).reshape(4, 1), init_mode=InitMode.NONE, init=None, need_miss_output=False)
    np.testing.assert_array_equal(y[:, 0], np.array([5.0, 3.0, 5.0, 7.0]))


@pytest.mark.stress
@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("runtime_k", [1, 2])
@pytest.mark.parametrize("ring_buffer_size", [1, 2])
def test_cusparse_large_stream_exact_two_pass_sequence(large_band_case, order, runtime_k, ring_buffer_size, monkeypatch):
    clear_gpu_state()
    backend = _make_stream_backend(ring_buffer_size=ring_buffer_size)
    _force_int64_slot_dtypes(monkeypatch, backend)
    setup = build_large_band_setup(large_band_case)
    try:
        backend.setup(setup)
        assert backend._slot_struct0_dtype == np.dtype(np.int64)
        assert backend._slot_struct1_dtype == np.dtype(np.int64)
        for run_idx, direction in enumerate(order):
            rng = np.random.default_rng(40_000 + 1_000 * ring_buffer_size + 100 * runtime_k + 10 * run_idx + (0 if direction == "up" else 1))
            primary = binary_pm1(rng, (large_band_case.n, int(runtime_k)), DATA_DTYPE)
            expected = _expected_band_direction(
                direction,
                primary,
                shifts=large_band_case.shifts,
                bandwidth=large_band_case.bandwidth,
            )
            actual = _run_stream_direction(backend, direction, primary)
            np.testing.assert_array_equal(actual, expected)
    finally:
        del backend
        del setup
        clear_gpu_state()


@pytest.mark.stress
@pytest.mark.parametrize("first_direction", ["up", "down"])
@pytest.mark.xfail(strict=True, reason="ring=3 must fail once three streamed slots cannot fit into total VRAM")
def test_cusparse_large_stream_ring3_xfail(large_band_case, first_direction, monkeypatch):
    clear_gpu_state()
    backend = _make_stream_backend(ring_buffer_size=3)
    _force_int64_slot_dtypes(monkeypatch, backend)
    setup = build_large_band_setup(large_band_case)
    try:
        backend.setup(setup)
        assert backend._slot_struct0_dtype == np.dtype(np.int64)
        assert backend._slot_struct1_dtype == np.dtype(np.int64)
        rng = np.random.default_rng(90_000 + (0 if first_direction == "up" else 1))
        primary = binary_pm1(rng, (large_band_case.n, 1), DATA_DTYPE)
        _ = _run_stream_direction(backend, first_direction, primary)
    finally:
        del backend
        del setup
        clear_gpu_state()


@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("runtime_k", [1, 2])
@pytest.mark.parametrize("ring_buffer_size", [1, 2])
def test_cusparse_stream_copy_overlaps_compute(order, runtime_k, ring_buffer_size, monkeypatch):
    clear_gpu_state()
    _warm_spin_kernel()
    backend = _make_stream_backend(ring_buffer_size=ring_buffer_size)
    ref_backend = _make_reference_backend()
    setup = build_overlap_band_setup()
    state = {
        "active": True,
        "compute": [],
        "copy": [],
    }
    original_launch = type(backend)._launch_spmm

    def _wrapped_launch(self, *, ws, op, sp_desc, dst_desc, beta_ptr, stream, ext_ptr):
        if state["active"]:
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record(stream)
            original_launch(
                self,
                ws=ws,
                op=op,
                sp_desc=sp_desc,
                dst_desc=dst_desc,
                beta_ptr=beta_ptr,
                stream=stream,
                ext_ptr=ext_ptr,
            )
            with stream:
                _spin_kernel()((1,), (1,), (10_000_000,))
            end.record(stream)
            state["compute"].append((start, end))
            return
        original_launch(
            self,
            ws=ws,
            op=op,
            sp_desc=sp_desc,
            dst_desc=dst_desc,
            beta_ptr=beta_ptr,
            stream=stream,
            ext_ptr=ext_ptr,
        )

    def _wrapped_copy(self, ws, dst_level, op_idx, op):
        if self._slot_pool is None:
            raise RuntimeError("slot pool is not initialized")
        copy_stream = self._slot_copy_streams[op.slot]
        slot_buffers = self._slot_pool.slots[op.slot]
        host0, host1 = op.host_block.struct_buffers
        with copy_stream:
            prev_event = self._prev_compute_event(ws, op)
            if prev_event is not None:
                copy_stream.wait_event(prev_event)
            start = None
            end = None
            if state["active"]:
                start = cp.cuda.Event()
                end = cp.cuda.Event()
                start.record(copy_stream)
            self._cp.cuda.runtime.memcpyAsync(
                slot_buffers.struct0.data.ptr,
                int(np.asarray(host0).ctypes.data),
                int(host0.nbytes),
                self._cp.cuda.runtime.memcpyHostToDevice,
                copy_stream.ptr,
            )
            self._cp.cuda.runtime.memcpyAsync(
                slot_buffers.struct1.data.ptr,
                int(np.asarray(host1).ctypes.data),
                int(host1.nbytes),
                self._cp.cuda.runtime.memcpyHostToDevice,
                copy_stream.ptr,
            )
            if state["active"]:
                assert start is not None and end is not None
                end.record(copy_stream)
                state["copy"].append((start, end))
            ws.copy_done_by_level[dst_level][op_idx].record(copy_stream)

    monkeypatch.setattr(type(backend), "_launch_spmm", _wrapped_launch)
    monkeypatch.setattr(type(backend), "_copy_host_block_to_slot", _wrapped_copy)

    def _has_overlap() -> bool:
        for compute_start, compute_end in state["compute"]:
            if cp.cuda.get_elapsed_time(compute_start, compute_end) <= 0.0:
                continue
            for copy_start, copy_end in state["copy"]:
                if cp.cuda.get_elapsed_time(copy_start, copy_end) <= 0.0:
                    continue
                if cp.cuda.get_elapsed_time(copy_start, compute_end) > 0.0 and cp.cuda.get_elapsed_time(compute_start, copy_end) > 0.0:
                    return True
        return False

    try:
        backend.setup(setup)
        ref_backend.setup(setup)
        for run_idx, direction in enumerate(order):
            rng = np.random.default_rng(50_000 + 1_000 * ring_buffer_size + 100 * runtime_k + 10 * run_idx + (0 if direction == "up" else 1))
            primary = binary_pm1(rng, (setup.num_samples, int(runtime_k)), DATA_DTYPE)
            expected = _run_stream_direction(ref_backend, direction, primary)
            actual = _run_stream_direction(backend, direction, primary)
            np.testing.assert_array_equal(actual, expected)
            cp.cuda.runtime.deviceSynchronize()
            if run_idx == 0:
                if int(ring_buffer_size) == 1:
                    assert not _has_overlap()
                else:
                    assert _has_overlap()
                state["active"] = False
    finally:
        del ref_backend
        del backend
        del setup
        clear_gpu_state()


def test_cusparse_accepts_raw_null_stream():
    backend = CusparseBackend(
        device=0,
        stream=0,
        pair=CusparsePlanPair.from_dicts(make_cusparse_plan(k_hint=None, store="N", fmt="CSR", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"), None),
        ring_buffer_size=2,
        log_level="WARNING",
    )
    try:
        assert backend._caller_stream_ptr == 0
    finally:
        del backend


def test_cusparse_accepts_protocol_null_stream():
    backend = CusparseBackend(
        device=0,
        stream=cp.cuda.Stream.null,
        pair=CusparsePlanPair.from_dicts(make_cusparse_plan(k_hint=None, store="N", fmt="CSR", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"), None),
        ring_buffer_size=2,
        log_level="WARNING",
    )
    try:
        assert backend._caller_stream_ptr == int(cp.cuda.Stream.null.ptr)
        assert backend._root_stream.ptr != backend._caller_stream_ptr
    finally:
        del backend


def test_cusparse_rejects_invalid_stream():
    with pytest.raises(TypeError, match="__cuda_stream__"):
        CusparseBackend(
            device=0,
            stream=object(),
            pair=CusparsePlanPair.from_dicts(make_cusparse_plan(k_hint=None, store="N", fmt="CSR", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"), None),
            ring_buffer_size=2,
            log_level="WARNING",
        )


def test_cusparse_rejects_foreign_device_stream():
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires >=2 CUDA devices")
    with cp.cuda.Device(1):
        master = cp.cuda.Stream(non_blocking=True)
    with pytest.raises(ValueError, match="requested CUDA device 0"):
        CusparseBackend(
            device=0,
            stream=master,
            pair=CusparsePlanPair.from_dicts(make_cusparse_plan(k_hint=None, store="N", fmt="CSR", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"), None),
            ring_buffer_size=2,
            log_level="WARNING",
        )


@pytest.mark.parametrize(
    "plan_up",
    [
        pytest.param(make_cusparse_plan(k_hint=None, store="N", fmt="CSR", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"), id="direct-row"),
        pytest.param(make_cusparse_plan(k_hint=None, store="N", fmt="CSR", op_a="N", op_b="T", order_b="COL", order_c="ROW", algo="DEFAULT"), id="reinterpret"),
        pytest.param(make_cusparse_plan(k_hint=None, store="N", fmt="CSR", op_a="N", op_b="N", order_b="COL", order_c="ROW", algo="DEFAULT"), id="repack"),
    ],
)
def test_forward_explicit_dense_view_plans(primary_grg_path, gt_small, plan_up):
    op = SpmvGRG(
        primary_grg_path,
        CusparseBackend(device=0, stream=0, pair=CusparsePlanPair.from_dicts(plan_up, None), ring_buffer_size=2, log_level="WARNING"),
        DATA_DTYPE,
        artifact_dir=_CACHE_DIR,
    )
    X, Y_expected = gt_small.get("forward", 4, seed=841, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(op.matmul(X.T, "up").T, Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "plan_down",
    [
        pytest.param(make_cusparse_plan(k_hint=None, store="T", fmt="CSC", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"), id="direct-row"),
        pytest.param(make_cusparse_plan(k_hint=None, store="T", fmt="CSC", op_a="N", op_b="T", order_b="COL", order_c="ROW", algo="DEFAULT"), id="reinterpret"),
        pytest.param(make_cusparse_plan(k_hint=None, store="T", fmt="CSC", op_a="N", op_b="N", order_b="COL", order_c="ROW", algo="DEFAULT"), id="repack"),
    ],
)
def test_backward_explicit_dense_view_plans(primary_grg_path, gt_small, plan_down):
    op = SpmvGRG(
        primary_grg_path,
        CusparseBackend(device=0, stream=0, pair=CusparsePlanPair.from_dicts(None, plan_down), ring_buffer_size=2, log_level="WARNING"),
        DATA_DTYPE,
        artifact_dir=_CACHE_DIR,
    )
    X, Y_expected = gt_small.get("backward", 4, seed=842, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(op.matmul(X.T, "down").T, Y_expected, atol=atol, rtol=rtol)


class TestSmokeCoreConfigMatrix:
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "cfg,k",
        [
            pytest.param(
                {"fmt_up": "csr", "fmt_down": None, "algo_up": "default", "algo_down": "default", "k_hint": 1},
                1,
                id="up-csr-down-auto-default-k1",
            ),
            pytest.param(
                {"fmt_up": None, "fmt_down": "csc", "algo_up": "default", "algo_down": "default", "k_hint": 4},
                4,
                id="up-auto-down-csc-default-k4",
            ),
            pytest.param(
                {"fmt_up": "coo", "fmt_down": "coo", "algo_up": "coo_alg1", "algo_down": "coo_alg2", "k_hint": None},
                2,
                id="up-coo-down-coo-mixed-dyn",
            ),
        ],
    )
    def test_forward(self, primary_grg_path, gt_small, cfg, k):
        op = _make_op(primary_grg_path, **cfg)
        X, Y_expected = gt_small.get("forward", k, seed=7201, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "cfg,k",
        [
            pytest.param(
                {"fmt_up": "csr", "fmt_down": None, "algo_up": "default", "algo_down": "default", "k_hint": 1},
                1,
                id="up-csr-down-auto-default-k1",
            ),
            pytest.param(
                {"fmt_up": None, "fmt_down": "csc", "algo_up": "default", "algo_down": "default", "k_hint": 4},
                4,
                id="up-auto-down-csc-default-k4",
            ),
            pytest.param(
                {"fmt_up": "coo", "fmt_down": "coo", "algo_up": "coo_alg1", "algo_down": "coo_alg2", "k_hint": None},
                2,
                id="up-coo-down-coo-mixed-dyn",
            ),
        ],
    )
    def test_backward(self, primary_grg_path, gt_small, cfg, k):
        op = _make_op(primary_grg_path, **cfg)
        X, Y_expected = gt_small.get("backward", k, seed=7202, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt,algo", _graph_fmt_algo_params())
def test_construct_for_valid_fmt_algo(primary_grg_path, fmt, algo):
    _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=4, algo_up=algo, algo_down=algo)


@pytest.mark.parametrize("fmt,algo", invalid_fmt_algo_params())
def test_construct_rejects_invalid_fmt_algo(primary_grg_path, fmt, algo):
    with pytest.raises(ValueError, match="Unsupported cuSPARSE"):
        _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=4, algo_up=algo, algo_down=algo)


def test_csr_alg3_transpose_validity_matrix():
    assert is_valid_combo("csr", False, "csr_alg3")
    assert not is_valid_combo("csc", False, "csr_alg3")
    assert not is_valid_combo("csr", True, "csr_alg3")
    assert not is_valid_combo("csc", True, "csr_alg3")
    assert not is_valid_combo("coo", False, "csr_alg3")
    assert not is_valid_combo("coo", True, "csr_alg3")


def test_csr_alg3_rejects_csc_non_transpose_path(primary_grg_path):
    with pytest.raises(ValueError, match="Unsupported cuSPARSE UP plan"):
        _make_op(primary_grg_path, fmt_up="csc", fmt_down="csc", k_hint=None, algo_up="csr_alg3", algo_down="default")


def test_csr_alg3_rejects_csc_transpose_path(primary_grg_path):
    with pytest.raises(ValueError, match="Unsupported cuSPARSE UP plan"):
        _make_op(primary_grg_path, fmt_up=None, fmt_down="csc", k_hint=None, algo_up="csr_alg3", algo_down="default")


def test_csr_alg3_rejects_csr_transpose_path(primary_grg_path):
    with pytest.raises(ValueError, match="Unsupported cuSPARSE DOWN plan"):
        SpmvGRG(
            primary_grg_path,
            CusparseBackend(
                device=0,
                stream=0,
                pair=CusparsePlanPair.from_dicts(
                    make_cusparse_plan(
                        k_hint=None,
                        store="N",
                        fmt="CSR",
                        op_a="N",
                        op_b="N",
                        order_b="ROW",
                        order_c="ROW",
                        algo="DEFAULT",
                    ),
                    make_cusparse_plan(
                        k_hint=None,
                        store="N",
                        fmt="CSR",
                        op_a="T",
                        op_b="N",
                        order_b="ROW",
                        order_c="ROW",
                        algo="CSR_ALG3",
                    ),
                ),
                ring_buffer_size=2,
                log_level="WARNING",
            ),
            DATA_DTYPE,
            artifact_dir=_CACHE_DIR,
        )


def test_fmt_inference_from_down(primary_grg_path):
    _make_op(primary_grg_path, fmt_up=None, fmt_down="csc", k_hint=4)


def test_both_fmts_none_rejected(primary_grg_path):
    with pytest.raises(ValueError, match="fmt_up/fmt_down"):
        _make_op(primary_grg_path, fmt_up=None, fmt_down=None)


def test_non_string_algo_rejected(primary_grg_path):
    with pytest.raises(TypeError, match="algo_up"):
        _make_op(primary_grg_path, algo_up=123, algo_down="default")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="algo_down"):
        _make_op(primary_grg_path, algo_up="default", algo_down=123)  # type: ignore[arg-type]


@pytest.mark.parametrize("fmt,algo", valid_fmt_algo_params())
def test_forward_dynamic(primary_grg_path, gt_small, fmt, algo):
    op = _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=None, algo_up=algo, algo_down=algo)
    X, Y_expected = gt_small.get("forward", 4, seed=42, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt,algo", valid_fmt_algo_params())
def test_backward_dynamic(primary_grg_path, gt_small, fmt, algo):
    op = _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=None, algo_up=algo, algo_down=algo)
    X, Y_expected = gt_small.get("backward", 4, seed=42, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt,algo", _graph_fmt_algo_params())
def test_forward_graph(primary_grg_path, gt_small, fmt, algo):
    op = _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=4, algo_up=algo, algo_down=algo)
    X, Y_expected = gt_small.get("forward", 4, seed=52, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt,algo", _graph_fmt_algo_params())
def test_backward_graph(primary_grg_path, gt_small, fmt, algo):
    op = _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=4, algo_up=algo, algo_down=algo)
    X, Y_expected = gt_small.get("backward", 4, seed=52, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt,algo,k", _graph_fmt_algo_k_params())
def test_k_sweep_graph(primary_grg_path, gt_small, fmt, algo, k):
    op = _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=k, algo_up=algo, algo_down=algo)
    Xf, Yf = gt_small.get("forward", k, seed=62, dtype=DATA_DTYPE)
    Xb, Yb = gt_small.get("backward", k, seed=63, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, Xf), Yf, atol=atol, rtol=rtol)
    np.testing.assert_allclose(_run_down(op, Xb), Yb, atol=atol, rtol=rtol)


def test_hint_none_keeps_graphs_disabled(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=None)
    X, _ = gt_small.get("forward", 4, seed=5000, dtype=DATA_DTYPE)
    _ = _run_up(op, X)
    ws = op._backend._workspaces.dynamic_up
    assert ws is not None
    assert op._backend._workspaces.graph_up is None
    assert op._backend._workspaces.graph_down is None
    assert ws.k == 4
    assert ws.graph is None
    assert op._backend._workspaces.dynamic_down is None


def test_graph_workspaces_built_during_setup(primary_grg_path):
    op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=4)
    backend = op._backend
    assert backend._workspaces.graph_up is not None
    assert backend._workspaces.graph_down is not None
    assert backend._workspaces.graph_up.graph is not None
    assert backend._workspaces.graph_down.graph is not None
    assert backend._workspaces.dynamic_up is None
    assert backend._workspaces.dynamic_down is None
    assert op.memory.retained is not None
    assert any(
        row.path[:4] == ("cuda_live", "retained", "up", "k=4") and row.retention == "captured"
        for row in tree_rows(op.memory.retained)
    )


def test_custom_master_stream_keeps_private_cusparse_root(primary_grg_path, gt_small):
    with cp.cuda.Device(0):
        master = cp.cuda.Stream(non_blocking=True)
    op = _make_op(primary_grg_path, stream=master, fmt_up="csr", fmt_down="csc", k_hint=4)
    backend = op._backend
    x_up, y_up = gt_small.get("forward", 4, seed=5113, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, x_up), y_up, atol=atol, rtol=rtol)
    assert backend._caller_stream_ptr == int(master.ptr)
    assert backend._root_stream.ptr != int(master.ptr)
    assert backend._workspaces.graph_up is not None
    assert backend._workspaces.graph_up.graph is not None


def test_cusparse_setup_and_run_stay_on_declared_device_after_device_switch(primary_grg_path, gt_small):
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires >=2 CUDA devices")
    with cp.cuda.Device(0):
        backend = make_cusparse_backend(device=0, fmt_up="csr", fmt_down="csc", k_hint=1)
    with cp.cuda.Device(1):
        op = SpmvGRG(
            primary_grg_path,
            backend,
            DATA_DTYPE,
            artifact_dir=_CACHE_DIR,
        )
        x_up, y_up = gt_small.get("forward", 1, seed=5114, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(_run_up(op, x_up), y_up, atol=atol, rtol=rtol)
    assert backend._alpha is not None
    assert int(backend._alpha.device.id) == 0
    ws = backend._workspaces.graph_up
    assert ws is not None
    assert int(ws.input_primary.device.id) == 0
    assert int(ws.dense.level_bufs[0].device.id) == 0


def test_cusparse_setup_gpu_allocations_run_inside_caller_root_scope(primary_grg_path, monkeypatch):
    backend = make_cusparse_backend(device=0, fmt_up="csr", fmt_down="csc", k_hint=4)
    depth = _install_scope_depth_probe(monkeypatch, backend)
    seen = {"blocks": 0, "workspace": 0}
    original_build_blocks = type(backend)._build_direction_host_blocks
    original_alloc_workspace = type(backend)._alloc_workspace

    def _wrapped_build_blocks(self, *, direction):
        assert depth["value"] > 0
        seen["blocks"] += 1
        return original_build_blocks(self, direction=direction)

    def _wrapped_alloc_workspace(self, direction, k):
        assert depth["value"] > 0
        seen["workspace"] += 1
        return original_alloc_workspace(self, direction, k)

    monkeypatch.setattr(type(backend), "_build_direction_host_blocks", _wrapped_build_blocks)
    monkeypatch.setattr(type(backend), "_alloc_workspace", _wrapped_alloc_workspace)

    op = SpmvGRG(
        primary_grg_path,
        backend,
        DATA_DTYPE,
        artifact_dir=_CACHE_DIR,
    )

    assert seen["blocks"] >= 2
    assert seen["workspace"] >= 2
    del op


def test_cusparse_run_direction_restores_caller_root_scope_on_missing_graph(primary_grg_path, monkeypatch):
    op = _make_op(primary_grg_path, fmt_up="csr", fmt_down=None, k_hint=4)
    backend = op._backend
    ws = backend._workspaces.graph_up
    assert ws is not None
    counts = _install_bridge_probe(monkeypatch, backend)
    ws.graph = None

    with pytest.raises(RuntimeError, match="Missing captured UP CUDA graph"):
        backend._run_direction(
            Direction.UP,
            np.ones((op.num_samples, 4), dtype=DATA_DTYPE),
            miss=None,
            init_mode=InitMode.NONE,
            init=None,
            need_miss_output=False,
            emit_all_nodes=False,
        )

    assert counts == {"entered": 1, "exited": 1}


def test_cusparse_dynamic_workspace_allocates_inside_caller_root_scope(primary_grg_path, gt_small, monkeypatch):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=None)
    backend = op._backend
    depth = _install_scope_depth_probe(monkeypatch, backend)
    calls = {"count": 0}
    original = type(backend)._alloc_workspace

    def _wrapped(self, direction, k):
        calls["count"] += 1
        assert depth["value"] > 0
        return original(self, direction, k)

    monkeypatch.setattr(type(backend), "_alloc_workspace", _wrapped)

    X, _ = gt_small.get("forward", 1, seed=5006, dtype=DATA_DTYPE)
    _ = _run_up(op, X)

    assert calls == {"count": 1}


def test_cusparse_build_wavefront_graph_restores_caller_root_scope_on_capture_failure(primary_grg_path, monkeypatch):
    op = _make_op(primary_grg_path, fmt_up="csr", fmt_down=None, k_hint=4)
    backend = op._backend
    with backend._root_stream:
        ws = backend._alloc_workspace(Direction.UP, 4)
    counts = _install_bridge_probe(monkeypatch, backend)
    original = backend._enqueue_wavefront_dispatch
    calls = {"count": 0}

    def _boom(ws):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("capture enqueue failed")
        return original(ws)

    monkeypatch.setattr(backend, "_enqueue_wavefront_dispatch", _boom)

    with pytest.raises(RuntimeError, match="capture enqueue failed"):
        backend._build_wavefront_graph(ws)

    assert counts == {"entered": 1, "exited": 1}
    ws.destroy(cslib=backend._cslib)


def test_cusparse_capture_failure_is_hard_error(primary_grg_path, monkeypatch):
    def _boom(self, ws):
        raise RuntimeError(f"capture failed for {ws.direction.value}")

    monkeypatch.setattr(CusparseBackend, "_build_wavefront_graph", _boom)
    with pytest.raises(RuntimeError, match=r"cuSPARSE graph capture failed dir=up .*capture failed for up"):
        _make_op(
            primary_grg_path,
            fmt_up="csr",
            fmt_down=None,
            k_hint=4,
            algo_up="csr_alg3",
            infer_missing=False,
        )


def test_cusparse_dynamic_runtime_never_calls_preprocess(primary_grg_path, monkeypatch):
    backend = make_cusparse_backend(device=0, fmt_up="csr", fmt_down=None, k_hint=None)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(backend._cslib, "spmm_preprocess", lambda *args: calls.append(args))
    op = SpmvGRG(
        primary_grg_path,
        backend,
        DATA_DTYPE,
        artifact_dir=_CACHE_DIR,
    )

    X = np.ones((op.num_samples, 1), dtype=DATA_DTYPE)
    _ = _run_up(op, X)

    assert calls == []


def test_cusparse_graph_setup_and_runtime_never_call_preprocess(primary_grg_path, monkeypatch):
    backend = make_cusparse_backend(device=0, fmt_up="csr", fmt_down=None, k_hint=4)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(backend._cslib, "spmm_preprocess", lambda *args: calls.append(args))
    op = SpmvGRG(
        primary_grg_path,
        backend,
        DATA_DTYPE,
        artifact_dir=_CACHE_DIR,
    )

    X = np.ones((op.num_samples, 4), dtype=DATA_DTYPE)
    _ = _run_up(op, X)

    assert calls == []


def test_cusparse_graph_workspace_memory_matches_dynamic_workspace(primary_grg_path):
    from pygrgl_spmv.backends.cusparse.backend import _workspace_nbytes

    op = _make_op(primary_grg_path, fmt_up="csr", fmt_down=None, k_hint=4)
    backend = op._backend
    ws_graph = backend._workspaces.graph_up
    assert ws_graph is not None
    assert len(ws_graph.slot_ext) == backend._ring_buffer_size

    with backend._caller_root_scope():
        with backend._root_stream:
            ws_dynamic = backend._alloc_workspace(Direction.UP, 4)
    try:
        assert len(ws_dynamic.slot_ext) == backend._ring_buffer_size
        assert _workspace_nbytes(ws_graph) == _workspace_nbytes(ws_dynamic)
    finally:
        ws_dynamic.destroy(cslib=backend._cslib)


def test_graph_workspace_keeps_optional_buffers_lazy(primary_grg_path):
    op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=1)
    backend = op._backend
    ws_up = backend._workspaces.graph_up
    ws_down = backend._workspaces.graph_down
    assert ws_up is not None and ws_down is not None
    before_up = id(ws_up)
    before_down = id(ws_down)
    assert backend._staging_up_by_k == {}

    X = np.ones((op.num_samples, 1), dtype=DATA_DTYPE)
    _ = op.matmul(X.T, "up", emit_all_nodes=True, init="xtx").T

    assert 1 in backend._staging_up_by_k
    assert backend._staging_up_by_k[1].xtx_bias is not None
    assert id(backend._workspaces.graph_up) == before_up
    assert id(backend._workspaces.graph_down) == before_down


def test_dynamic_workspace_reused_for_same_k(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4)
    X1, _ = gt_small.get("forward", 1, seed=5006, dtype=DATA_DTYPE)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _ = _run_up(op, X1)
    ws1 = op._backend._workspaces.dynamic_up
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _ = _run_up(op, X1)
    ws2 = op._backend._workspaces.dynamic_up
    assert ws1 is ws2
    assert ws2 is not None
    assert ws2.k == 1


def test_dynamic_workspace_recreated_when_k_changes(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4)
    X1, _ = gt_small.get("forward", 1, seed=5007, dtype=DATA_DTYPE)
    X2, _ = gt_small.get("forward", 2, seed=5008, dtype=DATA_DTYPE)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _ = _run_up(op, X1)
    ws1 = op._backend._workspaces.dynamic_up
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _ = _run_up(op, X2)
    ws2 = op._backend._workspaces.dynamic_up
    assert ws1 is not ws2
    assert ws2 is not None
    assert ws2.k == 2


def test_dynamic_k_mismatch_preserves_static_graph(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4)
    X4, Y4_exp = gt_small.get("backward", 4, seed=5001, dtype=DATA_DTYPE)
    X1, Y1_exp = gt_small.get("backward", 1, seed=5002, dtype=DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X4), Y4_exp, atol=1e-5, rtol=1e-5)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        np.testing.assert_allclose(_run_down(op, X1), Y1_exp, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_run_down(op, X4), Y4_exp, atol=1e-5, rtol=1e-5)
    assert op._backend._workspaces.graph_down is not None
    assert op._backend._workspaces.graph_down.graph is not None


def test_debug_log_level_keeps_graph_mode(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4, log_level="DEBUG")
    x_up, _ = gt_small.get("forward", 4, seed=5003, dtype=DATA_DTYPE)
    x_down, _ = gt_small.get("backward", 4, seed=5004, dtype=DATA_DTYPE)
    _ = _run_up(op, x_up)
    assert op.memory.last_call is not None
    assert op.memory.last_call.meta["mode"] == "graph"
    assert op.memory.last_call.active_alloc_keys
    merged = live_snapshot(op.memory.retained, op.memory.last_call)
    assert merged is not None
    assert any(
        row.owner == "backend"
        and row.direction == "up"
        and row.slot_k == 4
        and row.retention in {"captured", "staging"}
        and row.activity == "yes"
        for row in merged.allocations
    )
    _ = _run_down(op, x_down)
    assert op.memory.last_call is not None
    assert op.memory.last_call.meta["mode"] == "graph"
    assert op._backend._workspaces.graph_up is not None
    assert op._backend._workspaces.graph_down is not None
    assert op._backend._workspaces.graph_up.graph is not None
    assert op._backend._workspaces.graph_down.graph is not None


def test_debug_log_level_reports_block_memory(primary_grg_path, caplog):
    with caplog.at_level(logging.DEBUG):
        op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=1, log_level="DEBUG")
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("cuSPARSE host_blocks_up" in msg and "rows=" in msg and "nnz=" in msg and "struct0_bytes=" in msg for msg in messages)
    assert any("cuSPARSE host_blocks_up dir=up" in msg and "cols=" in msg and "nnz=" in msg for msg in messages)
    del op


def test_cusparse_setup_streams_materialized_blocks_before_slot_pool_alloc(monkeypatch):
    refs: list[weakref.ReferenceType[sp.spmatrix]] = []
    materialized = {"count": 0, "checked": 0}
    backend = make_cusparse_backend(device=0, fmt_up="csc", fmt_down="csc", k_hint=None, infer_missing=False)
    original_materialize = type(backend)._materialize_stored_block
    original_alloc_slot_pool = type(backend)._alloc_slot_pool

    def _wrapped_materialize(self, direction, *, dst_level, src_level):
        if refs:
            gc.collect()
            materialized["checked"] += 1
            assert all(ref() is None for ref in refs)
            refs.clear()
        sparse = original_materialize(self, direction, dst_level=dst_level, src_level=src_level)
        if sparse is not None:
            materialized["count"] += 1
            refs.append(weakref.ref(sparse))
        return sparse

    def _wrapped_alloc_slot_pool(self):
        gc.collect()
        assert refs
        assert all(ref() is None for ref in refs)
        return original_alloc_slot_pool(self)

    monkeypatch.setattr(type(backend), "_materialize_stored_block", _wrapped_materialize)
    monkeypatch.setattr(type(backend), "_alloc_slot_pool", _wrapped_alloc_slot_pool)

    backend.setup(_synthetic_setup())
    assert materialized["count"] >= 3
    assert materialized["checked"] >= 2


def test_cusparse_info_log_level_reports_setup_rss_checkpoints(monkeypatch, caplog):
    import pygrgl_spmv._rss as rss_mod

    values = iter([100, 140, 120, 160])
    monkeypatch.setattr(rss_mod, "rss_bytes", lambda: next(values))

    backend = make_cusparse_backend(device=0, fmt_up="csc", fmt_down="csc", k_hint=None, infer_missing=False, log_level="INFO")
    with caplog.at_level(logging.INFO, logger="pygrgl_spmv.backends.cusparse.backend.CusparseBackend"):
        backend.setup(_synthetic_setup())

    messages = [
        rec.getMessage()
        for rec in caplog.records
        if rec.name == "pygrgl_spmv.backends.cusparse.backend.CusparseBackend" and rec.getMessage().startswith("rss setup:")
    ]
    assert [msg.split()[1] for msg in messages] == [
        "setup:start",
        "setup:host_blocks_up_ready",
        "setup:host_blocks_down_ready",
        "setup:complete",
    ]
    assert "delta_bytes=" not in messages[0]
    assert all("delta_bytes=" in msg for msg in messages[1:])
    assert "stored_blocks=" in messages[1] and "struct0_bytes=" in messages[1] and "struct1_bytes=" in messages[1]
    assert "stored_blocks=" in messages[2] and "struct0_bytes=" in messages[2] and "struct1_bytes=" in messages[2]


def test_instrumentation_disables_graph_mode(primary_grg_path, gt_small):
    with pytest.warns(RuntimeWarning, match="instrumentation=True"):
        op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4, instrumentation=True)
    x_up, y_up = gt_small.get("forward", 4, seed=5005, dtype=DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, x_up), y_up, atol=1e-5, rtol=1e-5)
    assert op.memory.last_call is not None
    assert op.memory.last_call.meta["mode"] == "instrumented"
    assert op._backend._workspaces.dynamic_up is not None
    assert op._backend._workspaces.graph_up is None
    assert op._backend._workspaces.graph_down is None
    assert op.memory.retained is not None
    assert not any(row.retention == "captured" for row in tree_rows(op.memory.retained))


def test_transpose_compatible_storage_aliases_payload_but_not_descriptors(primary_grg_path):
    up = make_cusparse_plan(
        k_hint=None,
        store="N",
        fmt="CSR",
        op_a="N",
        op_b="N",
        order_b="ROW",
        order_c="ROW",
        algo="DEFAULT",
    )
    down = make_cusparse_plan(
        k_hint=None,
        store="T",
        fmt="CSC",
        op_a="N",
        op_b="N",
        order_b="ROW",
        order_c="ROW",
        algo="DEFAULT",
    )
    op = SpmvGRG(
        primary_grg_path,
        CusparseBackend(device=0, stream=0, pair=CusparsePlanPair.from_dicts(up, down), ring_buffer_size=2, log_level="WARNING"),
        DATA_DTYPE,
        artifact_dir=_CACHE_DIR,
    )
    backend = op._backend
    H = len(backend._level_offsets) - 1
    for dst_level in range(H):
        for row_index, down_block in enumerate(backend._host_blocks_down[dst_level]):
            if down_block is None:
                continue
            src_level = row_index + dst_level + 1
            up_block = backend._host_blocks_up[src_level][dst_level]
            assert up_block is not None
            assert np.shares_memory(down_block.struct_buffers[0], up_block.struct_buffers[0])
            assert np.shares_memory(down_block.struct_buffers[1], up_block.struct_buffers[1])
            return
    raise AssertionError("expected at least one shared cuSPARSE block alias")


def test_cusparse_slot_pool_allocated_for_requested_ring_size(primary_grg_path):
    op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=None)
    backend = op._backend
    assert backend._slot_pool is not None
    assert len(backend._slot_pool.slots) == 2


def test_cusparse_coo_slot_pool_allocated_for_requested_ring_size(primary_grg_path):
    op = _make_op(primary_grg_path, fmt_up="coo", fmt_down="coo", k_hint=None, algo_up="coo_alg1", algo_down="coo_alg2")
    backend = op._backend
    assert backend._slot_pool is not None
    assert len(backend._slot_pool.slots) == 2


def test_cusparse_shared_ones_use_materialized_array_when_vmm_unsupported(primary_grg_path, monkeypatch, caplog):
    import pygrgl_spmv.backends.cusparse.backend as cusparse_backend_mod

    monkeypatch.setattr(cusparse_backend_mod.CudaVmmDriver, "current_context", lambda self: 1)
    monkeypatch.setattr(cusparse_backend_mod.CudaVmmDriver, "vmm_supported", lambda self, device_id: False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.INFO):
            op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=None, log_level="INFO")
    backend = op._backend
    shared = backend._shared_ones
    assert shared is not None
    assert shared.vmm is False
    assert shared.physical_nbytes == shared.logical_nbytes
    assert caught == []
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("cuSPARSE shared ones: mode=materialized reason=vmm-unsupported" in msg for msg in messages)


def test_cusparse_shared_ones_log_materialized_when_vmm_saves_no_memory(primary_grg_path, monkeypatch, caplog):
    import pygrgl_spmv.backends.cusparse.backend as cusparse_backend_mod

    monkeypatch.setattr(cusparse_backend_mod.CudaVmmDriver, "current_context", lambda self: 1)
    monkeypatch.setattr(cusparse_backend_mod.CudaVmmDriver, "vmm_supported", lambda self, device_id: True)
    monkeypatch.setattr(
        cusparse_backend_mod.CudaVmmDriver,
        "allocation_granularity",
        lambda self, device_id, *, recommended: 1 << 30 if not recommended else 1 << 31,
    )
    monkeypatch.setattr(
        cusparse_backend_mod.CusparseBackend,
        "_build_vmm_shared_ones",
        lambda self, **kwargs: (_ for _ in ()).throw(AssertionError("VMM path should not be used")),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.INFO):
            op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=None, log_level="INFO")
    shared = op._backend._shared_ones
    assert shared is not None
    assert shared.vmm is False
    assert caught == []
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("cuSPARSE shared ones: mode=materialized reason=no-memory-savings" in msg for msg in messages)


def test_cusparse_shared_ones_log_vmm_when_physical_bytes_drop(primary_grg_path, monkeypatch, caplog):
    import pygrgl_spmv.backends.cusparse.backend as cusparse_backend_mod

    original_max_block_nnz = cusparse_backend_mod.CusparseBackend._max_block_nnz

    def fake_build_vmm_shared_ones(self, *, driver, device_id, logical_nbytes, tile_nbytes):
        arr = self._cp.ones((logical_nbytes // int(self._dtype.itemsize),), dtype=self._dtype)
        return cusparse_backend_mod._SharedOnes(
            ptr=int(arr.data.ptr),
            logical_nbytes=logical_nbytes,
            physical_nbytes=tile_nbytes,
            vmm=True,
            _materialized=arr,
            _reserved_nbytes=cusparse_backend_mod._round_up(logical_nbytes, tile_nbytes),
        )

    monkeypatch.setattr(cusparse_backend_mod.CusparseBackend, "_max_block_nnz", lambda self: max(original_max_block_nnz(self), 16))
    monkeypatch.setattr(cusparse_backend_mod.CudaVmmDriver, "current_context", lambda self: 1)
    monkeypatch.setattr(cusparse_backend_mod.CudaVmmDriver, "vmm_supported", lambda self, device_id: True)
    monkeypatch.setattr(
        cusparse_backend_mod.CudaVmmDriver,
        "allocation_granularity",
        lambda self, device_id, *, recommended: 64 if not recommended else 128,
    )
    monkeypatch.setattr(cusparse_backend_mod.CusparseBackend, "_build_vmm_shared_ones", fake_build_vmm_shared_ones)
    with caplog.at_level(logging.INFO):
        op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=None, log_level="INFO")
    shared = op._backend._shared_ones
    assert shared is not None
    assert shared.vmm is True
    messages = [rec.getMessage() for rec in caplog.records]
    assert any(
        "cuSPARSE shared ones: mode=vmm reason=physical-bytes-reduced" in msg and "reserve_alignment=64" in msg
        for msg in messages
    )


def test_cusparse_shared_ones_warn_when_vmm_build_fails(primary_grg_path, monkeypatch, caplog):
    import pygrgl_spmv.backends.cusparse.backend as cusparse_backend_mod

    original_max_block_nnz = cusparse_backend_mod.CusparseBackend._max_block_nnz

    monkeypatch.setattr(cusparse_backend_mod.CusparseBackend, "_max_block_nnz", lambda self: max(original_max_block_nnz(self), 16))
    monkeypatch.setattr(cusparse_backend_mod.CudaVmmDriver, "current_context", lambda self: 1)
    monkeypatch.setattr(cusparse_backend_mod.CudaVmmDriver, "vmm_supported", lambda self, device_id: True)
    monkeypatch.setattr(
        cusparse_backend_mod.CudaVmmDriver,
        "allocation_granularity",
        lambda self, device_id, *, recommended: 64 if not recommended else 128,
    )
    monkeypatch.setattr(
        cusparse_backend_mod.CusparseBackend,
        "_build_vmm_shared_ones",
        lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with caplog.at_level(logging.INFO):
        with pytest.warns(RuntimeWarning, match="cuSPARSE shared ones falling back to one materialized all-ones array: boom"):
            op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=None, log_level="INFO")
    shared = op._backend._shared_ones
    assert shared is not None
    assert shared.vmm is False
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("cuSPARSE shared ones: mode=materialized reason=vmm-build-failed" in msg for msg in messages)


def test_cusparse_vmm_driver_forwards_reserve_alignment(monkeypatch):
    import ctypes

    import pygrgl_spmv.backends.cusparse.ffi as cusparse_ffi

    class _FakeCudaDriverLib:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def cuInit(self, flags):
            return cusparse_ffi.CUDA_SUCCESS

        def cuMemAddressReserve(self, addr_ptr, size, alignment, addr, flags):
            self.calls.append((int(size.value), int(alignment.value)))
            ctypes.cast(addr_ptr, ctypes.POINTER(ctypes.c_ulonglong))[0] = 0x1234000
            return cusparse_ffi.CUDA_SUCCESS

    fake = _FakeCudaDriverLib()
    monkeypatch.setattr(cusparse_ffi, "_load_cuda_driver_library", lambda: fake)

    driver = cusparse_ffi.CudaVmmDriver()
    addr = driver.address_reserve(4096, alignment_bytes=2097152)

    assert addr == 0x1234000
    assert fake.calls == [(4096, 2097152)]


def test_setup_releases_host_sparse_payloads(primary_grg_path):
    op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=None)
    backend = op._backend
    assert backend._A_blocks == []
    assert backend._sel_mut.shape == (0, 0)
    assert backend._sel_miss.shape == (0, 0)


class _NvtxRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, object]]] = []

    @contextmanager
    def range(self, name: str, /, **fields: object):
        self.events.append(("range", name, dict(fields)))
        yield

    def mark(self, name: str, /, **fields: object) -> None:
        self.events.append(("mark", name, dict(fields)))


def test_instrumented_cusparse_emits_nvtx_markers(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=None, instrumentation=True)
    recorder = _NvtxRecorder()
    op._backend._nvtx = recorder
    x_up, _ = gt_small.get("forward", 2, seed=5009, dtype=DATA_DTYPE)
    _ = _run_up(op, x_up)
    names = [name for _, name, _ in recorder.events]
    assert "run_direction" in names
    assert "seed_direction" in names
    assert "wavefront" in names
    assert "level" in names
    assert "wait_ready" in names
    assert "launch" in names
    assert "publish_level" in names
    assert "event.record_ready" in names
    assert "join_ready" in names
    assert "collect_outputs" in names


def test_cusparse_scratch_buffers_allocated_for_enabled_levels(primary_grg_path):
    op = _make_op(
        primary_grg_path,
        fmt_up="csr",
        fmt_down="csc",
        k_hint=4,
        scratch_up="1",
        scratch_down="0",
    )
    backend = op._backend
    ws_up = backend._workspaces.graph_up
    ws_down = backend._workspaces.graph_down
    assert ws_up is not None and ws_down is not None
    assert len(ws_up.scratch_views_by_level[1]) == len(backend._wavefront_up[1]) > 0
    assert len(ws_up.scratch_dst_descs_by_level[1]) == len(backend._wavefront_up[1])
    assert len(ws_up.scratch_done_events_by_level[1]) == len(backend._wavefront_up[1])
    assert ws_up.scratch_views_by_level[0] == []
    assert len(ws_down.scratch_views_by_level[0]) == len(backend._wavefront_down[0]) > 0
    assert ws_down.scratch_views_by_level[1] == []


def test_instrumented_cusparse_scratch_emits_reduction_markers(primary_grg_path, gt_small):
    op = _make_op(
        primary_grg_path,
        fmt_up="csr",
        fmt_down=None,
        k_hint=None,
        scratch_up="1",
        instrumentation=True,
    )
    recorder = _NvtxRecorder()
    op._backend._nvtx = recorder
    x_up, _ = gt_small.get("forward", 2, seed=5010, dtype=DATA_DTYPE)
    _ = _run_up(op, x_up)
    names = [name for _, name, _ in recorder.events]
    assert "helper_launch" in names
    assert "event.record_scratch_done" in names
    assert "wait_scratch_done" in names
    assert "reduce_add" in names


def test_cusparse_scratch_enabled_up_matches_reference(primary_grg_path, gt_small):
    op = _make_op(
        primary_grg_path,
        fmt_up="csr",
        fmt_down=None,
        k_hint=None,
        scratch_up="1",
    )
    X, Y_expected = gt_small.get("forward", 4, seed=5011, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


def test_cusparse_scratch_enabled_down_matches_reference(primary_grg_path, gt_small):
    op = _make_op(
        primary_grg_path,
        fmt_up=None,
        fmt_down="csc",
        k_hint=None,
        scratch_down="0",
    )
    X, Y_expected = gt_small.get("backward", 4, seed=5012, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


def test_cusparse_graph_vs_dynamic_with_scratch(primary_grg_path):
    op_graph = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=4, scratch_up="1", scratch_down="0")
    op_dynamic = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=None, scratch_up="1", scratch_down="0")
    rng = np.random.default_rng(7010)
    V = rng.standard_normal((op_graph.num_samples, 4), dtype=DATA_DTYPE)
    W = rng.standard_normal((op_graph.num_mutations, 4), dtype=DATA_DTYPE)
    np.testing.assert_allclose(op_graph.matmul(V.T, "up").T, op_dynamic.matmul(V.T, "up").T, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(op_graph.matmul(W.T, "down").T, op_dynamic.matmul(W.T, "down").T, atol=1e-5, rtol=1e-5)


def test_graph_vs_dynamic(primary_grg_path):
    op_graph = _make_op(primary_grg_path, fmt_up="csr", k_hint=4)
    op_dynamic = _make_op(primary_grg_path, fmt_up="csr", k_hint=None)
    rng = np.random.default_rng(7001)
    V = rng.standard_normal((op_graph.num_samples, 4), dtype=DATA_DTYPE)
    W = rng.standard_normal((op_graph.num_mutations, 4), dtype=DATA_DTYPE)
    np.testing.assert_allclose(op_graph.matmul(V.T, "up").T, op_dynamic.matmul(V.T, "up").T, atol=1e-5)
    np.testing.assert_allclose(op_graph.matmul(W.T, "down").T, op_dynamic.matmul(W.T, "down").T, atol=1e-5)


def test_failure_modes(primary_grg_path, gt_small):
    for k, seed in [(1, 9001), (2, 9002), (3, 9003), (7, 9004)]:
        op = _make_op(primary_grg_path, fmt_up="csr", k_hint=k)
        X, Y_expected = gt_small.get("forward", k, seed=seed, dtype=DATA_DTYPE)
        np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=1e-5, rtol=1e-5)


def test_f_order_input(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=None)
    X, Y_expected = gt_small.get("forward", 4, seed=9005, dtype=DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, np.asfortranarray(X)), Y_expected, atol=1e-5, rtol=1e-5)


def test_none_hint_emits_no_warning(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=None)
    X, _ = gt_small.get("forward", 3, seed=9101, dtype=DATA_DTYPE)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _run_up(op, X)
    assert len(rec) == 0


def test_warn_every_mismatch_call(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4)
    X1, _ = gt_small.get("forward", 1, seed=9102, dtype=DATA_DTYPE)
    X2, _ = gt_small.get("forward", 2, seed=9103, dtype=DATA_DTYPE)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _run_up(op, X1)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _run_up(op, X1)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _run_up(op, X2)


def test_first_mismatch_call_matches_reference(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4, log_level="INFO")
    X, Y_expected = gt_small.get("forward", 1, seed=123, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        Y = _run_up(op, X)
    np.testing.assert_allclose(Y, Y_expected, atol=atol, rtol=rtol)


def test_repeated_mismatch_calls_match_reference(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4, log_level="INFO")
    X, Y_expected = gt_small.get("forward", 1, seed=123, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        Y_first = _run_up(op, X)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        Y_second = _run_up(op, X)
    np.testing.assert_allclose(Y_first, Y_expected, atol=atol, rtol=rtol)
    np.testing.assert_allclose(Y_second, Y_expected, atol=atol, rtol=rtol)


def test_runtime_logging_reports_versions_and_warns_for_non_doc_cuda(caplog, monkeypatch):
    import pygrgl_spmv.backends.cusparse as cusparse_backend

    cusparse_backend.cusparse_plan._runtime_cuda_version.cache_clear()
    monkeypatch.setattr(cusparse_backend.cusparse_plan, "_probe_cuda_version", lambda: (12, 8, 0))
    try:
        with caplog.at_level(logging.INFO):
            backend = cusparse_backend.CusparseBackend(
                device=0,
                stream=0,
                pair=cusparse_backend.CusparsePlanPair.from_dicts(
                    make_cusparse_plan(
                        k_hint=None,
                        store="N",
                        fmt="CSR",
                        op_a="N",
                        op_b="N",
                        order_b="ROW",
                        order_c="ROW",
                        algo="DEFAULT",
                    ),
                    None,
                ),
                ring_buffer_size=2,
                log_level="INFO",
            )
    finally:
        cusparse_backend.cusparse_plan._runtime_cuda_version.cache_clear()
    try:
        messages = [rec.getMessage() for rec in caplog.records]
        assert any("CUDA runtime=12.8.0" in msg and "cuSPARSE=" in msg for msg in messages)
        assert any("CUDA 12.9.0" in msg and "12.8.0" in msg for msg in messages)
    finally:
        del backend


def test_runtime_logging_skips_warning_for_doc_cuda(caplog, monkeypatch):
    import pygrgl_spmv.backends.cusparse as cusparse_backend

    cusparse_backend.cusparse_plan._runtime_cuda_version.cache_clear()
    monkeypatch.setattr(cusparse_backend.cusparse_plan, "_probe_cuda_version", lambda: (12, 9, 0))
    try:
        with caplog.at_level(logging.INFO):
            backend = cusparse_backend.CusparseBackend(
                device=0,
                stream=0,
                pair=cusparse_backend.CusparsePlanPair.from_dicts(
                    make_cusparse_plan(
                        k_hint=None,
                        store="N",
                        fmt="CSR",
                        op_a="N",
                        op_b="N",
                        order_b="ROW",
                        order_c="ROW",
                        algo="DEFAULT",
                    ),
                    None,
                ),
                ring_buffer_size=2,
                log_level="INFO",
            )
    finally:
        cusparse_backend.cusparse_plan._runtime_cuda_version.cache_clear()
    try:
        messages = [rec.getMessage() for rec in caplog.records]
        assert any("CUDA runtime=12.9.0" in msg and "cuSPARSE=" in msg for msg in messages)
        assert not any("CusparsePlan semantics are grounded in the CUDA 12.9.0" in msg for msg in messages)
    finally:
        del backend


def test_cusparse_buffer_size_warning_on_plan_disagreement(primary_grg_path, monkeypatch, caplog):
    op = _make_op(primary_grg_path, fmt_up="csr", fmt_down=None, k_hint=None)
    backend = op._backend
    monkeypatch.setattr(type(backend._plan_up), "need_buffer", property(lambda self: False))
    monkeypatch.setattr(backend._cslib, "spmm_buffer_size", lambda *args: 8)
    with caplog.at_level(logging.WARNING), backend._root_stream:
        ws = backend._alloc_workspace(Direction.UP, 1)
    try:
        messages = [rec.getMessage() for rec in caplog.records]
        assert any("bufferSize disagrees with plan.need_buffer" in msg for msg in messages)
    finally:
        ws.destroy(cslib=backend._cslib)


def test_cusparse_slot_ext_size_ignores_need_preprocess(primary_grg_path, monkeypatch):
    op = _make_op(primary_grg_path, fmt_up="csr", fmt_down=None, k_hint=None)
    backend = op._backend
    assert backend._plan_up is not None
    assert backend._plan_up.need_preprocess
    monkeypatch.setattr(backend._cslib, "spmm_buffer_size", lambda *args: 1)
    with backend._root_stream:
        ws = backend._alloc_workspace(Direction.UP, 1)
    try:
        assert len(ws.slot_ext) == backend._ring_buffer_size
        assert all(int(buf.nbytes) == 1 for buf in ws.slot_ext if buf is not None)
    finally:
        ws.destroy(cslib=backend._cslib)


def test_cusparse_runtime_loads_after_torch_import():
    pytest.importorskip("torch")
    from pygrgl_spmv.backends.cusparse.ffi import CuSparseLib

    lib = CuSparseLib()
    try:
        assert lib.version
    finally:
        lib.destroy()


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
def test_primary_grg_exact_binary_forward(primary_grg_path, gt_primary, fmt):
    op = _make_op(primary_grg_path, fmt_up=fmt, k_hint=None)
    X, Y_expected = gt_primary.get("forward", 4, seed=8001, dtype=DATA_DTYPE, input_fn=binary_pm1)
    np.testing.assert_array_equal(_run_up(op, X), Y_expected)


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
def test_primary_grg_exact_binary_backward(primary_grg_path, gt_primary, fmt):
    op = _make_op(primary_grg_path, fmt_up=fmt, k_hint=None)
    X, Y_expected = gt_primary.get("backward", 4, seed=8001, dtype=DATA_DTYPE, input_fn=binary_pm1)
    np.testing.assert_array_equal(_run_down(op, X), Y_expected)
