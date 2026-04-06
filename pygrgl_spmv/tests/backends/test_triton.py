"""Triton backend-specific tests."""

from __future__ import annotations

import gc
from contextlib import contextmanager
import logging
import numpy as np
import pytest
import scipy.sparse as sp
import weakref
import warnings

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.backends import BackendSetup, ReferenceBackend, ReferencePlanPair
from pygrgl_spmv.backends.triton import TritonBackend, TritonPlanPair
from pygrgl_spmv.backends import iter_direction_level_pairs
from pygrgl_spmv.backends.triton.backend import _AUTOTUNE_CACHE
from pygrgl_spmv.backends.triton.kernel import CscKernelConfig, CsrKernelConfig
from pygrgl_spmv.backends.types import Direction, InitMode
from pygrgl_spmv.grg.sparse import binary_csr_from_parts
from pygrgl_spmv.memory import live_snapshot
from pygrgl_spmv.tests.backends._streaming_stress import (
    LargeBandCase,
    build_large_band_setup,
    build_overlap_band_setup,
    clear_gpu_state,
    expected_down,
    expected_up,
    prepare_triton_large_band_case,
)
from pygrgl_spmv.tests.conftest import DATA_DTYPE, HAS_TRITON_RUNTIME, binary_pm1, make_triton_backend, make_triton_plan

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("Triton tests require CUDA", allow_module_level=True)

pytestmark = [pytest.mark.gpu, pytest.mark.triton]


class _ProtocolStream:
    def __init__(self, ptr: int) -> None:
        self._ptr = int(ptr)

    def __cuda_stream__(self) -> tuple[int, int]:
        return 0, self._ptr


def _make_op(
    grg_path,
    *,
    cache_dir,
    device=0,
    stream=0,
    fmt_up="csr",
    fmt_down="csc",
    k_hint=1,
    scratch_up="none",
    scratch_down="none",
    log_level="WARNING",
    instrumentation=False,
    infer_missing=True,
):
    return SpmvGRG(
        grg_path,
        make_triton_backend(
            device=device,
            stream=stream,
            fmt_up=fmt_up,
            fmt_down=fmt_down,
            k_hint=k_hint,
            scratch_up=scratch_up,
            scratch_down=scratch_down,
            log_level=log_level,
            instrumentation=instrumentation,
            infer_missing=infer_missing,
        ),
        DATA_DTYPE,
        artifact_dir=cache_dir,
    )


def _first_present_block(grid):
    for row in grid:
        for block in row:
            if block is not None:
                return block
    raise AssertionError("expected at least one non-empty Triton block")


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


@pytest.fixture(scope="module")
def large_band_case() -> LargeBandCase:
    return prepare_triton_large_band_case()


def _make_stream_backend(*, ring_buffer_size: int) -> object:
    return make_triton_backend(
        device=0,
        ring_buffer_size=int(ring_buffer_size),
        fmt_up="csr",
        fmt_down="csc",
        k_hint=None,
        infer_missing=False,
        log_level="WARNING",
    )


def _force_int64_slot_dtypes(monkeypatch, backend) -> None:
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


def test_triton_accepts_raw_null_stream():
    backend = TritonBackend(
        device=0,
        stream=0,
        pair=TritonPlanPair.from_dicts(make_triton_plan(k_hint=1, store="N", fmt="CSR"), None),
        ring_buffer_size=2,
        log_level="WARNING",
    )
    assert backend._caller_stream_ptr == 0
    assert backend._caller_stream_keepalive is None


def test_triton_accepts_protocol_null_stream_and_retains_owner():
    stream = _ProtocolStream(0)
    backend = TritonBackend(
        device=0,
        stream=stream,
        pair=TritonPlanPair.from_dicts(make_triton_plan(k_hint=1, store="N", fmt="CSR"), None),
        ring_buffer_size=2,
        log_level="WARNING",
    )
    assert backend._caller_stream_ptr == 0
    assert backend._caller_stream_keepalive is stream


def test_triton_accepts_torch_stream_and_retains_owner():
    with torch.cuda.device(0):
        master = torch.cuda.Stream()
    if not hasattr(master, "__cuda_stream__"):
        pytest.skip("torch.cuda.Stream() does not expose __cuda_stream__() in this build")
    backend = TritonBackend(
        device=0,
        stream=master,
        pair=TritonPlanPair.from_dicts(make_triton_plan(k_hint=1, store="N", fmt="CSR"), None),
        ring_buffer_size=2,
        log_level="WARNING",
    )
    assert backend._caller_stream_keepalive is master


def test_triton_rejects_invalid_stream():
    with pytest.raises(TypeError, match="__cuda_stream__"):
        TritonBackend(
            device=0,
            stream=object(),
            pair=TritonPlanPair.from_dicts(make_triton_plan(k_hint=1, store="N", fmt="CSR"), None),
            ring_buffer_size=2,
            log_level="WARNING",
        )


def test_triton_rejects_foreign_device_stream():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >=2 CUDA devices")
    with torch.cuda.device(1):
        master = torch.cuda.Stream()
    if not hasattr(master, "__cuda_stream__"):
        pytest.skip("torch.cuda.Stream() does not expose __cuda_stream__() in this build")
    with pytest.raises(ValueError, match="requested CUDA device 0"):
        TritonBackend(
            device=0,
            stream=master,
            pair=TritonPlanPair.from_dicts(make_triton_plan(k_hint=1, store="N", fmt="CSR"), None),
            ring_buffer_size=2,
            log_level="WARNING",
        )


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_host_pin_uses_separate_struct_dtypes_for_csc_blocks():
    backend = make_triton_backend(device=0, fmt_up="csc", fmt_down=None, k_hint=1, infer_missing=False)
    backend._apply_setup_state(_synthetic_setup())
    stored = _materialized_csc_block_with_large_row_index()

    backend._slot_indices_dtype = np.dtype(np.int64)
    backend._slot_indptr_dtype = np.dtype(np.int32)
    host_block = backend._pin_host_block(stored)
    assert host_block is not None
    assert host_block.indices.dtype == np.int64
    assert host_block.indptr.dtype == np.int32


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_down_only_scan_uses_stored_block_shape():
    huge = int(np.iinfo(np.int32).max) + 2
    block = sp.csr_matrix((1, huge), dtype=np.bool_)
    block.data = np.ones(1, dtype=np.bool_)
    block.indices = np.array([huge - 1], dtype=np.int64)
    block.indptr = np.array([0, 1], dtype=np.int32)
    block._shape = (1, huge)

    empty_selector = binary_csr_from_parts(
        indices=np.empty(0, dtype=np.int32),
        indptr=np.zeros(2, dtype=np.int32),
        shape=(1, huge + 1),
        shared_data=True,
    )
    backend = make_triton_backend(device=0, fmt_up=None, fmt_down="csc", k_hint=1, infer_missing=False)
    backend._apply_setup_state(
        BackendSetup(
            A_blocks=[[], [block]],
            level_offsets=np.array([0, huge, huge + 1], dtype=np.int64),
            num_samples=huge,
            num_mutations=1,
            num_nodes=huge + 1,
            sel_mut=empty_selector,
            sel_miss=empty_selector,
            coalescence_counts=None,
            dtype=np.float64,
        )
    )

    assert backend._scan_slot_struct_dtypes() == (np.dtype(np.int64), np.dtype(np.int32))
    backend._slot_indices_dtype, backend._slot_indptr_dtype = backend._scan_slot_struct_dtypes()
    host_block = backend._build_direction_host_blocks(Direction.DOWN)[0][0]
    assert host_block is not None
    assert host_block.indices.dtype == np.int64
    assert host_block.indptr.dtype == np.int32


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_slot_pool_compacts_safe_int64_indices_and_offsets():
    backend = make_triton_backend(device=0, fmt_up="csr", fmt_down="csc", k_hint=1)
    backend.setup(
        _synthetic_setup(
            block_indices_dtype=np.int64,
            block_indptr_dtype=np.int64,
            selector_indices_dtype=np.int64,
            selector_indptr_dtype=np.int64,
            level_offsets_dtype=np.int64,
        )
    )
    assert backend._slot_indices_dtype == np.dtype(np.int32)
    assert backend._slot_indptr_dtype == np.dtype(np.int32)
    assert backend._slot_pool is not None
    assert backend._slot_pool.slots[0].indices.dtype == torch.int32
    assert backend._slot_pool.slots[0].indptr.dtype == torch.int32
    y, _ = backend.run_up(np.arange(1, 5, dtype=np.float64).reshape(4, 1), init_mode=InitMode.NONE, init=None, need_miss_output=False)
    np.testing.assert_array_equal(y[:, 0], np.array([5.0, 3.0, 5.0, 7.0]))


@pytest.mark.stress
@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("ring_buffer_size", [1, 2])
def test_triton_large_stream_exact_two_pass_sequence(large_band_case, order, ring_buffer_size, monkeypatch):
    clear_gpu_state()
    backend = _make_stream_backend(ring_buffer_size=ring_buffer_size)
    _force_int64_slot_dtypes(monkeypatch, backend)
    setup = build_large_band_setup(large_band_case)
    try:
        backend.setup(setup)
        assert backend._slot_indices_dtype == np.dtype(np.int64)
        assert backend._slot_indptr_dtype == np.dtype(np.int64)
        for run_idx, direction in enumerate(order):
            rng = np.random.default_rng(60_000 + 1_000 * ring_buffer_size + 10 * run_idx + (0 if direction == "up" else 1))
            primary = binary_pm1(rng, (large_band_case.n, 1), DATA_DTYPE)
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
@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
@pytest.mark.parametrize("first_direction", ["up", "down"])
@pytest.mark.xfail(strict=True, reason="ring=3 must fail once three streamed slots cannot fit into total VRAM")
def test_triton_large_stream_ring3_xfail(large_band_case, first_direction, monkeypatch):
    clear_gpu_state()
    backend = _make_stream_backend(ring_buffer_size=3)
    _force_int64_slot_dtypes(monkeypatch, backend)
    setup = build_large_band_setup(large_band_case)
    try:
        backend.setup(setup)
        assert backend._slot_indices_dtype == np.dtype(np.int64)
        assert backend._slot_indptr_dtype == np.dtype(np.int64)
        rng = np.random.default_rng(80_000 + (0 if first_direction == "up" else 1))
        primary = binary_pm1(rng, (large_band_case.n, 1), DATA_DTYPE)
        _ = _run_stream_direction(backend, first_direction, primary)
    finally:
        del backend
        del setup
        clear_gpu_state()


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("ring_buffer_size", [1, 2])
def test_triton_stream_copy_overlaps_compute(order, ring_buffer_size, monkeypatch):
    clear_gpu_state()
    backend = _make_stream_backend(ring_buffer_size=ring_buffer_size)
    ref_backend = _make_reference_backend()
    setup = build_overlap_band_setup()
    state = {
        "active": True,
        "compute": [],
        "copy": [],
    }
    original_launch = type(backend)._launch_op

    def _wrapped_launch(self, op, *, x, y, config):
        if state["active"]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(device=self._torch_device)
            start.record(stream)
            original_launch(self, op, x=x, y=y, config=config)
            torch.cuda._sleep(10_000_000)
            end.record(stream)
            state["compute"].append((start, end))
            return
        original_launch(self, op, x=x, y=y, config=config)

    def _wrapped_copy(self, ws, dst_level, op_idx, op):
        copy_stream = self._slot_copy_streams[op.slot]
        host_indices, host_indptr = self._host_tensor_pair(op)
        with torch.cuda.stream(copy_stream):
            prev_event = self._prev_compute_event(ws, op)
            if prev_event is not None:
                copy_stream.wait_event(prev_event)
            start = None
            end = None
            if state["active"]:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(copy_stream)
            op.block.indices[: int(op.host_block.indices.size)].copy_(host_indices, non_blocking=True)
            op.block.indptr[: int(op.host_block.indptr.size)].copy_(host_indptr, non_blocking=True)
            if state["active"]:
                assert start is not None and end is not None
                end.record(copy_stream)
                state["copy"].append((start, end))
            ws.copy_done_by_level[dst_level][op_idx].record(copy_stream)

    monkeypatch.setattr(type(backend), "_launch_op", _wrapped_launch)
    monkeypatch.setattr(type(backend), "_copy_host_block_to_slot", _wrapped_copy)

    def _has_overlap() -> bool:
        for compute_start, compute_end in state["compute"]:
            if compute_start.elapsed_time(compute_end) <= 0.0:
                continue
            for copy_start, copy_end in state["copy"]:
                if copy_start.elapsed_time(copy_end) <= 0.0:
                    continue
                if copy_start.elapsed_time(compute_end) > 0.0 and compute_start.elapsed_time(copy_end) > 0.0:
                    return True
        return False

    try:
        backend.setup(setup)
        ref_backend.setup(setup)
        for run_idx, direction in enumerate(order):
            rng = np.random.default_rng(70_000 + 1_000 * ring_buffer_size + 10 * run_idx + (0 if direction == "up" else 1))
            primary = binary_pm1(rng, (setup.num_samples, 1), DATA_DTYPE)
            expected = _run_stream_direction(ref_backend, direction, primary)
            actual = _run_stream_direction(backend, direction, primary)
            np.testing.assert_array_equal(actual, expected)
            torch.cuda.synchronize()
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


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_graphs_created_after_setup(primary_grg_path, spmv_cache_dir):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir)
    backend = op._backend
    assert backend._workspaces.graph_up is not None
    assert backend._workspaces.graph_down is not None
    assert backend._workspaces.graph_up.graph is not None
    assert backend._workspaces.graph_down.graph is not None


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_accepts_foreign_cupy_stream_and_keeps_private_root(primary_grg_path, gt_small, spmv_cache_dir):
    cp = pytest.importorskip("cupy")

    with cp.cuda.Device(0):
        master = cp.cuda.Stream(non_blocking=True)
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, stream=master)
    backend = op._backend
    x_up, y_up = gt_small.get("forward", 1, seed=128, dtype=DATA_DTYPE)
    np.testing.assert_allclose(op.matmul(x_up.T, "up").T, y_up, atol=1e-5, rtol=1e-5)
    assert backend._workspaces.graph_up is not None
    assert backend._caller_stream_ptr == int(master.ptr)
    assert backend._caller_stream_keepalive is master
    assert int(backend._root_stream.cuda_stream) != int(master.ptr)


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_setup_and_run_stay_on_declared_device_after_device_switch(primary_grg_path, gt_small, spmv_cache_dir):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >=2 CUDA devices")
    with torch.cuda.device(0):
        backend = make_triton_backend(device=0, fmt_up="csr", fmt_down="csc", k_hint=1)
    with torch.cuda.device(1):
        op = SpmvGRG(
            primary_grg_path,
            backend,
            DATA_DTYPE,
            artifact_dir=spmv_cache_dir,
        )
        x_up, y_up = gt_small.get("forward", 1, seed=129, dtype=DATA_DTYPE)
        np.testing.assert_allclose(op.matmul(x_up.T, "up").T, y_up, atol=1e-5, rtol=1e-5)
    assert backend._sel_mut_rows_gpu.device.index == 0
    assert backend._sel_mut_cols_gpu.device.index == 0
    ws = backend._workspaces.graph_up
    assert ws is not None
    assert ws.node_state.device.index == 0
    assert ws.input_primary.device.index == 0


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_instrumentation_skips_graph_setup(primary_grg_path, spmv_cache_dir):
    with pytest.warns(RuntimeWarning, match="ignores k_hint"):
        op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, instrumentation=True)
    backend = op._backend
    assert backend._workspaces.dynamic_up is None
    assert backend._workspaces.dynamic_down is None
    assert backend._workspaces.graph_up is None
    assert backend._workspaces.graph_down is None


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_graph_workspace_keeps_optional_buffers_lazy(primary_grg_path, spmv_cache_dir):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="csr", fmt_down="csc")
    backend = op._backend
    ws_up = backend._workspaces.graph_up
    ws_down = backend._workspaces.graph_down
    assert ws_up is not None and ws_down is not None
    before_up = id(ws_up)
    before_down = id(ws_down)
    assert backend._staging_up is None
    assert backend._staging_down is None

    x_up = np.ones((1, op.num_samples), dtype=DATA_DTYPE)
    _ = op.matmul(x_up, "up", emit_all_nodes=True, init="xtx")

    assert backend._staging_up is not None
    assert backend._staging_up.xtx_bias is not None
    assert id(backend._workspaces.graph_up) == before_up
    assert id(backend._workspaces.graph_down) == before_down


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_hint_none_uses_dynamic_mode(primary_grg_path, spmv_cache_dir):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, k_hint=None)
    backend = op._backend
    assert backend._workspaces.graph_up is None
    assert backend._workspaces.graph_down is None
    x_up = np.ones((1, op.num_samples), dtype=DATA_DTYPE)
    _ = op.matmul(x_up, "up")
    assert op.memory.last_call is not None
    assert op.memory.last_call.meta["mode"] == "dynamic"
    assert op.memory.last_call.active_alloc_keys
    merged = live_snapshot(op.memory.retained, op.memory.last_call)
    assert merged is not None
    assert any(
        row.owner == "backend"
        and row.direction == "up"
        and row.retention in {"on_demand", "staging"}
        and row.activity == "yes"
        for row in merged.allocations
    )
    assert backend._workspaces.dynamic_up is not None


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_kernel_family_matches_format(primary_grg_path, spmv_cache_dir):
    op_csr = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="csr", fmt_down=None, infer_missing=False)
    op_csc = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="csc", fmt_down=None, infer_missing=False)
    assert isinstance(op_csr._backend._config_up, CsrKernelConfig)
    assert isinstance(op_csc._backend._config_up, CscKernelConfig)


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_transpose_compatible_storage_aliases_block_tensors(primary_grg_path, spmv_cache_dir):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="csr", fmt_down="csc")
    backend = op._backend
    H = len(backend._level_offsets) - 1
    for dst_level, src_level, row_index in iter_direction_level_pairs(Direction.DOWN, H):
        down_block = backend._host_blocks_down[dst_level][row_index]
        if down_block is None:
            continue
        up_block = backend._host_blocks_up[src_level][dst_level]
        assert up_block is not None
        assert np.shares_memory(down_block.indices, up_block.indices)
        assert np.shares_memory(down_block.indptr, up_block.indptr)
        return
    raise AssertionError("expected at least one shared Triton block alias")


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_incompatible_formats_do_not_alias_block_tensors(primary_grg_path, spmv_cache_dir):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="csr", fmt_down="csr")
    backend = op._backend
    H = len(backend._level_offsets) - 1
    for dst_level, src_level, row_index in iter_direction_level_pairs(Direction.DOWN, H):
        down_block = backend._host_blocks_down[dst_level][row_index]
        if down_block is None:
            continue
        up_block = backend._host_blocks_up[src_level][dst_level]
        assert up_block is not None
        assert not np.shares_memory(down_block.indices, up_block.indices)
        assert not np.shares_memory(down_block.indptr, up_block.indptr)
        return
    raise AssertionError("expected at least one non-shared Triton block")


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_capture_failure_is_hard_error(primary_grg_path, spmv_cache_dir, monkeypatch):
    from pygrgl_spmv.backends.triton import TritonBackend

    def _boom(self, ws, config):
        raise RuntimeError(f"capture failed for {ws.direction.value}")

    monkeypatch.setattr(TritonBackend, "_build_wavefront_graph", _boom)
    with pytest.raises(RuntimeError, match="capture failed"):
        _ = _make_op(primary_grg_path, cache_dir=spmv_cache_dir)


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_setup_gpu_allocations_run_inside_caller_root_scope(primary_grg_path, spmv_cache_dir, monkeypatch):
    backend = make_triton_backend(device=0, fmt_up="csr", fmt_down="csc", k_hint=1)
    depth = _install_scope_depth_probe(monkeypatch, backend)
    seen = {"blocks": 0, "workspace": 0}
    original_build_blocks = type(backend)._build_direction_host_blocks
    original_alloc_workspace = type(backend)._alloc_workspace

    def _wrapped_build_blocks(self, direction):
        assert depth["value"] > 0
        seen["blocks"] += 1
        return original_build_blocks(self, direction)

    def _wrapped_alloc_workspace(self, direction):
        assert depth["value"] > 0
        seen["workspace"] += 1
        return original_alloc_workspace(self, direction)

    monkeypatch.setattr(type(backend), "_build_direction_host_blocks", _wrapped_build_blocks)
    monkeypatch.setattr(type(backend), "_alloc_workspace", _wrapped_alloc_workspace)

    _ = SpmvGRG(
        primary_grg_path,
        backend,
        DATA_DTYPE,
        artifact_dir=spmv_cache_dir,
    )

    assert seen["blocks"] >= 2
    assert seen["workspace"] >= 4


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_tune_once_restores_caller_root_scope_on_failure(primary_grg_path, spmv_cache_dir, monkeypatch):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="csr", fmt_down=None, infer_missing=False)
    backend = op._backend
    with torch.cuda.stream(backend._root_stream):
        ws = backend._alloc_workspace(Direction.UP)
    counts = _install_bridge_probe(monkeypatch, backend)

    def _boom(ws, *, config):
        raise RuntimeError("wavefront failed")

    monkeypatch.setattr(backend, "_enqueue_wavefront_dispatch", _boom)

    with pytest.raises(RuntimeError, match="wavefront failed"):
        backend._tune_once(ws, backend._config_up)

    assert counts == {"entered": 1, "exited": 1}


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_tune_direction_restores_caller_root_scope_on_upload_failure(primary_grg_path, spmv_cache_dir, monkeypatch):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="csr", fmt_down=None, infer_missing=False)
    backend = op._backend
    with torch.cuda.stream(backend._root_stream):
        ws = backend._alloc_workspace(Direction.UP)
    counts = _install_bridge_probe(monkeypatch, backend)
    _AUTOTUNE_CACHE.clear()

    class _BrokenCopy:
        def copy_(self, other):
            raise RuntimeError("upload failed")

    ws.input_primary = _BrokenCopy()

    with pytest.raises(RuntimeError, match="upload failed"):
        backend._tune_direction(Direction.UP, ws)

    assert counts == {"entered": 1, "exited": 1}


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_build_wavefront_graph_restores_caller_root_scope_on_capture_failure(primary_grg_path, spmv_cache_dir, monkeypatch):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="csr", fmt_down=None, infer_missing=False)
    backend = op._backend
    with torch.cuda.stream(backend._root_stream):
        ws = backend._alloc_workspace(Direction.UP)
    counts = _install_bridge_probe(monkeypatch, backend)
    original = backend._enqueue_wavefront_dispatch
    calls = {"count": 0}

    def _boom(ws, *, config):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("capture enqueue failed")
        return original(ws, config=config)

    monkeypatch.setattr(backend, "_enqueue_wavefront_dispatch", _boom)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The CUDA Graph is empty.*", category=UserWarning)
        with pytest.raises(RuntimeError, match="capture enqueue failed"):
            backend._build_wavefront_graph(ws, backend._config_up)

    assert counts == {"entered": 2, "exited": 2}


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_run_column_restores_caller_root_scope_on_missing_graph(primary_grg_path, spmv_cache_dir, monkeypatch):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir)
    backend = op._backend
    ws = backend._workspaces.graph_up
    assert ws is not None
    counts = _install_bridge_probe(monkeypatch, backend)
    ws.graph = None

    with pytest.raises(RuntimeError, match="Missing Triton graph workspace"):
        backend._run_column(
            Direction.UP,
            primary_col=np.ones((op.num_samples,), dtype=DATA_DTYPE),
            miss_col=None,
            init_mode=InitMode.NONE,
            init_value=None,
            need_miss_output=False,
            emit_all_nodes=False,
        )

    assert counts == {"entered": 1, "exited": 1}


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_run_column_restores_caller_root_scope_on_gather_failure(primary_grg_path, spmv_cache_dir, monkeypatch):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, k_hint=None)
    backend = op._backend
    counts = _install_bridge_probe(monkeypatch, backend)

    def _boom(ws, staging, *, need_miss_output):
        raise RuntimeError("gather failed")

    monkeypatch.setattr(backend, "_enqueue_output_gather", _boom)

    with pytest.raises(RuntimeError, match="gather failed"):
        backend._run_column(
            Direction.UP,
            primary_col=np.ones((op.num_samples,), dtype=DATA_DTYPE),
            miss_col=None,
            init_mode=InitMode.NONE,
            init_value=None,
            need_miss_output=False,
            emit_all_nodes=False,
        )

    assert counts == {"entered": 1, "exited": 1}


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_dynamic_workspace_allocates_inside_caller_root_scope(primary_grg_path, spmv_cache_dir, monkeypatch):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, k_hint=None)
    backend = op._backend
    depth = _install_scope_depth_probe(monkeypatch, backend)
    calls = {"count": 0}
    original = type(backend)._alloc_workspace

    def _wrapped(self, direction):
        calls["count"] += 1
        assert depth["value"] > 0
        return original(self, direction)

    monkeypatch.setattr(type(backend), "_alloc_workspace", _wrapped)

    x_up = np.ones((1, op.num_samples), dtype=DATA_DTYPE)
    _ = op.matmul(x_up, "up")

    assert calls == {"count": 1}


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_structure_only_block_bytes_less_than_value_payload(primary_grg_path, spmv_cache_dir):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="csr", fmt_down=None, infer_missing=False)
    block = _first_present_block(op._backend._host_blocks_up)
    structure_only = int(block.nbytes())
    dense_value_payload = int(block.nnz * (int(np.dtype(DATA_DTYPE).itemsize) + 4) + (block.nrows + 1) * 4)
    assert structure_only < dense_value_payload


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_autotune_cache_reused(primary_grg_path, spmv_cache_dir):
    _AUTOTUNE_CACHE.clear()
    _ = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="csr", fmt_down="csc")
    size_after_first = len(_AUTOTUNE_CACHE)
    assert size_after_first >= 2
    _ = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="csr", fmt_down="csc")
    assert len(_AUTOTUNE_CACHE) == size_after_first


class _NvtxRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, object]]] = []

    @contextmanager
    def range(self, name: str, /, **fields: object):
        self.events.append(("range", name, dict(fields)))
        yield

    def mark(self, name: str, /, **fields: object) -> None:
        self.events.append(("mark", name, dict(fields)))


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_scratch_buffers_allocated_for_enabled_levels(primary_grg_path, spmv_cache_dir):
    op = _make_op(
        primary_grg_path,
        cache_dir=spmv_cache_dir,
        fmt_up="csr",
        fmt_down="csc",
        scratch_up="1",
        scratch_down="0",
    )
    backend = op._backend
    ws_up = backend._workspaces.graph_up
    ws_down = backend._workspaces.graph_down
    assert ws_up is not None and ws_down is not None
    assert not hasattr(ws_up, "level_streams")
    assert not hasattr(ws_up, "scratch_streams_by_level")
    assert len(backend._level_streams) == len(backend._level_offsets) - 1
    assert len(backend._scratch_streams_up_by_level[1]) == len(backend._ops_up[1])
    assert len(backend._scratch_streams_down_by_level[0]) == len(backend._ops_down[0])
    assert len(ws_up.scratch_views_by_level[1]) == len(backend._ops_up[1]) > 0
    assert len(ws_up.scratch_done_events_by_level[1]) == len(backend._ops_up[1])
    assert ws_up.scratch_views_by_level[0] == []
    assert len(ws_down.scratch_views_by_level[0]) == len(backend._ops_down[0]) > 0
    assert ws_down.scratch_views_by_level[1] == []


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_debug_log_level_keeps_graph_mode(primary_grg_path, gt_small, spmv_cache_dir):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, log_level="DEBUG")
    x_up, _ = gt_small.get("forward", 1, seed=124, dtype=DATA_DTYPE)
    x_down, _ = gt_small.get("backward", 1, seed=125, dtype=DATA_DTYPE)
    _ = op.matmul(x_up.T, "up")
    assert op.memory.last_call is not None
    assert op.memory.last_call.meta["mode"] == "graph"
    _ = op.matmul(x_down.T, "down")
    assert op.memory.last_call is not None
    assert op.memory.last_call.meta["mode"] == "graph"
    assert op._backend._workspaces.graph_up is not None
    assert op._backend._workspaces.graph_up.graph is not None
    assert op._backend._workspaces.graph_down is not None
    assert op._backend._workspaces.graph_down.graph is not None


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_debug_log_level_reports_block_memory(primary_grg_path, spmv_cache_dir, caplog):
    with caplog.at_level(logging.DEBUG):
        op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, log_level="DEBUG")
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("Triton host_blocks_up" in msg and "rows=" in msg and "nnz=" in msg and "indices_bytes=" in msg for msg in messages)
    assert any("Triton host_blocks_up dir=up" in msg and "cols=" in msg and "nnz=" in msg for msg in messages)
    del op


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_setup_streams_materialized_blocks_before_slot_pool_alloc(monkeypatch):
    refs: list[weakref.ReferenceType[sp.spmatrix]] = []
    materialized = {"count": 0, "checked": 0}
    backend = make_triton_backend(device=0, fmt_up="csc", fmt_down="csc", k_hint=None, infer_missing=False)
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


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_info_log_level_reports_setup_rss_checkpoints(monkeypatch, caplog):
    import pygrgl_spmv._rss as rss_mod

    values = iter([100, 140, 120, 160])
    monkeypatch.setattr(rss_mod, "rss_bytes", lambda: next(values))

    backend = make_triton_backend(device=0, fmt_up="csc", fmt_down="csc", k_hint=None, infer_missing=False, log_level="INFO")
    with caplog.at_level(logging.INFO, logger="pygrgl_spmv.backends.triton.backend.TritonBackend"):
        backend.setup(_synthetic_setup())

    messages = [
        rec.getMessage()
        for rec in caplog.records
        if rec.name == "pygrgl_spmv.backends.triton.backend.TritonBackend" and rec.getMessage().startswith("rss setup:")
    ]
    assert [msg.split()[1] for msg in messages] == [
        "setup:start",
        "setup:host_blocks_up_ready",
        "setup:host_blocks_down_ready",
        "setup:complete",
    ]
    assert "delta_bytes=" not in messages[0]
    assert all("delta_bytes=" in msg for msg in messages[1:])
    assert "stored_blocks=" in messages[1] and "indices_bytes=" in messages[1] and "indptr_bytes=" in messages[1]
    assert "stored_blocks=" in messages[2] and "indices_bytes=" in messages[2] and "indptr_bytes=" in messages[2]


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_instrumented_nvtx_markers(primary_grg_path, gt_small, spmv_cache_dir):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, instrumentation=True)
    recorder = _NvtxRecorder()
    op._backend._nvtx = recorder
    x_up, _ = gt_small.get("forward", 1, seed=126, dtype=DATA_DTYPE)
    _ = op.matmul(x_up.T, "up")
    names = [name for _, name, _ in recorder.events]
    assert "execute_singleton" in names
    assert "prepare_state" in names
    assert "wavefront" in names
    assert "level" in names
    assert "wait_ready" in names
    assert "launch" in names
    assert "event.record_ready" in names
    assert "gather_outputs" in names
    assert "join_ready" in names


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_instrumented_scratch_emits_reduction_markers(primary_grg_path, gt_small, spmv_cache_dir):
    op = _make_op(
        primary_grg_path,
        cache_dir=spmv_cache_dir,
        fmt_up="csr",
        fmt_down=None,
        scratch_up="1",
        instrumentation=True,
        infer_missing=False,
    )
    recorder = _NvtxRecorder()
    op._backend._nvtx = recorder
    x_up, _ = gt_small.get("forward", 1, seed=127, dtype=DATA_DTYPE)
    _ = op.matmul(x_up.T, "up")
    names = [name for _, name, _ in recorder.events]
    assert "helper_launch" in names
    assert "event.record_scratch_done" in names
    assert "wait_scratch_done" in names
    assert "reduce_add" in names


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_scratch_enabled_up_matches_reference(primary_grg_path, gt_small, spmv_cache_dir):
    op = _make_op(
        primary_grg_path,
        cache_dir=spmv_cache_dir,
        fmt_up="csr",
        fmt_down=None,
        scratch_up="1",
        infer_missing=False,
    )
    X, Y_expected = gt_small.get("forward", 1, seed=123, dtype=DATA_DTYPE)
    np.testing.assert_allclose(op.matmul(X.T, "up").T, Y_expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_scratch_enabled_down_matches_reference(primary_grg_path, gt_small, spmv_cache_dir):
    op = _make_op(
        primary_grg_path,
        cache_dir=spmv_cache_dir,
        fmt_up=None,
        fmt_down="csc",
        scratch_down="0",
        infer_missing=False,
    )
    X, Y_expected = gt_small.get("backward", 1, seed=123, dtype=DATA_DTYPE)
    np.testing.assert_allclose(op.matmul(X.T, "down").T, Y_expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_runtime_k_gt_1_raises_clear_error(primary_grg_path, spmv_cache_dir):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir)
    x_up = np.ones((2, op.num_samples), dtype=DATA_DTYPE)
    x_down = np.ones((2, op.num_mutations), dtype=DATA_DTYPE)
    with pytest.raises(RuntimeError, match="runtime k == 1 only"):
        op.matmul(x_up, "up")
    with pytest.raises(RuntimeError, match="runtime k == 1 only"):
        op.matmul(x_down, "down")
