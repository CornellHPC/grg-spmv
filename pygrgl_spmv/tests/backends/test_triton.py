"""Triton backend-specific tests."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import numpy as np
import pytest
import warnings

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.backends.triton import TritonBackend, TritonPlanPair
from pygrgl_spmv.backends import iter_direction_level_pairs
from pygrgl_spmv.backends.triton.backend import _AUTOTUNE_CACHE
from pygrgl_spmv.backends.triton.kernel import CscKernelConfig, CsrKernelConfig
from pygrgl_spmv.backends.types import Direction, InitMode
from pygrgl_spmv.memory import live_snapshot
from pygrgl_spmv.tests.conftest import DATA_DTYPE, HAS_TRITON_RUNTIME, INDEX_DTYPE, make_triton_backend, make_triton_plan

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
        INDEX_DTYPE,
        artifact_dir=cache_dir,
    )


def _first_present_block(grid):
    for row in grid:
        for block in row:
            if block is not None:
                return block
    raise AssertionError("expected at least one non-empty Triton block")


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
            INDEX_DTYPE,
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
        INDEX_DTYPE,
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
