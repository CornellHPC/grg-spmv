"""Triton backend-specific tests."""

from __future__ import annotations

from contextlib import contextmanager
import numpy as np
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.backends.triton import TritonBackend, TritonPlanPair
from pygrgl_spmv.backends import estimate_sparse_payload_bytes, iter_direction_level_pairs
from pygrgl_spmv.backends.triton.backend import _AUTOTUNE_CACHE
from pygrgl_spmv.backends.triton.kernel import CscKernelConfig, CsrKernelConfig
from pygrgl_spmv.backends.types import Direction
from pygrgl_spmv.tests.conftest import DATA_DTYPE, HAS_TRITON_RUNTIME, INDEX_DTYPE, make_triton_backend, make_triton_plan

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("Triton tests require CUDA", allow_module_level=True)

pytestmark = [pytest.mark.gpu, pytest.mark.triton]


def _make_op(
    grg_path,
    *,
    cache_dir,
    fmt_up="csr",
    fmt_down="csc",
    scratch_up="none",
    scratch_down="none",
    log_level="WARNING",
    instrumentation=False,
    infer_missing=True,
):
    return SpmvGRG(
        grg_path,
        make_triton_backend(
            fmt_up=fmt_up,
            fmt_down=fmt_down,
            k_hint=1,
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


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_graphs_created_after_setup(primary_grg_path, spmv_cache_dir):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir)
    backend = op._backend
    assert backend._workspaces.graph_up is not None
    assert backend._workspaces.graph_down is not None
    assert backend._workspaces.graph_up.graph is not None
    assert backend._workspaces.graph_down.graph is not None


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_instrumentation_skips_graph_setup(primary_grg_path, spmv_cache_dir):
    with pytest.warns(RuntimeWarning, match="instrumentation=True"):
        op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, instrumentation=True)
    backend = op._backend
    assert backend._workspaces.dynamic_up is not None
    assert backend._workspaces.dynamic_down is not None
    assert backend._workspaces.graph_up is None
    assert backend._workspaces.graph_down is None


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
        down_block = backend._blocks_down[dst_level][row_index]
        if down_block is None:
            continue
        up_block = backend._blocks_up[src_level][dst_level]
        assert up_block is not None
        assert down_block.indices.data_ptr() == up_block.indices.data_ptr()
        assert down_block.indptr.data_ptr() == up_block.indptr.data_ptr()
        return
    raise AssertionError("expected at least one shared Triton block alias")


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_incompatible_formats_do_not_alias_block_tensors(primary_grg_path, spmv_cache_dir):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="csr", fmt_down="csr")
    backend = op._backend
    H = len(backend._level_offsets) - 1
    for dst_level, src_level, row_index in iter_direction_level_pairs(Direction.DOWN, H):
        down_block = backend._blocks_down[dst_level][row_index]
        if down_block is None:
            continue
        up_block = backend._blocks_up[src_level][dst_level]
        assert up_block is not None
        assert down_block.indices.data_ptr() != up_block.indices.data_ptr()
        assert down_block.indptr.data_ptr() != up_block.indptr.data_ptr()
        return
    raise AssertionError("expected at least one non-shared Triton block")


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_capture_failure_is_hard_error(primary_grg_path, spmv_cache_dir, monkeypatch):
    from pygrgl_spmv.backends.triton import TritonBackend

    def _boom(self, direction, ws, config):
        raise RuntimeError(f"capture failed for {direction.value}")

    monkeypatch.setattr(TritonBackend, "_capture_wavefront_graph", _boom)
    with pytest.raises(RuntimeError, match="capture failed"):
        _ = _make_op(primary_grg_path, cache_dir=spmv_cache_dir)


@pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable")
def test_triton_structure_only_block_bytes_less_than_value_payload(primary_grg_path, spmv_cache_dir):
    op = _make_op(primary_grg_path, cache_dir=spmv_cache_dir, fmt_up="csr", fmt_down=None, infer_missing=False)
    block = _first_present_block(op._backend._blocks_up)
    structure_only = int(block.nbytes())
    dense_value_payload = estimate_sparse_payload_bytes(
        fmt=block.fmt.value.lower(),
        nrows=block.nrows,
        ncols=block.ncols,
        nnz=block.nnz,
        data_itemsize=int(np.dtype(DATA_DTYPE).itemsize),
        index_itemsize=4,
    )
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
    ws_up = backend._workspaces.dynamic_up
    ws_down = backend._workspaces.dynamic_down
    assert ws_up is not None and ws_down is not None
    assert len(ws_up.scratch_views_by_level[1]) == len(backend._ops_up[1]) > 0
    assert len(ws_up.scratch_streams_by_level[1]) == len(backend._ops_up[1])
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
    _ = op.matmul(x_down.T, "down")
    assert op._backend.mem_usage.calls[-2].meta["mode"] == "graph"
    assert op._backend.mem_usage.calls[-1].meta["mode"] == "graph"
    assert op._backend._workspaces.graph_up is not None
    assert op._backend._workspaces.graph_up.graph is not None
    assert op._backend._workspaces.graph_down is not None
    assert op._backend._workspaces.graph_down.graph is not None


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
    X, Y_expected = gt_small.get("forward", 4, seed=123, dtype=DATA_DTYPE)
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
    X, Y_expected = gt_small.get("backward", 4, seed=123, dtype=DATA_DTYPE)
    np.testing.assert_allclose(op.matmul(X.T, "down").T, Y_expected, atol=1e-5, rtol=1e-5)
