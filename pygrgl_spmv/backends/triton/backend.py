"""Triton backend using naive CSR/CSC unit-weight SpMV kernels."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha1
import logging
from time import perf_counter
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import triton

from pygrgl_spmv._rss import rss_checkpoint
from pygrgl_spmv.backends.base import (
    BackendBase,
    CallCapture,
    BackendSetup,
    _copy_struct_checked,
    _layout_struct_dtypes,
    _require_struct_dtype,
    effective_k_hint,
    iter_direction_level_pairs,
    warn_instrumentation_ignores_k_hint,
)
from pygrgl_spmv.backends._cuda_stream import (
    _cuda_stream_device,
    parse_cuda_device,
    parse_cuda_stream,
)
from pygrgl_spmv.backends._nvtx import make_torch_tracer
from pygrgl_spmv.memory import alloc_field, child_field, ignore_field
from pygrgl_spmv.backends.triton.kernel import (
    CSC_CANDIDATE_CONFIGS,
    CSR_CANDIDATE_CONFIGS,
    CscKernelConfig,
    CsrKernelConfig,
    format_config,
    launch_block,
)
from pygrgl_spmv.backends.triton.plan import TritonPlan, TritonPlanPair
from pygrgl_spmv.backends.types import (
    Direction,
    InitMode,
    SparseFormat,
    StoredMatrix,
    parse_init_mode,
    parse_sparse_format,
    parse_store,
    transpose_compatible_format,
)

_AUTOTUNE_CACHE: dict[tuple[object, ...], object] = {}
_TUNE_WARMUP_MS = 20
_TUNE_REP_MS = 80


def _torch_nbytes(value: torch.Tensor | None) -> int:
    if value is None:
        return 0
    return int(value.numel() * value.element_size())


def _torch_int_dtype(dtype: np.dtype) -> torch.dtype:
    dt = _require_struct_dtype(dtype, label="torch structural dtype")
    return torch.int32 if dt == np.dtype(np.int32) else torch.int64


def _selector_index_arrays(selector: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    coo = selector.tocoo()
    row = np.asarray(coo.row)
    col = np.asarray(coo.col)
    _require_struct_dtype(row.dtype, label="selector row indices")
    _require_struct_dtype(col.dtype, label="selector col indices")
    return row, col


def _structure_signature(ops_by_level: list[list["_TritonOp"]]) -> str:
    digest = sha1()
    for ops in ops_by_level:
        digest.update(len(ops).to_bytes(4, "little", signed=False))
        for op in ops:
            digest.update(op.block.fmt.value.encode("ascii"))
            digest.update(int(op.block.nrows).to_bytes(8, "little", signed=False))
            digest.update(int(op.block.ncols).to_bytes(8, "little", signed=False))
            digest.update(int(op.block.nnz).to_bytes(8, "little", signed=False))
    return digest.hexdigest()

@dataclass(frozen=True)
class _TritonHostBlock:
    indices: np.ndarray = alloc_field(label="host_blocks", kind="sparse")
    indptr: np.ndarray = alloc_field(label="host_blocks", kind="sparse")
    nrows: int = ignore_field()
    ncols: int = ignore_field()
    nnz: int = ignore_field()
    fmt: SparseFormat = ignore_field()
    keepalive_key: tuple[int, int] = ignore_field()

    def nbytes(self) -> int:
        return int(self.indices.nbytes + self.indptr.nbytes)

    def transpose_alias(self) -> "_TritonHostBlock":
        return _TritonHostBlock(
            indices=self.indices,
            indptr=self.indptr,
            nrows=int(self.ncols),
            ncols=int(self.nrows),
            nnz=int(self.nnz),
            fmt=transpose_compatible_format(self.fmt),
            keepalive_key=self.keepalive_key,
        )


@dataclass(frozen=True)
class _TritonSlotBlock:
    indices: torch.Tensor = ignore_field()
    indptr: torch.Tensor = ignore_field()
    nrows: int = ignore_field()
    ncols: int = ignore_field()
    nnz: int = ignore_field()
    fmt: SparseFormat = ignore_field()


@dataclass(frozen=True)
class _TritonOp:
    src_level: int
    dst_level: int
    slot: int
    block: _TritonSlotBlock
    host_block: _TritonHostBlock
    nnz: int
    prev_in_slot: tuple[int, int] | None


@dataclass(frozen=True)
class _TritonScratchLevelPlan:
    enabled: bool
    reduce_order: tuple[int, ...]


@dataclass
class _DirectionWorkspace:
    direction: Direction = ignore_field()
    launch_event: torch.cuda.Event = ignore_field()
    level_done_events: list[torch.cuda.Event] = ignore_field()
    copy_done_by_level: list[list[torch.cuda.Event]] = ignore_field()
    compute_done_by_level: list[list[torch.cuda.Event]] = ignore_field()
    node_state: torch.Tensor = alloc_field(label="node_state", kind="state")
    level_views: list[torch.Tensor] = alloc_field(label="level_view", kind="state")
    scratch_done_events_by_level: list[list[torch.cuda.Event]] = ignore_field()
    scratch_views_by_level: list[list[torch.Tensor]] = alloc_field(label="scratch_views", kind="scratch")
    input_primary: torch.Tensor = alloc_field(label="input_primary", kind="input")
    graph: torch.cuda.CUDAGraph | None = ignore_field(default=None)


@dataclass
class _TritonSlotBuffers:
    indices: torch.Tensor = alloc_field(label="slot_buffers", kind="sparse")
    indptr: torch.Tensor = alloc_field(label="slot_buffers", kind="sparse")


@dataclass
class _TritonSlotPool:
    slots: list[_TritonSlotBuffers] = child_field(default_factory=list)


@dataclass
class _WorkspaceCache:
    dynamic_up: _DirectionWorkspace | None = child_field(retention="on_demand", activity="no", direction="up", default=None)
    graph_up: _DirectionWorkspace | None = child_field(retention="captured", activity="no", direction="up", default=None)
    dynamic_down: _DirectionWorkspace | None = child_field(retention="on_demand", activity="no", direction="down", default=None)
    graph_down: _DirectionWorkspace | None = child_field(retention="captured", activity="no", direction="down", default=None)


@dataclass
class _DirectionStaging:
    input_miss: torch.Tensor | None = alloc_field(label="input_miss", kind="input", default=None)
    output_main: torch.Tensor | None = alloc_field(label="output_main", kind="output", default=None)
    output_miss: torch.Tensor | None = alloc_field(label="output_miss", kind="output", default=None)
    init_vector: torch.Tensor | None = alloc_field(label="init_vector", kind="init", default=None)
    init_matrix: torch.Tensor | None = alloc_field(label="init_matrix", kind="init", default=None)
    xtx_bias: torch.Tensor | None = alloc_field(label="xtx_bias", kind="init", default=None)


@dataclass
class TritonCall:
    node_values_host: np.ndarray | None = alloc_field(
        label="node_values_host", kind="state", owner="backend", retention="call", activity="yes", default=None
    )
    miss_output_host: np.ndarray | None = alloc_field(
        label="miss_output_host", kind="output", owner="backend", retention="call", activity="yes", default=None
    )


@dataclass
class TritonRetained:
    sel_mut_rows_gpu: torch.Tensor | None = alloc_field(
        label="selector_mut", kind="selector", owner="backend", retention="persistent", activity="always", default=None
    )
    sel_mut_cols_gpu: torch.Tensor | None = alloc_field(
        label="selector_mut", kind="selector", owner="backend", retention="persistent", activity="always", default=None
    )
    sel_miss_rows_gpu: torch.Tensor | None = alloc_field(
        label="selector_miss", kind="selector", owner="backend", retention="persistent", activity="always", default=None
    )
    sel_miss_cols_gpu: torch.Tensor | None = alloc_field(
        label="selector_miss", kind="selector", owner="backend", retention="persistent", activity="always", default=None
    )
    host_blocks_up: list[tuple[np.ndarray, np.ndarray]] = alloc_field(
        label="host_blocks_up", kind="sparse", owner="backend", retention="persistent", activity="always", default_factory=list
    )
    host_blocks_down: list[tuple[np.ndarray, np.ndarray]] = alloc_field(
        label="host_blocks_down", kind="sparse", owner="backend", retention="persistent", activity="always", default_factory=list
    )
    slot_pool_up: _TritonSlotPool | None = child_field(
        owner="backend", retention="persistent", activity="always", direction="up", default=None
    )
    slot_pool_down: _TritonSlotPool | None = child_field(
        owner="backend", retention="persistent", activity="always", direction="down", default=None
    )
    workspaces: _WorkspaceCache = child_field(owner="backend", default_factory=_WorkspaceCache)
    staging_up: _DirectionStaging | None = child_field(owner="backend", retention="staging", activity="no", direction="up", default=None)
    staging_down: _DirectionStaging | None = child_field(owner="backend", retention="staging", activity="no", direction="down", default=None)


class TritonBackend(BackendBase):
    """GPU backend using Triton naive CSR/CSC kernels on singleton vectors."""

    lock: int = 1

    _SETUP_MEMORY_POLICY = {
        "_A_blocks": "dropped",
        "_sel_mut": "dropped",
        "_sel_miss": "dropped",
        "_level_offsets": "borrowed",
        "_coalescence_counts": "borrowed",
        "_xtx_host": "dropped",
    }

    @staticmethod
    def plan(
        *,
        fmt: str | SparseFormat = SparseFormat.CSR,
        store: str | StoredMatrix = StoredMatrix.N,
        k_hint: int | None = 1,
    ) -> TritonPlan:
        return TritonPlan(store=parse_store(store), fmt=parse_sparse_format(fmt), k_hint=k_hint)

    def __init__(
        self,
        *,
        device: int,
        stream: object,
        pair: TritonPlanPair,
        ring_buffer_size: int,
        log_level: str = "WARNING",
        instrumentation: bool = False,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Triton backend requires torch.cuda.is_available()")
        if int(ring_buffer_size) < 1:
            raise ValueError(f"ring_buffer_size must be >= 1, got {ring_buffer_size}")

        super().__init__(
            plan_up=pair.plan_up,
            plan_down=pair.plan_down,
            log_level=log_level,
            instrumentation=instrumentation,
        )
        self._device_id = parse_cuda_device(device)
        self._torch_device = torch.device("cuda", self._device_id)
        self._ring_buffer_size = int(ring_buffer_size)
        self._caller_stream_ptr, self._caller_stream_keepalive = parse_cuda_stream(stream)
        if self._caller_stream_ptr != 0:
            stream_device = _cuda_stream_device(self._caller_stream_ptr)
            if stream_device != self._device_id:
                raise ValueError(
                    f"CUDA stream device {stream_device} does not match requested CUDA device {self._device_id}"
                )
        # PyTorch wraps the raw handle only, so protocol-backed stream owners
        # must stay alive on the backend for the wrapped stream to remain valid.
        with torch.cuda.device(self._torch_device):
            self._caller_stream = torch.cuda.get_stream_from_external(self._caller_stream_ptr, device=self._torch_device)
            self._root_stream = torch.cuda.Stream()
            self._caller_to_root_event = torch.cuda.Event()
            self._root_to_caller_event = torch.cuda.Event()
        self._level_streams: list[torch.cuda.Stream] = []
        self._slot_copy_streams: list[torch.cuda.Stream] = []
        self._scratch_streams_up_by_level: list[list[torch.cuda.Stream]] = []
        self._scratch_streams_down_by_level: list[list[torch.cuda.Stream]] = []
        self._torch_dtype = torch.float64
        self._host_blocks_up: list[list[_TritonHostBlock | None]] = []
        self._host_blocks_down: list[list[_TritonHostBlock | None]] = []
        self._ops_up: list[list[_TritonOp]] = []
        self._ops_down: list[list[_TritonOp]] = []
        self._slot_pool: _TritonSlotPool | None = None
        self._host_tensors: dict[int, torch.Tensor] = {}
        self._next_host_tensor_key = 0
        self._sel_mut_rows_gpu = torch.empty(0, device=self._torch_device, dtype=torch.int64)
        self._sel_mut_cols_gpu = torch.empty(0, device=self._torch_device, dtype=torch.int64)
        self._sel_miss_rows_gpu = torch.empty(0, device=self._torch_device, dtype=torch.int64)
        self._sel_miss_cols_gpu = torch.empty(0, device=self._torch_device, dtype=torch.int64)
        self._slot_indices_dtype = np.dtype(np.int32)
        self._slot_indptr_dtype = np.dtype(np.int32)
        self._workspaces = _WorkspaceCache()
        self._staging_up: _DirectionStaging | None = None
        self._staging_down: _DirectionStaging | None = None
        self._config_up: CsrKernelConfig | CscKernelConfig | None = None
        self._config_down: CsrKernelConfig | CscKernelConfig | None = None
        self._scratch_plan_up: list[_TritonScratchLevelPlan] = []
        self._scratch_plan_down: list[_TritonScratchLevelPlan] = []
        self._nvtx = make_torch_tracer("grg.triton", torch) if self._instrumentation else None
        self._install_memory(
            retained=TritonRetained(
                sel_mut_rows_gpu=None,
                sel_mut_cols_gpu=None,
                sel_miss_rows_gpu=None,
                sel_miss_cols_gpu=None,
                host_blocks_up=[],
                host_blocks_down=[],
                slot_pool_up=None,
                slot_pool_down=None,
                workspaces=self._workspaces,
                staging_up=self._staging_up,
                staging_down=self._staging_down,
            ),
            call_type=TritonCall,
        )

    def _configured_kernel_configs(self, direction: Direction) -> tuple[CsrKernelConfig | CscKernelConfig, ...]:
        plan = self._require_plan(direction)
        if plan.fmt == SparseFormat.CSR:
            return CSR_CANDIDATE_CONFIGS
        return CSC_CANDIDATE_CONFIGS

    def _grid_for(self, direction: Direction) -> list[list[_TritonHostBlock | None]]:
        return self._host_blocks_up if direction == Direction.UP else self._host_blocks_down

    def _ops_for(self, direction: Direction) -> list[list[_TritonOp]]:
        return self._ops_up if direction == Direction.UP else self._ops_down

    def _scratch_streams_for(self, direction: Direction) -> list[list[torch.cuda.Stream]]:
        return self._scratch_streams_up_by_level if direction == Direction.UP else self._scratch_streams_down_by_level

    def _config_for(self, direction: Direction) -> CsrKernelConfig | CscKernelConfig:
        config = self._config_up if direction == Direction.UP else self._config_down
        if config is None:
            raise RuntimeError(f"Triton {direction.value} kernel config is not initialized")
        return config

    def _scratch_plan_for(self, direction: Direction) -> list[_TritonScratchLevelPlan]:
        return self._scratch_plan_up if direction == Direction.UP else self._scratch_plan_down

    def _ensure_workspace(self, direction: Direction, *, graph: bool) -> _DirectionWorkspace:
        if direction == Direction.UP:
            ws = self._workspaces.graph_up if graph else self._workspaces.dynamic_up
        else:
            ws = self._workspaces.graph_down if graph else self._workspaces.dynamic_down
        if ws is None:
            if graph:
                mode = "graph_up" if direction == Direction.UP else "graph_down"
                raise RuntimeError(f"Triton {mode} workspace is not initialized")
            with torch.cuda.stream(self._root_stream):
                ws = self._alloc_workspace(direction)
            if direction == Direction.UP:
                self._workspaces.dynamic_up = ws
            else:
                self._workspaces.dynamic_down = ws
            self._sync_retained_root()
            self._bump_retained_epoch()
        return ws

    def _staging_for(self, direction: Direction) -> _DirectionStaging:
        staging = self._staging_up if direction == Direction.UP else self._staging_down
        if staging is None:
            staging = _DirectionStaging()
            if direction == Direction.UP:
                self._staging_up = staging
            else:
                self._staging_down = staging
            self._sync_retained_root()
            self._bump_retained_epoch()
        return staging

    def _clear_staging(self) -> None:
        had_staging = self._staging_up is not None or self._staging_down is not None
        self._staging_up = None
        self._staging_down = None
        self._sync_retained_root()
        if had_staging:
            self._bump_retained_epoch()

    def _sync_retained_root(self) -> None:
        retained = self._retained_mem
        retained.sel_mut_rows_gpu = self._sel_mut_rows_gpu
        retained.sel_mut_cols_gpu = self._sel_mut_cols_gpu
        retained.sel_miss_rows_gpu = self._sel_miss_rows_gpu
        retained.sel_miss_cols_gpu = self._sel_miss_cols_gpu
        retained.host_blocks_up = [
            (block.indices, block.indptr)
            for row in self._host_blocks_up
            for block in row
            if block is not None
        ]
        retained.host_blocks_down = [
            (block.indices, block.indptr)
            for row in self._host_blocks_down
            for block in row
            if block is not None
        ]
        retained.slot_pool_up = self._slot_pool if self._plan_up is not None else None
        retained.slot_pool_down = self._slot_pool if self._plan_down is not None else None
        retained.workspaces = self._workspaces
        retained.staging_up = self._staging_up
        retained.staging_down = self._staging_down

    @contextmanager
    def _caller_root_scope(self):
        with torch.cuda.device(self._torch_device):
            with torch.cuda.stream(self._caller_stream):
                self._caller_to_root_event.record(self._caller_stream)
            self._root_stream.wait_event(self._caller_to_root_event)
            try:
                yield
            finally:
                with torch.cuda.stream(self._root_stream):
                    self._root_to_caller_event.record(self._root_stream)
                self._caller_stream.wait_event(self._root_to_caller_event)

    def _ensure_staging_tensor(self, staging: _DirectionStaging, attr: str, shape: tuple[int, ...]) -> torch.Tensor:
        value = getattr(staging, attr)
        if value is None:
            value = torch.zeros(shape, device=self._torch_device, dtype=self._torch_dtype)
            setattr(staging, attr, value)
            self._bump_retained_epoch()
        return value

    def _materialize_stored_block(
        self,
        direction: Direction,
        *,
        dst_level: int,
        src_level: int,
    ) -> sp.spmatrix | None:
        plan = self._plan_for(direction)
        if plan is None:
            return None
        stored = self._stored_matrix(direction, dst_level=dst_level, src_level=src_level)
        if stored.nnz == 0:
            return None
        match plan.fmt:
            case SparseFormat.CSR:
                return stored.tocsr()
            case SparseFormat.CSC:
                return stored.tocsc()
            case _:
                raise ValueError(f"Unsupported Triton sparse format: {plan.fmt.value}")

    def _scan_slot_struct_dtypes(self) -> tuple[np.dtype, np.dtype]:
        indices_dtype = np.dtype(np.int32)
        indptr_dtype = np.dtype(np.int32)
        H = len(self._level_offsets) - 1
        for direction in self._configured_directions():
            plan = self._plan_for(direction)
            if plan is None:
                continue
            store_actual = self._store_blocks_up if direction == Direction.UP else self._store_blocks_down
            if not store_actual:
                continue
            for dst_level, src_level, _row_index in iter_direction_level_pairs(direction, H):
                stored = self._stored_matrix(direction, dst_level=dst_level, src_level=src_level)
                if stored.nnz == 0:
                    continue
                block_indptr_dtype, block_indices_dtype = _layout_struct_dtypes(
                    plan.fmt.value,
                    nrows=int(stored.shape[0]),
                    ncols=int(stored.shape[1]),
                    nnz=int(stored.nnz),
                )
                if block_indices_dtype == np.dtype(np.int64):
                    indices_dtype = np.dtype(np.int64)
                if block_indptr_dtype == np.dtype(np.int64):
                    indptr_dtype = np.dtype(np.int64)
        return indices_dtype, indptr_dtype

    def _register_host_tensor(self, values: np.ndarray, *, dtype: np.dtype, label: str) -> tuple[np.ndarray, int]:
        import cupyx

        arr = np.asarray(values)
        _require_struct_dtype(arr.dtype, label=label)
        host = cupyx.empty_pinned(arr.shape, dtype=_require_struct_dtype(dtype, label=label))
        _copy_struct_checked(host, arr, label=label)
        tensor = torch.from_numpy(host)
        key = int(self._next_host_tensor_key)
        self._next_host_tensor_key += 1
        self._host_tensors[key] = tensor
        return host, key

    def _pin_host_block(
        self,
        sparse: sp.spmatrix,
    ) -> _TritonHostBlock | None:
        if sparse.nnz == 0:
            return None
        fmt = str(getattr(sparse, "format", "")).lower()
        if fmt not in {"csr", "csc"}:
            raise ValueError(f"Unsupported Triton sparse format: {fmt!r}")
        indices = np.asarray(sparse.indices)
        indptr = np.asarray(sparse.indptr)
        nrows, ncols = sparse.shape
        indices_host, indices_key = self._register_host_tensor(
            indices,
            dtype=self._slot_indices_dtype,
            label="Triton host block indices",
        )
        indptr_host, indptr_key = self._register_host_tensor(
            indptr,
            dtype=self._slot_indptr_dtype,
            label="Triton host block indptr",
        )
        return _TritonHostBlock(
            indices=indices_host,
            indptr=indptr_host,
            nrows=int(nrows),
            ncols=int(ncols),
            nnz=int(sparse.nnz),
            fmt=parse_sparse_format(fmt.upper()),
            keepalive_key=(indices_key, indptr_key),
        )

    def _build_direction_host_blocks(self, direction: Direction) -> list[list[_TritonHostBlock | None]]:
        H = len(self._level_offsets) - 1
        rows = [
            [None] * (dst_level if direction == Direction.UP else max(H - dst_level - 1, 0))
            for dst_level in range(H)
        ]
        plan = self._plan_for(direction)
        if plan is None:
            return rows

        store_actual = self._store_blocks_up if direction == Direction.UP else self._store_blocks_down
        if store_actual:
            for dst_level, src_level, row_index in iter_direction_level_pairs(direction, H):
                sparse = self._materialize_stored_block(direction, dst_level=dst_level, src_level=src_level)
                if sparse is None:
                    continue
                rows[dst_level][row_index] = self._pin_host_block(sparse)
                del sparse
            return rows

        owner_direction = Direction.UP if (self._up_ops_owner if direction == Direction.UP else self._down_ops_owner) == "up" else Direction.DOWN
        owner_grid = self._grid_for(owner_direction)
        for dst_level, src_level, row_index in iter_direction_level_pairs(direction, H):
            owner_dst = dst_level if owner_direction == direction else src_level
            owner_src = src_level if owner_direction == direction else dst_level
            owner_row_index = owner_src if owner_direction == Direction.UP else owner_src - owner_dst - 1
            owner_block = owner_grid[owner_dst][owner_row_index]
            rows[dst_level][row_index] = None if owner_block is None else owner_block.transpose_alias()
        return rows

    def _slot_lengths(self) -> tuple[int, int]:
        max_indices = 0
        max_indptr = 0
        for grid in (self._host_blocks_up, self._host_blocks_down):
            for row in grid:
                for block in row:
                    if block is None:
                        continue
                    max_indices = max(max_indices, int(block.indices.size))
                    max_indptr = max(max_indptr, int(block.indptr.size))
        return max_indices, max_indptr

    def _alloc_slot_pool(self) -> _TritonSlotPool:
        max_indices, max_indptr = self._slot_lengths()
        with torch.cuda.device(self._torch_device):
            return _TritonSlotPool(
                slots=[
                    _TritonSlotBuffers(
                        indices=torch.zeros((max_indices,), device=self._torch_device, dtype=_torch_int_dtype(self._slot_indices_dtype)),
                        indptr=torch.zeros((max_indptr,), device=self._torch_device, dtype=_torch_int_dtype(self._slot_indptr_dtype)),
                    )
                    for _ in range(self._ring_buffer_size)
                ]
            )

    def _build_direction_ops(self, direction: Direction) -> list[list[_TritonOp]]:
        H = len(self._level_offsets) - 1
        ops: list[list[_TritonOp]] = [[] for _ in range(H)]
        grid = self._grid_for(direction)
        if self._slot_pool is None:
            raise RuntimeError("slot pool is not initialized")
        prev_in_slot: dict[int, tuple[int, int]] = {}
        seq_idx = 0
        if direction == Direction.UP:
            dst_iter = range(1, H)
        else:
            dst_iter = range(H - 2, -1, -1)
        for dst_level in dst_iter:
            if direction == Direction.UP:
                src_iter = range(dst_level)
            else:
                src_iter = range(H - 1, dst_level, -1)
            for src_level in src_iter:
                row_index = src_level if direction == Direction.UP else src_level - dst_level - 1
                block = grid[dst_level][row_index]
                if block is None:
                    continue
                slot = int(seq_idx % self._ring_buffer_size)
                slot_buffers = self._slot_pool.slots[slot]
                op_idx = len(ops[dst_level])
                ops[dst_level].append(
                    _TritonOp(
                        src_level=src_level,
                        dst_level=dst_level,
                        slot=slot,
                        block=_TritonSlotBlock(
                            indices=slot_buffers.indices,
                            indptr=slot_buffers.indptr,
                            nrows=int(block.nrows),
                            ncols=int(block.ncols),
                            nnz=int(block.nnz),
                            fmt=block.fmt,
                        ),
                        host_block=block,
                        nnz=int(block.nnz),
                        prev_in_slot=prev_in_slot.get(slot),
                    )
                )
                prev_in_slot[slot] = (dst_level, op_idx)
                seq_idx += 1
        return ops

    def _host_block_summary(self, direction: Direction) -> tuple[tuple[str, int], ...]:
        plan = self._plan_for(direction)
        if plan is None:
            return (
                ("stored_blocks", 0),
                ("alias_blocks", 0),
                ("empty_blocks", 0),
                ("indices_bytes", 0),
                ("indptr_bytes", 0),
            )
        store_actual = self._store_blocks_up if direction == Direction.UP else self._store_blocks_down
        grid = self._grid_for(direction)
        stored_blocks = 0
        alias_blocks = 0
        empty_blocks = 0
        indices_bytes = 0
        indptr_bytes = 0
        for dst_level, src_level, row_index in iter_direction_level_pairs(direction, len(self._level_offsets) - 1):
            block = grid[dst_level][row_index]
            if block is None:
                empty_blocks += 1
                continue
            if not store_actual:
                alias_blocks += 1
                continue
            stored_blocks += 1
            indices_bytes += int(block.indices.nbytes)
            indptr_bytes += int(block.indptr.nbytes)
        return (
            ("stored_blocks", stored_blocks),
            ("alias_blocks", alias_blocks),
            ("empty_blocks", empty_blocks),
            ("indices_bytes", indices_bytes),
            ("indptr_bytes", indptr_bytes),
        )

    def _log_block_memory(self, direction: Direction) -> None:
        if not self._logger.isEnabledFor(logging.DEBUG):
            return
        plan = self._plan_for(direction)
        if plan is None:
            return
        store_actual = self._store_blocks_up if direction == Direction.UP else self._store_blocks_down
        grid = self._grid_for(direction)
        bucket = "host_blocks_up" if direction == Direction.UP else "host_blocks_down"
        total_rows = 0
        total_cols = 0
        total_nnz = 0

        for dst_level, src_level, row_index in iter_direction_level_pairs(direction, len(self._level_offsets) - 1):
            block = grid[dst_level][row_index]
            if block is None:
                continue
            alias = not store_actual
            total_rows += int(block.nrows)
            total_cols += int(block.ncols)
            total_nnz += int(block.nnz)
            block_indices = 0 if alias else int(block.indices.nbytes)
            block_indptr = 0 if alias else int(block.indptr.nbytes)
            self._logger.debug(
                "Triton %s dir=%s dst=%d src=%d fmt=%s rows=%d cols=%d nnz=%d alias=%s indices_bytes=%d indptr_bytes=%d",
                bucket,
                direction.value,
                dst_level,
                src_level,
                block.fmt.value,
                block.nrows,
                block.ncols,
                block.nnz,
                alias,
                block_indices,
                block_indptr,
            )

        summary = dict(self._host_block_summary(direction))
        self._logger.debug(
            "Triton %s dir=%s rows=%d cols=%d nnz=%d stored_blocks=%d alias_blocks=%d empty_blocks=%d indices_bytes=%d indptr_bytes=%d",
            bucket,
            direction.value,
            total_rows,
            total_cols,
            total_nnz,
            int(summary["stored_blocks"]),
            int(summary["alias_blocks"]),
            int(summary["empty_blocks"]),
            int(summary["indices_bytes"]),
            int(summary["indptr_bytes"]),
        )
        if self._slot_pool is not None:
            slot_indices = int(sum(_torch_nbytes(slot.indices) for slot in self._slot_pool.slots))
            slot_indptr = int(sum(_torch_nbytes(slot.indptr) for slot in self._slot_pool.slots))
            self._logger.debug(
                "Triton slot_pool slots=%d indices_bytes=%d indptr_bytes=%d",
                len(self._slot_pool.slots),
                slot_indices,
                slot_indptr,
            )

    def _resolve_scratch_levels(self, direction: Direction) -> frozenset[int]:
        plan = self._require_plan(direction)
        token = str(plan.scratch)
        H = len(self._level_offsets) - 1
        if token == "none":
            return frozenset()
        if token == "all":
            return frozenset(range(H))
        levels = {int(piece) for piece in token.split("|")}
        invalid = sorted(level for level in levels if level < 0 or level >= H)
        if invalid:
            raise ValueError(
                f"Triton scratch levels out of range for {direction.value}: {invalid}; valid range is [0, {H})"
            )
        return frozenset(levels)

    def _build_scratch_level_plans(self, direction: Direction) -> list[_TritonScratchLevelPlan]:
        ops_by_level = self._ops_for(direction)
        enabled_levels = self._resolve_scratch_levels(direction)
        plans: list[_TritonScratchLevelPlan] = []
        for dst_level, ops in enumerate(ops_by_level):
            if dst_level not in enabled_levels or not ops:
                plans.append(
                    _TritonScratchLevelPlan(
                        enabled=False,
                        reduce_order=(),
                    )
                )
                continue
            reduce_order = tuple(
                sorted(
                    range(len(ops)),
                    key=lambda idx: (int(ops[idx].nnz), int(ops[idx].src_level)),
                )
            )
            plans.append(
                _TritonScratchLevelPlan(
                    enabled=True,
                    reduce_order=reduce_order,
                )
            )
        return plans

    def _build_scratch_streams(self, direction: Direction) -> list[list[torch.cuda.Stream]]:
        with torch.cuda.device(self._torch_device):
            return [
                [torch.cuda.Stream() for _ in self._ops_for(direction)[dst_level]]
                if self._scratch_plan_for(direction)[dst_level].enabled
                else []
                for dst_level in range(len(self._level_offsets) - 1)
            ]

    def _alloc_workspace(self, direction: Direction) -> _DirectionWorkspace:
        if self._torch_dtype not in {torch.float32, torch.float64}:
            raise ValueError(f"Unsupported Triton dtype: {self._torch_dtype}")

        with torch.cuda.device(self._torch_device):
            H = len(self._level_offsets) - 1
            scratch_plans = self._scratch_plan_for(direction)
            ops_by_level = self._ops_for(direction)
            level_offsets = [int(v) for v in self._level_offsets]
            node_state = torch.zeros((self._num_nodes,), device=self._torch_device, dtype=self._torch_dtype)
            level_views = [node_state[level_offsets[h] : level_offsets[h + 1]] for h in range(H)]
            copy_done_by_level = [[torch.cuda.Event() for _ in ops] for ops in ops_by_level]
            compute_done_by_level = [[torch.cuda.Event() for _ in ops] for ops in ops_by_level]
            scratch_done_events_by_level: list[list[torch.cuda.Event]] = []
            scratch_views_by_level: list[list[torch.Tensor]] = []
            for h in range(H):
                if h >= len(scratch_plans) or not scratch_plans[h].enabled:
                    scratch_done_events_by_level.append([])
                    scratch_views_by_level.append([])
                    continue
                level_size = level_views[h].shape[0]
                count = len(ops_by_level[h])
                scratch_done_events_by_level.append([torch.cuda.Event() for _ in range(count)])
                scratch_views_by_level.append(
                    [
                        torch.zeros((level_size,), device=self._torch_device, dtype=self._torch_dtype)
                        for _ in range(count)
                    ]
                )
            input_len = self._num_samples if direction == Direction.UP else self._num_mutations
            return _DirectionWorkspace(
                direction=direction,
                launch_event=torch.cuda.Event(),
                level_done_events=[torch.cuda.Event() for _ in range(H)],
                copy_done_by_level=copy_done_by_level,
                compute_done_by_level=compute_done_by_level,
                node_state=node_state,
                level_views=level_views,
                scratch_done_events_by_level=scratch_done_events_by_level,
                scratch_views_by_level=scratch_views_by_level,
                input_primary=torch.zeros((input_len,), device=self._torch_device, dtype=self._torch_dtype),
            )

    def _autotune_key(self, direction: Direction) -> tuple[object, ...]:
        device = torch.cuda.get_device_properties(self._torch_device)
        plan = self._require_plan(direction)
        ops_by_level = self._ops_for(direction)
        return (
            device.name,
            int(device.major),
            int(device.minor),
            str(self._torch_dtype),
            self._slot_indices_dtype.name,
            self._slot_indptr_dtype.name,
            direction.value,
            plan.fmt.value,
            plan.scratch,
            _structure_signature(ops_by_level),
        )

    def _seed_workspace(
        self,
        ws: _DirectionWorkspace,
        staging: _DirectionStaging,
        *,
        init_mode: InitMode,
        has_miss_input: bool,
    ) -> None:
        with torch.cuda.stream(self._root_stream):
            ws.node_state.zero_()
            match init_mode:
                case InitMode.NONE:
                    pass
                case InitMode.XTX:
                    if staging.xtx_bias is None:
                        if self._coalescence_counts is None:
                            raise ValueError("init_mode=xtx requires GRG coalescence counts")
                        staging.xtx_bias = torch.from_numpy(
                            (2.0 * self._coalescence_counts.astype(self._dtype, copy=False)).reshape(self._num_nodes)
                        ).to(device=self._torch_device, dtype=self._torch_dtype)
                        self._bump_retained_epoch()
                    ws.node_state.add_(staging.xtx_bias)
                case InitMode.VECTOR:
                    if staging.init_vector is None:
                        raise RuntimeError("Missing init vector buffer in Triton workspace")
                    ws.node_state.add_(staging.init_vector[0])
                case InitMode.MATRIX:
                    if staging.init_matrix is None:
                        raise RuntimeError("Missing init matrix buffer in Triton workspace")
                    ws.node_state.add_(staging.init_matrix)
                case _:
                    raise ValueError(f"Unknown init mode: {init_mode!r}")

            if ws.direction == Direction.UP:
                ws.node_state[: self._num_samples].add_(ws.input_primary)
            else:
                if self._sel_mut_rows_gpu.numel() > 0:
                    ws.node_state.index_add_(
                        0,
                        self._sel_mut_cols_gpu,
                        ws.input_primary.index_select(0, self._sel_mut_rows_gpu),
                    )
                if has_miss_input:
                    if staging.input_miss is None:
                        raise RuntimeError("Missing DOWN miss buffer in Triton workspace")
                    if self._sel_miss_rows_gpu.numel() > 0:
                        ws.node_state.index_add_(
                            0,
                            self._sel_miss_cols_gpu,
                            staging.input_miss.index_select(0, self._sel_miss_rows_gpu),
                        )

    def _launch_op(
        self,
        op: _TritonOp,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        config: CsrKernelConfig | CscKernelConfig,
    ) -> None:
        launch_block(
            block=op.block,
            x=x,
            y=y,
            config=config,
            fp64_acc=self._torch_dtype == torch.float64,
        )

    def _host_tensor_pair(self, op: _TritonOp) -> tuple[torch.Tensor, torch.Tensor]:
        idx_key, ptr_key = op.host_block.keepalive_key
        return self._host_tensors[idx_key], self._host_tensors[ptr_key]

    def _prev_compute_event(self, ws: _DirectionWorkspace, op: _TritonOp) -> torch.cuda.Event | None:
        if op.prev_in_slot is None:
            return None
        prev_dst, prev_op_idx = op.prev_in_slot
        return ws.compute_done_by_level[prev_dst][prev_op_idx]

    def _copy_host_block_to_slot(self, ws: _DirectionWorkspace, dst_level: int, op_idx: int, op: _TritonOp) -> None:
        copy_stream = self._slot_copy_streams[op.slot]
        host_indices, host_indptr = self._host_tensor_pair(op)
        with torch.cuda.stream(copy_stream):
            prev_event = self._prev_compute_event(ws, op)
            if prev_event is not None:
                copy_stream.wait_event(prev_event)
            op.block.indices[: int(op.host_block.indices.size)].copy_(host_indices, non_blocking=True)
            op.block.indptr[: int(op.host_block.indptr.size)].copy_(host_indptr, non_blocking=True)
            ws.copy_done_by_level[dst_level][op_idx].record(copy_stream)

    def _join_wavefront_to_root(self, ws: _DirectionWorkspace) -> None:
        with torch.cuda.stream(self._root_stream):
            for event in ws.level_done_events:
                self._root_stream.wait_event(event)

    def _enqueue_wavefront(
        self,
        ws: _DirectionWorkspace,
        *,
        config: CsrKernelConfig | CscKernelConfig,
    ) -> None:
        H = len(self._level_offsets) - 1
        ops_by_level = self._ops_for(ws.direction)
        scratch_plans = self._scratch_plan_for(ws.direction)
        scratch_streams_by_level = self._scratch_streams_for(ws.direction)
        if ws.direction == Direction.UP:
            seed_level = 0
            level_iter = range(1, H)
        else:
            seed_level = H - 1
            level_iter = range(H - 2, -1, -1)

        with torch.cuda.stream(self._root_stream):
            ws.launch_event.record(self._root_stream)
        for stream in self._slot_copy_streams:
            stream.wait_event(ws.launch_event)
        for stream in self._level_streams:
            stream.wait_event(ws.launch_event)
        for scratch_streams in scratch_streams_by_level:
            for stream in scratch_streams:
                stream.wait_event(ws.launch_event)

        if H > 0:
            seed_stream = self._level_streams[seed_level]
            with torch.cuda.stream(seed_stream):
                ws.level_done_events[seed_level].record(seed_stream)

        for dst_level in level_iter:
            stream = self._level_streams[dst_level]
            ops = ops_by_level[dst_level]
            scratch_plan = scratch_plans[dst_level]
            if scratch_plan.enabled:
                scratch_streams = scratch_streams_by_level[dst_level]
                scratch_done_events = ws.scratch_done_events_by_level[dst_level]
                scratch_views = ws.scratch_views_by_level[dst_level]
                for helper_idx, op in enumerate(ops):
                    self._copy_host_block_to_slot(ws, dst_level, helper_idx, op)
                    helper_stream = scratch_streams[helper_idx]
                    helper_view = scratch_views[helper_idx]
                    with torch.cuda.stream(helper_stream):
                        helper_stream.wait_event(ws.level_done_events[op.src_level])
                        helper_stream.wait_event(ws.copy_done_by_level[dst_level][helper_idx])
                        helper_view.zero_()
                        self._launch_op(op, x=ws.level_views[op.src_level], y=helper_view, config=config)
                        ws.compute_done_by_level[dst_level][helper_idx].record(helper_stream)
                        scratch_done_events[helper_idx].record(helper_stream)
                with torch.cuda.stream(stream):
                    for helper_idx in scratch_plan.reduce_order:
                        stream.wait_event(scratch_done_events[helper_idx])
                        ws.level_views[dst_level].add_(scratch_views[helper_idx])
                    ws.level_done_events[dst_level].record(stream)
                continue

            with torch.cuda.stream(stream):
                for op_idx, op in enumerate(ops):
                    self._copy_host_block_to_slot(ws, dst_level, op_idx, op)
                    stream.wait_event(ws.level_done_events[op.src_level])
                    stream.wait_event(ws.copy_done_by_level[dst_level][op_idx])
                    self._launch_op(op, x=ws.level_views[op.src_level], y=ws.level_views[dst_level], config=config)
                    ws.compute_done_by_level[dst_level][op_idx].record(stream)
                ws.level_done_events[dst_level].record(stream)

        self._join_wavefront_to_root(ws)

    def _enqueue_wavefront_nvtx(
        self,
        ws: _DirectionWorkspace,
        *,
        config: CsrKernelConfig | CscKernelConfig,
    ) -> None:
        tracer = self._nvtx
        if tracer is None:
            raise RuntimeError("Triton NVTX tracer is not initialized")

        H = len(self._level_offsets) - 1
        ops_by_level = self._ops_for(ws.direction)
        scratch_plans = self._scratch_plan_for(ws.direction)
        scratch_streams_by_level = self._scratch_streams_for(ws.direction)
        if ws.direction == Direction.UP:
            seed_level = 0
            level_iter = range(1, H)
        else:
            seed_level = H - 1
            level_iter = range(H - 2, -1, -1)

        with tracer.range("wavefront", dir=ws.direction.value):
            with torch.cuda.stream(self._root_stream):
                ws.launch_event.record(self._root_stream)
                tracer.mark("event.record_fork", dir=ws.direction.value)
            for stream in self._slot_copy_streams:
                stream.wait_event(ws.launch_event)
            for stream in self._level_streams:
                stream.wait_event(ws.launch_event)
            for scratch_streams in scratch_streams_by_level:
                for stream in scratch_streams:
                    stream.wait_event(ws.launch_event)

            if H > 0:
                seed_stream = self._level_streams[seed_level]
                with torch.cuda.stream(seed_stream):
                    ws.level_done_events[seed_level].record(seed_stream)
                    tracer.mark("event.record_ready", dir=ws.direction.value, level=seed_level)

            for dst_level in level_iter:
                stream = self._level_streams[dst_level]
                ops = ops_by_level[dst_level]
                scratch_plan = scratch_plans[dst_level]
                with tracer.range(
                    "level",
                    dir=ws.direction.value,
                    dst=dst_level,
                    ops=len(ops),
                    scratch=scratch_plan.enabled,
                ):
                    if scratch_plan.enabled:
                        scratch_streams = scratch_streams_by_level[dst_level]
                        scratch_done_events = ws.scratch_done_events_by_level[dst_level]
                        scratch_views = ws.scratch_views_by_level[dst_level]
                        for helper_idx, op in enumerate(ops):
                            with tracer.range(
                                "copy_to_slot",
                                dir=ws.direction.value,
                                dst=dst_level,
                                src=op.src_level,
                                slot=op.slot,
                            ):
                                self._copy_host_block_to_slot(ws, dst_level, helper_idx, op)
                            helper_stream = scratch_streams[helper_idx]
                            helper_view = scratch_views[helper_idx]
                            with torch.cuda.stream(helper_stream):
                                tracer.mark(
                                    "wait_ready",
                                    dir=ws.direction.value,
                                    dst=dst_level,
                                    src=op.src_level,
                                )
                                helper_stream.wait_event(ws.level_done_events[op.src_level])
                                tracer.mark(
                                    "wait_copy_done",
                                    dir=ws.direction.value,
                                    dst=dst_level,
                                    src=op.src_level,
                                    slot=op.slot,
                                )
                                helper_stream.wait_event(ws.copy_done_by_level[dst_level][helper_idx])
                                with tracer.range(
                                    "helper_launch",
                                    dir=ws.direction.value,
                                    dst=dst_level,
                                    src=op.src_level,
                                    helper=helper_idx,
                                ):
                                    helper_view.zero_()
                                    with tracer.range(
                                        "launch",
                                        dir=ws.direction.value,
                                        dst=dst_level,
                                        src=op.src_level,
                                        helper=helper_idx,
                                        fmt=op.block.fmt.value,
                                        rows=op.block.nrows,
                                        cols=op.block.ncols,
                                        nnz=op.nnz,
                                    ):
                                        self._launch_op(op, x=ws.level_views[op.src_level], y=helper_view, config=config)
                                    ws.compute_done_by_level[dst_level][helper_idx].record(helper_stream)
                                    scratch_done_events[helper_idx].record(helper_stream)
                                    tracer.mark(
                                        "event.record_scratch_done",
                                        dir=ws.direction.value,
                                        dst=dst_level,
                                        src=op.src_level,
                                        helper=helper_idx,
                                    )
                        with torch.cuda.stream(stream):
                            for helper_idx in scratch_plan.reduce_order:
                                src_level = ops[helper_idx].src_level
                                tracer.mark(
                                    "wait_scratch_done",
                                    dir=ws.direction.value,
                                    dst=dst_level,
                                    src=src_level,
                                    helper=helper_idx,
                                )
                                stream.wait_event(scratch_done_events[helper_idx])
                                with tracer.range(
                                    "reduce_add",
                                    dir=ws.direction.value,
                                    dst=dst_level,
                                    src=src_level,
                                    helper=helper_idx,
                                ):
                                    ws.level_views[dst_level].add_(scratch_views[helper_idx])
                            ws.level_done_events[dst_level].record(stream)
                            tracer.mark("event.record_ready", dir=ws.direction.value, level=dst_level)
                        continue

                    with torch.cuda.stream(stream):
                        for op_idx, op in enumerate(ops):
                            with tracer.range(
                                "copy_to_slot",
                                dir=ws.direction.value,
                                dst=dst_level,
                                src=op.src_level,
                                slot=op.slot,
                            ):
                                self._copy_host_block_to_slot(ws, dst_level, op_idx, op)
                            tracer.mark(
                                "wait_ready",
                                dir=ws.direction.value,
                                dst=dst_level,
                                src=op.src_level,
                            )
                            stream.wait_event(ws.level_done_events[op.src_level])
                            tracer.mark(
                                "wait_copy_done",
                                dir=ws.direction.value,
                                dst=dst_level,
                                src=op.src_level,
                                slot=op.slot,
                            )
                            stream.wait_event(ws.copy_done_by_level[dst_level][op_idx])
                            with tracer.range(
                                "launch",
                                dir=ws.direction.value,
                                dst=dst_level,
                                src=op.src_level,
                                fmt=op.block.fmt.value,
                                rows=op.block.nrows,
                                cols=op.block.ncols,
                                nnz=op.nnz,
                            ):
                                self._launch_op(op, x=ws.level_views[op.src_level], y=ws.level_views[dst_level], config=config)
                            ws.compute_done_by_level[dst_level][op_idx].record(stream)
                        ws.level_done_events[dst_level].record(stream)
                        tracer.mark("event.record_ready", dir=ws.direction.value, level=dst_level)

            with tracer.range("join_ready", dir=ws.direction.value):
                self._join_wavefront_to_root(ws)

    def _enqueue_wavefront_dispatch(
        self,
        ws: _DirectionWorkspace,
        *,
        config: CsrKernelConfig | CscKernelConfig,
    ) -> None:
        if self._instrumentation:
            self._enqueue_wavefront_nvtx(ws, config=config)
            return
        self._enqueue_wavefront(ws, config=config)

    def _enqueue_output_gather(self, ws: _DirectionWorkspace, staging: _DirectionStaging, *, need_miss_output: bool) -> None:
        with torch.cuda.stream(self._root_stream):
            if ws.direction == Direction.UP:
                if staging.output_main is None:
                    staging.output_main = self._ensure_staging_tensor(staging, "output_main", (self._num_mutations,))
                staging.output_main.zero_()
                if self._sel_mut_rows_gpu.numel() > 0:
                    staging.output_main.index_add_(
                        0,
                        self._sel_mut_rows_gpu,
                        ws.node_state.index_select(0, self._sel_mut_cols_gpu),
                    )
                if need_miss_output:
                    if staging.output_miss is None:
                        staging.output_miss = self._ensure_staging_tensor(staging, "output_miss", (self._num_mutations,))
                    staging.output_miss.zero_()
                    if self._sel_miss_rows_gpu.numel() > 0:
                        staging.output_miss.index_add_(
                            0,
                            self._sel_miss_rows_gpu,
                            ws.node_state.index_select(0, self._sel_miss_cols_gpu),
                        )
            else:
                if staging.output_main is None:
                    staging.output_main = self._ensure_staging_tensor(staging, "output_main", (self._num_samples,))
                staging.output_main.copy_(ws.node_state[: self._num_samples])

    def _tune_once(self, ws: _DirectionWorkspace, config: CsrKernelConfig | CscKernelConfig) -> None:
        with self._caller_root_scope():
            self._seed_workspace(ws, _DirectionStaging(), init_mode=InitMode.NONE, has_miss_input=False)
            self._enqueue_wavefront_dispatch(ws, config=config)
        with torch.cuda.device(self._torch_device):
            self._root_stream.synchronize()

    def _tune_direction(self, direction: Direction, ws: _DirectionWorkspace) -> CsrKernelConfig | CscKernelConfig:
        key = self._autotune_key(direction)
        cached = _AUTOTUNE_CACHE.get(key)
        if cached is not None:
            self._logger.info("Reusing cached Triton autotune config for %s: %s", direction.value, format_config(cached))
            return cached

        with self._caller_root_scope():
            with torch.cuda.stream(self._root_stream):
                tune_input = torch.randn(
                    (self._num_samples if direction == Direction.UP else self._num_mutations,),
                    device=self._torch_device,
                    dtype=self._torch_dtype,
                )
                ws.input_primary.copy_(tune_input)
        with torch.cuda.device(self._torch_device):
            self._root_stream.synchronize()

        results: list[tuple[CsrKernelConfig | CscKernelConfig, float]] = []
        for config in self._configured_kernel_configs(direction):
            self._tune_once(ws, config)
            timing_ms = float(
                triton.testing.do_bench(
                    lambda config=config: self._tune_once(ws, config),
                    warmup=_TUNE_WARMUP_MS,
                    rep=_TUNE_REP_MS,
                )
            )
            results.append((config, timing_ms))

        best_config, best_ms = min(results, key=lambda item: item[1])
        _AUTOTUNE_CACHE[key] = best_config
        self._logger.info(
            "Triton autotune[%s] selected %s (median %.3fms)",
            direction.value,
            format_config(best_config),
            best_ms,
        )
        return best_config

    def _build_wavefront_graph(self, ws: _DirectionWorkspace, config: CsrKernelConfig | CscKernelConfig) -> torch.cuda.CUDAGraph:
        # Warm up outside capture so Triton compilation is not part of the graph.
        with self._caller_root_scope():
            self._seed_workspace(ws, _DirectionStaging(), init_mode=InitMode.NONE, has_miss_input=False)
            self._enqueue_wavefront_dispatch(ws, config=config)
        with torch.cuda.device(self._torch_device):
            self._root_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
        with self._caller_root_scope():
            with torch.cuda.graph(graph, stream=self._root_stream):
                self._enqueue_wavefront_dispatch(ws, config=config)
        with torch.cuda.device(self._torch_device):
            self._root_stream.synchronize()
        return graph

    def _run_column(
        self,
        direction: Direction,
        *,
        primary_col: np.ndarray,
        miss_col: np.ndarray | None,
        init_mode: InitMode,
        init_value: np.ndarray | None,
        need_miss_output: bool,
        emit_all_nodes: bool,
    ) -> tuple[np.ndarray, np.ndarray | None] | np.ndarray:
        config = self._config_for(direction)
        hint = effective_k_hint(instrumentation=self._instrumentation, k_hint=self._require_plan(direction).k_hint)
        use_graph = bool(not self._instrumentation and hint is not None)
        primary_cpu = torch.from_numpy(np.ascontiguousarray(primary_col)).to(dtype=self._torch_dtype)
        miss_cpu = None if miss_col is None else torch.from_numpy(np.ascontiguousarray(miss_col)).to(dtype=self._torch_dtype)
        init_matrix_cpu = None
        if init_mode == InitMode.MATRIX:
            assert init_value is not None
            init_matrix_cpu = torch.from_numpy(np.ascontiguousarray(init_value)).to(dtype=self._torch_dtype)

        tracer = self._nvtx
        with self._caller_root_scope():
            ws = self._ensure_workspace(direction, graph=use_graph)
            staging = self._staging_for(direction)
            with torch.cuda.stream(self._root_stream):
                ws.input_primary.copy_(primary_cpu)
                if miss_cpu is not None:
                    if staging.input_miss is None:
                        staging.input_miss = self._ensure_staging_tensor(staging, "input_miss", (self._num_mutations,))
                    staging.input_miss.copy_(miss_cpu)
                if init_mode == InitMode.VECTOR:
                    assert init_value is not None
                    if staging.init_vector is None:
                        staging.init_vector = self._ensure_staging_tensor(staging, "init_vector", (1,))
                    staging.init_vector.fill_(float(np.asarray(init_value).reshape(())))
                if init_matrix_cpu is not None:
                    if staging.init_matrix is None:
                        staging.init_matrix = self._ensure_staging_tensor(staging, "init_matrix", (self._num_nodes,))
                    staging.init_matrix.copy_(init_matrix_cpu)

            if tracer is not None:
                with tracer.range("execute_singleton", dir=direction.value):
                    with tracer.range("prepare_state", dir=direction.value):
                        self._seed_workspace(ws, staging, init_mode=init_mode, has_miss_input=miss_col is not None)
                    self._enqueue_wavefront_dispatch(ws, config=config)
                    if not emit_all_nodes:
                        with tracer.range("gather_outputs", dir=direction.value):
                            self._enqueue_output_gather(ws, staging, need_miss_output=need_miss_output)
            else:
                self._seed_workspace(ws, staging, init_mode=init_mode, has_miss_input=miss_col is not None)
                if use_graph:
                    if ws.graph is None:
                        raise RuntimeError(f"Missing Triton graph workspace for {direction.value}")
                    with torch.cuda.stream(self._root_stream):
                        ws.graph.replay()
                else:
                    self._enqueue_wavefront_dispatch(ws, config=config)
                if not emit_all_nodes:
                    self._enqueue_output_gather(ws, staging, need_miss_output=need_miss_output)

        if tracer is not None:
            with tracer.range("await_outputs", dir=direction.value):
                with torch.cuda.device(self._torch_device):
                    self._root_stream.synchronize()
        else:
            with torch.cuda.device(self._torch_device):
                self._root_stream.synchronize()

        if emit_all_nodes:
            return ws.node_state.cpu().numpy().copy()
        if direction == Direction.UP:
            if staging.output_main is None:
                raise RuntimeError("Missing Triton main output buffer after gather")
            out_mut = staging.output_main.cpu().numpy().copy()
            out_miss = staging.output_miss.cpu().numpy().copy() if need_miss_output and staging.output_miss is not None else None
            return out_mut, out_miss
        if staging.output_main is None:
            raise RuntimeError("Missing Triton main output buffer after gather")
        return staging.output_main.cpu().numpy().copy()

    def setup(self, setup: BackendSetup) -> None:
        self._workspaces = _WorkspaceCache()
        self._clear_staging()
        self._slot_pool = None
        self._slot_copy_streams = []
        self._host_tensors.clear()
        self._next_host_tensor_key = 0
        self._apply_setup_state(setup)
        if np.dtype(setup.dtype) == np.float64:
            self._torch_dtype = torch.float64
        elif np.dtype(setup.dtype) == np.float32:
            self._torch_dtype = torch.float32
        else:
            raise ValueError(f"Triton backend supports only float32/float64, got {setup.dtype}")

        H = len(self._level_offsets) - 1
        prev_rss_bytes = rss_checkpoint(self._logger, "setup:start", None)
        self._slot_indices_dtype, self._slot_indptr_dtype = self._scan_slot_struct_dtypes()
        self._host_blocks_up = [[] for _ in range(H)]
        self._host_blocks_down = [[] for _ in range(H)]
        self._ops_up = [[] for _ in range(H)]
        self._ops_down = [[] for _ in range(H)]

        with self._caller_root_scope():
            if self._plan_up is not None:
                self._host_blocks_up = self._build_direction_host_blocks(Direction.UP)
                prev_rss_bytes = rss_checkpoint(
                    self._logger,
                    "setup:host_blocks_up_ready",
                    prev_rss_bytes,
                    extras=self._host_block_summary(Direction.UP),
                )
            if self._plan_down is not None:
                self._host_blocks_down = self._build_direction_host_blocks(Direction.DOWN)
                prev_rss_bytes = rss_checkpoint(
                    self._logger,
                    "setup:host_blocks_down_ready",
                    prev_rss_bytes,
                    extras=self._host_block_summary(Direction.DOWN),
                )
            with torch.cuda.stream(self._root_stream):
                self._slot_pool = self._alloc_slot_pool()
                mut_rows, mut_cols = _selector_index_arrays(self._sel_mut)
                miss_rows, miss_cols = _selector_index_arrays(self._sel_miss)
                self._sel_mut_rows_gpu = torch.from_numpy(mut_rows).to(device=self._torch_device)
                self._sel_mut_cols_gpu = torch.from_numpy(mut_cols).to(device=self._torch_device)
                self._sel_miss_rows_gpu = torch.from_numpy(miss_rows).to(device=self._torch_device)
                self._sel_miss_cols_gpu = torch.from_numpy(miss_cols).to(device=self._torch_device)
            if self._plan_up is not None:
                self._ops_up = self._build_direction_ops(Direction.UP)
                self._scratch_plan_up = self._build_scratch_level_plans(Direction.UP)
            if self._plan_down is not None:
                self._ops_down = self._build_direction_ops(Direction.DOWN)
                self._scratch_plan_down = self._build_scratch_level_plans(Direction.DOWN)

        with torch.cuda.device(self._torch_device):
            self._level_streams = [torch.cuda.Stream() for _ in range(H)]
            self._slot_copy_streams = [torch.cuda.Stream() for _ in range(self._ring_buffer_size)]
            self._scratch_streams_up_by_level = self._build_scratch_streams(Direction.UP) if self._plan_up is not None else [[] for _ in range(H)]
            self._scratch_streams_down_by_level = self._build_scratch_streams(Direction.DOWN) if self._plan_down is not None else [[] for _ in range(H)]
        for direction in self._configured_directions():
            self._log_block_memory(direction)
        self._xtx_host = None

        if self._plan_up is not None:
            with self._caller_root_scope():
                with torch.cuda.stream(self._root_stream):
                    tune_up = self._alloc_workspace(Direction.UP)
            self._config_up = self._tune_direction(Direction.UP, tune_up)
            hint = effective_k_hint(instrumentation=self._instrumentation, k_hint=self._plan_up.k_hint)
            if self._instrumentation and self._plan_up.k_hint is not None:
                warn_instrumentation_ignores_k_hint(backend="Triton", direction=Direction.UP, k_hint=int(self._plan_up.k_hint))
            if hint is not None:
                with self._caller_root_scope():
                    with torch.cuda.stream(self._root_stream):
                        self._workspaces.graph_up = self._alloc_workspace(Direction.UP)
                self._workspaces.graph_up.graph = self._build_wavefront_graph(self._workspaces.graph_up, self._config_up)

        if self._plan_down is not None:
            with self._caller_root_scope():
                with torch.cuda.stream(self._root_stream):
                    tune_down = self._alloc_workspace(Direction.DOWN)
            self._config_down = self._tune_direction(Direction.DOWN, tune_down)
            hint = effective_k_hint(instrumentation=self._instrumentation, k_hint=self._plan_down.k_hint)
            if self._instrumentation and self._plan_down.k_hint is not None:
                warn_instrumentation_ignores_k_hint(backend="Triton", direction=Direction.DOWN, k_hint=int(self._plan_down.k_hint))
            if hint is not None:
                with self._caller_root_scope():
                    with torch.cuda.stream(self._root_stream):
                        self._workspaces.graph_down = self._alloc_workspace(Direction.DOWN)
                self._workspaces.graph_down.graph = self._build_wavefront_graph(self._workspaces.graph_down, self._config_down)

        self._A_blocks = []
        self._sel_mut = sp.csr_matrix((0, 0))
        self._sel_miss = sp.csr_matrix((0, 0))
        self._sync_retained_root()
        self._bump_retained_epoch()
        self._assert_setup_memory_contract()
        rss_checkpoint(self._logger, "setup:complete", prev_rss_bytes)

    def _run_direction(
        self,
        direction: Direction,
        primary: np.ndarray,
        *,
        miss: np.ndarray | None,
        init_mode: InitMode,
        init: np.ndarray | None,
        need_miss_output: bool,
        emit_all_nodes: bool,
    ) -> tuple[np.ndarray, np.ndarray | None] | np.ndarray:
        self._require_plan(direction)
        x, k = self._normalize_primary_input(direction=direction, primary=primary)
        if int(k) != 1:
            raise RuntimeError("Triton streamed backend supports runtime k == 1 only; runtime k > 1 is not implemented")
        miss_arr = self._normalize_down_miss_input(miss, k=k) if direction == Direction.DOWN else None
        mode = parse_init_mode(init_mode)
        payload = self._validate_init(mode, init, k)
        hint = effective_k_hint(instrumentation=self._instrumentation, k_hint=self._require_plan(direction).k_hint)
        use_graph = bool(not self._instrumentation and hint is not None)

        init_value = None
        if mode == InitMode.VECTOR:
            assert payload is not None
            init_value = payload[0]
        elif mode == InitMode.MATRIX:
            assert payload is not None
            init_value = payload[:, 0]

        t0 = perf_counter()
        result = self._run_column(
            direction,
            primary_col=x[:, 0],
            miss_col=None if miss_arr is None else miss_arr[:, 0],
            init_mode=mode,
            init_value=init_value,
            need_miss_output=need_miss_output,
            emit_all_nodes=emit_all_nodes,
        )
        self._logger.info("grg=%s direction=%s time=%.3fms", self._grg_name or "<unknown>", direction.value, (perf_counter() - t0) * 1000.0)
        if emit_all_nodes:
            out = np.asarray(result, dtype=self._dtype).reshape(-1, 1)
            out_miss = None
        elif direction == Direction.UP:
            mut_col, miss_col = result
            out = np.asarray(mut_col, dtype=self._dtype).reshape(-1, 1)
            out_miss = None if miss_col is None else np.asarray(miss_col, dtype=self._dtype).reshape(-1, 1)
        else:
            out = np.asarray(result, dtype=self._dtype).reshape(-1, 1)
            out_miss = None

        ws = self._ensure_workspace(direction, graph=use_graph)
        staging = self._staging_for(direction)
        if self._capture_active:
            call = self._call_mem
            assert isinstance(call, TritonCall)
            call.node_values_host = out if emit_all_nodes else None
            call.miss_output_host = out_miss if (direction == Direction.UP and not emit_all_nodes) else None
            active_values: list[object] = [ws]
            if direction == Direction.DOWN and miss_arr is not None and staging.input_miss is not None:
                active_values.append(staging.input_miss)
            if mode == InitMode.VECTOR and staging.init_vector is not None:
                active_values.append(staging.init_vector)
            elif mode == InitMode.MATRIX and staging.init_matrix is not None:
                active_values.append(staging.init_matrix)
            elif mode == InitMode.XTX and staging.xtx_bias is not None:
                active_values.append(staging.xtx_bias)
            if not emit_all_nodes and staging.output_main is not None:
                active_values.append(staging.output_main)
            if direction == Direction.UP and need_miss_output and staging.output_miss is not None:
                active_values.append(staging.output_miss)
            self._publish_call_capture(
                CallCapture(
                    nonce=self._capture_nonce,
                    direction=direction.value,
                    runtime_k=k,
                    active_alloc_keys=self._alloc_keys(*active_values),
                    meta={
                        "emit_all_nodes": bool(emit_all_nodes),
                        "need_miss_output": bool(need_miss_output) if direction == Direction.UP else False,
                        "has_miss_input": bool(miss_arr is not None) if direction == Direction.DOWN else False,
                        "mode": "instrumented" if self._instrumentation else ("graph" if use_graph else "dynamic"),
                    },
                )
            )
        if emit_all_nodes or direction == Direction.DOWN:
            return out
        return out, out_miss

    def run_up(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
        need_miss_output: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        out_mut, out_miss = self._run_direction(
            Direction.UP,
            primary,
            miss=None,
            init_mode=init_mode,
            init=init,
            need_miss_output=need_miss_output,
            emit_all_nodes=False,
        )
        return out_mut, out_miss

    def run_down(
        self,
        primary: np.ndarray,
        *,
        miss: np.ndarray | None = None,
        init_mode: InitMode,
        init: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._run_direction(
            Direction.DOWN,
            primary,
            miss=miss,
            init_mode=init_mode,
            init=init,
            need_miss_output=False,
            emit_all_nodes=False,
        )

    def run_up_nodes(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._run_direction(
            Direction.UP,
            primary,
            miss=None,
            init_mode=init_mode,
            init=init,
            need_miss_output=False,
            emit_all_nodes=True,
        )

    def run_down_nodes(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._run_direction(
            Direction.DOWN,
            primary,
            miss=None,
            init_mode=init_mode,
            init=init,
            need_miss_output=False,
            emit_all_nodes=True,
        )
__all__ = ["TritonBackend"]
