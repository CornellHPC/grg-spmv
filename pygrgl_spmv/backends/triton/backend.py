"""Triton backend using naive CSR/CSC unit-weight SpMV kernels."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from types import SimpleNamespace
from typing import Any
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import triton

from pygrgl_spmv.backends.base import (
    BackendBase,
    BackendSetup,
    estimate_common_host_static_bytes,
    iter_direction_level_pairs,
)
from pygrgl_spmv.backends.memory import RuntimeBytes, StaticBytes
from pygrgl_spmv.backends._nvtx import make_torch_tracer
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


def _ensure_int32_array(values: np.ndarray, *, label: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size:
        max_value = int(arr.max())
        if max_value >= np.iinfo(np.int32).max:
            raise ValueError(f"{label} exceeds int32 range required by Triton kernels")
    return np.asarray(arr, dtype=np.int32)


def _selector_index_arrays(selector: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    coo = selector.tocoo()
    return np.asarray(coo.row, dtype=np.int64), np.asarray(coo.col, dtype=np.int64)


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
class _TritonBlock:
    indices: torch.Tensor
    indptr: torch.Tensor
    nrows: int
    ncols: int
    nnz: int
    fmt: SparseFormat

    def nbytes(self) -> int:
        return _torch_nbytes(self.indices) + _torch_nbytes(self.indptr)

    def transpose_alias(self) -> "_TritonBlock":
        return _TritonBlock(
            indices=self.indices,
            indptr=self.indptr,
            nrows=int(self.ncols),
            ncols=int(self.nrows),
            nnz=int(self.nnz),
            fmt=transpose_compatible_format(self.fmt),
        )


@dataclass(frozen=True)
class _TritonOp:
    src_level: int
    block: _TritonBlock
    nnz: int


@dataclass(frozen=True)
class _TritonScratchLevelPlan:
    enabled: bool
    helper_src_levels: tuple[int, ...]
    helper_op_indices: tuple[int, ...]
    reduce_order: tuple[int, ...]


@dataclass
class _TritonWorkspace:
    direction: Direction
    capture_stream: torch.cuda.Stream
    level_streams: list[torch.cuda.Stream]
    fork_event: torch.cuda.Event
    ready_events: list[torch.cuda.Event]
    node_state: torch.Tensor
    level_views: list[torch.Tensor]
    scratch_streams_by_level: list[list[torch.cuda.Stream]]
    scratch_done_events_by_level: list[list[torch.cuda.Event]]
    scratch_views_by_level: list[list[torch.Tensor]]
    input_primary: torch.Tensor
    input_miss: torch.Tensor | None
    output_main: torch.Tensor
    output_aux: torch.Tensor | None
    init_scalar: torch.Tensor
    init_matrix: torch.Tensor
    graph: torch.cuda.CUDAGraph | None = None

    def nbytes(self) -> int:
        return (
            _torch_nbytes(self.node_state)
            + _torch_nbytes(self.input_primary)
            + _torch_nbytes(self.input_miss)
            + _torch_nbytes(self.output_main)
            + _torch_nbytes(self.output_aux)
            + _torch_nbytes(self.init_scalar)
            + _torch_nbytes(self.init_matrix)
            + sum(_torch_nbytes(buf) for level_bufs in self.scratch_views_by_level for buf in level_bufs)
        )


class TritonBackend(BackendBase):
    """GPU backend using Triton naive CSR/CSC kernels on singleton vectors."""

    @staticmethod
    def plan(
        *,
        fmt: str | SparseFormat = SparseFormat.CSR,
        store: str | StoredMatrix = StoredMatrix.N,
        k_hint: int = 1,
    ) -> TritonPlan:
        return TritonPlan(store=parse_store(store), fmt=parse_sparse_format(fmt), k_hint=k_hint)

    def __init__(
        self,
        *,
        pair: TritonPlanPair,
        log_level: str = "WARNING",
        instrumentation: bool = False,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Triton backend requires torch.cuda.is_available()")

        super().__init__(
            plan_up=pair.plan_up,
            plan_down=pair.plan_down,
            log_level=log_level,
            instrumentation=instrumentation,
        )
        self._device = torch.device("cuda")
        self._torch_dtype = torch.float64
        self._blocks_up: list[list[_TritonBlock | None]] = []
        self._blocks_down: list[list[_TritonBlock | None]] = []
        self._ops_up: list[list[_TritonOp]] = []
        self._ops_down: list[list[_TritonOp]] = []
        self._sample_perm_gpu = torch.empty(0, device=self._device, dtype=torch.int64)
        self._inv_sample_perm_gpu = torch.empty(0, device=self._device, dtype=torch.int64)
        self._sel_mut_rows_gpu = torch.empty(0, device=self._device, dtype=torch.int64)
        self._sel_mut_cols_gpu = torch.empty(0, device=self._device, dtype=torch.int64)
        self._sel_miss_rows_gpu = torch.empty(0, device=self._device, dtype=torch.int64)
        self._sel_miss_cols_gpu = torch.empty(0, device=self._device, dtype=torch.int64)
        self._xtx_gpu: torch.Tensor | None = None
        self._workspace_bytes_up = 0
        self._workspace_bytes_down = 0
        self._workspaces = SimpleNamespace(
            dynamic_up=None,
            graph_up=None,
            dynamic_down=None,
            graph_down=None,
        )
        self._config_up: CsrKernelConfig | CscKernelConfig | None = None
        self._config_down: CsrKernelConfig | CscKernelConfig | None = None
        self._scratch_plan_up: list[_TritonScratchLevelPlan] = []
        self._scratch_plan_down: list[_TritonScratchLevelPlan] = []
        self._nvtx = make_torch_tracer("grg.triton", torch) if self._instrumentation else None

    def _configured_kernel_configs(self, direction: Direction) -> tuple[CsrKernelConfig | CscKernelConfig, ...]:
        plan = self._require_plan(direction)
        if plan.fmt == SparseFormat.CSR:
            return CSR_CANDIDATE_CONFIGS
        return CSC_CANDIDATE_CONFIGS

    def _grid_for(self, direction: Direction) -> list[list[_TritonBlock | None]]:
        return self._blocks_up if direction == Direction.UP else self._blocks_down

    def _ops_for(self, direction: Direction) -> list[list[_TritonOp]]:
        return self._ops_up if direction == Direction.UP else self._ops_down

    def _config_for(self, direction: Direction) -> CsrKernelConfig | CscKernelConfig:
        config = self._config_up if direction == Direction.UP else self._config_down
        if config is None:
            raise RuntimeError(f"Triton {direction.value} kernel config is not initialized")
        return config

    def _scratch_plan_for(self, direction: Direction) -> list[_TritonScratchLevelPlan]:
        return self._scratch_plan_up if direction == Direction.UP else self._scratch_plan_down

    def _workspace_for(self, direction: Direction, *, graph: bool) -> _TritonWorkspace:
        key = f"{'graph' if graph else 'dynamic'}_{direction.value}"
        ws = getattr(self._workspaces, key)
        if ws is None:
            raise RuntimeError(f"Triton {key} workspace is not initialized")
        return ws

    def _operator_matrix(self, direction: Direction, *, dst_level: int, src_level: int) -> sp.spmatrix:
        if direction == Direction.UP:
            return self._A_blocks[dst_level][src_level]
        return self._A_blocks[src_level][dst_level].T

    def _upload_block(self, matrix: sp.spmatrix, *, fmt: SparseFormat) -> _TritonBlock | None:
        if matrix.nnz == 0:
            return None
        match fmt:
            case SparseFormat.CSR:
                sparse = matrix.tocsr()
                indices = _ensure_int32_array(sparse.indices, label="CSR indices")
                indptr = _ensure_int32_array(sparse.indptr, label="CSR indptr")
                nrows, ncols = sparse.shape
            case SparseFormat.CSC:
                sparse = matrix.tocsc()
                indices = _ensure_int32_array(sparse.indices, label="CSC rowidx")
                indptr = _ensure_int32_array(sparse.indptr, label="CSC colptr")
                nrows, ncols = sparse.shape
            case _:
                raise ValueError(f"Unsupported Triton sparse format: {fmt.value}")
        return _TritonBlock(
            indices=torch.from_numpy(indices).to(device=self._device),
            indptr=torch.from_numpy(indptr).to(device=self._device),
            nrows=int(nrows),
            ncols=int(ncols),
            nnz=int(sparse.nnz),
            fmt=fmt,
        )

    def _build_direction_blocks(self, direction: Direction) -> list[list[_TritonBlock | None]]:
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
                matrix = self._operator_matrix(direction, dst_level=dst_level, src_level=src_level)
                rows[dst_level][row_index] = self._upload_block(matrix, fmt=plan.fmt)
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

    def _build_direction_ops(self, direction: Direction) -> list[list[_TritonOp]]:
        H = len(self._level_offsets) - 1
        ops: list[list[_TritonOp]] = [[] for _ in range(H)]
        grid = self._grid_for(direction)
        for dst_level, src_level, row_index in iter_direction_level_pairs(direction, H):
            block = grid[dst_level][row_index]
            if block is None:
                continue
            ops[dst_level].append(_TritonOp(src_level=src_level, block=block, nnz=int(block.nnz)))
        return ops

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
                        helper_src_levels=(),
                        helper_op_indices=(),
                        reduce_order=(),
                    )
                )
                continue
            helper_op_indices = tuple(range(len(ops)))
            reduce_order = tuple(
                sorted(
                    helper_op_indices,
                    key=lambda idx: (int(ops[idx].nnz), int(ops[idx].src_level)),
                )
            )
            plans.append(
                _TritonScratchLevelPlan(
                    enabled=True,
                    helper_src_levels=tuple(int(op.src_level) for op in ops),
                    helper_op_indices=helper_op_indices,
                    reduce_order=reduce_order,
                )
            )
        return plans

    def _build_workspace(self, direction: Direction) -> _TritonWorkspace:
        if self._torch_dtype not in {torch.float32, torch.float64}:
            raise ValueError(f"Unsupported Triton dtype: {self._torch_dtype}")

        H = len(self._level_offsets) - 1
        capture_stream = torch.cuda.Stream()
        level_streams = [torch.cuda.Stream() for _ in range(H)]
        scratch_plans = self._scratch_plan_for(direction)
        level_offsets = [int(v) for v in self._level_offsets]
        node_state = torch.zeros((self._num_nodes,), device=self._device, dtype=self._torch_dtype)
        level_views = [node_state[level_offsets[h] : level_offsets[h + 1]] for h in range(H)]
        scratch_streams_by_level: list[list[torch.cuda.Stream]] = []
        scratch_done_events_by_level: list[list[torch.cuda.Event]] = []
        scratch_views_by_level: list[list[torch.Tensor]] = []
        for h in range(H):
            if h >= len(scratch_plans) or not scratch_plans[h].enabled:
                scratch_streams_by_level.append([])
                scratch_done_events_by_level.append([])
                scratch_views_by_level.append([])
                continue
            level_size = level_views[h].shape[0]
            count = len(scratch_plans[h].helper_op_indices)
            scratch_streams_by_level.append([torch.cuda.Stream() for _ in range(count)])
            scratch_done_events_by_level.append([torch.cuda.Event() for _ in range(count)])
            scratch_views_by_level.append(
                [
                    torch.zeros((level_size,), device=self._device, dtype=self._torch_dtype)
                    for _ in range(count)
                ]
            )
        input_len = self._num_samples if direction == Direction.UP else self._num_mutations
        output_len = self._num_mutations if direction == Direction.UP else self._num_samples
        return _TritonWorkspace(
            direction=direction,
            capture_stream=capture_stream,
            level_streams=level_streams,
            fork_event=torch.cuda.Event(),
            ready_events=[torch.cuda.Event() for _ in range(H)],
            node_state=node_state,
            level_views=level_views,
            scratch_streams_by_level=scratch_streams_by_level,
            scratch_done_events_by_level=scratch_done_events_by_level,
            scratch_views_by_level=scratch_views_by_level,
            input_primary=torch.zeros((input_len,), device=self._device, dtype=self._torch_dtype),
            input_miss=(
                torch.zeros((self._num_mutations,), device=self._device, dtype=self._torch_dtype)
                if direction == Direction.DOWN
                else None
            ),
            output_main=torch.zeros((output_len,), device=self._device, dtype=self._torch_dtype),
            output_aux=(
                torch.zeros((self._num_mutations,), device=self._device, dtype=self._torch_dtype)
                if direction == Direction.UP
                else None
            ),
            init_scalar=torch.zeros((1,), device=self._device, dtype=self._torch_dtype),
            init_matrix=torch.zeros((self._num_nodes,), device=self._device, dtype=self._torch_dtype),
        )

    def _autotune_key(self, direction: Direction) -> tuple[object, ...]:
        device = torch.cuda.get_device_properties(self._device)
        plan = self._require_plan(direction)
        ops_by_level = self._ops_for(direction)
        return (
            device.name,
            int(device.major),
            int(device.minor),
            str(self._torch_dtype),
            direction.value,
            plan.fmt.value,
            plan.scratch,
            _structure_signature(ops_by_level),
        )

    def _prepare_workspace_state(self, ws: _TritonWorkspace, *, init_mode: InitMode, has_miss_input: bool) -> None:
        with torch.cuda.stream(ws.capture_stream):
            ws.node_state.zero_()
            match init_mode:
                case InitMode.NONE:
                    pass
                case InitMode.XTX:
                    if self._xtx_gpu is None:
                        raise ValueError("init_mode=xtx requires GRG coalescence counts")
                    ws.node_state.add_(self._xtx_gpu)
                case InitMode.VECTOR:
                    ws.node_state.add_(ws.init_scalar[0])
                case InitMode.MATRIX:
                    ws.node_state.add_(ws.init_matrix)
                case _:
                    raise ValueError(f"Unknown init mode: {init_mode!r}")

            if ws.direction == Direction.UP:
                ws.node_state[: self._num_samples].add_(ws.input_primary.index_select(0, self._sample_perm_gpu))
            else:
                if self._sel_mut_rows_gpu.numel() > 0:
                    ws.node_state.index_add_(
                        0,
                        self._sel_mut_cols_gpu,
                        ws.input_primary.index_select(0, self._sel_mut_rows_gpu),
                    )
                if has_miss_input:
                    if ws.input_miss is None:
                        raise RuntimeError("Missing DOWN miss buffer in Triton workspace")
                    if self._sel_miss_rows_gpu.numel() > 0:
                        ws.node_state.index_add_(
                            0,
                            self._sel_miss_cols_gpu,
                            ws.input_miss.index_select(0, self._sel_miss_rows_gpu),
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

    def _enqueue_wavefront(
        self,
        ws: _TritonWorkspace,
        *,
        config: CsrKernelConfig | CscKernelConfig,
    ) -> None:
        H = len(self._level_offsets) - 1
        ops_by_level = self._ops_for(ws.direction)
        scratch_plans = self._scratch_plan_for(ws.direction)
        if ws.direction == Direction.UP:
            seed_level = 0
            level_iter = range(1, H)
        else:
            seed_level = H - 1
            level_iter = range(H - 2, -1, -1)

        with torch.cuda.stream(ws.capture_stream):
            ws.fork_event.record(ws.capture_stream)
        for stream in ws.level_streams:
            stream.wait_event(ws.fork_event)
        for scratch_streams in ws.scratch_streams_by_level:
            for stream in scratch_streams:
                stream.wait_event(ws.fork_event)

        if H > 0:
            seed_stream = ws.level_streams[seed_level]
            with torch.cuda.stream(seed_stream):
                ws.ready_events[seed_level].record(seed_stream)

        for dst_level in level_iter:
            stream = ws.level_streams[dst_level]
            ops = ops_by_level[dst_level]
            scratch_plan = scratch_plans[dst_level]
            if scratch_plan.enabled:
                scratch_streams = ws.scratch_streams_by_level[dst_level]
                scratch_done_events = ws.scratch_done_events_by_level[dst_level]
                scratch_views = ws.scratch_views_by_level[dst_level]
                for helper_idx, op_idx in enumerate(scratch_plan.helper_op_indices):
                    op = ops[op_idx]
                    helper_stream = scratch_streams[helper_idx]
                    helper_view = scratch_views[helper_idx]
                    with torch.cuda.stream(helper_stream):
                        helper_stream.wait_event(ws.ready_events[op.src_level])
                        helper_view.zero_()
                        self._launch_op(op, x=ws.level_views[op.src_level], y=helper_view, config=config)
                        scratch_done_events[helper_idx].record(helper_stream)
                with torch.cuda.stream(stream):
                    for helper_idx in scratch_plan.reduce_order:
                        stream.wait_event(scratch_done_events[helper_idx])
                        ws.level_views[dst_level].add_(scratch_views[helper_idx])
                    ws.ready_events[dst_level].record(stream)
                continue

            with torch.cuda.stream(stream):
                for op in ops:
                    stream.wait_event(ws.ready_events[op.src_level])
                    self._launch_op(op, x=ws.level_views[op.src_level], y=ws.level_views[dst_level], config=config)
                ws.ready_events[dst_level].record(stream)

        with torch.cuda.stream(ws.capture_stream):
            for event in ws.ready_events:
                ws.capture_stream.wait_event(event)

    def _enqueue_wavefront_nvtx(
        self,
        ws: _TritonWorkspace,
        *,
        config: CsrKernelConfig | CscKernelConfig,
    ) -> None:
        tracer = self._nvtx
        if tracer is None:
            raise RuntimeError("Triton NVTX tracer is not initialized")

        H = len(self._level_offsets) - 1
        ops_by_level = self._ops_for(ws.direction)
        scratch_plans = self._scratch_plan_for(ws.direction)
        if ws.direction == Direction.UP:
            seed_level = 0
            level_iter = range(1, H)
        else:
            seed_level = H - 1
            level_iter = range(H - 2, -1, -1)

        with tracer.range("wavefront", dir=ws.direction.value):
            with torch.cuda.stream(ws.capture_stream):
                ws.fork_event.record(ws.capture_stream)
                tracer.mark("event.record_fork", dir=ws.direction.value)
            for stream in ws.level_streams:
                stream.wait_event(ws.fork_event)
            for scratch_streams in ws.scratch_streams_by_level:
                for stream in scratch_streams:
                    stream.wait_event(ws.fork_event)

            if H > 0:
                seed_stream = ws.level_streams[seed_level]
                with torch.cuda.stream(seed_stream):
                    ws.ready_events[seed_level].record(seed_stream)
                    tracer.mark("event.record_ready", dir=ws.direction.value, level=seed_level)

            for dst_level in level_iter:
                stream = ws.level_streams[dst_level]
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
                        scratch_streams = ws.scratch_streams_by_level[dst_level]
                        scratch_done_events = ws.scratch_done_events_by_level[dst_level]
                        scratch_views = ws.scratch_views_by_level[dst_level]
                        for helper_idx, op_idx in enumerate(scratch_plan.helper_op_indices):
                            op = ops[op_idx]
                            helper_stream = scratch_streams[helper_idx]
                            helper_view = scratch_views[helper_idx]
                            with torch.cuda.stream(helper_stream):
                                tracer.mark(
                                    "wait_ready",
                                    dir=ws.direction.value,
                                    dst=dst_level,
                                    src=op.src_level,
                                )
                                helper_stream.wait_event(ws.ready_events[op.src_level])
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
                                op_idx = scratch_plan.helper_op_indices[helper_idx]
                                src_level = ops[op_idx].src_level
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
                            ws.ready_events[dst_level].record(stream)
                            tracer.mark("event.record_ready", dir=ws.direction.value, level=dst_level)
                        continue

                    with torch.cuda.stream(stream):
                        for op in ops:
                            tracer.mark(
                                "wait_ready",
                                dir=ws.direction.value,
                                dst=dst_level,
                                src=op.src_level,
                            )
                            stream.wait_event(ws.ready_events[op.src_level])
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
                        ws.ready_events[dst_level].record(stream)
                        tracer.mark("event.record_ready", dir=ws.direction.value, level=dst_level)

            with tracer.range("join_ready", dir=ws.direction.value):
                with torch.cuda.stream(ws.capture_stream):
                    for event in ws.ready_events:
                        ws.capture_stream.wait_event(event)

    def _run_wavefront(
        self,
        ws: _TritonWorkspace,
        *,
        config: CsrKernelConfig | CscKernelConfig,
    ) -> None:
        if self._instrumentation:
            self._enqueue_wavefront_nvtx(ws, config=config)
            return
        self._enqueue_wavefront(ws, config=config)

    def _enqueue_output_gather(self, ws: _TritonWorkspace, *, need_miss_output: bool) -> None:
        with torch.cuda.stream(ws.capture_stream):
            if ws.direction == Direction.UP:
                ws.output_main.zero_()
                if self._sel_mut_rows_gpu.numel() > 0:
                    ws.output_main.index_add_(
                        0,
                        self._sel_mut_rows_gpu,
                        ws.node_state.index_select(0, self._sel_mut_cols_gpu),
                    )
                if need_miss_output:
                    if ws.output_aux is None:
                        raise RuntimeError("Missing UP miss output buffer in Triton workspace")
                    ws.output_aux.zero_()
                    if self._sel_miss_rows_gpu.numel() > 0:
                        ws.output_aux.index_add_(
                            0,
                            self._sel_miss_rows_gpu,
                            ws.node_state.index_select(0, self._sel_miss_cols_gpu),
                        )
            else:
                ws.output_main.copy_(ws.node_state.index_select(0, self._inv_sample_perm_gpu))

    def _tune_once(self, direction: Direction, ws: _TritonWorkspace, config: CsrKernelConfig | CscKernelConfig) -> None:
        self._prepare_workspace_state(ws, init_mode=InitMode.NONE, has_miss_input=False)
        self._enqueue_wavefront(ws, config=config)
        ws.capture_stream.synchronize()

    def _tune_direction(self, direction: Direction, ws: _TritonWorkspace) -> CsrKernelConfig | CscKernelConfig:
        key = self._autotune_key(direction)
        cached = _AUTOTUNE_CACHE.get(key)
        if cached is not None:
            self._logger.info("Reusing cached Triton autotune config for %s: %s", direction.value, format_config(cached))
            return cached

        tune_input = torch.randn(
            (self._num_samples if direction == Direction.UP else self._num_mutations,),
            device=self._device,
            dtype=self._torch_dtype,
        )
        with torch.cuda.stream(ws.capture_stream):
            ws.input_primary.copy_(tune_input)
            if ws.input_miss is not None:
                ws.input_miss.zero_()
            ws.init_scalar.zero_()
            ws.init_matrix.zero_()
        ws.capture_stream.synchronize()

        results: list[tuple[CsrKernelConfig | CscKernelConfig, float]] = []
        for config in self._configured_kernel_configs(direction):
            self._tune_once(direction, ws, config)
            timing_ms = float(
                triton.testing.do_bench(
                    lambda config=config: self._tune_once(direction, ws, config),
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

    def _capture_wavefront_graph(self, direction: Direction, ws: _TritonWorkspace, config: CsrKernelConfig | CscKernelConfig) -> torch.cuda.CUDAGraph:
        # Warm up outside capture so Triton compilation is not part of the graph.
        self._prepare_workspace_state(ws, init_mode=InitMode.NONE, has_miss_input=False)
        self._enqueue_wavefront(ws, config=config)
        ws.capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=ws.capture_stream):
            self._enqueue_wavefront(ws, config=config)
        ws.capture_stream.synchronize()
        return graph

    def _run_singleton_column(
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
        use_graph = not self._instrumentation
        ws = self._workspace_for(direction, graph=use_graph)
        primary_cpu = torch.from_numpy(np.ascontiguousarray(primary_col)).to(dtype=self._torch_dtype)
        miss_cpu = None if miss_col is None else torch.from_numpy(np.ascontiguousarray(miss_col)).to(dtype=self._torch_dtype)
        init_matrix_cpu = None
        if init_mode == InitMode.MATRIX:
            assert init_value is not None
            init_matrix_cpu = torch.from_numpy(np.ascontiguousarray(init_value)).to(dtype=self._torch_dtype)

        with torch.cuda.stream(ws.capture_stream):
            ws.input_primary.copy_(primary_cpu)
            if ws.input_miss is not None and miss_cpu is not None:
                ws.input_miss.copy_(miss_cpu)
            if init_mode == InitMode.VECTOR:
                assert init_value is not None
                ws.init_scalar.fill_(float(np.asarray(init_value).reshape(())))
            else:
                ws.init_scalar.zero_()
            if init_matrix_cpu is not None:
                ws.init_matrix.copy_(init_matrix_cpu)
            else:
                ws.init_matrix.zero_()

        tracer = self._nvtx
        if tracer is not None:
            with tracer.range("execute_singleton", dir=direction.value):
                with tracer.range("prepare_state", dir=direction.value):
                    self._prepare_workspace_state(ws, init_mode=init_mode, has_miss_input=miss_col is not None)
                self._run_wavefront(ws, config=config)
                if not emit_all_nodes:
                    with tracer.range("gather_outputs", dir=direction.value):
                        self._enqueue_output_gather(ws, need_miss_output=need_miss_output)
                with tracer.range("await_outputs", dir=direction.value):
                    ws.capture_stream.synchronize()
        else:
            self._prepare_workspace_state(ws, init_mode=init_mode, has_miss_input=miss_col is not None)
            if use_graph:
                if ws.graph is None:
                    raise RuntimeError(f"Missing Triton graph workspace for {direction.value}")
                with torch.cuda.stream(ws.capture_stream):
                    ws.graph.replay()
            else:
                self._run_wavefront(ws, config=config)
            if not emit_all_nodes:
                self._enqueue_output_gather(ws, need_miss_output=need_miss_output)
            ws.capture_stream.synchronize()

        if emit_all_nodes:
            return ws.node_state.cpu().numpy().copy()
        if direction == Direction.UP:
            out_mut = ws.output_main.cpu().numpy().copy()
            out_miss = ws.output_aux.cpu().numpy().copy() if need_miss_output and ws.output_aux is not None else None
            return out_mut, out_miss
        return ws.output_main.cpu().numpy().copy()

    def setup(self, setup: BackendSetup) -> None:
        self._apply_setup_state(setup)
        if np.dtype(setup.dtype) == np.float64:
            self._torch_dtype = torch.float64
        elif np.dtype(setup.dtype) == np.float32:
            self._torch_dtype = torch.float32
        else:
            raise ValueError(f"Triton backend supports only float32/float64, got {setup.dtype}")

        H = len(self._level_offsets) - 1
        self._blocks_up = [[] for _ in range(H)]
        self._blocks_down = [[] for _ in range(H)]
        self._ops_up = [[] for _ in range(H)]
        self._ops_down = [[] for _ in range(H)]

        if self._plan_up is not None:
            self._blocks_up = self._build_direction_blocks(Direction.UP)
        if self._plan_down is not None:
            self._blocks_down = self._build_direction_blocks(Direction.DOWN)
        if self._plan_up is not None:
            self._ops_up = self._build_direction_ops(Direction.UP)
            self._scratch_plan_up = self._build_scratch_level_plans(Direction.UP)
        if self._plan_down is not None:
            self._ops_down = self._build_direction_ops(Direction.DOWN)
            self._scratch_plan_down = self._build_scratch_level_plans(Direction.DOWN)

        self._sample_perm_gpu = torch.from_numpy(np.asarray(self._sample_perm, dtype=np.int64)).to(device=self._device)
        self._inv_sample_perm_gpu = torch.from_numpy(np.asarray(self._inv_sample_perm, dtype=np.int64)).to(device=self._device)
        mut_rows, mut_cols = _selector_index_arrays(self._sel_mut)
        miss_rows, miss_cols = _selector_index_arrays(self._sel_miss)
        self._sel_mut_rows_gpu = torch.from_numpy(mut_rows).to(device=self._device)
        self._sel_mut_cols_gpu = torch.from_numpy(mut_cols).to(device=self._device)
        self._sel_miss_rows_gpu = torch.from_numpy(miss_rows).to(device=self._device)
        self._sel_miss_cols_gpu = torch.from_numpy(miss_cols).to(device=self._device)
        self._xtx_gpu = None if self._xtx_init is None else torch.from_numpy(np.asarray(self._xtx_init)).to(
            device=self._device, dtype=self._torch_dtype
        )

        if self._plan_up is not None:
            self._workspaces.dynamic_up = self._build_workspace(Direction.UP)
            self._config_up = self._tune_direction(Direction.UP, self._workspaces.dynamic_up)
            if self._instrumentation:
                if self._plan_up.k_hint is not None:
                    warnings.warn(
                        (
                            "Triton up graph capture/replay disabled because instrumentation=True "
                            "uses the dynamic scheduler for observability."
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                self._workspaces.graph_up = None
                self._workspace_bytes_up = self._workspaces.dynamic_up.nbytes()
            else:
                self._workspaces.graph_up = self._build_workspace(Direction.UP)
                self._workspaces.graph_up.graph = self._capture_wavefront_graph(
                    Direction.UP,
                    self._workspaces.graph_up,
                    self._config_up,
                )
                self._workspace_bytes_up = self._workspaces.dynamic_up.nbytes() + self._workspaces.graph_up.nbytes()
        else:
            self._workspace_bytes_up = 0

        if self._plan_down is not None:
            self._workspaces.dynamic_down = self._build_workspace(Direction.DOWN)
            self._config_down = self._tune_direction(Direction.DOWN, self._workspaces.dynamic_down)
            if self._instrumentation:
                if self._plan_down.k_hint is not None:
                    warnings.warn(
                        (
                            "Triton down graph capture/replay disabled because instrumentation=True "
                            "uses the dynamic scheduler for observability."
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                self._workspaces.graph_down = None
                self._workspace_bytes_down = self._workspaces.dynamic_down.nbytes()
            else:
                self._workspaces.graph_down = self._build_workspace(Direction.DOWN)
                self._workspaces.graph_down.graph = self._capture_wavefront_graph(
                    Direction.DOWN,
                    self._workspaces.graph_down,
                    self._config_down,
                )
                self._workspace_bytes_down = self._workspaces.dynamic_down.nbytes() + self._workspaces.graph_down.nbytes()
        else:
            self._workspace_bytes_down = 0

        # Release host sparse structures after upload so host memory accounting matches retained state.
        self._A_blocks = []
        self._sel_mut = sp.csr_matrix((0, 0))
        self._sel_miss = sp.csr_matrix((0, 0))

        self.mem_usage.reset()
        self.mem_usage.host_static = estimate_common_host_static_bytes(
            level_offsets=self._level_offsets,
            sample_perm=self._sample_perm,
            inv_sample_perm=self._inv_sample_perm,
            coalescence_counts=self._coalescence_counts,
            xtx_init=self._xtx_init,
        )
        self.mem_usage.device_static = self.estimate_static_bytes()[1]

    def _block_grid_bytes(self, direction: Direction) -> int:
        store_actual = self._store_blocks_up if direction == Direction.UP else self._store_blocks_down
        if not store_actual:
            return 0
        grid = self._grid_for(direction)
        return int(sum(block.nbytes() for row in grid for block in row if block is not None))

    def estimate_static_bytes(self) -> tuple[StaticBytes, StaticBytes]:
        host = estimate_common_host_static_bytes(
            level_offsets=self._level_offsets,
            sample_perm=self._sample_perm,
            inv_sample_perm=self._inv_sample_perm,
            coalescence_counts=self._coalescence_counts,
            xtx_init=self._xtx_init,
        )
        device = StaticBytes()
        device.sample_perm = _torch_nbytes(self._sample_perm_gpu)
        device.inv_sample_perm = _torch_nbytes(self._inv_sample_perm_gpu)
        device.blocks_up = self._block_grid_bytes(Direction.UP)
        device.blocks_down = self._block_grid_bytes(Direction.DOWN)
        device.selector_mut = _torch_nbytes(self._sel_mut_rows_gpu) + _torch_nbytes(self._sel_mut_cols_gpu)
        device.selector_miss = _torch_nbytes(self._sel_miss_rows_gpu) + _torch_nbytes(self._sel_miss_cols_gpu)
        device.xtx_init = _torch_nbytes(self._xtx_gpu)
        device.workspace = int(self._workspace_bytes_up + self._workspace_bytes_down)
        return host, device

    def run_up(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
        need_miss_output: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self._require_plan(Direction.UP)
        X, k = self._normalize_primary_input(direction=Direction.UP, primary=primary)
        mode = parse_init_mode(init_mode)
        payload = self._validate_init(mode, init, k)
        out_mut = np.empty((self._num_mutations, k), dtype=self._dtype)
        out_miss = np.empty((self._num_mutations, k), dtype=self._dtype) if need_miss_output else None

        for col in range(k):
            init_value = None
            if mode == InitMode.VECTOR:
                assert payload is not None
                init_value = payload[col]
            elif mode == InitMode.MATRIX:
                assert payload is not None
                init_value = payload[:, col]

            result = self._run_singleton_column(
                Direction.UP,
                primary_col=X[:, col],
                miss_col=None,
                init_mode=mode,
                init_value=init_value,
                need_miss_output=need_miss_output,
                emit_all_nodes=False,
            )

            mut_col, miss_col = result
            out_mut[:, col] = mut_col
            if need_miss_output and out_miss is not None:
                assert miss_col is not None
                out_miss[:, col] = miss_col

        self.mem_usage.record(
            stage="run_up",
            runtime_k=k,
            host_runtime=RuntimeBytes(
                inputs=int(X.nbytes),
                outputs=int(out_mut.nbytes + (0 if out_miss is None else out_miss.nbytes)),
                aux=0 if payload is None else int(payload.nbytes),
            ),
            device_runtime=RuntimeBytes(),
            meta={
                "direction": "up",
                "need_miss_output": bool(need_miss_output),
                "mode": "instrumented" if self._instrumentation else "graph",
            },
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
        self._require_plan(Direction.DOWN)
        X, k = self._normalize_primary_input(direction=Direction.DOWN, primary=primary)
        miss_arr = self._normalize_down_miss_input(miss, k=k)
        mode = parse_init_mode(init_mode)
        payload = self._validate_init(mode, init, k)
        out = np.empty((self._num_samples, k), dtype=self._dtype)

        for col in range(k):
            init_value = None
            if mode == InitMode.VECTOR:
                assert payload is not None
                init_value = payload[col]
            elif mode == InitMode.MATRIX:
                assert payload is not None
                init_value = payload[:, col]
            miss_col = None if miss_arr is None else miss_arr[:, col]

            result = self._run_singleton_column(
                Direction.DOWN,
                primary_col=X[:, col],
                miss_col=miss_col,
                init_mode=mode,
                init_value=init_value,
                need_miss_output=False,
                emit_all_nodes=False,
            )
            out[:, col] = result

        self.mem_usage.record(
            stage="run_down",
            runtime_k=k,
            host_runtime=RuntimeBytes(
                inputs=int(X.nbytes + (0 if miss_arr is None else miss_arr.nbytes)),
                outputs=int(out.nbytes),
                aux=0 if payload is None else int(payload.nbytes),
            ),
            device_runtime=RuntimeBytes(),
            meta={
                "direction": "down",
                "has_miss_input": bool(miss_arr is not None),
                "mode": "instrumented" if self._instrumentation else "graph",
            },
        )
        return out

    def run_up_nodes(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
    ) -> np.ndarray:
        self._require_plan(Direction.UP)
        X, k = self._normalize_primary_input(direction=Direction.UP, primary=primary)
        mode = parse_init_mode(init_mode)
        payload = self._validate_init(mode, init, k)
        out = np.empty((self._num_nodes, k), dtype=self._dtype)

        for col in range(k):
            init_value = None
            if mode == InitMode.VECTOR:
                assert payload is not None
                init_value = payload[col]
            elif mode == InitMode.MATRIX:
                assert payload is not None
                init_value = payload[:, col]
            out[:, col] = self._run_singleton_column(
                Direction.UP,
                primary_col=X[:, col],
                miss_col=None,
                init_mode=mode,
                init_value=init_value,
                need_miss_output=False,
                emit_all_nodes=True,
            )
        return out

    def run_down_nodes(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
    ) -> np.ndarray:
        self._require_plan(Direction.DOWN)
        X, k = self._normalize_primary_input(direction=Direction.DOWN, primary=primary)
        mode = parse_init_mode(init_mode)
        payload = self._validate_init(mode, init, k)
        out = np.empty((self._num_nodes, k), dtype=self._dtype)

        for col in range(k):
            init_value = None
            if mode == InitMode.VECTOR:
                assert payload is not None
                init_value = payload[col]
            elif mode == InitMode.MATRIX:
                assert payload is not None
                init_value = payload[:, col]
            out[:, col] = self._run_singleton_column(
                Direction.DOWN,
                primary_col=X[:, col],
                miss_col=None,
                init_mode=mode,
                init_value=init_value,
                need_miss_output=False,
                emit_all_nodes=True,
            )
        return out


__all__ = ["TritonBackend"]
