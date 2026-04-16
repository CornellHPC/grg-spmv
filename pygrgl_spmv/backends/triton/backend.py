"""Triton runtime and layout planner."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import triton

from pygrgl_spmv.backends._cuda_stream import _cuda_stream_device, parse_cuda_device, parse_cuda_stream
from pygrgl_spmv.backends.base import (
    BudgetItem,
    _copy_struct_checked,
    _layout_struct_dtypes,
    _require_struct_dtype,
    iter_direction_level_pairs,
    materialize_sparse_block,
    sparse_structure_lengths,
    stored_block_shape,
)
from pygrgl_spmv.backends.triton.kernel import (
    CSC_CANDIDATE_CONFIGS,
    CSR_CANDIDATE_CONFIGS,
    CscKernelConfig,
    CsrKernelConfig,
    launch_block,
)
from pygrgl_spmv.backends.types import Direction, InitMode, SparseFormat, StoredMatrix, transpose_compatible_format
from pygrgl_spmv.grg import BoundGRG, RuntimeRequirements
from pygrgl_spmv.grg.artifact import _load_grg_spmv_host, iter_artifact_blocks, scan_grg_spmv

from .plan import TritonPlan, TritonPlanPair

_TUNE_WARMUP_MS = 20
_TUNE_REP_MS = 80


@dataclass
class _TritonBlockPlan:
    artifact_index: int
    owner: Direction
    shared: bool
    dst_level: int
    src_level: int
    fmt: SparseFormat
    store: StoredMatrix
    stored_shape: tuple[int, int]
    nnz: int
    indices_dtype: np.dtype
    indptr_dtype: np.dtype
    indices_len: int
    indptr_len: int
    nbytes: int
    resident: bool = True
    slot: int | None = None


@dataclass(frozen=True)
class _TritonArtifactLayout:
    path: Path
    share_storage: bool
    up_owner: Direction | None
    down_owner: Direction | None
    blocks_up: tuple[_TritonBlockPlan, ...]
    blocks_down: tuple[_TritonBlockPlan, ...]


@dataclass(frozen=True)
class _SlotPlan:
    indices_dtype: np.dtype
    indices_len: int
    indptr_dtype: np.dtype
    indptr_len: int

    @property
    def nbytes(self) -> int:
        return int(self.indices_len * self.indices_dtype.itemsize + self.indptr_len * self.indptr_dtype.itemsize)


@dataclass
class TritonLayout:
    artifacts: tuple[_TritonArtifactLayout, ...]
    pair: TritonPlanPair
    dtype: np.dtype
    requirements: RuntimeRequirements
    device: int
    stream_ptr: int
    stream_owner: object | None
    allow_residency: bool
    requested_ring_buffer_size: int
    allocated_ring_buffer_size: int
    vram_budget_bytes: int
    max_num_samples: int
    max_num_mutations: int
    max_num_nodes: int
    max_levels: int
    max_rows_by_level: tuple[int, ...]
    max_up_ops_by_level: tuple[int, ...]
    max_down_ops_by_level: tuple[int, ...]
    scratch_up_enabled: tuple[bool, ...]
    scratch_down_enabled: tuple[bool, ...]
    slot_plans: tuple[_SlotPlan, ...]
    budget_items: tuple[BudgetItem, ...]
    required_budget_for_full_residency: int
    bytes_by_category: dict[str, int]
    bytes_total: int


@dataclass(frozen=True)
class _TritonLaunchBlock:
    indices: torch.Tensor
    indptr: torch.Tensor
    nrows: int
    ncols: int
    nnz: int
    fmt: SparseFormat

    def transpose_alias(self) -> "_TritonLaunchBlock":
        return _TritonLaunchBlock(
            indices=self.indices,
            indptr=self.indptr,
            nrows=self.ncols,
            ncols=self.nrows,
            nnz=self.nnz,
            fmt=transpose_compatible_format(self.fmt),
        )


@dataclass(frozen=True)
class _TritonStreamSource:
    slot: int
    indices_host: torch.Tensor
    indptr_host: torch.Tensor
    nrows: int
    ncols: int
    nnz: int
    fmt: SparseFormat

    def transpose_alias(self) -> "_TritonStreamSource":
        return _TritonStreamSource(
            slot=self.slot,
            indices_host=self.indices_host,
            indptr_host=self.indptr_host,
            nrows=self.ncols,
            ncols=self.nrows,
            nnz=self.nnz,
            fmt=transpose_compatible_format(self.fmt),
        )


@dataclass(frozen=True)
class _TritonOp:
    src_level: int
    launch_block: _TritonLaunchBlock
    slot: int | None
    indices_host: torch.Tensor | None
    indptr_host: torch.Tensor | None
    prev_in_slot: tuple[int, int] | None


@dataclass
class _TritonArtifact:
    path: Path
    state: object
    sel_mut_rows: torch.Tensor
    sel_mut_cols: torch.Tensor
    sel_miss_rows: torch.Tensor
    sel_miss_cols: torch.Tensor
    xtx_bias: torch.Tensor | None
    up_ops: list[list[_TritonOp]]
    down_ops: list[list[_TritonOp]]


def _torch_int_dtype(dtype: np.dtype) -> torch.dtype:
    dt = _require_struct_dtype(dtype, label="torch structural dtype")
    return torch.int32 if dt == np.dtype(np.int32) else torch.int64


def _resolve_artifacts(artifacts) -> tuple[Path, ...]:
    paths = tuple(Path(path) for path in artifacts)
    if not paths:
        raise ValueError("artifacts must be non-empty")
    for path in paths:
        if path.suffix != ".grg_spmv":
            raise ValueError(f"planner expects .grg_spmv artifacts, got {path}")
        if not path.exists():
            raise FileNotFoundError(path)
    return paths


def _block_plan(artifact_index: int, owner: Direction, shared: bool, block, plan: TritonPlan) -> _TritonBlockPlan:
    nrows, ncols = stored_block_shape(block.shape[0], block.shape[1], store=plan.store)
    indptr_dtype, indices_dtype = _layout_struct_dtypes(plan.fmt, nrows=nrows, ncols=ncols, nnz=block.nnz)
    indptr_len, indices_len = sparse_structure_lengths(plan.fmt, nrows=nrows, ncols=ncols, nnz=block.nnz)
    nbytes = int(indptr_len * indptr_dtype.itemsize + indices_len * indices_dtype.itemsize)
    return _TritonBlockPlan(
        artifact_index=artifact_index,
        owner=owner,
        shared=shared,
        dst_level=block.dst_level,
        src_level=block.src_level,
        fmt=plan.fmt,
        store=plan.store,
        stored_shape=(nrows, ncols),
        nnz=int(block.nnz),
        indices_dtype=np.dtype(indices_dtype),
        indptr_dtype=np.dtype(indptr_dtype),
        indices_len=int(indices_len),
        indptr_len=int(indptr_len),
        nbytes=nbytes,
    )


def _resolve_scratch_levels(plan: TritonPlan | None, height: int) -> tuple[bool, ...]:
    if plan is None:
        return tuple(False for _ in range(height))
    token = str(plan.scratch)
    if token == "none":
        return tuple(False for _ in range(height))
    if token == "all":
        return tuple(True for _ in range(height))
    levels = {int(piece) for piece in token.split("|")}
    invalid = sorted(level for level in levels if level < 0 or level >= height)
    if invalid:
        raise ValueError(f"Triton scratch levels out of range: {invalid}; valid range is [0, {height})")
    return tuple(level in levels for level in range(height))


def _assign_slots(blocks: list[_TritonBlockPlan], ring_buffer_size: int) -> tuple[tuple[_SlotPlan, ...], int]:
    for block in blocks:
        block.slot = None
    streamed = sorted(
        (block for block in blocks if not block.resident),
        key=lambda block: (block.artifact_index, block.owner.value, block.dst_level, block.src_level),
    )
    if not streamed:
        return (), 0
    allocated_ring_buffer_size = min(int(ring_buffer_size), len(streamed))
    if allocated_ring_buffer_size < 1:
        raise ValueError("ring_buffer_size must be >= 1 when any Triton block is streamed")
    slot_indices_dtype = [np.dtype(np.int32) for _ in range(allocated_ring_buffer_size)]
    slot_indptr_dtype = [np.dtype(np.int32) for _ in range(allocated_ring_buffer_size)]
    slot_indices_len = [0 for _ in range(allocated_ring_buffer_size)]
    slot_indptr_len = [0 for _ in range(allocated_ring_buffer_size)]
    for idx, block in enumerate(streamed):
        slot = int(idx % allocated_ring_buffer_size)
        block.slot = slot
        if block.indices_dtype == np.dtype(np.int64):
            slot_indices_dtype[slot] = np.dtype(np.int64)
        if block.indptr_dtype == np.dtype(np.int64):
            slot_indptr_dtype[slot] = np.dtype(np.int64)
        slot_indices_len[slot] = max(slot_indices_len[slot], int(block.indices_len))
        slot_indptr_len[slot] = max(slot_indptr_len[slot], int(block.indptr_len))
    slot_plans = tuple(
        _SlotPlan(slot_indices_dtype[slot], slot_indices_len[slot], slot_indptr_dtype[slot], slot_indptr_len[slot])
        for slot in range(allocated_ring_buffer_size)
    )
    return slot_plans, int(sum(slot.nbytes for slot in slot_plans))


def _promotion_key(block: _TritonBlockPlan) -> tuple[int, int, int, int, int]:
    owner_rank = 0 if block.owner == Direction.UP else 1
    return (-int(block.nbytes), -int(block.artifact_index), -int(block.dst_level), -int(block.src_level), owner_rank)


def _warn_ring_mismatch(*, requested: int, allocated: int) -> None:
    if int(allocated) == int(requested):
        return
    if int(allocated) == 0 and int(requested) > 0:
        warnings.warn(
            f"requested ring_buffer_size={requested}, but all sparse blocks fit resident so no ring slots were allocated",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    warnings.warn(
        f"requested ring_buffer_size={requested}, but only {allocated} ring slot(s) were allocated",
        RuntimeWarning,
        stacklevel=2,
    )


def _relink_stream_dependencies(ops_by_level: list[list[_TritonOp]], *, direction: Direction) -> list[list[_TritonOp]]:
    prev_by_slot: dict[int, tuple[int, int]] = {}
    height = len(ops_by_level)
    level_iter = range(1, height) if direction == Direction.UP else range(height - 2, -1, -1)
    for dst_level in level_iter:
        for op_idx, op in enumerate(ops_by_level[dst_level]):
            if op.slot is None:
                continue
            ops_by_level[dst_level][op_idx] = replace(op, prev_in_slot=prev_by_slot.get(op.slot))
            prev_by_slot[op.slot] = (dst_level, op_idx)
    return ops_by_level


def plan_triton_layout(
    *,
    artifacts,
    pair: TritonPlanPair,
    dtype,
    requirements: RuntimeRequirements,
    vram_budget_bytes: int,
    ring_buffer_size: int,
    allow_residency: bool = True,
    device,
    stream,
) -> TritonLayout:
    dtype = np.dtype(dtype)
    if dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError(f"Triton runtime supports only float32/float64, got {dtype}")
    paths = _resolve_artifacts(artifacts)
    device_id = parse_cuda_device(device)
    stream_ptr, stream_owner = parse_cuda_stream(stream)
    requested_ring_buffer_size = int(ring_buffer_size)
    allow_residency = bool(allow_residency)
    if stream_ptr != 0:
        stream_device = _cuda_stream_device(stream_ptr)
        if stream_device != device_id:
            raise ValueError(f"CUDA stream device {stream_device} does not match requested CUDA device {device_id}")
    scans = tuple(scan_grg_spmv(path) for path in paths)
    max_levels = max(scan.num_levels for scan in scans)
    max_rows_by_level = tuple(
        max((scan.level_sizes[level] if level < scan.num_levels else 0) for scan in scans)
        for level in range(max_levels)
    )
    max_up_ops_by_level = tuple(
        max(sum(1 for block in scan.blocks if block.nnz > 0 and block.dst_level == level) for scan in scans)
        for level in range(max_levels)
    )
    max_down_ops_by_level = tuple(
        max(sum(1 for block in scan.blocks if block.nnz > 0 and block.src_level == level) for scan in scans)
        for level in range(max_levels)
    )
    scratch_up_enabled = _resolve_scratch_levels(pair.plan_up, max_levels)
    scratch_down_enabled = _resolve_scratch_levels(pair.plan_down, max_levels)
    max_samples = max(scan.num_samples for scan in scans)
    max_mutations = max(scan.num_mutations for scan in scans)
    max_nodes = max(scan.num_nodes for scan in scans)

    selector_bytes = 0
    xtx_bias_bytes = 0
    for path in paths:
        state = _load_grg_spmv_host(path, dtype)
        sel_mut = state.sel_mut.tocoo()
        sel_miss = state.sel_miss.tocoo()
        selector_bytes += int(np.asarray(sel_mut.row).nbytes + np.asarray(sel_mut.col).nbytes)
        selector_bytes += int(np.asarray(sel_miss.row).nbytes + np.asarray(sel_miss.col).nbytes)
        if requirements.need_init_xtx and state.coalescence_counts is not None:
            xtx_bias_bytes += int(state.num_nodes * dtype.itemsize)

    staging_bytes = 0
    workspace_up = 0
    workspace_down = 0
    scratch_bytes = 0
    if pair.plan_up is not None:
        workspace_up = int(max_nodes * int(requirements.max_k_up) * dtype.itemsize)
        staging_bytes += int(max_samples * int(requirements.max_k_up) * dtype.itemsize)
        staging_bytes += int(max_mutations * int(requirements.max_k_up) * dtype.itemsize)
        if requirements.need_up_miss_output:
            staging_bytes += int(max_mutations * int(requirements.max_k_up) * dtype.itemsize)
        if requirements.need_init_vector:
            staging_bytes += int(int(requirements.max_k_up) * dtype.itemsize)
        if requirements.need_init_matrix:
            staging_bytes += int(max_nodes * int(requirements.max_k_up) * dtype.itemsize)
        for level, enabled in enumerate(scratch_up_enabled):
            if enabled:
                scratch_bytes += int(max_rows_by_level[level] * max_up_ops_by_level[level] * int(requirements.max_k_up) * dtype.itemsize)
    if pair.plan_down is not None:
        workspace_down = int(max_nodes * int(requirements.max_k_down) * dtype.itemsize)
        staging_bytes += int(max_mutations * int(requirements.max_k_down) * dtype.itemsize)
        staging_bytes += int(max_samples * int(requirements.max_k_down) * dtype.itemsize)
        if requirements.need_down_miss_input:
            staging_bytes += int(max_mutations * int(requirements.max_k_down) * dtype.itemsize)
        if requirements.need_init_vector:
            staging_bytes += int(int(requirements.max_k_down) * dtype.itemsize)
        if requirements.need_init_matrix:
            staging_bytes += int(max_nodes * int(requirements.max_k_down) * dtype.itemsize)
        for level, enabled in enumerate(scratch_down_enabled):
            if enabled:
                scratch_bytes += int(max_rows_by_level[level] * max_down_ops_by_level[level] * int(requirements.max_k_down) * dtype.itemsize)

    blocks: list[_TritonBlockPlan] = []
    planned: list[_TritonArtifactLayout] = []
    for artifact_index, (path, scan) in enumerate(zip(paths, scans, strict=True)):
        share_storage = bool(pair.plan_up is not None and pair.plan_down is not None and pair.plan_up.can_share_storage_with(pair.plan_down))
        blocks_up = () if pair.plan_up is None else tuple(_block_plan(artifact_index, Direction.UP, share_storage, block, pair.plan_up) for block in scan.blocks if block.nnz > 0)
        blocks_down = () if pair.plan_down is None or share_storage else tuple(_block_plan(artifact_index, Direction.DOWN, False, block, pair.plan_down) for block in scan.blocks if block.nnz > 0)
        blocks.extend(blocks_up)
        blocks.extend(blocks_down)
        planned.append(
            _TritonArtifactLayout(
                path=path,
                share_storage=share_storage,
                up_owner=Direction.UP if pair.plan_up is not None else None,
                down_owner=None if pair.plan_down is None else (Direction.UP if share_storage else Direction.DOWN),
                blocks_up=blocks_up,
                blocks_down=blocks_down,
            )
        )

    fixed_bytes = int(selector_bytes + xtx_bias_bytes + staging_bytes + workspace_up + workspace_down + scratch_bytes)
    resident_bytes_full = int(sum(block.nbytes for block in blocks))
    required_budget_for_full_residency = int(fixed_bytes + resident_bytes_full)
    if not allow_residency:
        if blocks and requested_ring_buffer_size < 1:
            raise ValueError("ring_buffer_size must be >= 1 when any Triton block is streamed")
        for block in blocks:
            block.resident = False
        slot_plans, ring_bytes = _assign_slots(blocks, requested_ring_buffer_size) if blocks else ((), 0)
        resident_bytes = 0
        total_bytes = int(fixed_bytes + ring_bytes)
        allocated_ring_buffer_size = len(slot_plans)
        if total_bytes > int(vram_budget_bytes):
            raise ValueError(
                f"Triton layout requires at least {total_bytes} owned device bytes, exceeds vram_budget_bytes={vram_budget_bytes}"
            )
        if allocated_ring_buffer_size != requested_ring_buffer_size:
            _warn_ring_mismatch(requested=requested_ring_buffer_size, allocated=allocated_ring_buffer_size)
    elif required_budget_for_full_residency <= int(vram_budget_bytes):
        for block in blocks:
            block.resident = True
        slot_plans = ()
        ring_bytes = 0
        resident_bytes = resident_bytes_full
        total_bytes = required_budget_for_full_residency
        allocated_ring_buffer_size = 0
        if requested_ring_buffer_size > 0:
            _warn_ring_mismatch(requested=requested_ring_buffer_size, allocated=allocated_ring_buffer_size)
    else:
        if requested_ring_buffer_size < 1:
            raise ValueError("ring_buffer_size must be >= 1 when any Triton block is streamed")
        for block in blocks:
            block.resident = False
        slot_plans, ring_bytes = _assign_slots(blocks, requested_ring_buffer_size)
        resident_bytes = 0
        total_bytes = int(fixed_bytes + ring_bytes)
        if total_bytes > int(vram_budget_bytes):
            raise ValueError(
                f"Triton layout requires at least {total_bytes} owned device bytes, exceeds vram_budget_bytes={vram_budget_bytes}"
            )
        for block in sorted(blocks, key=_promotion_key):
            block.resident = True
            resident_bytes = int(sum(item.nbytes for item in blocks if item.resident))
            slot_plans, ring_bytes = _assign_slots(blocks, requested_ring_buffer_size)
            total_bytes = int(fixed_bytes + resident_bytes + ring_bytes)
            if total_bytes > int(vram_budget_bytes):
                block.resident = False
        resident_bytes = int(sum(item.nbytes for item in blocks if item.resident))
        slot_plans, ring_bytes = _assign_slots(blocks, requested_ring_buffer_size)
        total_bytes = int(fixed_bytes + resident_bytes + ring_bytes)
        allocated_ring_buffer_size = len(slot_plans)
        if allocated_ring_buffer_size != requested_ring_buffer_size:
            _warn_ring_mismatch(requested=requested_ring_buffer_size, allocated=allocated_ring_buffer_size)
    if total_bytes > int(vram_budget_bytes):
        raise ValueError(
            f"Triton layout requires {total_bytes} owned device bytes, exceeds vram_budget_bytes={vram_budget_bytes}"
        )
    bytes_by_category = {
        "resident_sparse": resident_bytes,
        "ring_slots": ring_bytes,
        "selectors": int(selector_bytes),
        "xtx_bias": int(xtx_bias_bytes),
        "staging": int(staging_bytes),
        "workspace_up": int(workspace_up),
        "workspace_down": int(workspace_down),
        "scratch": int(scratch_bytes),
    }
    budget_items: list[BudgetItem] = [
        BudgetItem(kind="fixed", name="selectors", nbytes=int(selector_bytes)),
        BudgetItem(kind="fixed", name="xtx_bias", nbytes=int(xtx_bias_bytes)),
        BudgetItem(kind="fixed", name="staging", nbytes=int(staging_bytes)),
        BudgetItem(kind="fixed", name="workspace_up", nbytes=int(workspace_up)),
        BudgetItem(kind="fixed", name="workspace_down", nbytes=int(workspace_down)),
        BudgetItem(kind="fixed", name="scratch", nbytes=int(scratch_bytes)),
    ]
    for artifact_index, artifact_layout in enumerate(planned):
        for block in (*artifact_layout.blocks_up, *artifact_layout.blocks_down):
            if not block.resident:
                continue
            budget_items.append(
                BudgetItem(
                    kind="resident_sparse",
                    name="resident_sparse",
                    nbytes=int(block.nbytes),
                    artifact_index=artifact_index,
                    dst_level=int(block.dst_level),
                    src_level=int(block.src_level),
                )
            )
    for slot_idx, slot in enumerate(slot_plans):
        budget_items.append(BudgetItem(kind="ring_slot", name="ring_slots", nbytes=int(slot.nbytes), slot=slot_idx))
    return TritonLayout(
        artifacts=tuple(planned),
        pair=pair,
        dtype=dtype,
        requirements=requirements,
        device=device_id,
        stream_ptr=stream_ptr,
        stream_owner=stream_owner,
        allow_residency=allow_residency,
        requested_ring_buffer_size=requested_ring_buffer_size,
        allocated_ring_buffer_size=len(slot_plans),
        vram_budget_bytes=int(vram_budget_bytes),
        max_num_samples=max_samples,
        max_num_mutations=max_mutations,
        max_num_nodes=max_nodes,
        max_levels=max_levels,
        max_rows_by_level=max_rows_by_level,
        max_up_ops_by_level=max_up_ops_by_level,
        max_down_ops_by_level=max_down_ops_by_level,
        scratch_up_enabled=scratch_up_enabled,
        scratch_down_enabled=scratch_down_enabled,
        slot_plans=slot_plans,
        budget_items=tuple(item for item in budget_items if item.nbytes > 0),
        required_budget_for_full_residency=required_budget_for_full_residency,
        bytes_by_category=bytes_by_category,
        bytes_total=int(sum(item.nbytes for item in budget_items)),
    )


class TritonRuntime:
    """Runtime-owned Triton execution with resident or streamed sparse blocks."""

    def __init__(self, layout: TritonLayout) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Triton runtime requires CUDA")
        self.layout = layout
        self.device = torch.device("cuda", int(layout.device))
        self.stream_ptr = int(layout.stream_ptr)
        self.stream = None
        self._caller_stream_keepalive = layout.stream_owner
        self._caller_stream = None
        self._root_stream = None
        self._caller_to_root_event = None
        self._root_to_caller_event = None
        self._level_streams: list[torch.cuda.Stream] = []
        self._slot_copy_streams: list[torch.cuda.Stream] = []
        self._scratch_streams_up: list[list[torch.cuda.Stream]] = []
        self._scratch_streams_down: list[list[torch.cuda.Stream]] = []
        self._slot_indices: list[torch.Tensor] = []
        self._slot_indptr: list[torch.Tensor] = []
        self._up_state = None
        self._down_state = None
        self._up_scratch: list[list[torch.Tensor]] = []
        self._down_scratch: list[list[torch.Tensor]] = []
        self._artifacts: tuple[_TritonArtifact, ...] = ()
        self._grgs: tuple[BoundGRG, ...] = ()
        self._entered = False
        self._active_call = False
        self._config_up: CsrKernelConfig | CscKernelConfig | None = None
        self._config_down: CsrKernelConfig | CscKernelConfig | None = None
        self._staging_up: dict[str, torch.Tensor] = {}
        self._staging_down: dict[str, torch.Tensor] = {}

    @property
    def grgs(self) -> tuple[BoundGRG, ...]:
        if not self._entered:
            raise RuntimeError("TritonRuntime must be entered before accessing grgs")
        return self._grgs

    def _reset_owned_state(self) -> None:
        self._artifacts = ()
        self._grgs = ()
        self._slot_indices = []
        self._slot_indptr = []
        self._level_streams = []
        self._slot_copy_streams = []
        self._scratch_streams_up = []
        self._scratch_streams_down = []
        self._up_state = None
        self._down_state = None
        self._up_scratch = []
        self._down_scratch = []
        self._staging_up = {}
        self._staging_down = {}
        self._config_up = None
        self._config_down = None
        self.stream = None
        self._caller_stream = None
        self._root_stream = None
        self._caller_to_root_event = None
        self._root_to_caller_event = None
        self._entered = False
        self._active_call = False

    def __enter__(self) -> "TritonRuntime":
        artifacts: list[_TritonArtifact] = []
        try:
            with torch.cuda.device(self.device):
                self._caller_stream = torch.cuda.get_stream_from_external(self.stream_ptr, device=self.device)
                self.stream = self._caller_stream
                self._root_stream = torch.cuda.Stream()
                self._caller_to_root_event = torch.cuda.Event()
                self._root_to_caller_event = torch.cuda.Event()
                self._level_streams = [torch.cuda.Stream() for _ in range(self.layout.max_levels)]
                self._slot_copy_streams = [torch.cuda.Stream() for _ in range(self.layout.allocated_ring_buffer_size)]
                self._scratch_streams_up = [
                    [torch.cuda.Stream() for _ in range(self.layout.max_up_ops_by_level[level])] if self.layout.scratch_up_enabled[level] else []
                    for level in range(self.layout.max_levels)
                ]
                self._scratch_streams_down = [
                    [torch.cuda.Stream() for _ in range(self.layout.max_down_ops_by_level[level])] if self.layout.scratch_down_enabled[level] else []
                    for level in range(self.layout.max_levels)
                ]
                self._slot_indices = [
                    torch.zeros((slot.indices_len,), device=self.device, dtype=_torch_int_dtype(slot.indices_dtype))
                    for slot in self.layout.slot_plans
                ]
                self._slot_indptr = [
                    torch.zeros((slot.indptr_len,), device=self.device, dtype=_torch_int_dtype(slot.indptr_dtype))
                    for slot in self.layout.slot_plans
                ]
                if self.layout.pair.plan_up is not None:
                    max_k_up = int(self.layout.requirements.max_k_up)
                    self._up_state = torch.zeros((self.layout.max_num_nodes, max_k_up), device=self.device, dtype=self._torch_dtype())
                    self._staging_up["input_primary"] = torch.zeros((self.layout.max_num_samples, max_k_up), device=self.device, dtype=self._torch_dtype())
                    self._staging_up["output_main"] = torch.zeros((self.layout.max_num_mutations, max_k_up), device=self.device, dtype=self._torch_dtype())
                    if self.layout.requirements.need_up_miss_output:
                        self._staging_up["output_miss"] = torch.zeros((self.layout.max_num_mutations, max_k_up), device=self.device, dtype=self._torch_dtype())
                    if self.layout.requirements.need_init_vector:
                        self._staging_up["init_vector"] = torch.zeros((1, max_k_up), device=self.device, dtype=self._torch_dtype())
                    if self.layout.requirements.need_init_matrix:
                        self._staging_up["init_matrix"] = torch.zeros((self.layout.max_num_nodes, max_k_up), device=self.device, dtype=self._torch_dtype())
                    self._up_scratch = [
                        [torch.zeros((self.layout.max_rows_by_level[level], max_k_up), device=self.device, dtype=self._torch_dtype()) for _ in range(self.layout.max_up_ops_by_level[level])]
                        if self.layout.scratch_up_enabled[level]
                        else []
                        for level in range(self.layout.max_levels)
                    ]
                if self.layout.pair.plan_down is not None:
                    max_k_down = int(self.layout.requirements.max_k_down)
                    self._down_state = torch.zeros((self.layout.max_num_nodes, max_k_down), device=self.device, dtype=self._torch_dtype())
                    self._staging_down["input_primary"] = torch.zeros((self.layout.max_num_mutations, max_k_down), device=self.device, dtype=self._torch_dtype())
                    self._staging_down["output_main"] = torch.zeros((self.layout.max_num_samples, max_k_down), device=self.device, dtype=self._torch_dtype())
                    if self.layout.requirements.need_down_miss_input:
                        self._staging_down["input_miss"] = torch.zeros((self.layout.max_num_mutations, max_k_down), device=self.device, dtype=self._torch_dtype())
                    if self.layout.requirements.need_init_vector:
                        self._staging_down["init_vector"] = torch.zeros((1, max_k_down), device=self.device, dtype=self._torch_dtype())
                    if self.layout.requirements.need_init_matrix:
                        self._staging_down["init_matrix"] = torch.zeros((self.layout.max_num_nodes, max_k_down), device=self.device, dtype=self._torch_dtype())
                    self._down_scratch = [
                        [torch.zeros((self.layout.max_rows_by_level[level], max_k_down), device=self.device, dtype=self._torch_dtype()) for _ in range(self.layout.max_down_ops_by_level[level])]
                        if self.layout.scratch_down_enabled[level]
                        else []
                        for level in range(self.layout.max_levels)
                    ]

            states = tuple(_load_grg_spmv_host(artifact.path, self.layout.dtype) for artifact in self.layout.artifacts)
            for artifact_layout, state in zip(self.layout.artifacts, states, strict=True):
                artifacts.append(self._build_artifact(artifact_layout, state))
            self._artifacts = tuple(artifacts)
            self._grgs = tuple(BoundGRG(self, idx, artifact.state, artifact.path) for idx, artifact in enumerate(self._artifacts))
            if self.layout.pair.plan_up is not None:
                self._config_up = self._tune_direction(Direction.UP)
            if self.layout.pair.plan_down is not None:
                self._config_down = self._tune_direction(Direction.DOWN)
            self._entered = True
            return self
        except Exception:
            self._reset_owned_state()
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        self._reset_owned_state()

    def _torch_dtype(self) -> torch.dtype:
        return torch.float64 if np.dtype(self.layout.dtype) == np.float64 else torch.float32

    @contextmanager
    def _call_scope(self):
        if not self._entered:
            raise RuntimeError("TritonRuntime must be entered before matmul")
        if self._active_call:
            raise RuntimeError("concurrent runtime.grgs calls are not supported")
        self._active_call = True
        try:
            yield
        finally:
            self._active_call = False

    @contextmanager
    def _caller_root_scope(self):
        assert self._caller_stream is not None
        assert self._root_stream is not None
        assert self._caller_to_root_event is not None
        assert self._root_to_caller_event is not None
        with torch.cuda.device(self.device):
            with torch.cuda.stream(self._caller_stream):
                self._caller_to_root_event.record(self._caller_stream)
            self._root_stream.wait_event(self._caller_to_root_event)
            try:
                yield
            finally:
                with torch.cuda.stream(self._root_stream):
                    self._root_to_caller_event.record(self._root_stream)
                self._caller_stream.wait_event(self._root_to_caller_event)

    def _pinned_tensor(self, values: np.ndarray, *, dtype: np.dtype, label: str) -> torch.Tensor:
        host = torch.empty(values.shape, dtype=_torch_int_dtype(dtype), pin_memory=True)
        _copy_struct_checked(host.numpy(), values, label=label)
        return host

    def _build_artifact(self, artifact_layout: _TritonArtifactLayout, state) -> _TritonArtifact:
        up_owner: dict[tuple[int, int], _TritonLaunchBlock | _TritonStreamSource] = {}
        down_owner: dict[tuple[int, int], _TritonLaunchBlock | _TritonStreamSource] = {}
        plan_up = self.layout.pair.plan_up
        plan_down = self.layout.pair.plan_down
        up_plan_map = {(block.dst_level, block.src_level): block for block in artifact_layout.blocks_up}
        down_plan_map = {(block.dst_level, block.src_level): block for block in artifact_layout.blocks_down}
        with torch.cuda.device(self.device):
            for block in iter_artifact_blocks(artifact_layout.path):
                base = sp.csr_matrix(
                    (
                        np.ones(block.nnz, dtype=np.bool_),
                        np.asarray(block.indices),
                        np.asarray(block.indptr),
                    ),
                    shape=block.shape,
                )
                plan = up_plan_map.get((block.dst_level, block.src_level))
                if plan is not None and plan_up is not None:
                    sparse = materialize_sparse_block(base, store=plan_up.store, fmt=plan_up.fmt)
                    if plan.resident:
                        indices = torch.from_numpy(np.asarray(sparse.indices)).to(device=self.device, dtype=_torch_int_dtype(plan.indices_dtype))
                        indptr = torch.from_numpy(np.asarray(sparse.indptr)).to(device=self.device, dtype=_torch_int_dtype(plan.indptr_dtype))
                        up_owner[(block.dst_level, block.src_level)] = _TritonLaunchBlock(indices, indptr, plan.stored_shape[0], plan.stored_shape[1], plan.nnz, plan.fmt)
                    else:
                        assert plan.slot is not None
                        up_owner[(block.dst_level, block.src_level)] = _TritonStreamSource(
                            slot=plan.slot,
                            indices_host=self._pinned_tensor(np.asarray(sparse.indices), dtype=plan.indices_dtype, label="Triton streamed indices"),
                            indptr_host=self._pinned_tensor(np.asarray(sparse.indptr), dtype=plan.indptr_dtype, label="Triton streamed indptr"),
                            nrows=plan.stored_shape[0],
                            ncols=plan.stored_shape[1],
                            nnz=plan.nnz,
                            fmt=plan.fmt,
                        )
                plan = down_plan_map.get((block.dst_level, block.src_level))
                if plan is not None and plan_down is not None:
                    sparse = materialize_sparse_block(base, store=plan_down.store, fmt=plan_down.fmt)
                    if plan.resident:
                        indices = torch.from_numpy(np.asarray(sparse.indices)).to(device=self.device, dtype=_torch_int_dtype(plan.indices_dtype))
                        indptr = torch.from_numpy(np.asarray(sparse.indptr)).to(device=self.device, dtype=_torch_int_dtype(plan.indptr_dtype))
                        down_owner[(block.dst_level, block.src_level)] = _TritonLaunchBlock(indices, indptr, plan.stored_shape[0], plan.stored_shape[1], plan.nnz, plan.fmt)
                    else:
                        assert plan.slot is not None
                        down_owner[(block.dst_level, block.src_level)] = _TritonStreamSource(
                            slot=plan.slot,
                            indices_host=self._pinned_tensor(np.asarray(sparse.indices), dtype=plan.indices_dtype, label="Triton streamed indices"),
                            indptr_host=self._pinned_tensor(np.asarray(sparse.indptr), dtype=plan.indptr_dtype, label="Triton streamed indptr"),
                            nrows=plan.stored_shape[0],
                            ncols=plan.stored_shape[1],
                            nnz=plan.nnz,
                            fmt=plan.fmt,
                        )
        sel_mut = state.sel_mut.tocoo()
        sel_miss = state.sel_miss.tocoo()
        with torch.cuda.device(self.device):
            xtx_bias = None
            if self.layout.requirements.need_init_xtx and state.coalescence_counts is not None:
                xtx_bias = torch.from_numpy(2.0 * state.coalescence_counts.astype(self.layout.dtype, copy=False)).to(device=self.device, dtype=self._torch_dtype())
            return _TritonArtifact(
                path=artifact_layout.path,
                state=state,
                sel_mut_rows=torch.from_numpy(np.asarray(sel_mut.row)).to(device=self.device),
                sel_mut_cols=torch.from_numpy(np.asarray(sel_mut.col)).to(device=self.device),
                sel_miss_rows=torch.from_numpy(np.asarray(sel_miss.row)).to(device=self.device),
                sel_miss_cols=torch.from_numpy(np.asarray(sel_miss.col)).to(device=self.device),
                xtx_bias=xtx_bias,
                up_ops=self._build_ops(Direction.UP, artifact_layout, state, up_owner, down_owner),
                down_ops=self._build_ops(Direction.DOWN, artifact_layout, state, up_owner, down_owner),
            )

    def _build_ops(self, direction: Direction, artifact_layout: _TritonArtifactLayout, state, up_owner, down_owner) -> list[list[_TritonOp]]:
        h = len(state.level_offsets) - 1
        ops: list[list[_TritonOp]] = [[] for _ in range(h)]
        if (direction == Direction.UP and self.layout.pair.plan_up is None) or (direction == Direction.DOWN and self.layout.pair.plan_down is None):
            return ops
        owner_direction = artifact_layout.up_owner if direction == Direction.UP else artifact_layout.down_owner
        assert owner_direction is not None
        owner_blocks = up_owner if owner_direction == Direction.UP else down_owner
        for dst_level, src_level, _row_index in iter_direction_level_pairs(direction, h):
            owner_dst = dst_level if direction == Direction.UP else src_level
            owner_src = src_level if direction == Direction.UP else dst_level
            owner_value = owner_blocks.get((owner_dst, owner_src))
            if owner_value is None:
                continue
            if isinstance(owner_value, _TritonLaunchBlock):
                block = owner_value if owner_direction == direction else owner_value.transpose_alias()
                ops[dst_level].append(_TritonOp(src_level, block, None, None, None, None))
                continue
            source = owner_value if owner_direction == direction else owner_value.transpose_alias()
            slot = int(source.slot)
            block = _TritonLaunchBlock(
                indices=self._slot_indices[slot],
                indptr=self._slot_indptr[slot],
                nrows=source.nrows,
                ncols=source.ncols,
                nnz=source.nnz,
                fmt=source.fmt,
            )
            ops[dst_level].append(
                _TritonOp(
                    src_level=src_level,
                    launch_block=block,
                    slot=slot,
                    indices_host=source.indices_host,
                    indptr_host=source.indptr_host,
                    prev_in_slot=None,
                )
            )
        return _relink_stream_dependencies(ops, direction=direction)

    def _view(self, state_tensor: torch.Tensor, state, level: int, k: int) -> torch.Tensor:
        lo = int(state.level_offsets[level])
        hi = int(state.level_offsets[level + 1])
        return state_tensor[lo:hi, :k]

    def _prev_compute_event(self, compute_done, op: _TritonOp):
        if op.prev_in_slot is None:
            return None
        prev_dst, prev_idx = op.prev_in_slot
        return compute_done[prev_dst][prev_idx]

    def _copy_host_block_to_slot(self, copy_done, compute_done, dst_level: int, op_idx: int, op: _TritonOp) -> None:
        assert op.slot is not None
        assert op.indices_host is not None and op.indptr_host is not None
        stream = self._slot_copy_streams[op.slot]
        with torch.cuda.stream(stream):
            prev_event = self._prev_compute_event(compute_done, op)
            if prev_event is not None:
                stream.wait_event(prev_event)
            self._slot_indices[op.slot][: op.indices_host.numel()].copy_(op.indices_host, non_blocking=True)
            self._slot_indptr[op.slot][: op.indptr_host.numel()].copy_(op.indptr_host, non_blocking=True)
            copy_done[dst_level][op_idx].record(stream)

    def _enqueue_wavefront(self, artifact: _TritonArtifact, direction: Direction, k: int) -> None:
        state_tensor = self._up_state if direction == Direction.UP else self._down_state
        assert state_tensor is not None
        ops_by_level = artifact.up_ops if direction == Direction.UP else artifact.down_ops
        scratch_enabled = self.layout.scratch_up_enabled if direction == Direction.UP else self.layout.scratch_down_enabled
        scratch_streams = self._scratch_streams_up if direction == Direction.UP else self._scratch_streams_down
        scratch_buffers = self._up_scratch if direction == Direction.UP else self._down_scratch
        h = len(artifact.state.level_offsets) - 1
        if direction == Direction.UP:
            seed_level = 0
            level_iter = range(1, h)
        else:
            seed_level = h - 1
            level_iter = range(h - 2, -1, -1)

        copy_done = [[torch.cuda.Event() for _ in ops] for ops in ops_by_level]
        compute_done = [[torch.cuda.Event() for _ in ops] for ops in ops_by_level]
        level_ready = [torch.cuda.Event() for _ in range(h)]
        launch_event = torch.cuda.Event()
        with torch.cuda.stream(self._root_stream):
            launch_event.record(self._root_stream)
        for stream in self._slot_copy_streams:
            stream.wait_event(launch_event)
        for stream in self._level_streams:
            stream.wait_event(launch_event)
        for per_level in scratch_streams:
            for stream in per_level:
                stream.wait_event(launch_event)
        if h > 0:
            with torch.cuda.stream(self._level_streams[seed_level]):
                level_ready[seed_level].record(self._level_streams[seed_level])
        for dst_level in level_iter:
            stream = self._level_streams[dst_level]
            ops = ops_by_level[dst_level]
            if scratch_enabled[dst_level]:
                done_events = [torch.cuda.Event() for _ in ops]
                with torch.cuda.stream(stream):
                    pass
                for op_idx, op in enumerate(ops):
                    helper = scratch_streams[dst_level][op_idx]
                    if op.slot is not None:
                        self._copy_host_block_to_slot(copy_done, compute_done, dst_level, op_idx, op)
                    dst_view = self._view(state_tensor, artifact.state, dst_level, k)
                    helper_view = scratch_buffers[dst_level][op_idx][: dst_view.shape[0], :k]
                    with torch.cuda.stream(helper):
                        helper.wait_event(level_ready[op.src_level])
                        if op.slot is not None:
                            helper.wait_event(copy_done[dst_level][op_idx])
                        helper_view.zero_()
                        launch_block(
                            block=op.launch_block,
                            x=self._view(state_tensor, artifact.state, op.src_level, k),
                            y=helper_view,
                            config=self._config_for(direction),
                            fp64_acc=self._torch_dtype() == torch.float64,
                        )
                        compute_done[dst_level][op_idx].record(helper)
                        done_events[op_idx].record(helper)
                with torch.cuda.stream(stream):
                    dst_view = self._view(state_tensor, artifact.state, dst_level, k)
                    for op_idx, done in enumerate(done_events):
                        stream.wait_event(done)
                        dst_view.add_(scratch_buffers[dst_level][op_idx][: dst_view.shape[0], :k])
                    level_ready[dst_level].record(stream)
                continue
            with torch.cuda.stream(stream):
                dst_view = self._view(state_tensor, artifact.state, dst_level, k)
                for op_idx, op in enumerate(ops):
                    if op.slot is not None:
                        self._copy_host_block_to_slot(copy_done, compute_done, dst_level, op_idx, op)
                    stream.wait_event(level_ready[op.src_level])
                    if op.slot is not None:
                        stream.wait_event(copy_done[dst_level][op_idx])
                    launch_block(
                        block=op.launch_block,
                        x=self._view(state_tensor, artifact.state, op.src_level, k),
                        y=dst_view,
                        config=self._config_for(direction),
                        fp64_acc=self._torch_dtype() == torch.float64,
                    )
                    compute_done[dst_level][op_idx].record(stream)
                level_ready[dst_level].record(stream)
        with torch.cuda.stream(self._root_stream):
            for event in level_ready:
                self._root_stream.wait_event(event)

    def _config_for(self, direction: Direction):
        config = self._config_up if direction == Direction.UP else self._config_down
        if config is None:
            raise RuntimeError(f"Triton {direction.value} kernel config is not initialized")
        return config

    def _candidate_configs(self, direction: Direction):
        plan = self.layout.pair.plan_up if direction == Direction.UP else self.layout.pair.plan_down
        assert plan is not None
        return CSR_CANDIDATE_CONFIGS if plan.fmt == SparseFormat.CSR else CSC_CANDIDATE_CONFIGS

    def _tune_direction(self, direction: Direction):
        dummy = self._artifacts[0]
        best = None
        best_ms = None
        for config in self._candidate_configs(direction):
            self._bench_once(dummy, direction, config)
            timing = float(
                triton.testing.do_bench(
                    lambda config=config: self._bench_once(dummy, direction, config),
                    warmup=_TUNE_WARMUP_MS,
                    rep=_TUNE_REP_MS,
                )
            )
            if best_ms is None or timing < best_ms:
                best = config
                best_ms = timing
        assert best is not None
        return best

    def _bench_once(self, artifact: _TritonArtifact, direction: Direction, config):
        old = self._config_up if direction == Direction.UP else self._config_down
        if direction == Direction.UP:
            self._config_up = config
        else:
            self._config_down = config
        try:
            with self._caller_root_scope():
                active_k = self._max_k(direction)
                self._seed_for_tune(artifact, direction, active_k)
                self._enqueue_wavefront(artifact, direction, active_k)
            torch.cuda.synchronize(self.device)
        finally:
            if direction == Direction.UP:
                self._config_up = old
            else:
                self._config_down = old

    def _max_k(self, direction: Direction) -> int:
        return int(self.layout.requirements.max_k_up if direction == Direction.UP else self.layout.requirements.max_k_down)

    def _seed_for_tune(self, artifact: _TritonArtifact, direction: Direction, k: int) -> None:
        state_tensor = self._up_state if direction == Direction.UP else self._down_state
        staging = self._staging_up if direction == Direction.UP else self._staging_down
        assert state_tensor is not None
        with torch.cuda.stream(self._root_stream):
            state_tensor.zero_()
            staging["input_primary"].zero_()
            staging["input_primary"][:, :k].fill_(1.0)
            if direction == Direction.UP:
                state_tensor[: artifact.state.num_samples, :k].add_(staging["input_primary"][: artifact.state.num_samples, :k])
            else:
                if artifact.sel_mut_rows.numel() > 0:
                    state_tensor[:, :k].index_add_(0, artifact.sel_mut_cols, staging["input_primary"][:, :k].index_select(0, artifact.sel_mut_rows))

    def _stage_inputs(self, direction: Direction, primary: np.ndarray, miss: np.ndarray | None, init_mode: InitMode, init_payload: np.ndarray | None) -> None:
        staging = self._staging_up if direction == Direction.UP else self._staging_down
        input_primary = staging["input_primary"]
        k = int(primary.shape[1])
        with torch.cuda.stream(self._root_stream):
            input_primary.zero_()
            input_primary[: primary.shape[0], :k].copy_(torch.from_numpy(np.asarray(primary, dtype=self.layout.dtype)), non_blocking=False)
            if direction == Direction.DOWN and miss is not None and "input_miss" in staging:
                staging["input_miss"].zero_()
                staging["input_miss"][: miss.shape[0], :k].copy_(torch.from_numpy(np.asarray(miss, dtype=self.layout.dtype)), non_blocking=False)
            if init_mode == InitMode.VECTOR and "init_vector" in staging and init_payload is not None:
                staging["init_vector"].zero_()
                staging["init_vector"][0, :k].copy_(torch.from_numpy(np.asarray(init_payload, dtype=self.layout.dtype)), non_blocking=False)
            if init_mode == InitMode.MATRIX and "init_matrix" in staging and init_payload is not None:
                staging["init_matrix"].zero_()
                staging["init_matrix"][: init_payload.shape[0], :k].copy_(torch.from_numpy(np.asarray(init_payload, dtype=self.layout.dtype)), non_blocking=False)

    def _seed_workspace(self, artifact: _TritonArtifact, direction: Direction, init_mode: InitMode, has_miss_input: bool, k: int) -> None:
        state_tensor = self._up_state if direction == Direction.UP else self._down_state
        staging = self._staging_up if direction == Direction.UP else self._staging_down
        assert state_tensor is not None
        with torch.cuda.stream(self._root_stream):
            state_tensor.zero_()
            if init_mode == InitMode.XTX:
                if artifact.xtx_bias is None:
                    raise ValueError("init_mode=xtx requires GRG coalescence counts")
                state_tensor[: artifact.state.num_nodes, :k].add_(artifact.xtx_bias[: artifact.state.num_nodes, None])
            elif init_mode == InitMode.VECTOR and "init_vector" in staging:
                state_tensor[: artifact.state.num_nodes, :k].add_(staging["init_vector"][0, :k])
            elif init_mode == InitMode.MATRIX and "init_matrix" in staging:
                state_tensor[: artifact.state.num_nodes, :k].add_(staging["init_matrix"][: artifact.state.num_nodes, :k])
            if direction == Direction.UP:
                state_tensor[: artifact.state.num_samples, :k].add_(staging["input_primary"][: artifact.state.num_samples, :k])
            else:
                if artifact.sel_mut_rows.numel() > 0:
                    state_tensor[:, :k].index_add_(0, artifact.sel_mut_cols, staging["input_primary"][:, :k].index_select(0, artifact.sel_mut_rows))
                if has_miss_input and artifact.sel_miss_rows.numel() > 0 and "input_miss" in staging:
                    state_tensor[:, :k].index_add_(0, artifact.sel_miss_cols, staging["input_miss"][:, :k].index_select(0, artifact.sel_miss_rows))

    def _run(
        self,
        artifact_index: int,
        direction: Direction,
        primary: np.ndarray,
        *,
        miss: np.ndarray | None,
        init_mode: InitMode,
        init_payload: np.ndarray | None,
        need_miss_output: bool,
        emit_all_nodes: bool,
    ):
        if direction == Direction.UP and self.layout.pair.plan_up is None:
            raise ValueError("UP plan is not configured")
        if direction == Direction.DOWN and self.layout.pair.plan_down is None:
            raise ValueError("DOWN plan is not configured")
        k = int(primary.shape[1])
        artifact = self._artifacts[int(artifact_index)]
        with self._caller_root_scope():
            self._stage_inputs(direction, primary, miss, init_mode, init_payload)
            self._seed_workspace(artifact, direction, init_mode, has_miss_input=miss is not None, k=k)
            self._enqueue_wavefront(artifact, direction, k)
            torch.cuda.synchronize(self.device)
        state_tensor = self._up_state if direction == Direction.UP else self._down_state
        assert state_tensor is not None
        if emit_all_nodes:
            return state_tensor[: artifact.state.num_nodes, :k].cpu().numpy().copy()
        if direction == Direction.UP:
            staging = self._staging_up
            output_main = staging["output_main"]
            output_main.zero_()
            node_view = state_tensor[: artifact.state.num_nodes, :k]
            if artifact.sel_mut_rows.numel() > 0:
                output_main[: artifact.state.num_mutations, :k].index_add_(0, artifact.sel_mut_rows, node_view.index_select(0, artifact.sel_mut_cols))
            out_miss = None
            if need_miss_output:
                output_miss = staging["output_miss"]
                output_miss.zero_()
                if artifact.sel_miss_rows.numel() > 0:
                    output_miss[: artifact.state.num_mutations, :k].index_add_(0, artifact.sel_miss_rows, node_view.index_select(0, artifact.sel_miss_cols))
                out_miss = output_miss[: artifact.state.num_mutations, :k].cpu().numpy().copy()
            return output_main[: artifact.state.num_mutations, :k].cpu().numpy().copy(), out_miss
        output_main = self._staging_down["output_main"]
        output_main.zero_()
        output_main[: artifact.state.num_samples, :k].copy_(state_tensor[: artifact.state.num_samples, :k])
        return output_main[: artifact.state.num_samples, :k].cpu().numpy().copy()


__all__ = ["TritonLayout", "TritonRuntime", "plan_triton_layout"]
