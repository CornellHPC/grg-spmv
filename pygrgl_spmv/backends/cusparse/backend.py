"""cuSPARSE runtime and layout planner."""

from __future__ import annotations

from contextlib import contextmanager
from ctypes import c_void_p
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends._cuda_stream import CudaStreamToken, _cuda_stream_device, parse_cuda_device, parse_cuda_stream
from pygrgl_spmv.backends.base import (
    BudgetItem,
    _copy_struct_checked,
    _layout_struct_dtypes,
    _require_struct_dtype,
    iter_direction_level_pairs,
    materialize_sparse_block,
    relink_stream_dependencies,
    sparse_structure_lengths,
    split_selector_by_level,
    stored_block_shape,
)
from pygrgl_spmv.backends.cusparse.ffi import (
    CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_64I,
    CuSparseLib,
    CudaVmmDriver,
    cuda_dtype,
)
from pygrgl_spmv.backends.types import Direction, InitMode, SparseFormat, StoredMatrix
from pygrgl_spmv.grg import BoundGRG, RuntimeRequirements, _CudaMatmulSpec
from pygrgl_spmv.grg.artifact import _load_grg_spmv_host, iter_artifact_blocks, scan_grg_spmv

from .plan import CusparsePlan, CusparsePlanPair, DenseOrder, Operation


@dataclass
class _CuBlockPlan:
    artifact_index: int
    owner: Direction
    shared: bool
    dst_level: int
    src_level: int
    fmt: SparseFormat
    store: StoredMatrix
    stored_shape: tuple[int, int]
    nnz: int
    struct0_dtype: np.dtype
    struct1_dtype: np.dtype
    struct0_len: int
    struct1_len: int
    nbytes: int
    resident: bool = True
    slot: int | None = None


@dataclass(frozen=True)
class _CuArtifactLayout:
    path: Path
    share_storage: bool
    up_owner: Direction | None
    down_owner: Direction | None
    blocks_up: tuple[_CuBlockPlan, ...]
    blocks_down: tuple[_CuBlockPlan, ...]


@dataclass(frozen=True)
class _CuSlotPlan:
    struct0_dtype: np.dtype
    struct0_len: int
    struct1_dtype: np.dtype
    struct1_len: int

    @property
    def nbytes(self) -> int:
        return int(self.struct0_len * self.struct0_dtype.itemsize + self.struct1_len * self.struct1_dtype.itemsize)


@dataclass(frozen=True)
class _SharedOnesPlan:
    mode: str
    logical_bytes: int
    physical_bytes: int


@dataclass
class CusparseLayout:
    artifacts: tuple[_CuArtifactLayout, ...]
    pair: CusparsePlanPair
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
    max_selector_nnz: int
    max_levels: int
    max_rows_by_level: tuple[int, ...]
    max_up_ops_by_level: tuple[int, ...]
    max_down_ops_by_level: tuple[int, ...]
    scratch_up_enabled: tuple[bool, ...]
    scratch_down_enabled: tuple[bool, ...]
    slot_plans: tuple[_CuSlotPlan, ...]
    ext_main_up: tuple[int, ...]
    ext_main_down: tuple[int, ...]
    ext_scratch_up: tuple[tuple[int, ...], ...]
    ext_scratch_down: tuple[tuple[int, ...], ...]
    shared_ones: _SharedOnesPlan
    budget_items: tuple[BudgetItem, ...]
    required_budget_for_full_residency: int
    bytes_by_category: dict[str, int]
    bytes_total: int


@dataclass
class _SharedOnes:
    ptr: int
    logical_nbytes: int
    physical_nbytes: int
    vmm: bool
    _materialized: Any | None = None
    _driver: CudaVmmDriver | None = None
    _vaddr: int = 0
    _reserved_nbytes: int = 0
    _handle: int = 0

    def destroy(self) -> None:
        self._materialized = None
        driver = self._driver
        if driver is not None and self._vaddr and self._reserved_nbytes:
            try:
                driver.mem_unmap(self._vaddr, self._reserved_nbytes)
            finally:
                try:
                    if self._handle:
                        driver.mem_release(self._handle)
                finally:
                    driver.address_free(self._vaddr, self._reserved_nbytes)
        self.ptr = 0


@dataclass(frozen=True)
class _CuRuntimeBlock:
    struct0: Any
    struct1: Any
    nrows: int
    ncols: int
    nnz: int
    fmt: SparseFormat

    def transpose_alias(self) -> "_CuRuntimeBlock":
        return _CuRuntimeBlock(
            struct0=self.struct0,
            struct1=self.struct1,
            nrows=self.ncols,
            ncols=self.nrows,
            nnz=self.nnz,
            fmt=SparseFormat.COO if self.fmt == SparseFormat.COO else (SparseFormat.CSC if self.fmt == SparseFormat.CSR else SparseFormat.CSR),
        )


@dataclass(frozen=True)
class _CuOp:
    src_level: int
    sp_desc: c_void_p
    block: _CuRuntimeBlock
    slot: int | None
    host0: np.ndarray | None
    host1: np.ndarray | None
    prev_in_slot: tuple[int, int] | None


@dataclass(frozen=True)
class _SelectorLevels:
    rows_by_level: list[Any]
    cols_by_level: list[Any]
    row_unique: bool


@dataclass
class _DenseDescriptorSet:
    dst_descs: list[c_void_p]
    src_descs: list[c_void_p]
    src_bufs: list[Any] | None
    scratch_descs_by_level: list[list[c_void_p]]


@dataclass
class _PreparedDenseEntry:
    state: _DenseDescriptorSet
    refs: int = 0


@dataclass
class _CuArtifact:
    path: Path
    state: object
    mut_selector: _SelectorLevels
    miss_selector: _SelectorLevels
    node_perm: Any
    sample_to_individual: Any
    xtx_bias: Any | None
    init_vector_up_bias: Any | None
    init_vector_down_bias: Any | None
    init_xtx_up_bias: Any | None
    init_xtx_down_bias: Any | None
    up_ops: list[list[_CuOp]]
    down_ops: list[list[_CuOp]]
    up_dense_by_k: dict[int, _PreparedDenseEntry]
    down_dense_by_k: dict[int, _PreparedDenseEntry]


def _round_up(value: int, alignment: int) -> int:
    return int(((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment))


def _numpy_ptr(value: np.ndarray) -> int:
    return int(np.asarray(value).__array_interface__["data"][0])


def _torch_from_cupy(array):
    import torch

    return torch.from_dlpack(array)


def _dense_entries(artifact: _CuArtifact, direction: Direction) -> dict[int, _PreparedDenseEntry]:
    return artifact.up_dense_by_k if direction == Direction.UP else artifact.down_dense_by_k


class _CusparsePreparedMatmul:
    def __init__(self, runtime: "CusparseRuntime", artifact: _CuArtifact, spec: _CudaMatmulSpec) -> None:
        self._runtime = runtime
        self._artifact = artifact
        self._spec = spec
        k = int(spec.k)
        self._input_internal = runtime._io0[: spec.input_cols, :k]
        self.input = runtime._io0_torch[: spec.input_cols, :k].T
        if spec.emit_all_nodes:
            self._output_internal = runtime._aux[: artifact.state.num_nodes, :k]
            self.output = runtime._aux_torch[: artifact.state.num_nodes, :k].T
        else:
            self._output_internal = runtime._io0[: spec.output_cols, :k]
            self.output = runtime._io0_torch[: spec.output_cols, :k].T
        if spec.use_miss:
            assert runtime._io1 is not None and runtime._io1_torch is not None
            miss_view = runtime._io1[: artifact.state.num_mutations, :k]
            miss_torch = runtime._io1_torch[: artifact.state.num_mutations, :k].T
            if spec.direction == Direction.DOWN:
                self._miss_input_internal = miss_view
                self.miss_input = miss_torch
            else:
                self._miss_output_internal = miss_view
                self.miss_output = miss_torch
        if spec.init_mode == InitMode.VECTOR:
            assert runtime._io1 is not None and runtime._io1_torch is not None
            self._init_vector_internal = runtime._io1[0, :k]
            self.init_vector = runtime._io1_torch[0, :k]
        if spec.init_mode == InitMode.MATRIX:
            assert runtime._io1 is not None and runtime._io1_torch is not None
            self._init_matrix_internal = runtime._io1[: artifact.state.num_nodes, :k]
            self.init_matrix = runtime._io1_torch[: artifact.state.num_nodes, :k].T
        self._dense_entry: _PreparedDenseEntry | None = None
        self._dense_state: _DenseDescriptorSet | None = None

    def __enter__(self) -> "_CusparsePreparedMatmul":
        if self._dense_entry is not None:
            raise RuntimeError("cuSPARSE prepared matmul is already entered")
        entry = self._runtime._acquire_dense_descriptors(self._artifact, self._spec.direction, self._spec.k)
        self._dense_entry = entry
        self._dense_state = entry.state
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        entry = self._dense_entry
        try:
            if entry is not None:
                self._runtime._release_dense_descriptors(self._artifact, self._spec.direction, self._spec.k, entry)
        finally:
            self._dense_entry = None
            self._dense_state = None

    def __call__(self) -> None:
        with self._runtime._call_scope():
            self._call_prelocked()

    def _call_prelocked(self) -> None:
        self._runtime._execute_prepared(self._artifact, self._spec, self)


def _cusparse_index_type(dtype: np.dtype) -> int:
    dt = _require_struct_dtype(dtype, label="cuSPARSE structural dtype")
    return CUSPARSE_INDEX_32I if dt == np.dtype(np.int32) else CUSPARSE_INDEX_64I


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


def _block_plan(artifact_index: int, owner: Direction, shared: bool, block, plan: CusparsePlan) -> _CuBlockPlan:
    nrows, ncols = stored_block_shape(block.shape[0], block.shape[1], store=plan.store)
    struct0_dtype, struct1_dtype = _layout_struct_dtypes(plan.fmt, nrows=nrows, ncols=ncols, nnz=block.nnz)
    struct0_len, struct1_len = sparse_structure_lengths(plan.fmt, nrows=nrows, ncols=ncols, nnz=block.nnz)
    nbytes = int(struct0_len * struct0_dtype.itemsize + struct1_len * struct1_dtype.itemsize)
    return _CuBlockPlan(
        artifact_index=artifact_index,
        owner=owner,
        shared=shared,
        dst_level=block.dst_level,
        src_level=block.src_level,
        fmt=plan.fmt,
        store=plan.store,
        stored_shape=(nrows, ncols),
        nnz=int(block.nnz),
        struct0_dtype=np.dtype(struct0_dtype),
        struct1_dtype=np.dtype(struct1_dtype),
        struct0_len=int(struct0_len),
        struct1_len=int(struct1_len),
        nbytes=nbytes,
    )


def _resolve_scratch_levels(plan: CusparsePlan | None, height: int) -> tuple[bool, ...]:
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
        raise ValueError(f"cuSPARSE scratch levels out of range: {invalid}; valid range is [0, {height})")
    return tuple(level in levels for level in range(height))


def _assign_slots(blocks: list[_CuBlockPlan], ring_buffer_size: int) -> tuple[tuple[_CuSlotPlan, ...], int]:
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
        raise ValueError("ring_buffer_size must be >= 1 when any cuSPARSE block is streamed")
    slot0_dtype = [np.dtype(np.int32) for _ in range(allocated_ring_buffer_size)]
    slot1_dtype = [np.dtype(np.int32) for _ in range(allocated_ring_buffer_size)]
    slot0_len = [0 for _ in range(allocated_ring_buffer_size)]
    slot1_len = [0 for _ in range(allocated_ring_buffer_size)]
    for idx, block in enumerate(streamed):
        slot = int(idx % allocated_ring_buffer_size)
        block.slot = slot
        if block.struct0_dtype == np.dtype(np.int64):
            slot0_dtype[slot] = np.dtype(np.int64)
        if block.struct1_dtype == np.dtype(np.int64):
            slot1_dtype[slot] = np.dtype(np.int64)
        if block.fmt == SparseFormat.COO and (slot0_dtype[slot] == np.dtype(np.int64) or slot1_dtype[slot] == np.dtype(np.int64)):
            slot0_dtype[slot] = np.dtype(np.int64)
            slot1_dtype[slot] = np.dtype(np.int64)
        slot0_len[slot] = max(slot0_len[slot], int(block.struct0_len))
        slot1_len[slot] = max(slot1_len[slot], int(block.struct1_len))
    slots = tuple(
        _CuSlotPlan(slot0_dtype[slot], slot0_len[slot], slot1_dtype[slot], slot1_len[slot])
        for slot in range(allocated_ring_buffer_size)
    )
    return slots, int(sum(slot.nbytes for slot in slots))


def _promotion_key(block: _CuBlockPlan) -> tuple[int, int, int, int, int]:
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


def _enabled_max_k(pair: CusparsePlanPair, requirements: RuntimeRequirements) -> int:
    values = []
    if pair.plan_up is not None:
        values.append(int(requirements.max_k_up))
    if pair.plan_down is not None:
        values.append(int(requirements.max_k_down))
    if not values:
        raise ValueError("cuSPARSE layout requires at least one configured direction plan")
    return max(values)


def _side_io_rows(pair: CusparsePlanPair, requirements: RuntimeRequirements, *, max_num_mutations: int, max_num_nodes: int) -> int:
    need_miss_rows = (
        (pair.plan_down is not None and requirements.need_down_miss_input)
        or (pair.plan_up is not None and requirements.need_up_miss_output)
    )
    return max(
        int(max_num_mutations) if need_miss_rows else 0,
        1 if requirements.need_init_vector else 0,
        int(max_num_nodes) if requirements.need_init_matrix else 0,
    )


def _metadata_bytes(state, pair: CusparsePlanPair, requirements: RuntimeRequirements, dtype: np.dtype) -> int:
    total = int(np.asarray(state.node_perm).nbytes + np.asarray(state.sample_to_individual).nbytes)
    if requirements.need_init_vector:
        if pair.plan_up is not None:
            total += int(np.asarray(state.init_vector_up_bias, dtype=dtype).nbytes)
        if pair.plan_down is not None:
            total += int(np.asarray(state.init_vector_down_bias, dtype=dtype).nbytes)
    if requirements.need_init_xtx:
        if pair.plan_up is not None and state.init_xtx_up_bias is not None:
            total += int(np.asarray(state.init_xtx_up_bias, dtype=dtype).nbytes)
        if pair.plan_down is not None and state.init_xtx_down_bias is not None:
            total += int(np.asarray(state.init_xtx_down_bias, dtype=dtype).nbytes)
    return total


def _dense_order_char(order: DenseOrder) -> str:
    return "C" if order == DenseOrder.ROW else "F"


def _dense_ld(buf, *, order: DenseOrder) -> int:
    strides = tuple(int(value) for value in buf.strides)
    itemsize = int(np.dtype(buf.dtype).itemsize)
    unit0 = strides[0] == itemsize
    unit1 = strides[1] == itemsize
    match (order, unit0, unit1):
        case (DenseOrder.ROW, _, True):
            axis = 0
        case (DenseOrder.COL, True, _):
            axis = 1
        case (DenseOrder.ROW, True, False):
            axis = 1
        case (DenseOrder.COL, False, True):
            axis = 0
        case _:
            raise ValueError(
                "dense descriptor requires unit stride along one matrix axis, "
                f"got order={order} shape={tuple(int(v) for v in buf.shape)} strides={strides}"
            )
    return int(strides[axis] // itemsize)


def _build_shared_ones_plan(dtype: np.dtype, max_nnz: int, *, device_id: int) -> _SharedOnesPlan:
    if max_nnz <= 0:
        return _SharedOnesPlan(mode="disabled", logical_bytes=0, physical_bytes=0)
    logical_bytes = int(max_nnz) * int(dtype.itemsize)
    try:
        driver = CudaVmmDriver()
        if driver.current_context() is None or not driver.vmm_supported(device_id):
            return _SharedOnesPlan(mode="materialized", logical_bytes=logical_bytes, physical_bytes=logical_bytes)
        tile = int(driver.allocation_granularity(device_id, recommended=False))
        if tile <= 0 or tile % int(dtype.itemsize) != 0:
            return _SharedOnesPlan(mode="materialized", logical_bytes=logical_bytes, physical_bytes=logical_bytes)
        if tile >= logical_bytes:
            return _SharedOnesPlan(mode="materialized", logical_bytes=logical_bytes, physical_bytes=logical_bytes)
        return _SharedOnesPlan(mode="vmm", logical_bytes=logical_bytes, physical_bytes=tile)
    except Exception:
        return _SharedOnesPlan(mode="materialized", logical_bytes=logical_bytes, physical_bytes=logical_bytes)


def _create_dense_desc(*, cslib: CuSparseLib, buf, order: DenseOrder, cuda_dtype_id: int) -> c_void_p:
    return cslib.create_dnmat(
        int(buf.shape[0]),
        int(buf.shape[1]),
        _dense_ld(buf, order=order),
        buf.data.ptr,
        cuda_dtype_id,
        int(order),
    )


def _publish_level_source_view(cp, *, level_bufs, src_bufs, plan: CusparsePlan, level: int) -> None:
    if src_bufs is None:
        return
    src = src_bufs[level]
    state = level_bufs[level]
    if plan.op_b == Operation.N:
        cp.copyto(src, state)
    else:
        cp.copyto(src, state.T)


def _needs_explicit_source(plan: CusparsePlan) -> bool:
    if plan.op_b == Operation.N and plan.order_b == plan.order_c:
        return False
    if plan.op_b == Operation.T and plan.order_b != plan.order_c:
        return False
    return True


def _selector_levels(cp, selector: sp.csr_matrix, level_offsets: np.ndarray) -> _SelectorLevels:
    pairs = split_selector_by_level(selector, level_offsets)
    return _SelectorLevels(
        rows_by_level=[cp.asarray(rows) for rows, _ in pairs],
        cols_by_level=[cp.asarray(cols) for _, cols in pairs],
        row_unique=bool(np.all(np.diff(np.asarray(selector.indptr)) <= 1)),
    )


def _query_ext_sizes(
    *,
    cp,
    cslib: CuSparseLib,
    shared_ones_ptr: int,
    layout_artifacts: tuple[_CuArtifactLayout, ...],
    scans,
    pair: CusparsePlanPair,
    dtype: np.dtype,
    max_k_up: int,
    max_k_down: int,
    scratch_up_enabled: tuple[bool, ...],
    scratch_down_enabled: tuple[bool, ...],
):
    cuda_dtype_id = cuda_dtype(dtype)
    alpha = cp.ones(1, dtype=dtype)
    beta_zero = cp.zeros(1, dtype=dtype)
    beta_one = cp.ones(1, dtype=dtype)
    max_h = max(scan.num_levels for scan in scans)
    ext_main_up = [0 for _ in range(max_h)]
    ext_main_down = [0 for _ in range(max_h)]
    # Normalize scratch tuples to global maxima.
    max_up_ops = [max(sum(1 for block in scan.blocks if block.nnz > 0 and block.dst_level == level) for scan in scans) for level in range(max_h)]
    max_down_ops = [max(sum(1 for block in scan.blocks if block.nnz > 0 and block.src_level == level) for scan in scans) for level in range(max_h)]
    ext_scratch_up = [list(0 for _ in range(max_up_ops[level])) for level in range(max_h)]
    ext_scratch_down = [list(0 for _ in range(max_down_ops[level])) for level in range(max_h)]
    try:
        for artifact_layout, scan in zip(layout_artifacts, scans, strict=True):
            for direction, plan, max_k, scratch_enabled, ext_main, ext_scratch in (
                (Direction.UP, pair.plan_up, max_k_up, scratch_up_enabled, ext_main_up, ext_scratch_up),
                (Direction.DOWN, pair.plan_down, max_k_down, scratch_down_enabled, ext_main_down, ext_scratch_down),
            ):
                if plan is None:
                    continue
                level_sizes = [int(scan.level_offsets[level + 1] - scan.level_offsets[level]) for level in range(scan.num_levels)]
                level_bufs = [
                    cp.zeros((level_sizes[level], max_k), dtype=dtype, order=_dense_order_char(plan.order_c))
                    for level in range(scan.num_levels)
                ]
                dst_descs = [
                    _create_dense_desc(cslib=cslib, buf=buf, order=plan.order_c, cuda_dtype_id=cuda_dtype_id)
                    for buf in level_bufs
                ]
                if not _needs_explicit_source(plan):
                    if plan.op_b == Operation.N:
                        src_bufs = None
                        src_descs = dst_descs
                    else:
                        src_bufs = None
                        src_descs = [
                            _create_dense_desc(cslib=cslib, buf=buf.T, order=plan.order_b, cuda_dtype_id=cuda_dtype_id)
                            for buf in level_bufs
                        ]
                else:
                    src_bufs = []
                    src_descs = []
                    for level, rows in enumerate(level_sizes):
                        src_shape = (rows, max_k) if plan.op_b == Operation.N else (max_k, rows)
                        src = cp.zeros(src_shape, dtype=dtype, order=_dense_order_char(plan.order_b))
                        src_bufs.append(src)
                        src_descs.append(
                            _create_dense_desc(cslib=cslib, buf=src, order=plan.order_b, cuda_dtype_id=cuda_dtype_id)
                        )
                up_plan_map = {(block.dst_level, block.src_level): block for block in artifact_layout.blocks_up}
                down_plan_map = {(block.dst_level, block.src_level): block for block in artifact_layout.blocks_down}
                op_index_by_level = [0 for _ in range(scan.num_levels)]
                for block in iter_artifact_blocks(artifact_layout.path):
                    shared_from_up = False
                    if direction == Direction.UP:
                        owner = up_plan_map.get((block.dst_level, block.src_level))
                        if owner is None:
                            continue
                        dst_level = block.dst_level
                        src_level = block.src_level
                    else:
                        shared_from_up = artifact_layout.share_storage
                        owner = (up_plan_map if shared_from_up else down_plan_map).get((block.dst_level, block.src_level))
                        if owner is None:
                            continue
                        dst_level = block.src_level
                        src_level = block.dst_level
                    sparse = materialize_sparse_block(
                        sp.csr_matrix((np.ones(block.nnz, dtype=np.bool_), np.asarray(block.indices), np.asarray(block.indptr)), shape=block.shape),
                        store=owner.store,
                        fmt=owner.fmt,
                    )
                    if owner.fmt == SparseFormat.CSR:
                        struct0 = cp.asarray(np.asarray(sparse.indptr, dtype=owner.struct0_dtype))
                        struct1 = cp.asarray(np.asarray(sparse.indices, dtype=owner.struct1_dtype))
                        if shared_from_up:
                            sp_desc = cslib.create_csc(
                                owner.stored_shape[1],
                                owner.stored_shape[0],
                                owner.nnz,
                                struct0.data.ptr,
                                struct1.data.ptr,
                                shared_ones_ptr,
                                _cusparse_index_type(owner.struct0_dtype),
                                _cusparse_index_type(owner.struct1_dtype),
                                cuda_dtype_id,
                            )
                        else:
                            sp_desc = cslib.create_csr(
                                owner.stored_shape[0],
                                owner.stored_shape[1],
                                owner.nnz,
                                struct0.data.ptr,
                                struct1.data.ptr,
                                shared_ones_ptr,
                                _cusparse_index_type(owner.struct0_dtype),
                                _cusparse_index_type(owner.struct1_dtype),
                                cuda_dtype_id,
                            )
                    elif owner.fmt == SparseFormat.CSC:
                        struct0 = cp.asarray(np.asarray(sparse.indptr, dtype=owner.struct0_dtype))
                        struct1 = cp.asarray(np.asarray(sparse.indices, dtype=owner.struct1_dtype))
                        if shared_from_up:
                            sp_desc = cslib.create_csr(
                                owner.stored_shape[1],
                                owner.stored_shape[0],
                                owner.nnz,
                                struct0.data.ptr,
                                struct1.data.ptr,
                                shared_ones_ptr,
                                _cusparse_index_type(owner.struct0_dtype),
                                _cusparse_index_type(owner.struct1_dtype),
                                cuda_dtype_id,
                            )
                        else:
                            sp_desc = cslib.create_csc(
                                owner.stored_shape[0],
                                owner.stored_shape[1],
                                owner.nnz,
                                struct0.data.ptr,
                                struct1.data.ptr,
                                shared_ones_ptr,
                                _cusparse_index_type(owner.struct0_dtype),
                                _cusparse_index_type(owner.struct1_dtype),
                                cuda_dtype_id,
                            )
                    else:
                        struct0 = cp.asarray(np.asarray(sparse.row, dtype=owner.struct0_dtype))
                        struct1 = cp.asarray(np.asarray(sparse.col, dtype=owner.struct1_dtype))
                        sp_desc = cslib.create_coo(owner.stored_shape[0], owner.stored_shape[1], owner.nnz, struct0.data.ptr, struct1.data.ptr, shared_ones_ptr, _cusparse_index_type(owner.struct0_dtype), cuda_dtype_id)
                    try:
                        op_idx = op_index_by_level[dst_level]
                        op_index_by_level[dst_level] += 1
                        if scratch_enabled[dst_level]:
                            scratch = cp.zeros((level_sizes[dst_level], max_k), dtype=dtype, order=_dense_order_char(plan.order_c))
                            dst_desc = _create_dense_desc(cslib=cslib, buf=scratch, order=plan.order_c, cuda_dtype_id=cuda_dtype_id)
                            size = cslib.spmm_buffer_size(int(plan.algo), int(plan.op_a), int(plan.op_b), alpha.data.ptr, sp_desc, src_descs[src_level], beta_zero.data.ptr, dst_desc, cuda_dtype_id)
                            ext_scratch[dst_level][op_idx] = max(int(ext_scratch[dst_level][op_idx]), int(size))
                            cslib.destroy_dn_mat(dst_desc)
                        else:
                            size = cslib.spmm_buffer_size(int(plan.algo), int(plan.op_a), int(plan.op_b), alpha.data.ptr, sp_desc, src_descs[src_level], beta_one.data.ptr, dst_descs[dst_level], cuda_dtype_id)
                            ext_main[dst_level] = max(int(ext_main[dst_level]), int(size))
                    finally:
                        cslib.destroy_sp_mat(sp_desc)
                for desc in dst_descs:
                    cslib.destroy_dn_mat(desc)
                seen = {id(desc) for desc in dst_descs}
                for desc in src_descs:
                    if id(desc) in seen:
                        continue
                    cslib.destroy_dn_mat(desc)
    finally:
        pass
    return (
        tuple(int(v) for v in ext_main_up),
        tuple(int(v) for v in ext_main_down),
        tuple(tuple(int(v) for v in row) for row in ext_scratch_up),
        tuple(tuple(int(v) for v in row) for row in ext_scratch_down),
    )


def plan_cusparse_layout(
    *,
    artifacts,
    pair: CusparsePlanPair,
    dtype,
    requirements: RuntimeRequirements,
    vram_budget_bytes: int,
    ring_buffer_size: int,
    allow_residency: bool = True,
    device,
    stream,
) -> CusparseLayout:
    dtype = np.dtype(dtype)
    if dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError(f"cuSPARSE runtime supports only float32/float64, got {dtype}")
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError("CuPy required: pip install cupy-cuda12x") from exc

    requested_ring_buffer_size = int(ring_buffer_size)
    allow_residency = bool(allow_residency)
    device_id = parse_cuda_device(device)
    stream_ptr, stream_owner = parse_cuda_stream(stream)
    if stream_ptr != 0:
        stream_device = _cuda_stream_device(stream_ptr)
        if stream_device != device_id:
            raise ValueError(f"CUDA stream device {stream_device} does not match requested CUDA device {device_id}")
    paths = _resolve_artifacts(artifacts)
    scans = tuple(scan_grg_spmv(path) for path in paths)
    max_levels = max(scan.num_levels for scan in scans)
    max_rows_by_level = tuple(max((scan.level_sizes[level] if level < scan.num_levels else 0) for scan in scans) for level in range(max_levels))
    max_up_ops_by_level = tuple(max(sum(1 for block in scan.blocks if block.nnz > 0 and block.dst_level == level) for scan in scans) for level in range(max_levels))
    max_down_ops_by_level = tuple(max(sum(1 for block in scan.blocks if block.nnz > 0 and block.src_level == level) for scan in scans) for level in range(max_levels))
    scratch_up_enabled = _resolve_scratch_levels(pair.plan_up, max_levels)
    scratch_down_enabled = _resolve_scratch_levels(pair.plan_down, max_levels)
    max_num_samples = max(scan.num_samples for scan in scans)
    max_num_mutations = max(scan.num_mutations for scan in scans)
    max_num_nodes = max(scan.num_nodes for scan in scans)
    max_selector_nnz = max(
        max(int(scan.selector_mut_nnz), int(scan.selector_miss_nnz))
        for scan in scans
    )

    blocks: list[_CuBlockPlan] = []
    selector_bytes = 0
    metadata_bytes = 0
    xtx_bias_bytes = 0
    planned: list[_CuArtifactLayout] = []
    max_block_nnz = 0
    for artifact_index, (path, scan) in enumerate(zip(paths, scans, strict=True)):
        share_storage = bool(pair.plan_up is not None and pair.plan_down is not None and pair.plan_up.can_share_storage_with(pair.plan_down))
        blocks_up = () if pair.plan_up is None else tuple(_block_plan(artifact_index, Direction.UP, share_storage, block, pair.plan_up) for block in scan.blocks if block.nnz > 0)
        blocks_down = () if pair.plan_down is None or share_storage else tuple(_block_plan(artifact_index, Direction.DOWN, False, block, pair.plan_down) for block in scan.blocks if block.nnz > 0)
        blocks.extend(blocks_up)
        blocks.extend(blocks_down)
        state = _load_grg_spmv_host(path, dtype)
        mut_pairs = split_selector_by_level(state.sel_mut, state.level_offsets)
        miss_pairs = split_selector_by_level(state.sel_miss, state.level_offsets)
        selector_bytes += sum(int(rows.nbytes + cols.nbytes) for rows, cols in mut_pairs)
        selector_bytes += sum(int(rows.nbytes + cols.nbytes) for rows, cols in miss_pairs)
        metadata_bytes += _metadata_bytes(state, pair, requirements, dtype)
        if requirements.need_init_xtx and state.coalescence_counts is not None:
            xtx_bias_bytes += int(state.num_nodes * dtype.itemsize)
        max_block_nnz = max(max_block_nnz, max((block.nnz for block in scan.blocks), default=0))
        planned.append(
            _CuArtifactLayout(
                path=path,
                share_storage=share_storage,
                up_owner=Direction.UP if pair.plan_up is not None else None,
                down_owner=None if pair.plan_down is None else (Direction.UP if share_storage else Direction.DOWN),
                blocks_up=blocks_up,
                blocks_down=blocks_down,
            )
        )

    max_k_any = _enabled_max_k(pair, requirements)
    io0_rows = int(max(max_num_samples, max_num_mutations))
    side_io_rows = _side_io_rows(pair, requirements, max_num_mutations=max_num_mutations, max_num_nodes=max_num_nodes)
    aux_rows = int(max(max_num_nodes, max_num_samples, max_num_mutations, max_selector_nnz))
    dense_arena_bytes = int((io0_rows + side_io_rows + aux_rows) * max_k_any * dtype.itemsize)
    workspace_up = 0
    workspace_down = 0
    source_up = 0
    source_down = 0
    scratch_bytes = 0
    if pair.plan_up is not None:
        workspace_up = int(sum(rows * int(requirements.max_k_up) * dtype.itemsize for rows in max_rows_by_level))
        if _needs_explicit_source(pair.plan_up):
            source_up = workspace_up
        for level, enabled in enumerate(scratch_up_enabled):
            if enabled:
                scratch_bytes += int(max_rows_by_level[level] * max_up_ops_by_level[level] * int(requirements.max_k_up) * dtype.itemsize)
    if pair.plan_down is not None:
        workspace_down = int(sum(rows * int(requirements.max_k_down) * dtype.itemsize for rows in max_rows_by_level))
        if _needs_explicit_source(pair.plan_down):
            source_down = workspace_down
        for level, enabled in enumerate(scratch_down_enabled):
            if enabled:
                scratch_bytes += int(max_rows_by_level[level] * max_down_ops_by_level[level] * int(requirements.max_k_down) * dtype.itemsize)

    with cp.cuda.Device(device_id):
        cslib = CuSparseLib()
        shared_vals = cp.ones((max(max_block_nnz, 1),), dtype=dtype)
        try:
            ext_main_up, ext_main_down, ext_scratch_up, ext_scratch_down = _query_ext_sizes(
                cp=cp,
                cslib=cslib,
                shared_ones_ptr=int(shared_vals.data.ptr),
                layout_artifacts=tuple(planned),
                scans=scans,
                pair=pair,
                dtype=dtype,
                max_k_up=int(requirements.max_k_up),
                max_k_down=int(requirements.max_k_down),
                scratch_up_enabled=scratch_up_enabled,
                scratch_down_enabled=scratch_down_enabled,
            )
        finally:
            cslib.destroy()
            del shared_vals

    ext_bytes = int(sum(ext_main_up) + sum(ext_main_down) + sum(sum(row) for row in ext_scratch_up) + sum(sum(row) for row in ext_scratch_down))
    shared_ones_plan = _build_shared_ones_plan(dtype, max_block_nnz, device_id=device_id)
    scalar_bytes = int(3 * dtype.itemsize)
    fixed_bytes = int(
        selector_bytes
        + metadata_bytes
        + xtx_bias_bytes
        + dense_arena_bytes
        + workspace_up
        + workspace_down
        + source_up
        + source_down
        + scratch_bytes
        + shared_ones_plan.physical_bytes
        + scalar_bytes
        + ext_bytes
    )
    resident_bytes_full = int(sum(block.nbytes for block in blocks))
    required_budget_for_full_residency = int(fixed_bytes + resident_bytes_full)
    if not allow_residency:
        if blocks and requested_ring_buffer_size < 1:
            raise ValueError("ring_buffer_size must be >= 1 when any cuSPARSE block is streamed")
        for block in blocks:
            block.resident = False
        slot_plans, ring_bytes = _assign_slots(blocks, requested_ring_buffer_size) if blocks else ((), 0)
        resident_bytes = 0
        total_bytes = int(fixed_bytes + ring_bytes)
        if total_bytes > int(vram_budget_bytes):
            raise ValueError(
                f"cuSPARSE layout requires at least {total_bytes} owned device bytes, exceeds vram_budget_bytes={vram_budget_bytes}"
            )
        if len(slot_plans) != requested_ring_buffer_size:
            _warn_ring_mismatch(requested=requested_ring_buffer_size, allocated=len(slot_plans))
    elif required_budget_for_full_residency <= int(vram_budget_bytes):
        for block in blocks:
            block.resident = True
        slot_plans = ()
        ring_bytes = 0
        resident_bytes = resident_bytes_full
        total_bytes = required_budget_for_full_residency
        if requested_ring_buffer_size > 0:
            _warn_ring_mismatch(requested=requested_ring_buffer_size, allocated=0)
    else:
        if requested_ring_buffer_size < 1:
            raise ValueError("ring_buffer_size must be >= 1 when any cuSPARSE block is streamed")
        for block in blocks:
            block.resident = False
        slot_plans, ring_bytes = _assign_slots(blocks, requested_ring_buffer_size)
        resident_bytes = 0
        total_bytes = int(fixed_bytes + ring_bytes)
        if total_bytes > int(vram_budget_bytes):
            raise ValueError(
                f"cuSPARSE layout requires at least {total_bytes} owned device bytes, exceeds vram_budget_bytes={vram_budget_bytes}"
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
        if len(slot_plans) != requested_ring_buffer_size:
            _warn_ring_mismatch(requested=requested_ring_buffer_size, allocated=len(slot_plans))

    if total_bytes > int(vram_budget_bytes):
        raise ValueError(
            f"cuSPARSE layout requires {total_bytes} owned device bytes, exceeds vram_budget_bytes={vram_budget_bytes}"
        )

    bytes_by_category = {
        "resident_sparse": resident_bytes,
        "ring_slots": ring_bytes,
        "selectors": int(selector_bytes),
        "metadata": int(metadata_bytes),
        "xtx_bias": int(xtx_bias_bytes),
        "dense_arena": int(dense_arena_bytes),
        "workspace_up": int(workspace_up + source_up),
        "workspace_down": int(workspace_down + source_down),
        "scratch": int(scratch_bytes),
        "shared_ones": int(shared_ones_plan.physical_bytes),
        "scalars": int(scalar_bytes),
        "ext": int(ext_bytes),
    }
    budget_items: list[BudgetItem] = [
        BudgetItem(kind="fixed", name="selectors", nbytes=int(selector_bytes)),
        BudgetItem(kind="fixed", name="metadata", nbytes=int(metadata_bytes)),
        BudgetItem(kind="fixed", name="xtx_bias", nbytes=int(xtx_bias_bytes)),
        BudgetItem(kind="fixed", name="dense_arena", nbytes=int(dense_arena_bytes)),
        BudgetItem(kind="fixed", name="workspace_up", nbytes=int(workspace_up + source_up)),
        BudgetItem(kind="fixed", name="workspace_down", nbytes=int(workspace_down + source_down)),
        BudgetItem(kind="fixed", name="scratch", nbytes=int(scratch_bytes)),
        BudgetItem(kind="fixed", name="shared_ones", nbytes=int(shared_ones_plan.physical_bytes)),
        BudgetItem(kind="fixed", name="scalars", nbytes=int(scalar_bytes)),
        BudgetItem(kind="fixed", name="ext", nbytes=int(ext_bytes)),
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
    return CusparseLayout(
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
        max_num_samples=max_num_samples,
        max_num_mutations=max_num_mutations,
        max_num_nodes=max_num_nodes,
        max_selector_nnz=max_selector_nnz,
        max_levels=max_levels,
        max_rows_by_level=max_rows_by_level,
        max_up_ops_by_level=max_up_ops_by_level,
        max_down_ops_by_level=max_down_ops_by_level,
        scratch_up_enabled=scratch_up_enabled,
        scratch_down_enabled=scratch_down_enabled,
        slot_plans=slot_plans,
        ext_main_up=ext_main_up,
        ext_main_down=ext_main_down,
        ext_scratch_up=ext_scratch_up,
        ext_scratch_down=ext_scratch_down,
        shared_ones=shared_ones_plan,
        budget_items=tuple(item for item in budget_items if item.nbytes > 0),
        required_budget_for_full_residency=required_budget_for_full_residency,
        bytes_by_category=bytes_by_category,
        bytes_total=int(sum(item.nbytes for item in budget_items)),
    )


class CusparseRuntime:
    """Runtime-owned cuSPARSE execution."""

    def __init__(self, layout: CusparseLayout) -> None:
        try:
            import cupy as cp
            import cupyx
        except ImportError as exc:
            raise ImportError("CuPy required: pip install cupy-cuda12x") from exc
        self._cp = cp
        self._cupyx = cupyx
        self.layout = layout
        self.device = cp.cuda.Device(int(layout.device))
        self.stream_ptr = int(layout.stream_ptr)
        self.stream = None
        self._stream_owner = layout.stream_owner
        self._cslib: CuSparseLib | None = None
        self._spmm_lib_by_stream_ptr: dict[int, CuSparseLib] = {}
        self._shared_ones: _SharedOnes | None = None
        self._alpha = None
        self._beta_zero = None
        self._beta_one = None
        self._caller_stream = None
        self._root_stream = None
        self._caller_to_root_event = None
        self._root_to_caller_event = None
        self._level_streams = []
        self._slot_copy_streams = []
        self._scratch_streams_up = []
        self._scratch_streams_down = []
        self._slot_struct0 = []
        self._slot_struct1 = []
        self._up_level_bufs = []
        self._down_level_bufs = []
        self._up_src_bufs = None
        self._down_src_bufs = None
        self._up_scratch = []
        self._down_scratch = []
        self._io0 = None
        self._io1 = None
        self._aux = None
        self._io0_torch = None
        self._io1_torch = None
        self._aux_torch = None
        self._ext_main_up = []
        self._ext_main_down = []
        self._ext_scratch_up = []
        self._ext_scratch_down = []
        self._copy_done = []
        self._compute_done = []
        self._ready = []
        self._launch_event = None
        self._artifacts: tuple[_CuArtifact, ...] = ()
        self._grgs: tuple[BoundGRG, ...] = ()
        self._entered = False
        self._active_call = False

    @property
    def grgs(self) -> tuple[BoundGRG, ...]:
        if not self._entered:
            raise RuntimeError("CusparseRuntime must be entered before accessing grgs")
        return self._grgs

    def _destroy_direction_state(self, direction_state: _DenseDescriptorSet | None) -> None:
        if direction_state is None or self._cslib is None:
            return
        seen: set[int] = set()
        for desc in [*direction_state.dst_descs, *direction_state.src_descs]:
            key = int(desc.value)
            if key in seen:
                continue
            seen.add(key)
            self._cslib.destroy_dn_mat(desc)
        for row in direction_state.scratch_descs_by_level:
            for desc in row:
                key = int(desc.value)
                if key in seen:
                    continue
                seen.add(key)
                self._cslib.destroy_dn_mat(desc)

    def _destroy_dense_entries(self, entries: dict[int, _PreparedDenseEntry]) -> None:
        for entry in entries.values():
            self._destroy_direction_state(entry.state)
            entry.refs = 0
        entries.clear()

    def _destroy_ops(self, ops_by_level: list[list[_CuOp]] | None) -> None:
        if ops_by_level is None or self._cslib is None:
            return
        seen: set[int] = set()
        for ops in ops_by_level:
            for op in ops:
                if op.sp_desc.value is None:
                    continue
                key = int(op.sp_desc.value)
                if key in seen:
                    continue
                seen.add(key)
                self._cslib.destroy_sp_mat(op.sp_desc)

    def _destroy_artifact_resources(self, artifact: _CuArtifact) -> None:
        self._destroy_dense_entries(artifact.up_dense_by_k)
        self._destroy_dense_entries(artifact.down_dense_by_k)
        self._destroy_ops(artifact.up_ops)
        self._destroy_ops(artifact.down_ops)

    def _release_owned_state(self, artifacts: tuple[_CuArtifact, ...] | list[_CuArtifact]) -> None:
        for artifact in artifacts:
            self._destroy_artifact_resources(artifact)
        if self._shared_ones is not None:
            self._shared_ones.destroy()
            self._shared_ones = None
        for lib in self._spmm_lib_by_stream_ptr.values():
            lib.destroy()
        self._spmm_lib_by_stream_ptr = {}
        if self._cslib is not None:
            self._cslib.destroy()
            self._cslib = None
        self._alpha = None
        self._beta_zero = None
        self._beta_one = None
        self._caller_stream = None
        self._root_stream = None
        self._caller_to_root_event = None
        self._root_to_caller_event = None
        self._level_streams = []
        self._slot_copy_streams = []
        self._scratch_streams_up = []
        self._scratch_streams_down = []
        self._slot_struct0 = []
        self._slot_struct1 = []
        self._up_level_bufs = []
        self._down_level_bufs = []
        self._up_src_bufs = None
        self._down_src_bufs = None
        self._up_scratch = []
        self._down_scratch = []
        self._io0 = None
        self._io1 = None
        self._aux = None
        self._io0_torch = None
        self._io1_torch = None
        self._aux_torch = None
        self._ext_main_up = []
        self._ext_main_down = []
        self._ext_scratch_up = []
        self._ext_scratch_down = []
        self._copy_done = []
        self._compute_done = []
        self._ready = []
        self._launch_event = None
        self._artifacts = ()
        self._grgs = ()
        self._entered = False
        self._active_call = False
        self.stream = None

    def __enter__(self) -> "CusparseRuntime":
        artifacts: list[_CuArtifact] = []
        try:
            with self.device:
                self._cslib = CuSparseLib()
                self._caller_stream = self._make_caller_stream()
                self.stream = self._caller_stream
                self._root_stream = self._cp.cuda.Stream(non_blocking=True)
                self._caller_to_root_event = self._cp.cuda.Event()
                self._root_to_caller_event = self._cp.cuda.Event()
                self._level_streams = [self._cp.cuda.Stream(non_blocking=True) for _ in range(self.layout.max_levels)]
                self._slot_copy_streams = [self._cp.cuda.Stream(non_blocking=True) for _ in range(self.layout.allocated_ring_buffer_size)]
                self._scratch_streams_up = [
                    [self._cp.cuda.Stream(non_blocking=True) for _ in range(self.layout.max_up_ops_by_level[level])] if self.layout.scratch_up_enabled[level] else []
                    for level in range(self.layout.max_levels)
                ]
                self._scratch_streams_down = [
                    [self._cp.cuda.Stream(non_blocking=True) for _ in range(self.layout.max_down_ops_by_level[level])] if self.layout.scratch_down_enabled[level] else []
                    for level in range(self.layout.max_levels)
                ]
                stream_ptrs = {
                    int(stream.ptr)
                    for stream in [
                        *self._level_streams,
                        *(stream for level_streams in self._scratch_streams_up for stream in level_streams),
                        *(stream for level_streams in self._scratch_streams_down for stream in level_streams),
                    ]
                }
                self._spmm_lib_by_stream_ptr = {ptr: CuSparseLib() for ptr in stream_ptrs}
                self._alpha = self._cp.ones(1, dtype=self.layout.dtype)
                self._beta_zero = self._cp.zeros(1, dtype=self.layout.dtype)
                self._beta_one = self._cp.ones(1, dtype=self.layout.dtype)
                self._shared_ones = self._materialize_shared_ones()
                self._slot_struct0 = [self._cp.zeros((slot.struct0_len,), dtype=slot.struct0_dtype) for slot in self.layout.slot_plans]
                self._slot_struct1 = [self._cp.zeros((slot.struct1_len,), dtype=slot.struct1_dtype) for slot in self.layout.slot_plans]
                max_k_any = _enabled_max_k(self.layout.pair, self.layout.requirements)
                io0_rows = int(max(self.layout.max_num_samples, self.layout.max_num_mutations))
                side_io_rows = _side_io_rows(
                    self.layout.pair,
                    self.layout.requirements,
                    max_num_mutations=self.layout.max_num_mutations,
                    max_num_nodes=self.layout.max_num_nodes,
                )
                aux_rows = int(max(self.layout.max_num_nodes, self.layout.max_num_samples, self.layout.max_num_mutations, self.layout.max_selector_nnz))
                self._io0 = self._cp.zeros((io0_rows, max_k_any), dtype=self.layout.dtype, order="C")
                self._io1 = None if side_io_rows == 0 else self._cp.zeros((side_io_rows, max_k_any), dtype=self.layout.dtype, order="C")
                self._aux = self._cp.zeros((aux_rows, max_k_any), dtype=self.layout.dtype, order="C")
                self._io0_torch = _torch_from_cupy(self._io0)
                self._io1_torch = None if self._io1 is None else _torch_from_cupy(self._io1)
                self._aux_torch = _torch_from_cupy(self._aux)
                max_ops_by_level = [
                    max(int(self.layout.max_up_ops_by_level[level]), int(self.layout.max_down_ops_by_level[level]))
                    for level in range(self.layout.max_levels)
                ]
                self._copy_done = [[self._cp.cuda.Event() for _ in range(max_ops_by_level[level])] for level in range(self.layout.max_levels)]
                self._compute_done = [[self._cp.cuda.Event() for _ in range(max_ops_by_level[level])] for level in range(self.layout.max_levels)]
                self._ready = [self._cp.cuda.Event() for _ in range(self.layout.max_levels)]
                self._launch_event = self._cp.cuda.Event()
                if self.layout.pair.plan_up is not None:
                    self._up_level_bufs = [self._cp.zeros((rows, int(self.layout.requirements.max_k_up)), dtype=self.layout.dtype, order=_dense_order_char(self.layout.pair.plan_up.order_c)) for rows in self.layout.max_rows_by_level]
                    self._up_src_bufs = None if not _needs_explicit_source(self.layout.pair.plan_up) else [
                        self._cp.zeros(((rows, int(self.layout.requirements.max_k_up)) if self.layout.pair.plan_up.op_b == Operation.N else (int(self.layout.requirements.max_k_up), rows)), dtype=self.layout.dtype, order=_dense_order_char(self.layout.pair.plan_up.order_b))
                        for rows in self.layout.max_rows_by_level
                    ]
                    self._up_scratch = [
                        [self._cp.zeros((rows, int(self.layout.requirements.max_k_up)), dtype=self.layout.dtype, order=_dense_order_char(self.layout.pair.plan_up.order_c)) for _ in range(self.layout.max_up_ops_by_level[level])]
                        if self.layout.scratch_up_enabled[level]
                        else []
                        for level, rows in enumerate(self.layout.max_rows_by_level)
                    ]
                    self._ext_main_up = [None if size == 0 else self._cp.zeros((size,), dtype=self._cp.uint8) for size in self.layout.ext_main_up]
                    self._ext_scratch_up = [[None if size == 0 else self._cp.zeros((size,), dtype=self._cp.uint8) for size in row] for row in self.layout.ext_scratch_up]
                if self.layout.pair.plan_down is not None:
                    self._down_level_bufs = [self._cp.zeros((rows, int(self.layout.requirements.max_k_down)), dtype=self.layout.dtype, order=_dense_order_char(self.layout.pair.plan_down.order_c)) for rows in self.layout.max_rows_by_level]
                    self._down_src_bufs = None if not _needs_explicit_source(self.layout.pair.plan_down) else [
                        self._cp.zeros(((rows, int(self.layout.requirements.max_k_down)) if self.layout.pair.plan_down.op_b == Operation.N else (int(self.layout.requirements.max_k_down), rows)), dtype=self.layout.dtype, order=_dense_order_char(self.layout.pair.plan_down.order_b))
                        for rows in self.layout.max_rows_by_level
                    ]
                    self._down_scratch = [
                        [self._cp.zeros((rows, int(self.layout.requirements.max_k_down)), dtype=self.layout.dtype, order=_dense_order_char(self.layout.pair.plan_down.order_c)) for _ in range(self.layout.max_down_ops_by_level[level])]
                        if self.layout.scratch_down_enabled[level]
                        else []
                        for level, rows in enumerate(self.layout.max_rows_by_level)
                    ]
                    self._ext_main_down = [None if size == 0 else self._cp.zeros((size,), dtype=self._cp.uint8) for size in self.layout.ext_main_down]
                    self._ext_scratch_down = [[None if size == 0 else self._cp.zeros((size,), dtype=self._cp.uint8) for size in row] for row in self.layout.ext_scratch_down]
            states = tuple(_load_grg_spmv_host(artifact.path, self.layout.dtype) for artifact in self.layout.artifacts)
            for artifact_layout, state in zip(self.layout.artifacts, states, strict=True):
                artifacts.append(self._build_artifact(artifact_layout, state))
            self._artifacts = tuple(artifacts)
            self._grgs = tuple(BoundGRG(self, idx, artifact.state, artifact.path, self.layout.device) for idx, artifact in enumerate(self._artifacts))
            self._entered = True
            return self
        except Exception:
            self._release_owned_state(artifacts)
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        self._release_owned_state(self._artifacts)

    def _make_caller_stream(self):
        if self.stream_ptr == 0:
            return self._cp.cuda.Stream.null
        if hasattr(self._cp.cuda.Stream, "from_external"):
            token = self._stream_owner
            if token is None:
                token = CudaStreamToken(self.stream_ptr)
                self._stream_owner = token
            return self._cp.cuda.Stream.from_external(token)
        return self._cp.cuda.ExternalStream(self.stream_ptr, device_id=self.layout.device)

    def _torch_caller_stream(self):
        import torch

        device = torch.device("cuda", int(self.layout.device))
        if self.stream_ptr == 0:
            return torch.cuda.default_stream(device=device)
        return torch.cuda.get_stream_from_external(self.stream_ptr, device=device)

    @contextmanager
    def _call_scope(self):
        if not self._entered:
            raise RuntimeError("CusparseRuntime must be entered before matmul")
        if self._active_call:
            raise RuntimeError("concurrent runtime.grgs calls are not supported")
        self._active_call = True
        try:
            yield
        finally:
            self._active_call = False

    @contextmanager
    def _caller_root_scope(self):
        with self.device:
            with self._caller_stream:
                self._caller_to_root_event.record(self._caller_stream)
            self._root_stream.wait_event(self._caller_to_root_event)
            try:
                yield
            finally:
                with self._root_stream:
                    self._root_to_caller_event.record(self._root_stream)
                self._caller_stream.wait_event(self._root_to_caller_event)

    def _materialize_shared_ones(self) -> _SharedOnes:
        if self.layout.shared_ones.mode == "disabled":
            return _SharedOnes(ptr=0, logical_nbytes=0, physical_nbytes=0, vmm=False)
        logical = int(self.layout.shared_ones.logical_bytes)
        if self.layout.shared_ones.mode == "materialized":
            arr = self._cp.ones((logical // int(np.dtype(self.layout.dtype).itemsize),), dtype=self.layout.dtype)
            return _SharedOnes(ptr=int(arr.data.ptr), logical_nbytes=logical, physical_nbytes=int(arr.nbytes), vmm=False, _materialized=arr)
        driver = CudaVmmDriver()
        tile = int(self.layout.shared_ones.physical_bytes)
        reserved = _round_up(logical, tile)
        handle = int(driver.mem_create(self.layout.device, tile))
        vaddr = int(driver.address_reserve(reserved, alignment_bytes=tile))
        for offset in range(0, reserved, tile):
            driver.mem_map(vaddr + offset, tile, handle)
        driver.mem_set_access(vaddr, reserved, self.layout.device)
        init_owner = object()
        init_mem = self._cp.cuda.UnownedMemory(vaddr, tile, init_owner, self.layout.device)
        init_ptr = self._cp.cuda.MemoryPointer(init_mem, 0)
        arr = self._cp.ndarray((tile // int(np.dtype(self.layout.dtype).itemsize),), dtype=self.layout.dtype, memptr=init_ptr)
        arr.fill(1)
        return _SharedOnes(ptr=vaddr, logical_nbytes=logical, physical_nbytes=tile, vmm=True, _driver=driver, _vaddr=vaddr, _reserved_nbytes=reserved, _handle=handle)

    def _pin_struct_buffer(self, values: np.ndarray, *, dtype: np.dtype, label: str) -> np.ndarray:
        host = self._cupyx.empty_pinned(values.shape, dtype=_require_struct_dtype(dtype, label=label))
        _copy_struct_checked(host, values, label=label)
        return host

    def _create_sparse_desc(self, block: _CuRuntimeBlock) -> c_void_p:
        assert self._shared_ones is not None
        cuda_dtype_id = cuda_dtype(self.layout.dtype)
        if block.fmt == SparseFormat.CSR:
            return self._cslib.create_csr(block.nrows, block.ncols, block.nnz, block.struct0.data.ptr, block.struct1.data.ptr, int(self._shared_ones.ptr), _cusparse_index_type(block.struct0.dtype), _cusparse_index_type(block.struct1.dtype), cuda_dtype_id)
        if block.fmt == SparseFormat.CSC:
            return self._cslib.create_csc(block.nrows, block.ncols, block.nnz, block.struct0.data.ptr, block.struct1.data.ptr, int(self._shared_ones.ptr), _cusparse_index_type(block.struct0.dtype), _cusparse_index_type(block.struct1.dtype), cuda_dtype_id)
        if block.fmt == SparseFormat.COO:
            return self._cslib.create_coo(block.nrows, block.ncols, block.nnz, block.struct0.data.ptr, block.struct1.data.ptr, int(self._shared_ones.ptr), _cusparse_index_type(block.struct0.dtype), cuda_dtype_id)
        raise ValueError(f"unsupported sparse format: {block.fmt}")

    def _build_artifact_direction_state(self, state, plan: CusparsePlan | None, level_bufs, src_bufs, scratch_bufs, max_k: int):
        if plan is None:
            return None
        cuda_dtype_id = cuda_dtype(self.layout.dtype)
        level_sizes = [int(state.level_offsets[level + 1] - state.level_offsets[level]) for level in range(len(state.level_offsets) - 1)]
        dst_descs = []
        src_descs = []
        scratch_descs_by_level = []
        try:
            for level, rows in enumerate(level_sizes):
                dst_view = level_bufs[level][:rows, :max_k]
                dst_descs.append(_create_dense_desc(cslib=self._cslib, buf=dst_view, order=plan.order_c, cuda_dtype_id=cuda_dtype_id))
                if src_bufs is None:
                    if plan.op_b == Operation.N:
                        src_descs.append(dst_descs[-1])
                    else:
                        src_descs.append(_create_dense_desc(cslib=self._cslib, buf=dst_view.T, order=plan.order_b, cuda_dtype_id=cuda_dtype_id))
                else:
                    src_view = src_bufs[level][:rows, :max_k] if plan.op_b == Operation.N else src_bufs[level][:max_k, :rows]
                    src_descs.append(_create_dense_desc(cslib=self._cslib, buf=src_view, order=plan.order_b, cuda_dtype_id=cuda_dtype_id))
                scratch_row = []
                for scratch in scratch_bufs[level]:
                    scratch_view = scratch[:rows, :max_k]
                    scratch_row.append(_create_dense_desc(cslib=self._cslib, buf=scratch_view, order=plan.order_c, cuda_dtype_id=cuda_dtype_id))
                scratch_descs_by_level.append(scratch_row)
            return _DenseDescriptorSet(dst_descs=dst_descs, src_descs=src_descs, src_bufs=src_bufs, scratch_descs_by_level=scratch_descs_by_level)
        except Exception:
            self._destroy_direction_state(
                _DenseDescriptorSet(dst_descs=dst_descs, src_descs=src_descs, src_bufs=src_bufs, scratch_descs_by_level=scratch_descs_by_level)
            )
            raise

    def _prepare_matmul_cuda(self, grg: BoundGRG, spec: _CudaMatmulSpec) -> _CusparsePreparedMatmul:
        if spec.direction == Direction.UP and self.layout.pair.plan_up is None:
            raise ValueError("UP plan is not configured")
        if spec.direction == Direction.DOWN and self.layout.pair.plan_down is None:
            raise ValueError("DOWN plan is not configured")
        return _CusparsePreparedMatmul(self, self._artifacts[int(grg._artifact_index)], spec)

    def _build_prepared_direction_state(self, artifact: _CuArtifact, direction: Direction, k: int) -> _DenseDescriptorSet:
        if direction == Direction.UP:
            return self._build_artifact_direction_state(
                artifact.state,
                self.layout.pair.plan_up,
                self._up_level_bufs,
                self._up_src_bufs,
                self._up_scratch,
                int(k),
            )
        return self._build_artifact_direction_state(
            artifact.state,
            self.layout.pair.plan_down,
            self._down_level_bufs,
            self._down_src_bufs,
            self._down_scratch,
            int(k),
        )

    def _acquire_dense_descriptors(self, artifact: _CuArtifact, direction: Direction, k: int) -> _PreparedDenseEntry:
        key = int(k)
        entries = _dense_entries(artifact, direction)
        entry = entries.get(key)
        if entry is None:
            entry = _PreparedDenseEntry(self._build_prepared_direction_state(artifact, direction, key))
            entries[key] = entry
        entry.refs += 1
        return entry

    def _release_dense_descriptors(self, artifact: _CuArtifact, direction: Direction, k: int, entry: _PreparedDenseEntry) -> None:
        key = int(k)
        entries = _dense_entries(artifact, direction)
        if entries.get(key) is not entry:
            raise RuntimeError("cuSPARSE prepared dense descriptor entry mismatch")
        if entry.refs <= 0:
            raise RuntimeError("cuSPARSE prepared dense descriptor refcount underflow")
        entry.refs -= 1
        if entry.refs == 0:
            self._destroy_direction_state(entry.state)
            del entries[key]

    def _build_artifact(self, artifact_layout: _CuArtifactLayout, state) -> _CuArtifact:
        up_owner: dict[tuple[int, int], tuple[_CuRuntimeBlock, np.ndarray | None, np.ndarray | None, int | None]] = {}
        down_owner: dict[tuple[int, int], tuple[_CuRuntimeBlock, np.ndarray | None, np.ndarray | None, int | None]] = {}
        plan_up = self.layout.pair.plan_up
        plan_down = self.layout.pair.plan_down
        up_plan_map = {(block.dst_level, block.src_level): block for block in artifact_layout.blocks_up}
        down_plan_map = {(block.dst_level, block.src_level): block for block in artifact_layout.blocks_down}
        up_ops: list[list[_CuOp]] | None = None
        down_ops: list[list[_CuOp]] | None = None
        with self.device:
            try:
                mut_selector = _selector_levels(self._cp, state.sel_mut, state.level_offsets)
                miss_selector = _selector_levels(self._cp, state.sel_miss, state.level_offsets)
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
                        if plan_up.fmt == SparseFormat.CSR:
                            arr0 = np.asarray(sparse.indptr, dtype=plan.struct0_dtype)
                            arr1 = np.asarray(sparse.indices, dtype=plan.struct1_dtype)
                        elif plan_up.fmt == SparseFormat.CSC:
                            arr0 = np.asarray(sparse.indptr, dtype=plan.struct0_dtype)
                            arr1 = np.asarray(sparse.indices, dtype=plan.struct1_dtype)
                        else:
                            arr0 = np.asarray(sparse.row, dtype=plan.struct0_dtype)
                            arr1 = np.asarray(sparse.col, dtype=plan.struct1_dtype)
                        if plan.resident:
                            block_runtime = _CuRuntimeBlock(self._cp.asarray(arr0), self._cp.asarray(arr1), plan.stored_shape[0], plan.stored_shape[1], plan.nnz, plan.fmt)
                            up_owner[(block.dst_level, block.src_level)] = (block_runtime, None, None, None)
                        else:
                            block_runtime = _CuRuntimeBlock(self._slot_struct0[plan.slot], self._slot_struct1[plan.slot], plan.stored_shape[0], plan.stored_shape[1], plan.nnz, plan.fmt)
                            up_owner[(block.dst_level, block.src_level)] = (
                                block_runtime,
                                self._pin_struct_buffer(arr0, dtype=plan.struct0_dtype, label="cuSPARSE streamed struct0"),
                                self._pin_struct_buffer(arr1, dtype=plan.struct1_dtype, label="cuSPARSE streamed struct1"),
                                plan.slot,
                            )
                    plan = down_plan_map.get((block.dst_level, block.src_level))
                    if plan is not None and plan_down is not None:
                        sparse = materialize_sparse_block(base, store=plan_down.store, fmt=plan_down.fmt)
                        if plan_down.fmt == SparseFormat.CSR:
                            arr0 = np.asarray(sparse.indptr, dtype=plan.struct0_dtype)
                            arr1 = np.asarray(sparse.indices, dtype=plan.struct1_dtype)
                        elif plan_down.fmt == SparseFormat.CSC:
                            arr0 = np.asarray(sparse.indptr, dtype=plan.struct0_dtype)
                            arr1 = np.asarray(sparse.indices, dtype=plan.struct1_dtype)
                        else:
                            arr0 = np.asarray(sparse.row, dtype=plan.struct0_dtype)
                            arr1 = np.asarray(sparse.col, dtype=plan.struct1_dtype)
                        if plan.resident:
                            block_runtime = _CuRuntimeBlock(self._cp.asarray(arr0), self._cp.asarray(arr1), plan.stored_shape[0], plan.stored_shape[1], plan.nnz, plan.fmt)
                            down_owner[(block.dst_level, block.src_level)] = (block_runtime, None, None, None)
                        else:
                            block_runtime = _CuRuntimeBlock(self._slot_struct0[plan.slot], self._slot_struct1[plan.slot], plan.stored_shape[0], plan.stored_shape[1], plan.nnz, plan.fmt)
                            down_owner[(block.dst_level, block.src_level)] = (
                                block_runtime,
                                self._pin_struct_buffer(arr0, dtype=plan.struct0_dtype, label="cuSPARSE streamed struct0"),
                                self._pin_struct_buffer(arr1, dtype=plan.struct1_dtype, label="cuSPARSE streamed struct1"),
                                plan.slot,
                            )
                xtx_bias = None
                if self.layout.requirements.need_init_xtx and state.coalescence_counts is not None:
                    xtx_bias = self._cp.asarray(2.0 * state.coalescence_counts.astype(self.layout.dtype, copy=False), dtype=self.layout.dtype)
                init_vector_up_bias = None
                init_vector_down_bias = None
                if self.layout.requirements.need_init_vector:
                    if self.layout.pair.plan_up is not None:
                        init_vector_up_bias = self._cp.asarray(state.init_vector_up_bias, dtype=self.layout.dtype)
                    if self.layout.pair.plan_down is not None:
                        init_vector_down_bias = self._cp.asarray(state.init_vector_down_bias, dtype=self.layout.dtype)
                init_xtx_up_bias = None
                init_xtx_down_bias = None
                if self.layout.requirements.need_init_xtx:
                    if self.layout.pair.plan_up is not None and state.init_xtx_up_bias is not None:
                        init_xtx_up_bias = self._cp.asarray(state.init_xtx_up_bias, dtype=self.layout.dtype)
                    if self.layout.pair.plan_down is not None and state.init_xtx_down_bias is not None:
                        init_xtx_down_bias = self._cp.asarray(state.init_xtx_down_bias, dtype=self.layout.dtype)
                up_ops = self._build_ops(Direction.UP, artifact_layout, state, up_owner, down_owner)
                down_ops = self._build_ops(Direction.DOWN, artifact_layout, state, up_owner, down_owner)
                return _CuArtifact(
                    path=artifact_layout.path,
                    state=state,
                    mut_selector=mut_selector,
                    miss_selector=miss_selector,
                    node_perm=self._cp.asarray(state.node_perm),
                    sample_to_individual=self._cp.asarray(state.sample_to_individual),
                    xtx_bias=xtx_bias,
                    init_vector_up_bias=init_vector_up_bias,
                    init_vector_down_bias=init_vector_down_bias,
                    init_xtx_up_bias=init_xtx_up_bias,
                    init_xtx_down_bias=init_xtx_down_bias,
                    up_ops=up_ops,
                    down_ops=down_ops,
                    up_dense_by_k={},
                    down_dense_by_k={},
                )
            except Exception:
                self._destroy_ops(up_ops)
                self._destroy_ops(down_ops)
                raise

    def _build_ops(self, direction: Direction, artifact_layout: _CuArtifactLayout, state, up_owner, down_owner) -> list[list[_CuOp]]:
        h = len(state.level_offsets) - 1
        ops: list[list[_CuOp]] = [[] for _ in range(h)]
        if (direction == Direction.UP and self.layout.pair.plan_up is None) or (direction == Direction.DOWN and self.layout.pair.plan_down is None):
            return ops
        owner_direction = artifact_layout.up_owner if direction == Direction.UP else artifact_layout.down_owner
        assert owner_direction is not None
        owner_plan = self.layout.pair.plan_up if owner_direction == Direction.UP else self.layout.pair.plan_down
        assert owner_plan is not None
        owner_blocks = up_owner if owner_direction == Direction.UP else down_owner
        try:
            for dst_level, src_level, _row_index in iter_direction_level_pairs(direction, h):
                owner_dst = dst_level if direction == Direction.UP else src_level
                owner_src = src_level if direction == Direction.UP else dst_level
                owner_value = owner_blocks.get((owner_dst, owner_src))
                if owner_value is None:
                    continue
                block_runtime, host0, host1, slot = owner_value
                block_view = block_runtime if owner_direction == direction else block_runtime.transpose_alias()
                sp_desc = self._create_sparse_desc(block_view)
                op_idx = len(ops[dst_level])
                ops[dst_level].append(
                    _CuOp(
                        src_level=src_level,
                        sp_desc=sp_desc,
                        block=block_view,
                        slot=slot,
                        host0=host0,
                        host1=host1,
                        prev_in_slot=None,
                    )
                )
            return relink_stream_dependencies(ops, direction=direction)
        except Exception:
            self._destroy_ops(ops)
            raise

    def _copy_to_slot(self, copy_done, compute_done, dst_level: int, op_idx: int, op: _CuOp) -> None:
        if op.slot is None or op.host0 is None or op.host1 is None:
            return
        stream = self._slot_copy_streams[op.slot]
        with stream:
            if op.prev_in_slot is not None:
                prev_dst, prev_idx = op.prev_in_slot
                stream.wait_event(compute_done[prev_dst][prev_idx])
            self._cp.cuda.runtime.memcpyAsync(self._slot_struct0[op.slot].data.ptr, _numpy_ptr(op.host0), int(op.host0.nbytes), self._cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
            self._cp.cuda.runtime.memcpyAsync(self._slot_struct1[op.slot].data.ptr, _numpy_ptr(op.host1), int(op.host1.nbytes), self._cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
            copy_done[dst_level][op_idx].record(stream)

    def _ext_for(self, direction: Direction, dst_level: int, op_idx: int, scratch: bool):
        if direction == Direction.UP:
            return self._ext_scratch_up[dst_level][op_idx] if scratch else self._ext_main_up[dst_level]
        return self._ext_scratch_down[dst_level][op_idx] if scratch else self._ext_main_down[dst_level]

    def _launch_spmm(self, artifact: _CuArtifact, direction: Direction, op: _CuOp, dst_desc: c_void_p, beta_ptr: int, src_desc: c_void_p, ext) -> None:
        plan = self.layout.pair.plan_up if direction == Direction.UP else self.layout.pair.plan_down
        assert plan is not None
        stream_ptr = int(self._cp.cuda.get_current_stream().ptr)
        cslib = self._spmm_lib_by_stream_ptr.get(stream_ptr, self._cslib)
        cslib.set_stream(stream_ptr)
        cslib.spmm(
            int(plan.algo),
            int(plan.op_a),
            int(plan.op_b),
            self._alpha.data.ptr,
            op.sp_desc,
            src_desc,
            beta_ptr,
            dst_desc,
            cuda_dtype(self.layout.dtype),
            0 if ext is None else int(ext.data.ptr),
        )

    def _publish_level_source(self, direction: Direction, dense: _DenseDescriptorSet, level: int) -> None:
        plan = self.layout.pair.plan_up if direction == Direction.UP else self.layout.pair.plan_down
        level_bufs = self._up_level_bufs if direction == Direction.UP else self._down_level_bufs
        if plan is None:
            return
        _publish_level_source_view(self._cp, level_bufs=level_bufs, src_bufs=dense.src_bufs, plan=plan, level=level)

    def _enqueue_wavefront(self, artifact: _CuArtifact, direction: Direction, dense: _DenseDescriptorSet) -> None:
        ops_by_level = artifact.up_ops if direction == Direction.UP else artifact.down_ops
        level_bufs = self._up_level_bufs if direction == Direction.UP else self._down_level_bufs
        scratch_enabled = self.layout.scratch_up_enabled if direction == Direction.UP else self.layout.scratch_down_enabled
        scratch_streams = self._scratch_streams_up if direction == Direction.UP else self._scratch_streams_down
        scratch_bufs = self._up_scratch if direction == Direction.UP else self._down_scratch
        h = len(artifact.state.level_offsets) - 1
        assert self._launch_event is not None
        if direction == Direction.UP:
            seed_level = 0
            level_iter = range(1, h)
        else:
            seed_level = h - 1
            level_iter = range(h - 2, -1, -1)
        copy_done = self._copy_done
        compute_done = self._compute_done
        ready = self._ready
        with self._root_stream:
            self._launch_event.record(self._root_stream)
        for stream in self._slot_copy_streams:
            stream.wait_event(self._launch_event)
        for stream in self._level_streams:
            stream.wait_event(self._launch_event)
        for level_streams in scratch_streams:
            for stream in level_streams:
                stream.wait_event(self._launch_event)
        if h > 0:
            with self._level_streams[seed_level]:
                self._publish_level_source(direction, dense, seed_level)
                ready[seed_level].record(self._level_streams[seed_level])
        for dst_level in level_iter:
            stream = self._level_streams[dst_level]
            if scratch_enabled[dst_level]:
                for op_idx, op in enumerate(ops_by_level[dst_level]):
                    self._copy_to_slot(copy_done, compute_done, dst_level, op_idx, op)
                    helper = scratch_streams[dst_level][op_idx]
                    with helper:
                        helper.wait_event(ready[op.src_level])
                        if op.slot is not None:
                            helper.wait_event(copy_done[dst_level][op_idx])
                        rows = int(artifact.state.level_offsets[dst_level + 1] - artifact.state.level_offsets[dst_level])
                        scratch_bufs[dst_level][op_idx][:rows].fill(0)
                        ext = self._ext_for(direction, dst_level, op_idx, True)
                        self._launch_spmm(artifact, direction, op, dense.scratch_descs_by_level[dst_level][op_idx], self._beta_zero.data.ptr, dense.src_descs[op.src_level], ext)
                        compute_done[dst_level][op_idx].record(helper)
                with stream:
                    rows = int(artifact.state.level_offsets[dst_level + 1] - artifact.state.level_offsets[dst_level])
                    for op_idx in range(len(ops_by_level[dst_level])):
                        stream.wait_event(compute_done[dst_level][op_idx])
                        level_bufs[dst_level][:rows] += scratch_bufs[dst_level][op_idx][:rows]
                    self._publish_level_source(direction, dense, dst_level)
                    ready[dst_level].record(stream)
                continue
            with stream:
                for op_idx, op in enumerate(ops_by_level[dst_level]):
                    self._copy_to_slot(copy_done, compute_done, dst_level, op_idx, op)
                    stream.wait_event(ready[op.src_level])
                    if op.slot is not None:
                        stream.wait_event(copy_done[dst_level][op_idx])
                    ext = self._ext_for(direction, dst_level, op_idx, False)
                    self._launch_spmm(artifact, direction, op, dense.dst_descs[dst_level], self._beta_one.data.ptr, dense.src_descs[op.src_level], ext)
                    compute_done[dst_level][op_idx].record(stream)
                self._publish_level_source(direction, dense, dst_level)
                ready[dst_level].record(stream)
        with self._root_stream:
            for event in ready[:h]:
                self._root_stream.wait_event(event)

    def _seed_prepared(self, artifact: _CuArtifact, spec: _CudaMatmulSpec, prepared: _CusparsePreparedMatmul) -> None:
        level_bufs = self._up_level_bufs if spec.direction == Direction.UP else self._down_level_bufs
        assert level_bufs
        k = int(spec.k)
        with self._root_stream:
            for level in range(len(artifact.state.level_offsets) - 1):
                rows = int(artifact.state.level_offsets[level + 1] - artifact.state.level_offsets[level])
                level_bufs[level][:rows, :k].fill(0)
            if spec.backend_init_mode == InitMode.XTX:
                if artifact.xtx_bias is None:
                    raise ValueError("init_mode=xtx requires GRG coalescence counts")
                for level in range(len(artifact.state.level_offsets) - 1):
                    lo = int(artifact.state.level_offsets[level])
                    hi = int(artifact.state.level_offsets[level + 1])
                    level_bufs[level][: hi - lo, :k] += artifact.xtx_bias[lo:hi, None]
            elif spec.backend_init_mode == InitMode.VECTOR:
                for level in range(len(artifact.state.level_offsets) - 1):
                    rows = int(artifact.state.level_offsets[level + 1] - artifact.state.level_offsets[level])
                    level_bufs[level][:rows, :k] += prepared._init_vector_internal[None, :]
            elif spec.backend_init_mode == InitMode.MATRIX:
                for level in range(len(artifact.state.level_offsets) - 1):
                    lo = int(artifact.state.level_offsets[level])
                    hi = int(artifact.state.level_offsets[level + 1])
                    self._cp.take(prepared._init_matrix_internal, artifact.node_perm[lo:hi], axis=0, out=level_bufs[level][: hi - lo, :k])

            if spec.direction == Direction.UP:
                if spec.by_individual:
                    temp = self._aux[: artifact.state.num_samples, :k]
                    self._cp.take(prepared._input_internal, artifact.sample_to_individual, axis=0, out=temp)
                    level_bufs[0][: artifact.state.num_samples, :k] += temp
                else:
                    level_bufs[0][: artifact.state.num_samples, :k] += prepared._input_internal
                return

            for level, rows in enumerate(artifact.mut_selector.rows_by_level):
                if rows.size == 0:
                    continue
                count = int(rows.size)
                level_rows = int(artifact.state.level_offsets[level + 1] - artifact.state.level_offsets[level])
                temp = self._aux[:count, :k]
                self._cp.take(prepared._input_internal, rows, axis=0, out=temp)
                self._cp.add.at(level_bufs[level][:level_rows, :k], artifact.mut_selector.cols_by_level[level], temp)
            if spec.use_miss:
                for level, rows in enumerate(artifact.miss_selector.rows_by_level):
                    if rows.size == 0:
                        continue
                    count = int(rows.size)
                    level_rows = int(artifact.state.level_offsets[level + 1] - artifact.state.level_offsets[level])
                    temp = self._aux[:count, :k]
                    self._cp.take(prepared._miss_input_internal, rows, axis=0, out=temp)
                    self._cp.add.at(level_bufs[level][:level_rows, :k], artifact.miss_selector.cols_by_level[level], temp)

    def _apply_endpoint_bias_prepared(
        self,
        artifact: _CuArtifact,
        spec: _CudaMatmulSpec,
        prepared: _CusparsePreparedMatmul,
        output,
    ) -> None:
        if not spec.apply_endpoint_bias:
            return
        rows = int(output.shape[0])
        if spec.init_mode == InitMode.XTX:
            bias = artifact.init_xtx_up_bias if spec.direction == Direction.UP else artifact.init_xtx_down_bias
            if bias is None:
                raise RuntimeError(f"missing cuSPARSE {spec.direction.value} XTX endpoint bias")
            output += bias[:rows, None]
            return
        bias = artifact.init_vector_up_bias if spec.direction == Direction.UP else artifact.init_vector_down_bias
        if bias is None:
            raise RuntimeError(f"missing cuSPARSE {spec.direction.value} vector endpoint bias")
        temp = self._aux[:rows, : spec.k]
        temp.fill(0)
        temp += bias[:rows, None]
        temp *= prepared._init_vector_internal[None, :]
        output += temp

    def _write_prepared_output(self, artifact: _CuArtifact, spec: _CudaMatmulSpec, prepared: _CusparsePreparedMatmul) -> None:
        level_bufs = self._up_level_bufs if spec.direction == Direction.UP else self._down_level_bufs
        assert level_bufs
        k = int(spec.k)
        with self._root_stream:
            if spec.emit_all_nodes:
                for level in range(len(artifact.state.level_offsets) - 1):
                    lo = int(artifact.state.level_offsets[level])
                    hi = int(artifact.state.level_offsets[level + 1])
                    prepared._output_internal[artifact.node_perm[lo:hi]] = level_bufs[level][: hi - lo, :k]
                return

            if spec.direction == Direction.UP:
                output = prepared._output_internal
                output.fill(0)
                for level, rows in enumerate(artifact.mut_selector.rows_by_level):
                    if rows.size == 0:
                        continue
                    count = int(rows.size)
                    level_rows = int(artifact.state.level_offsets[level + 1] - artifact.state.level_offsets[level])
                    temp = self._aux[:count, :k]
                    self._cp.take(level_bufs[level][:level_rows, :k], artifact.mut_selector.cols_by_level[level], axis=0, out=temp)
                    if artifact.mut_selector.row_unique:
                        output[rows] = temp
                    else:
                        self._cp.add.at(output, rows, temp)
                if spec.use_miss:
                    miss_output = prepared._miss_output_internal
                    miss_output.fill(0)
                    for level, rows in enumerate(artifact.miss_selector.rows_by_level):
                        if rows.size == 0:
                            continue
                        count = int(rows.size)
                        level_rows = int(artifact.state.level_offsets[level + 1] - artifact.state.level_offsets[level])
                        temp = self._aux[:count, :k]
                        self._cp.take(level_bufs[level][:level_rows, :k], artifact.miss_selector.cols_by_level[level], axis=0, out=temp)
                        if artifact.miss_selector.row_unique:
                            miss_output[rows] = temp
                        else:
                            self._cp.add.at(miss_output, rows, temp)
                self._apply_endpoint_bias_prepared(artifact, spec, prepared, output)
                return

            if spec.by_individual:
                output = level_bufs[0][: artifact.state.num_samples, :k]
                self._apply_endpoint_bias_prepared(artifact, spec, prepared, output)
                prepared._output_internal.fill(0)
                self._cp.add.at(prepared._output_internal, artifact.sample_to_individual, output)
            else:
                self._cp.copyto(prepared._output_internal[: artifact.state.num_samples, :k], level_bufs[0][: artifact.state.num_samples, :k])
                self._apply_endpoint_bias_prepared(artifact, spec, prepared, prepared._output_internal)

    def _execute_prepared(self, artifact: _CuArtifact, spec: _CudaMatmulSpec, prepared: _CusparsePreparedMatmul) -> None:
        dense = prepared._dense_state
        if dense is None:
            raise RuntimeError("cuSPARSE prepared matmul must be entered before use")
        with self.device:
            with self._caller_root_scope():
                self._seed_prepared(artifact, spec, prepared)
                self._enqueue_wavefront(artifact, spec.direction, dense)
                self._write_prepared_output(artifact, spec, prepared)

__all__ = ["CusparseLayout", "CusparseRuntime", "plan_cusparse_layout"]
