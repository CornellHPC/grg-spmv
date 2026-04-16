"""MKL runtime and layout planner."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import mmap
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends.base import (
    BudgetItem,
    iter_direction_level_pairs,
    materialize_sparse_block,
    sparse_structure_lengths,
    stored_block_shape,
)
import pygrgl_spmv.backends.mkl.ffi as mkl_ffi
from pygrgl_spmv.backends.mkl.ffi import MklSparseHandle, _address_array, _mmap, _munmap, mkl_set_num_threads
from pygrgl_spmv.backends.types import Direction, InitMode, StoredMatrix
from pygrgl_spmv.grg import BoundGRG, RuntimeRequirements
from pygrgl_spmv.grg.artifact import _load_grg_spmv_host, iter_artifact_blocks, scan_grg_spmv

from .plan import MklPlan, MklPlanPair

_HEADROOM_MAPS = 4096
_MKL_EXPECTED_CALLS = 1000
_PROT_NONE = 0
_PROT_READ = int(mmap.PROT_READ)
_PROT_WRITE = int(mmap.PROT_WRITE)
_MAP_SHARED = int(mmap.MAP_SHARED)
_MAP_PRIVATE = int(mmap.MAP_PRIVATE)
_MAP_ANONYMOUS = int(getattr(mmap, "MAP_ANONYMOUS", 0x20))
_MAP_FIXED = int(getattr(mmap, "MAP_FIXED", 0x10))


@dataclass(frozen=True)
class _MklSharedValuesPlan:
    mode: str
    logical_bytes: int
    physical_bytes: int
    tile_bytes: int


@dataclass(frozen=True)
class _MklBlockPlan:
    dst_level: int
    src_level: int
    stored_shape: tuple[int, int]
    nnz: int
    struct_bytes: int


@dataclass(frozen=True)
class _MklArtifactLayout:
    path: Path
    share_storage: bool
    up_owner: Direction | None
    down_owner: Direction | None
    blocks_up: tuple[_MklBlockPlan, ...]
    blocks_down: tuple[_MklBlockPlan, ...]


@dataclass
class MklLayout:
    artifacts: tuple[_MklArtifactLayout, ...]
    pair: MklPlanPair
    dtype: np.dtype
    struct_dtype: np.dtype
    shared_values: _MklSharedValuesPlan
    requirements: RuntimeRequirements
    budget_items: tuple[BudgetItem, ...]
    required_budget_for_full_residency: int
    bytes_by_category: dict[str, int]
    bytes_total: int


@dataclass(frozen=True)
class _MklOp:
    src_level: int
    handle: MklSparseHandle
    transpose: bool


@dataclass
class _MklArtifact:
    path: Path
    state: object
    up_grid: list[list[MklSparseHandle | None]]
    down_grid: list[list[MklSparseHandle | None]]
    up_ops: list[list[_MklOp]]
    down_ops: list[list[_MklOp]]


@dataclass
class _SharedValues:
    array: np.ndarray
    logical_bytes: int
    physical_bytes: int
    mode: str
    fd: int = -1
    base_addr: int = 0
    reserved_bytes: int = 0

    def destroy(self) -> None:
        self.array = np.empty(0, dtype=self.array.dtype)
        if self.base_addr and self.reserved_bytes:
            _munmap(self.base_addr, self.reserved_bytes)
            self.base_addr = 0
            self.reserved_bytes = 0
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


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


def _round_up(value: int, alignment: int) -> int:
    return int(((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment))


def _page_size() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        return None


def _vm_max_map_count() -> int | None:
    try:
        with open("/proc/sys/vm/max_map_count", encoding="ascii") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _current_map_count() -> int | None:
    try:
        with open("/proc/self/maps", encoding="ascii") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def _shared_values_plan(dtype: np.dtype, max_nnz: int) -> _MklSharedValuesPlan:
    logical_bytes = int(max(int(max_nnz), 0) * int(np.dtype(dtype).itemsize))
    if logical_bytes == 0:
        return _MklSharedValuesPlan(mode="disabled", logical_bytes=0, physical_bytes=0, tile_bytes=0)
    if not hasattr(os, "memfd_create"):
        return _MklSharedValuesPlan(mode="materialized", logical_bytes=logical_bytes, physical_bytes=logical_bytes, tile_bytes=0)
    page_size = _page_size()
    max_maps = _vm_max_map_count()
    current_maps = _current_map_count()
    if page_size is None or max_maps is None or current_maps is None:
        return _MklSharedValuesPlan(mode="materialized", logical_bytes=logical_bytes, physical_bytes=logical_bytes, tile_bytes=0)
    usable_maps = int(max_maps) - int(current_maps) - _HEADROOM_MAPS
    if usable_maps < 2:
        return _MklSharedValuesPlan(mode="materialized", logical_bytes=logical_bytes, physical_bytes=logical_bytes, tile_bytes=0)
    alignment = math.lcm(int(page_size), int(np.dtype(dtype).itemsize))
    tile_bytes = _round_up(max(alignment, math.ceil(logical_bytes / usable_maps)), alignment)
    if tile_bytes >= logical_bytes:
        return _MklSharedValuesPlan(mode="materialized", logical_bytes=logical_bytes, physical_bytes=logical_bytes, tile_bytes=0)
    return _MklSharedValuesPlan(mode="alias", logical_bytes=logical_bytes, physical_bytes=tile_bytes, tile_bytes=tile_bytes)


def _plan_blocks(scan, plan: MklPlan, *, struct_dtype: np.dtype) -> tuple[_MklBlockPlan, ...]:
    blocks: list[_MklBlockPlan] = []
    itemsize = int(np.dtype(struct_dtype).itemsize)
    for block in scan.blocks:
        nrows, ncols = stored_block_shape(block.shape[0], block.shape[1], store=plan.store)
        len0, len1 = sparse_structure_lengths(plan.fmt, nrows=nrows, ncols=ncols, nnz=block.nnz)
        blocks.append(
            _MklBlockPlan(
                dst_level=int(block.dst_level),
                src_level=int(block.src_level),
                stored_shape=(int(nrows), int(ncols)),
                nnz=int(block.nnz),
                struct_bytes=int((len0 + len1) * itemsize),
            )
        )
    return tuple(blocks)


def plan_mkl_layout(
    *,
    artifacts,
    pair: MklPlanPair,
    dtype,
    requirements: RuntimeRequirements,
) -> MklLayout:
    dtype = np.dtype(dtype)
    if dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError(f"MKL runtime supports only float32/float64, got {dtype}")
    _, struct_dtype, _ = mkl_ffi._ensure_loaded()
    paths = _resolve_artifacts(artifacts)
    scans = tuple(scan_grg_spmv(path) for path in paths)
    max_nodes = max(scan.num_nodes for scan in scans)
    sparse_bytes = 0
    selector_bytes = 0
    max_owned_block_nnz = 0
    planned: list[_MklArtifactLayout] = []
    for path, scan in zip(paths, scans, strict=True):
        share_storage = bool(pair.plan_up is not None and pair.plan_down is not None and pair.plan_up.can_share_storage_with(pair.plan_down))
        blocks_up = () if pair.plan_up is None else _plan_blocks(scan, pair.plan_up, struct_dtype=struct_dtype)
        blocks_down = () if pair.plan_down is None or share_storage else _plan_blocks(scan, pair.plan_down, struct_dtype=struct_dtype)
        sparse_bytes += sum(block.struct_bytes for block in blocks_up)
        sparse_bytes += sum(block.struct_bytes for block in blocks_down)
        max_owned_block_nnz = max(max_owned_block_nnz, max((block.nnz for block in blocks_up), default=0))
        max_owned_block_nnz = max(max_owned_block_nnz, max((block.nnz for block in blocks_down), default=0))
        state = _load_grg_spmv_host(path, dtype)
        selector_bytes += int(state.sel_mut.indices.nbytes + state.sel_mut.indptr.nbytes + state.sel_mut.data.nbytes)
        selector_bytes += int(state.sel_miss.indices.nbytes + state.sel_miss.indptr.nbytes + state.sel_miss.data.nbytes)
        planned.append(
            _MklArtifactLayout(
                path=path,
                share_storage=share_storage,
                up_owner=Direction.UP if pair.plan_up is not None else None,
                down_owner=None if pair.plan_down is None else (Direction.UP if share_storage else Direction.DOWN),
                blocks_up=blocks_up,
                blocks_down=blocks_down,
            )
        )
    workspace_up = 0 if pair.plan_up is None else int(max_nodes * int(requirements.max_k_up) * dtype.itemsize)
    workspace_down = 0 if pair.plan_down is None else int(max_nodes * int(requirements.max_k_down) * dtype.itemsize)
    shared_values = _shared_values_plan(dtype, max_owned_block_nnz)
    bytes_by_category = {
        "resident_sparse": int(sparse_bytes),
        "selectors": int(selector_bytes),
        "workspace_up": int(workspace_up),
        "workspace_down": int(workspace_down),
        "shared_values": int(shared_values.physical_bytes),
    }
    budget_items: list[BudgetItem] = [
        BudgetItem(kind="fixed", name="selectors", nbytes=int(selector_bytes)),
        BudgetItem(kind="fixed", name="workspace_up", nbytes=int(workspace_up)),
        BudgetItem(kind="fixed", name="workspace_down", nbytes=int(workspace_down)),
        BudgetItem(kind="fixed", name="shared_values", nbytes=int(shared_values.physical_bytes)),
    ]
    for artifact_index, artifact_layout in enumerate(planned):
        for block in artifact_layout.blocks_up:
            budget_items.append(
                BudgetItem(
                    kind="resident_sparse",
                    name="resident_sparse",
                    nbytes=int(block.struct_bytes),
                    artifact_index=artifact_index,
                    dst_level=int(block.dst_level),
                    src_level=int(block.src_level),
                )
            )
        for block in artifact_layout.blocks_down:
            budget_items.append(
                BudgetItem(
                    kind="resident_sparse",
                    name="resident_sparse",
                    nbytes=int(block.struct_bytes),
                    artifact_index=artifact_index,
                    dst_level=int(block.dst_level),
                    src_level=int(block.src_level),
                )
            )
    budget_items = [item for item in budget_items if item.nbytes > 0]
    bytes_total = int(sum(item.nbytes for item in budget_items))
    return MklLayout(
        artifacts=tuple(planned),
        pair=pair,
        dtype=dtype,
        struct_dtype=np.dtype(struct_dtype),
        shared_values=shared_values,
        requirements=requirements,
        budget_items=tuple(budget_items),
        required_budget_for_full_residency=bytes_total,
        bytes_by_category=bytes_by_category,
        bytes_total=bytes_total,
    )


def _needs_transpose(direction: Direction, store: StoredMatrix) -> bool:
    return (direction == Direction.DOWN) != (store == StoredMatrix.T)


def _thread_count(plan: MklPlan | None) -> int | None:
    if plan is None:
        return None
    count = os.cpu_count() or 1
    return count if int(plan.n_threads) == 0 else int(plan.n_threads)


class MklRuntime:
    """Runtime-owned MKL execution."""

    def __init__(self, layout: MklLayout) -> None:
        self.layout = layout
        self.device = None
        self.stream = None
        self.stream_ptr = None
        self._artifacts: tuple[_MklArtifact, ...] = ()
        self._up_workspace: np.ndarray | None = None
        self._down_workspace: np.ndarray | None = None
        self._shared_values: _SharedValues | None = None
        self._entered = False
        self._active_call = False
        self._grgs: tuple[BoundGRG, ...] = ()
        self._threads_up = _thread_count(layout.pair.plan_up)
        self._threads_down = _thread_count(layout.pair.plan_down)

    @property
    def grgs(self) -> tuple[BoundGRG, ...]:
        if not self._entered:
            raise RuntimeError("MklRuntime must be entered before accessing grgs")
        return self._grgs

    def __enter__(self) -> "MklRuntime":
        dtype = np.dtype(self.layout.dtype)
        states = tuple(_load_grg_spmv_host(artifact.path, dtype) for artifact in self.layout.artifacts)
        max_nodes = max(state.num_nodes for state in states)
        if self.layout.pair.plan_up is not None:
            self._up_workspace = np.zeros((max_nodes, int(self.layout.requirements.max_k_up)), dtype=dtype)
        if self.layout.pair.plan_down is not None:
            self._down_workspace = np.zeros((max_nodes, int(self.layout.requirements.max_k_down)), dtype=dtype)
        setup_threads = max(count for count in (self._threads_up, self._threads_down) if count is not None)
        mkl_set_num_threads(int(setup_threads))
        self._shared_values = self._materialize_shared_values()
        artifacts: list[_MklArtifact] = []
        try:
            for artifact_layout, state in zip(self.layout.artifacts, states, strict=True):
                h = len(state.level_offsets) - 1
                up_grid = [[None] * dst_level for dst_level in range(h)]
                down_grid = [[None] * dst_level for dst_level in range(h)]
                for block in iter_artifact_blocks(artifact_layout.path):
                    base = sp.csr_matrix(
                        (
                            np.ones(block.nnz, dtype=np.bool_),
                            np.asarray(block.indices),
                            np.asarray(block.indptr),
                        ),
                        shape=block.shape,
                    )
                    if self.layout.pair.plan_up is not None:
                        matrix = materialize_sparse_block(
                            base,
                            store=self.layout.pair.plan_up.store,
                            fmt=self.layout.pair.plan_up.fmt,
                        )
                        up_grid[block.dst_level][block.src_level] = self._build_handle(matrix, self.layout.pair.plan_up)
                    if self.layout.pair.plan_down is not None and not artifact_layout.share_storage:
                        matrix = materialize_sparse_block(
                            base,
                            store=self.layout.pair.plan_down.store,
                            fmt=self.layout.pair.plan_down.fmt,
                        )
                        down_grid[block.dst_level][block.src_level] = self._build_handle(matrix, self.layout.pair.plan_down)
                artifacts.append(
                    _MklArtifact(
                        path=artifact_layout.path,
                        state=state,
                        up_grid=up_grid,
                        down_grid=down_grid,
                        up_ops=self._build_ops(Direction.UP, state, up_grid, down_grid, artifact_layout),
                        down_ops=self._build_ops(Direction.DOWN, state, up_grid, down_grid, artifact_layout),
                    )
                )
            self._artifacts = tuple(artifacts)
            self._configure_handle_hints()
            self._grgs = tuple(BoundGRG(self, idx, artifact.state, artifact.path) for idx, artifact in enumerate(self._artifacts))
            self._entered = True
            return self
        except Exception:
            self._destroy_handles(artifacts)
            if self._shared_values is not None:
                self._shared_values.destroy()
                self._shared_values = None
            self._up_workspace = None
            self._down_workspace = None
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        self._destroy_handles(self._artifacts)
        self._artifacts = ()
        self._up_workspace = None
        self._down_workspace = None
        if self._shared_values is not None:
            self._shared_values.destroy()
            self._shared_values = None
        self._grgs = ()
        self._entered = False
        self._active_call = False

    @contextmanager
    def _call_scope(self):
        if not self._entered:
            raise RuntimeError("MklRuntime must be entered before matmul")
        if self._active_call:
            raise RuntimeError("concurrent runtime.grgs calls are not supported")
        self._active_call = True
        try:
            yield
        finally:
            self._active_call = False

    def _materialize_shared_values(self) -> _SharedValues:
        plan = self.layout.shared_values
        dtype = np.dtype(self.layout.dtype)
        logical_len = int(plan.logical_bytes // int(dtype.itemsize))
        if plan.mode == "disabled":
            return _SharedValues(array=np.empty(0, dtype=dtype), logical_bytes=0, physical_bytes=0, mode="disabled")
        if plan.mode == "materialized":
            arr = np.ones(logical_len, dtype=dtype)
            arr.setflags(write=False)
            return _SharedValues(
                array=arr,
                logical_bytes=int(plan.logical_bytes),
                physical_bytes=int(arr.nbytes),
                mode="materialized",
            )
        if plan.mode != "alias":
            raise ValueError(f"unknown shared-values mode {plan.mode!r}")
        fd = os.memfd_create("pygrgl_spmv_mkl_shared_values", getattr(os, "MFD_CLOEXEC", 0))
        tile_bytes = int(plan.tile_bytes)
        reserved_bytes = _round_up(int(plan.logical_bytes), tile_bytes)
        seed_addr = 0
        base_addr = 0
        try:
            os.ftruncate(fd, tile_bytes)
            seed_addr = _mmap(None, tile_bytes, _PROT_READ | _PROT_WRITE, _MAP_SHARED, fd, 0)
            seed = _address_array(seed_addr, tile_bytes // int(dtype.itemsize), dtype)
            seed.fill(1)
            base_addr = _mmap(None, reserved_bytes, _PROT_NONE, _MAP_PRIVATE | _MAP_ANONYMOUS, -1, 0)
            for offset in range(0, reserved_bytes, tile_bytes):
                _mmap(base_addr + offset, tile_bytes, _PROT_READ, _MAP_SHARED | _MAP_FIXED, fd, 0)
            _munmap(seed_addr, tile_bytes)
            seed_addr = 0
            arr = _address_array(base_addr, logical_len, dtype)
            arr.setflags(write=False)
            return _SharedValues(
                array=arr,
                logical_bytes=int(plan.logical_bytes),
                physical_bytes=int(plan.physical_bytes),
                mode="alias",
                fd=fd,
                base_addr=base_addr,
                reserved_bytes=reserved_bytes,
            )
        except Exception:
            if seed_addr:
                _munmap(seed_addr, tile_bytes)
            if base_addr:
                _munmap(base_addr, reserved_bytes)
            os.close(fd)
            raise

    def _build_handle(self, matrix, plan: MklPlan) -> MklSparseHandle | None:
        if matrix.nnz == 0:
            return None
        assert self._shared_values is not None
        matrix.data = self._shared_values.array[: int(matrix.nnz)]
        return MklSparseHandle(matrix, plan.fmt.value.lower(), dtype=self.layout.dtype)

    def _destroy_handles(self, artifacts) -> None:
        seen: set[int] = set()
        for artifact in artifacts:
            for grid in (artifact.up_grid, artifact.down_grid):
                for row in grid:
                    for handle in row:
                        if handle is None:
                            continue
                        key = id(handle)
                        if key in seen:
                            continue
                        seen.add(key)
                        handle.destroy()

    def _configure_handle_hints(self) -> None:
        usage: dict[tuple[int, bool], dict[str, object]] = {}
        for artifact in self._artifacts:
            for ops_by_level, plan, max_k in (
                (artifact.up_ops, self.layout.pair.plan_up, int(self.layout.requirements.max_k_up)),
                (artifact.down_ops, self.layout.pair.plan_down, int(self.layout.requirements.max_k_down)),
            ):
                if plan is None or not plan.optimize:
                    continue
                for ops in ops_by_level:
                    for op in ops:
                        key = (id(op.handle), bool(op.transpose))
                        entry = usage.get(key)
                        if entry is None:
                            usage[key] = {
                                "handle": op.handle,
                                "transpose": bool(op.transpose),
                                "max_k": int(max_k),
                            }
                            continue
                        entry["max_k"] = max(int(entry["max_k"]), int(max_k))
        for entry in usage.values():
            handle = entry["handle"]
            assert isinstance(handle, MklSparseHandle)
            transpose = bool(entry["transpose"])
            handle.set_mv_hint(transpose=transpose, expected_calls=_MKL_EXPECTED_CALLS)
            max_k = int(entry["max_k"])
            if max_k > 1:
                handle.set_mm_hint(max_k, transpose=transpose, expected_calls=_MKL_EXPECTED_CALLS)
            handle.optimize()

    def _build_ops(
        self,
        direction: Direction,
        state,
        up_grid: list[list[MklSparseHandle | None]],
        down_grid: list[list[MklSparseHandle | None]],
        artifact_layout: _MklArtifactLayout,
    ) -> list[list[_MklOp]]:
        h = len(state.level_offsets) - 1
        ops: list[list[_MklOp]] = [[] for _ in range(h)]
        if (direction == Direction.UP and self.layout.pair.plan_up is None) or (direction == Direction.DOWN and self.layout.pair.plan_down is None):
            return ops
        owner_direction = artifact_layout.up_owner if direction == Direction.UP else artifact_layout.down_owner
        assert owner_direction is not None
        owner_plan = self.layout.pair.plan_up if owner_direction == Direction.UP else self.layout.pair.plan_down
        assert owner_plan is not None
        owner_grid = up_grid if owner_direction == Direction.UP else down_grid
        for dst_level, src_level, _row_index in iter_direction_level_pairs(direction, h):
            owner_dst = dst_level if direction == Direction.UP else src_level
            owner_src = src_level if direction == Direction.UP else dst_level
            handle = owner_grid[owner_dst][owner_src]
            if handle is None:
                continue
            ops[dst_level].append(
                _MklOp(
                    src_level=src_level,
                    handle=handle,
                    transpose=_needs_transpose(direction, owner_plan.store),
                )
            )
        return ops

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
        artifact = self._artifacts[int(artifact_index)]
        state = artifact.state
        workspace = self._up_workspace if direction == Direction.UP else self._down_workspace
        if workspace is None:
            raise ValueError(f"{direction.value.upper()} plan is not configured")
        thread_count = self._threads_up if direction == Direction.UP else self._threads_down
        assert thread_count is not None
        mkl_set_num_threads(int(thread_count))
        x = np.asarray(primary, dtype=self.layout.dtype, order="C")
        k = int(x.shape[1])
        node_values = workspace[: state.num_nodes, :k]
        node_values.fill(0)
        if init_mode == InitMode.XTX:
            if state.coalescence_counts is None:
                raise ValueError("init_mode=xtx requires GRG coalescence counts")
            node_values += (2.0 * state.coalescence_counts.astype(self.layout.dtype, copy=False))[:, None]
        elif init_mode == InitMode.VECTOR:
            assert init_payload is not None
            node_values += init_payload[None, :]
        elif init_mode == InitMode.MATRIX:
            assert init_payload is not None
            node_values += init_payload

        use_mv = k == 1 and workspace.shape[1] == 1
        if direction == Direction.UP:
            node_values[: state.num_samples] += x
            ops = artifact.up_ops
            for dst_level in range(1, len(state.level_offsets) - 1):
                lo = int(state.level_offsets[dst_level])
                hi = int(state.level_offsets[dst_level + 1])
                dst = node_values[lo:hi]
                for op in ops[dst_level]:
                    src_lo = int(state.level_offsets[op.src_level])
                    src_hi = int(state.level_offsets[op.src_level + 1])
                    src = node_values[src_lo:src_hi]
                    if use_mv:
                        op.handle.mv(src[:, 0], dst[:, 0], alpha=1.0, beta=1.0, transpose=op.transpose)
                    else:
                        op.handle.mm(src, dst, alpha=1.0, beta=1.0, transpose=op.transpose)
            if emit_all_nodes:
                return np.array(node_values, copy=True)
            out_mut = (
                np.asarray(state.sel_mut @ node_values, dtype=self.layout.dtype)
                if state.sel_mut.nnz
                else np.zeros((state.num_mutations, k), dtype=self.layout.dtype)
            )
            out_miss = None
            if need_miss_output:
                out_miss = (
                    np.asarray(state.sel_miss @ node_values, dtype=self.layout.dtype)
                    if state.sel_miss.nnz
                    else np.zeros((state.num_mutations, k), dtype=self.layout.dtype)
                )
            return out_mut, out_miss

        if state.sel_mut.nnz:
            node_values += state.sel_mut.T @ x
        if miss is not None and state.sel_miss.nnz:
            node_values += state.sel_miss.T @ np.asarray(miss, dtype=self.layout.dtype, order="C")
        ops = artifact.down_ops
        for dst_level in range(len(state.level_offsets) - 2, -1, -1):
            lo = int(state.level_offsets[dst_level])
            hi = int(state.level_offsets[dst_level + 1])
            dst = node_values[lo:hi]
            for op in ops[dst_level]:
                src_lo = int(state.level_offsets[op.src_level])
                src_hi = int(state.level_offsets[op.src_level + 1])
                src = node_values[src_lo:src_hi]
                if use_mv:
                    op.handle.mv(src[:, 0], dst[:, 0], alpha=1.0, beta=1.0, transpose=op.transpose)
                else:
                    op.handle.mm(src, dst, alpha=1.0, beta=1.0, transpose=op.transpose)
        if emit_all_nodes:
            return np.array(node_values, copy=True)
        return np.array(node_values[: state.num_samples], copy=True)


__all__ = ["MklLayout", "MklRuntime", "plan_mkl_layout"]
