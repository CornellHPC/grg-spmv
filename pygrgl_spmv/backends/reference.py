"""Reference runtime and layout planner."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends.base import (
    BudgetItem,
    iter_direction_level_pairs,
    materialize_sparse_block,
    sparse_structure_nbytes,
    stored_block_shape,
)
from pygrgl_spmv.backends.types import (
    Direction,
    InitMode,
    SparseFormat,
    StoredMatrix,
    parse_sparse_format,
    parse_store,
    transpose_compatible_format,
)
from pygrgl_spmv.grg import BoundGRG, RuntimeRequirements
from pygrgl_spmv.grg.artifact import _load_grg_spmv_host, iter_artifact_blocks, scan_grg_spmv


@dataclass(frozen=True)
class ReferencePlan:
    store: StoredMatrix
    fmt: SparseFormat

    def __post_init__(self) -> None:
        object.__setattr__(self, "store", parse_store(self.store))
        object.__setattr__(self, "fmt", parse_sparse_format(self.fmt))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ReferencePlan":
        allowed = {"store", "fmt"}
        extra = sorted(set(value) - allowed)
        if extra:
            raise ValueError(f"unknown ReferencePlan field(s): {extra}")
        return cls(
            store=parse_store(value["store"]),
            fmt=parse_sparse_format(value["fmt"]),
        )

    def can_share_storage_with(self, other: "ReferencePlan") -> bool:
        if self.store == other.store:
            return self.fmt == other.fmt
        return transpose_compatible_format(self.fmt) == other.fmt


@dataclass(frozen=True)
class ReferencePlanPair:
    plan_up: ReferencePlan | None
    plan_down: ReferencePlan | None

    def __post_init__(self) -> None:
        if self.plan_up is None and self.plan_down is None:
            raise ValueError("at least one of plan_up/plan_down must be provided")

    @classmethod
    def from_dicts(
        cls,
        plan_up: Mapping[str, object] | None,
        plan_down: Mapping[str, object] | None,
    ) -> "ReferencePlanPair":
        return cls(
            plan_up=None if plan_up is None else ReferencePlan.from_dict(plan_up),
            plan_down=None if plan_down is None else ReferencePlan.from_dict(plan_down),
        )


@dataclass(frozen=True)
class _ReferenceBlockPlan:
    dst_level: int
    src_level: int
    stored_shape: tuple[int, int]
    nnz: int
    nbytes: int


@dataclass(frozen=True)
class _ReferenceArtifactLayout:
    path: Path
    share_storage: bool
    up_owner: Direction | None
    down_owner: Direction | None
    blocks_up: tuple[_ReferenceBlockPlan, ...]
    blocks_down: tuple[_ReferenceBlockPlan, ...]


@dataclass
class ReferenceLayout:
    artifacts: tuple[_ReferenceArtifactLayout, ...]
    pair: ReferencePlanPair
    dtype: np.dtype
    requirements: RuntimeRequirements
    budget_items: tuple[BudgetItem, ...]
    required_budget_for_full_residency: int
    bytes_by_category: dict[str, int]
    bytes_total: int


@dataclass(frozen=True)
class _ReferenceOp:
    src_level: int
    matrix: sp.spmatrix
    transpose: bool


@dataclass
class _ReferenceArtifact:
    path: Path
    state: object
    up_grid: list[list[sp.spmatrix | None]]
    down_grid: list[list[sp.spmatrix | None]]
    up_ops: list[list[_ReferenceOp]]
    down_ops: list[list[_ReferenceOp]]


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


def _plan_blocks(scan, plan: ReferencePlan) -> tuple[_ReferenceBlockPlan, ...]:
    blocks: list[_ReferenceBlockPlan] = []
    for block in scan.blocks:
        nrows, ncols = stored_block_shape(block.shape[0], block.shape[1], store=plan.store)
        nbytes = int(sparse_structure_nbytes(plan.fmt, nrows=nrows, ncols=ncols, nnz=block.nnz) + block.nnz)
        blocks.append(
            _ReferenceBlockPlan(
                dst_level=block.dst_level,
                src_level=block.src_level,
                stored_shape=(nrows, ncols),
                nnz=block.nnz,
                nbytes=nbytes,
            )
        )
    return tuple(blocks)


def plan_reference_layout(
    *,
    artifacts,
    pair: ReferencePlanPair,
    dtype,
    requirements: RuntimeRequirements,
) -> ReferenceLayout:
    dtype = np.dtype(dtype)
    paths = _resolve_artifacts(artifacts)
    scans = tuple(scan_grg_spmv(path) for path in paths)
    max_nodes = max(scan.num_nodes for scan in scans)
    sparse_bytes = 0
    selector_bytes = 0
    planned: list[_ReferenceArtifactLayout] = []
    for path, scan in zip(paths, scans, strict=True):
        share_storage = bool(pair.plan_up is not None and pair.plan_down is not None and pair.plan_up.can_share_storage_with(pair.plan_down))
        blocks_up = () if pair.plan_up is None else _plan_blocks(scan, pair.plan_up)
        blocks_down = () if pair.plan_down is None or share_storage else _plan_blocks(scan, pair.plan_down)
        sparse_bytes += sum(block.nbytes for block in blocks_up)
        sparse_bytes += sum(block.nbytes for block in blocks_down)
        state = _load_grg_spmv_host(path, dtype)
        selector_bytes += int(state.sel_mut.indices.nbytes + state.sel_mut.indptr.nbytes + state.sel_mut.data.nbytes)
        selector_bytes += int(state.sel_miss.indices.nbytes + state.sel_miss.indptr.nbytes + state.sel_miss.data.nbytes)
        planned.append(
            _ReferenceArtifactLayout(
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
    bytes_by_category = {
        "resident_sparse": int(sparse_bytes),
        "selectors": int(selector_bytes),
        "workspace_up": workspace_up,
        "workspace_down": workspace_down,
    }
    budget_items: list[BudgetItem] = [
        BudgetItem(kind="fixed", name="selectors", nbytes=int(selector_bytes)),
        BudgetItem(kind="fixed", name="workspace_up", nbytes=int(workspace_up)),
        BudgetItem(kind="fixed", name="workspace_down", nbytes=int(workspace_down)),
    ]
    for artifact_index, artifact_layout in enumerate(planned):
        for block in artifact_layout.blocks_up:
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
        for block in artifact_layout.blocks_down:
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
    bytes_total = int(sum(item.nbytes for item in budget_items))
    return ReferenceLayout(
        artifacts=tuple(planned),
        pair=pair,
        dtype=dtype,
        requirements=requirements,
        budget_items=tuple(item for item in budget_items if item.nbytes > 0),
        required_budget_for_full_residency=bytes_total,
        bytes_by_category=bytes_by_category,
        bytes_total=bytes_total,
    )


def _needs_transpose(direction: Direction, store: StoredMatrix) -> bool:
    return (direction == Direction.DOWN) != (store == StoredMatrix.T)


class ReferenceRuntime:
    """Runtime-owned CPU reference execution."""

    def __init__(self, layout: ReferenceLayout) -> None:
        self.layout = layout
        self.device = None
        self.stream = None
        self.stream_ptr = None
        self._artifacts: tuple[_ReferenceArtifact, ...] = ()
        self._up_workspace: np.ndarray | None = None
        self._down_workspace: np.ndarray | None = None
        self._entered = False
        self._active_call = False
        self._grgs: tuple[BoundGRG, ...] = ()

    @property
    def grgs(self) -> tuple[BoundGRG, ...]:
        if not self._entered:
            raise RuntimeError("ReferenceRuntime must be entered before accessing grgs")
        return self._grgs

    def __enter__(self) -> "ReferenceRuntime":
        dtype = np.dtype(self.layout.dtype)
        states = tuple(_load_grg_spmv_host(artifact.path, dtype) for artifact in self.layout.artifacts)
        max_nodes = max(state.num_nodes for state in states)
        if self.layout.pair.plan_up is not None:
            self._up_workspace = np.zeros((max_nodes, int(self.layout.requirements.max_k_up)), dtype=dtype)
        if self.layout.pair.plan_down is not None:
            self._down_workspace = np.zeros((max_nodes, int(self.layout.requirements.max_k_down)), dtype=dtype)
        artifacts: list[_ReferenceArtifact] = []
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
                    up_grid[block.dst_level][block.src_level] = materialize_sparse_block(
                        base,
                        store=self.layout.pair.plan_up.store,
                        fmt=self.layout.pair.plan_up.fmt,
                    )
                if self.layout.pair.plan_down is not None and not artifact_layout.share_storage:
                    down_grid[block.dst_level][block.src_level] = materialize_sparse_block(
                        base,
                        store=self.layout.pair.plan_down.store,
                        fmt=self.layout.pair.plan_down.fmt,
                    )
            artifacts.append(
                _ReferenceArtifact(
                    path=artifact_layout.path,
                    state=state,
                    up_grid=up_grid,
                    down_grid=down_grid,
                    up_ops=self._build_ops(Direction.UP, state, up_grid, down_grid, artifact_layout),
                    down_ops=self._build_ops(Direction.DOWN, state, up_grid, down_grid, artifact_layout),
                )
            )
        self._artifacts = tuple(artifacts)
        self._grgs = tuple(BoundGRG(self, idx, artifact.state, artifact.path) for idx, artifact in enumerate(self._artifacts))
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._artifacts = ()
        self._up_workspace = None
        self._down_workspace = None
        self._grgs = ()
        self._entered = False
        self._active_call = False

    @contextmanager
    def _call_scope(self):
        if not self._entered:
            raise RuntimeError("ReferenceRuntime must be entered before matmul")
        if self._active_call:
            raise RuntimeError("concurrent runtime.grgs calls are not supported")
        self._active_call = True
        try:
            yield
        finally:
            self._active_call = False

    def _build_ops(
        self,
        direction: Direction,
        state,
        up_grid: list[list[sp.spmatrix | None]],
        down_grid: list[list[sp.spmatrix | None]],
        artifact_layout: _ReferenceArtifactLayout,
    ) -> list[list[_ReferenceOp]]:
        h = len(state.level_offsets) - 1
        ops: list[list[_ReferenceOp]] = [[] for _ in range(h)]
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
            matrix = owner_grid[owner_dst][owner_src]
            if matrix is None or matrix.nnz == 0:
                continue
            ops[dst_level].append(
                _ReferenceOp(
                    src_level=src_level,
                    matrix=matrix,
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
        x = np.asarray(primary, dtype=self.layout.dtype, order="C")
        node_values = workspace[: state.num_nodes, : x.shape[1]]
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

        if direction == Direction.UP:
            node_values[: state.num_samples] += x
            ops = artifact.up_ops
            for dst_level in range(1, len(state.level_offsets) - 1):
                lo = int(state.level_offsets[dst_level])
                hi = int(state.level_offsets[dst_level + 1])
                for op in ops[dst_level]:
                    src_lo = int(state.level_offsets[op.src_level])
                    src_hi = int(state.level_offsets[op.src_level + 1])
                    src = node_values[src_lo:src_hi]
                    dst = node_values[lo:hi]
                    dst += op.matrix.T @ src if op.transpose else op.matrix @ src
            if emit_all_nodes:
                return np.array(node_values, copy=True)
            out_mut = np.asarray(state.sel_mut @ node_values, dtype=self.layout.dtype) if state.sel_mut.nnz else np.zeros((state.num_mutations, x.shape[1]), dtype=self.layout.dtype)
            out_miss = None
            if need_miss_output:
                out_miss = np.asarray(state.sel_miss @ node_values, dtype=self.layout.dtype) if state.sel_miss.nnz else np.zeros((state.num_mutations, x.shape[1]), dtype=self.layout.dtype)
            return out_mut, out_miss

        if state.sel_mut.nnz:
            node_values += state.sel_mut.T @ x
        if miss is not None and state.sel_miss.nnz:
            node_values += state.sel_miss.T @ np.asarray(miss, dtype=self.layout.dtype, order="C")
        ops = artifact.down_ops
        for dst_level in range(len(state.level_offsets) - 2, -1, -1):
            lo = int(state.level_offsets[dst_level])
            hi = int(state.level_offsets[dst_level + 1])
            for op in ops[dst_level]:
                src_lo = int(state.level_offsets[op.src_level])
                src_hi = int(state.level_offsets[op.src_level + 1])
                src = node_values[src_lo:src_hi]
                dst = node_values[lo:hi]
                dst += op.matrix.T @ src if op.transpose else op.matrix @ src
        if emit_all_nodes:
            return np.array(node_values, copy=True)
        return np.array(node_values[: state.num_samples], copy=True)


__all__ = [
    "ReferenceLayout",
    "ReferencePlan",
    "ReferencePlanPair",
    "ReferenceRuntime",
    "plan_reference_layout",
]
