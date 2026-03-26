"""cuSPARSE backend for level-wise GRG sparse matmul traversal.

The implementation follows the level-wise equations in ``text/math.tex``:

- UP traversal seeds level 0 with sample values, then for each level ``h`` runs
  ``u[h] += A[h, src] @ u[src]`` over all lower source levels.
- DOWN traversal seeds node buffers with selector scatter-add, then for each
  level ``h`` runs ``r[h] += A[src, h].T @ r[src]`` over all higher source
  levels.

The backend is intentionally organized around a small set of concrete runtime
objects:

- ``_CuBlock`` owns one physical sparse block plus the cuSPARSE descriptors
  that alias its immutable payload.
- ``_CuOp`` is one logical wavefront contribution.
- ``_SelectorLevels`` and ``_SampleLevels`` hold the only two scatter/gather
  schemes needed at the GRG boundary.
- ``_DirectionWorkspace`` is the reusable device-state cache for one direction
  and one runtime ``k``.

Everything else is plain lists and helper methods so the hot path stays direct.
"""

from __future__ import annotations

import logging
import warnings
from ctypes import c_void_p
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends import (
    BackendBase,
    CallCapture,
    BackendSetup,
    effective_k_hint,
    iter_direction_level_pairs,
    selector_rows_unique_from_csr_indptr,
    warn_instrumentation_ignores_k_hint,
    warn_k_hint_mismatch,
)
from pygrgl_spmv.backends._nvtx import make_cupy_tracer
from pygrgl_spmv.backends.cusparse.ffi import (
    CuSparseLib,
    CudaVmmDriver,
    cuda_dtype,
)
from pygrgl_spmv.memory import VmmAliasedAlloc, alloc_field, child_field, ignore_field
from pygrgl_spmv.backends.types import Direction, InitMode, SparseFormat, parse_init_mode
from . import plan as cusparse_plan
from .plan import CusparsePlan, CusparsePlanPair, DenseOrder, Operation, SpMMAlgorithm

if TYPE_CHECKING:
    import cupy as cp

    CupyArray = cp.ndarray
    CupyEvent = cp.cuda.Event
    CupyGraph = cp.cuda.graph.Graph
    CupyStream = cp.cuda.Stream
else:
    CupyArray = Any
    CupyEvent = Any
    CupyGraph = Any
    CupyStream = Any


def _round_up(value: int, alignment: int) -> int:
    return int(((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment))


def is_valid_combo(fmt: str, transpose_bool: bool, algo: str) -> bool:
    """Return whether a combo is executable on the current backend/runtime path."""
    plan = CusparsePlan.from_dict(
        {
            "k_hint": None,
            "store": "N",
            "fmt": str(fmt).upper(),
            "opA": "T" if bool(transpose_bool) else "N",
            "opB": "N",
            "orderB": "ROW",
            "orderC": "ROW",
            "algo": str(algo).upper(),
        }
    )
    return plan.supported


@dataclass
class _CuBlock:
    """One physical sparse block plus its cuSPARSE descriptors."""

    fmt: str = ignore_field()
    nrows: int = ignore_field()
    ncols: int = ignore_field()
    nnz: int = ignore_field()
    index_buffers: tuple[CupyArray, CupyArray] = alloc_field(label="blocks", kind="sparse")
    data_ptr: int = ignore_field()
    graph_desc: c_void_p = ignore_field()
    dynamic_desc: c_void_p = ignore_field()
    payload_key: tuple[int, int, int] = ignore_field()

    @classmethod
    def from_scipy(
        cls,
        matrix: sp.spmatrix,
        *,
        fmt: str,
        cp: Any,
        data_ptr: int,
        cslib: CuSparseLib,
        cuda_dtype_id: int,
    ) -> _CuBlock:
        if fmt == "csr":
            mat = sp.csr_matrix(matrix)
            index_buffers = (
                cp.asarray(mat.indptr.astype(np.int32, copy=False)),
                cp.asarray(mat.indices.astype(np.int32, copy=False)),
            )
        elif fmt == "csc":
            mat = matrix.tocsc()
            index_buffers = (
                cp.asarray(mat.indptr.astype(np.int32, copy=False)),
                cp.asarray(mat.indices.astype(np.int32, copy=False)),
            )
        elif fmt == "coo":
            mat = matrix.tocoo()
            index_buffers = (
                cp.asarray(mat.row.astype(np.int32, copy=False)),
                cp.asarray(mat.col.astype(np.int32, copy=False)),
            )
        else:
            raise ValueError(f"Unknown sparse format: {fmt!r}")

        block = cls(
            fmt=fmt,
            nrows=int(mat.shape[0]),
            ncols=int(mat.shape[1]),
            nnz=int(mat.nnz),
            index_buffers=index_buffers,
            data_ptr=int(data_ptr),
            graph_desc=c_void_p(),
            dynamic_desc=c_void_p(),
            payload_key=(int(index_buffers[0].data.ptr), int(index_buffers[1].data.ptr), int(data_ptr)),
        )
        block.graph_desc = block._create_desc(cslib=cslib, cuda_dtype_id=cuda_dtype_id)
        block.dynamic_desc = block._create_desc(cslib=cslib, cuda_dtype_id=cuda_dtype_id)
        return block

    @classmethod
    def from_buffers(
        cls,
        *,
        fmt: str,
        nrows: int,
        ncols: int,
        nnz: int,
        index_buffers: tuple[CupyArray, CupyArray],
        data_ptr: int,
        payload_key: tuple[int, int, int],
        cslib: CuSparseLib,
        cuda_dtype_id: int,
    ) -> _CuBlock:
        block = cls(
            fmt=fmt,
            nrows=int(nrows),
            ncols=int(ncols),
            nnz=int(nnz),
            index_buffers=index_buffers,
            data_ptr=int(data_ptr),
            graph_desc=c_void_p(),
            dynamic_desc=c_void_p(),
            payload_key=payload_key,
        )
        block.graph_desc = block._create_desc(cslib=cslib, cuda_dtype_id=cuda_dtype_id)
        block.dynamic_desc = block._create_desc(cslib=cslib, cuda_dtype_id=cuda_dtype_id)
        return block

    def _create_desc(self, *, cslib: CuSparseLib, cuda_dtype_id: int) -> c_void_p:
        b0, b1 = self.index_buffers
        if self.fmt == "csr":
            return cslib.create_csr(
                self.nrows,
                self.ncols,
                self.nnz,
                b0.data.ptr,
                b1.data.ptr,
                self.data_ptr,
                cuda_dtype_id,
            )
        if self.fmt == "csc":
            return cslib.create_csc(
                self.nrows,
                self.ncols,
                self.nnz,
                b0.data.ptr,
                b1.data.ptr,
                self.data_ptr,
                cuda_dtype_id,
            )
        if self.fmt == "coo":
            return cslib.create_coo(
                self.nrows,
                self.ncols,
                self.nnz,
                b0.data.ptr,
                b1.data.ptr,
                self.data_ptr,
                cuda_dtype_id,
            )
        raise ValueError(f"Unknown sparse format: {self.fmt!r}")

    def nbytes(self) -> int:
        return int(sum(int(buf.nbytes) for buf in self.index_buffers))

    def estimate_nbytes(self, *, index_itemsize: int) -> int:
        if self.fmt == "csr":
            return int(self.nnz * index_itemsize + (self.nrows + 1) * index_itemsize)
        if self.fmt == "csc":
            return int(self.nnz * index_itemsize + (self.ncols + 1) * index_itemsize)
        if self.fmt == "coo":
            return int(self.nnz * 2 * index_itemsize)
        raise ValueError(f"Unknown sparse format: {self.fmt!r}")

    def destroy(self, *, cslib: CuSparseLib) -> None:
        for desc in (self.graph_desc, self.dynamic_desc):
            cslib.destroy_sp_mat(desc)


@dataclass(frozen=True)
class _CuOp:
    """One logical block application inside a level wavefront."""

    src_level: int
    block: _CuBlock
    nnz: int


@dataclass
class _SharedOnes:
    ptr: int
    logical_nbytes: int
    physical_nbytes: int
    vmm: bool
    _materialized: CupyArray | None = None
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
class _ScratchLevelPlan:
    enabled: bool
    reduce_order: tuple[int, ...]


@dataclass
class _SelectorLevels:
    """Selector rows/cols split by level for gather and scatter-add."""

    rows_by_level: list[CupyArray]
    cols_by_level: list[CupyArray]
    row_unique: bool

    @classmethod
    def from_csr(
        cls,
        *,
        cp: Any,
        selector: sp.csr_matrix,
        level_offsets: np.ndarray,
        H: int,
    ) -> _SelectorLevels:
        rows_by_level: list[CupyArray] = []
        cols_by_level: list[CupyArray] = []
        for h in range(H):
            lo = int(level_offsets[h])
            hi = int(level_offsets[h + 1])
            block = selector[:, lo:hi].tocoo()
            rows_by_level.append(cp.asarray(block.row.astype(np.int32, copy=False)))
            cols_by_level.append(cp.asarray(block.col.astype(np.int32, copy=False)))
        return cls(
            rows_by_level=rows_by_level,
            cols_by_level=cols_by_level,
            row_unique=selector_rows_unique_from_csr_indptr(selector.indptr),
        )

    def nnz(self) -> int:
        return int(sum(int(rows.size) for rows in self.rows_by_level))

    def nbytes(self) -> int:
        total = 0
        for rows, cols in zip(self.rows_by_level, self.cols_by_level, strict=True):
            total += int(rows.nbytes)
            total += int(cols.nbytes)
        return int(total)

    def scatter_add(
        self,
        *,
        cp: Any,
        stream: Any,
        level_buffers: list[CupyArray],
        x_gpu: CupyArray,
    ) -> None:
        with stream:
            for h, rows in enumerate(self.rows_by_level):
                if rows.size == 0:
                    continue
                cp.add.at(level_buffers[h], self.cols_by_level[h], x_gpu[rows])

    def gather(
        self,
        *,
        cp: Any,
        stream: Any,
        level_buffers: list[CupyArray],
        out_gpu: CupyArray,
    ) -> None:
        with stream:
            out_gpu.fill(0)
            for h, rows in enumerate(self.rows_by_level):
                if rows.size == 0:
                    continue
                values = level_buffers[h][self.cols_by_level[h]]
                if self.row_unique:
                    out_gpu[rows] = values
                else:
                    cp.add.at(out_gpu, rows, values)


@dataclass
class _SampleLevels:
    """Sample ids and local row ids split by compiled level."""

    sample_ids_by_level: list[CupyArray]
    row_ids_by_level: list[CupyArray]

    @classmethod
    def from_sample_rows(
        cls,
        *,
        cp: Any,
        sample_rows: np.ndarray,
        level_offsets: np.ndarray,
        H: int,
    ) -> _SampleLevels:
        sample_rows_arr = np.asarray(sample_rows, dtype=np.int64)
        sample_ids_by_level: list[CupyArray] = []
        row_ids_by_level: list[CupyArray] = []

        for h in range(H):
            lo = int(level_offsets[h])
            hi = int(level_offsets[h + 1])
            mask = (sample_rows_arr >= lo) & (sample_rows_arr < hi)
            sample_ids = np.flatnonzero(mask).astype(np.int32, copy=False)
            row_ids = (sample_rows_arr[sample_ids] - lo).astype(np.int32, copy=False)
            sample_ids_by_level.append(cp.asarray(sample_ids))
            row_ids_by_level.append(cp.asarray(row_ids))

        return cls(sample_ids_by_level=sample_ids_by_level, row_ids_by_level=row_ids_by_level)

    def scatter(
        self,
        *,
        cp: Any,
        stream: Any,
        x_gpu: CupyArray,
        level_buffers: list[CupyArray],
    ) -> None:
        with stream:
            for h, sample_ids in enumerate(self.sample_ids_by_level):
                if sample_ids.size == 0:
                    continue
                level_buffers[h][self.row_ids_by_level[h]] = x_gpu[sample_ids]

    def gather(
        self,
        *,
        cp: Any,
        stream: Any,
        level_buffers: list[CupyArray],
        out_gpu: CupyArray,
    ) -> None:
        with stream:
            out_gpu.fill(0)
            for h, sample_ids in enumerate(self.sample_ids_by_level):
                if sample_ids.size == 0:
                    continue
                out_gpu[sample_ids] = level_buffers[h][self.row_ids_by_level[h]]


@dataclass
class _DenseState:
    """Per-direction dense state, destination views, and source views."""

    level_bufs: list[CupyArray] = alloc_field(label="dense_state", kind="state")
    dst_descs: list[c_void_p] = ignore_field()
    src_descs: list[c_void_p] = ignore_field()
    src_bufs: list[CupyArray] | None = alloc_field(label="source_state", kind="state", default=None)


@dataclass
class _DirectionWorkspace:
    """Reusable device state for one direction and one runtime ``k``."""

    direction: Direction = ignore_field()
    k: int = ignore_field()
    use_graph_descs: bool = ignore_field()
    dense: _DenseState = child_field()
    fork_event: CupyEvent = ignore_field()
    ready_events: list[CupyEvent] = ignore_field()
    spmm_ext_by_level: list[list[CupyArray | None]] = alloc_field(label="spmm_ext", kind="auxiliary")
    scratch_views_by_level: list[list[CupyArray]] = alloc_field(label="scratch_views", kind="scratch")
    scratch_dst_descs_by_level: list[list[c_void_p]] = ignore_field()
    scratch_done_events_by_level: list[list[CupyEvent]] = ignore_field()
    input_primary: CupyArray = alloc_field(label="input_primary", kind="input")
    graph: CupyGraph | None = ignore_field(default=None)

    def destroy(self, *, cslib: CuSparseLib) -> None:
        seen: set[int] = set()
        for desc in [*self.dense.dst_descs, *self.dense.src_descs]:
            key = id(desc)
            if key in seen:
                continue
            seen.add(key)
            cslib.destroy_dn_mat(desc)
        for row in self.scratch_dst_descs_by_level:
            for desc in row:
                key = id(desc)
                if key in seen:
                    continue
                seen.add(key)
                cslib.destroy_dn_mat(desc)


@dataclass
class _WorkspaceCache:
    dynamic_up: _DirectionWorkspace | None = child_field(retention="on_demand", activity="no", direction="up", slot_k_from_attr="k", default=None)
    graph_up: _DirectionWorkspace | None = child_field(retention="captured", activity="no", direction="up", slot_k_from_attr="k", default=None)
    dynamic_down: _DirectionWorkspace | None = child_field(retention="on_demand", activity="no", direction="down", slot_k_from_attr="k", default=None)
    graph_down: _DirectionWorkspace | None = child_field(retention="captured", activity="no", direction="down", slot_k_from_attr="k", default=None)


@dataclass
class _DirectionStaging:
    k: int = ignore_field()
    input_miss: CupyArray | None = alloc_field(label="input_miss", kind="input", default=None)
    output_main: CupyArray | None = alloc_field(label="output_main", kind="output", default=None)
    output_miss: CupyArray | None = alloc_field(label="output_miss", kind="output", default=None)
    init_vector: CupyArray | None = alloc_field(label="init_vector", kind="init", default=None)
    init_matrix: CupyArray | None = alloc_field(label="init_matrix", kind="init", default=None)
    xtx_bias: CupyArray | None = alloc_field(label="xtx_bias", kind="init", default=None)


@dataclass
class CusparseCall:
    node_values_host: np.ndarray | None = alloc_field(
        label="node_values_host", kind="state", owner="backend", retention="call", activity="yes", default=None
    )
    miss_output_host: np.ndarray | None = alloc_field(
        label="miss_output_host", kind="output", owner="backend", retention="call", activity="yes", default=None
    )


@dataclass
class CusparseRetained:
    alpha: CupyArray | None = alloc_field(
        label="alpha", kind="auxiliary", owner="backend", retention="persistent", activity="always", default=None
    )
    beta_zero: CupyArray | None = alloc_field(
        label="beta_zero", kind="auxiliary", owner="backend", retention="persistent", activity="always", default=None
    )
    beta_one: CupyArray | None = alloc_field(
        label="beta_one", kind="auxiliary", owner="backend", retention="persistent", activity="always", default=None
    )
    shared_values_materialized: CupyArray | None = alloc_field(
        label="shared_values", kind="sparse", owner="backend", retention="persistent", activity="always", default=None
    )
    shared_values_vmm: VmmAliasedAlloc | None = alloc_field(
        label="shared_values", kind="sparse", owner="backend", retention="persistent", activity="always", default=None
    )
    blocks_up: list[tuple[CupyArray, CupyArray]] = alloc_field(
        label="blocks_up", kind="sparse", owner="backend", retention="persistent", activity="always", default_factory=list
    )
    blocks_down: list[tuple[CupyArray, CupyArray]] = alloc_field(
        label="blocks_down", kind="sparse", owner="backend", retention="persistent", activity="always", default_factory=list
    )
    mut_selector_rows: list[CupyArray] | None = alloc_field(
        label="selector_mut", kind="selector", owner="backend", retention="persistent", activity="always", default=None
    )
    mut_selector_cols: list[CupyArray] | None = alloc_field(
        label="selector_mut", kind="selector", owner="backend", retention="persistent", activity="always", default=None
    )
    miss_selector_rows: list[CupyArray] | None = alloc_field(
        label="selector_miss", kind="selector", owner="backend", retention="persistent", activity="always", default=None
    )
    miss_selector_cols: list[CupyArray] | None = alloc_field(
        label="selector_miss", kind="selector", owner="backend", retention="persistent", activity="always", default=None
    )
    sample_level_ids: list[CupyArray] | None = alloc_field(
        label="sample_level_ids", kind="mapping", owner="backend", retention="persistent", activity="always", default=None
    )
    sample_level_rows: list[CupyArray] | None = alloc_field(
        label="sample_level_rows", kind="mapping", owner="backend", retention="persistent", activity="always", default=None
    )
    workspaces: _WorkspaceCache = child_field(owner="backend", default_factory=_WorkspaceCache)
    staging_up_by_k: dict[int, _DirectionStaging] = child_field(
        owner="backend", retention="staging", activity="no", direction="up", slot_k_from_dict_key=True, default_factory=dict
    )
    staging_down_by_k: dict[int, _DirectionStaging] = child_field(
        owner="backend", retention="staging", activity="no", direction="down", slot_k_from_dict_key=True, default_factory=dict
    )


def _dense_order_char(order: DenseOrder) -> str:
    return "C" if order == DenseOrder.ROW else "F"


def _dense_ld(*, rows: int, cols: int, order: DenseOrder) -> int:
    return int(cols if order == DenseOrder.ROW else rows)


def _create_dense_desc(
    *,
    cslib: CuSparseLib,
    buf: CupyArray,
    rows: int,
    cols: int,
    order: DenseOrder,
    cuda_dtype_id: int,
) -> c_void_p:
    return cslib.create_dnmat(rows, cols, _dense_ld(rows=rows, cols=cols, order=order), buf.data.ptr, cuda_dtype_id, int(order))


def _build_dense_state(
    *,
    cp: Any,
    cslib: CuSparseLib,
    plan: CusparsePlan | None,
    level_sizes: list[int],
    k: int,
    dtype: np.dtype,
    cuda_dtype_id: int,
) -> _DenseState:
    if plan is None:
        return _DenseState(level_bufs=[], dst_descs=[], src_descs=[], src_bufs=None)

    level_bufs = [cp.zeros((nrows, k), dtype=dtype, order=_dense_order_char(plan.order_c)) for nrows in level_sizes]

    dst_descs = [
        _create_dense_desc(cslib=cslib, buf=buf, rows=buf.shape[0], cols=buf.shape[1], order=plan.order_c, cuda_dtype_id=cuda_dtype_id)
        for buf in level_bufs
    ]

    if plan.op_b == Operation.N and plan.order_b == plan.order_c:
        return _DenseState(level_bufs=level_bufs, dst_descs=dst_descs, src_descs=dst_descs, src_bufs=None)

    if plan.op_b == Operation.T and plan.order_b != plan.order_c:
        src_descs = [
            _create_dense_desc(cslib=cslib, buf=buf, rows=k, cols=buf.shape[0], order=plan.order_b, cuda_dtype_id=cuda_dtype_id)
            for buf in level_bufs
        ]
        return _DenseState(level_bufs=level_bufs, dst_descs=dst_descs, src_descs=src_descs, src_bufs=None)

    src_bufs: list[CupyArray] = []
    src_descs: list[c_void_p] = []
    for nrows in level_sizes:
        src_rows = nrows if plan.op_b == Operation.N else k
        src_cols = k if plan.op_b == Operation.N else nrows
        src = cp.zeros((src_rows, src_cols), dtype=dtype, order=_dense_order_char(plan.order_b))
        src_bufs.append(src)
        src_descs.append(
            _create_dense_desc(cslib=cslib, buf=src, rows=src_rows, cols=src_cols, order=plan.order_b, cuda_dtype_id=cuda_dtype_id)
        )
    return _DenseState(level_bufs=level_bufs, dst_descs=dst_descs, src_descs=src_descs, src_bufs=src_bufs)


def _dense_state_nbytes(dense: _DenseState) -> int:
    total = int(sum(int(buf.nbytes) for buf in dense.level_bufs))
    if dense.src_bufs is not None:
        total += int(sum(int(buf.nbytes) for buf in dense.src_bufs))
    return total


def _iter_unique_blocks(*grids: list[list[_CuBlock | None]]) -> Any:
    seen: set[tuple[int, int, int, str, int, int]] = set()
    for grid in grids:
        for row in grid:
            for block in row:
                if block is None:
                    continue
                key = (*block.payload_key, block.fmt, block.nrows, block.ncols)
                if key in seen:
                    continue
                seen.add(key)
                yield block


def _estimate_block_grid_bytes(
    grid: list[list[_CuBlock | None]],
    *,
    index_itemsize: int,
) -> int:
    return int(
        sum(
            block.estimate_nbytes(index_itemsize=index_itemsize)
            for block in _iter_unique_blocks(grid)
        )
    )


def _publish_level_source_view(*, cp: Any, dense: _DenseState, plan: CusparsePlan, level: int) -> None:
    if dense.src_bufs is None:
        return
    src = dense.src_bufs[level]
    state = dense.level_bufs[level]
    if plan.op_b == Operation.N:
        cp.copyto(src, state)
    else:
        cp.copyto(src, state.T)


def _destroy_block_grid(grid: list[list[_CuBlock | None]], *, cslib: CuSparseLib) -> None:
    for block in _iter_unique_blocks(grid):
        block.destroy(cslib=cslib)


class CusparseBackend(BackendBase):
    """GPU backend using cuSPARSE for block-wise level traversal."""

    _SETUP_MEMORY_POLICY = {
        "_A_blocks": "dropped",
        "_sel_mut": "dropped",
        "_sel_miss": "dropped",
        "_level_offsets": "borrowed",
        "_sample_rows": "borrowed",
        "_coalescence_counts": "borrowed",
        "_xtx_host": "dropped",
    }

    def __init__(
        self,
        *,
        pair: CusparsePlanPair,
        log_level: str = "WARNING",
        instrumentation: bool = False,
    ):
        for name, plan in (("UP", pair.plan_up), ("DOWN", pair.plan_down)):
            if plan is not None and not plan.supported:
                raise ValueError(f"Unsupported cuSPARSE {name} plan: {plan}")

        try:
            import cupy as cp

            self._cp = cp
        except ImportError as exc:
            raise ImportError("CuPy required: pip install cupy-cuda12x") from exc

        super().__init__(
            plan_up=pair.plan_up,
            plan_down=pair.plan_down,
            log_level=log_level,
            instrumentation=instrumentation,
        )

        self._cslib = CuSparseLib()
        runtime_version = cusparse_plan._runtime_cuda_version()
        runtime_token = ".".join(str(part) for part in runtime_version)
        self._capture_stream: CupyStream = self._cp.cuda.Stream(non_blocking=True)
        self._level_streams: list[CupyStream] = []
        self._scratch_streams_up_by_level: list[list[CupyStream]] = []
        self._scratch_streams_down_by_level: list[list[CupyStream]] = []

        self._dtype = np.float64
        self._cuda_dtype: int | None = None
        self._alpha: CupyArray | None = None
        self._beta_zero: CupyArray | None = None
        self._beta_one: CupyArray | None = None

        self._H = 0
        self._num_samples = 0
        self._num_mutations = 0
        self._num_nodes = 0
        self._level_offsets = np.empty(0, dtype=np.int64)

        self._shared_ones: _SharedOnes | None = None

        self._blocks_up: list[list[_CuBlock | None]] = []
        self._blocks_down: list[list[_CuBlock | None]] = []
        self._ops_up: list[list[_CuOp]] = []
        self._ops_down: list[list[_CuOp]] = []
        self._scratch_plan_up: list[_ScratchLevelPlan] = []
        self._scratch_plan_down: list[_ScratchLevelPlan] = []

        self._mut_selector: _SelectorLevels | None = None
        self._miss_selector: _SelectorLevels | None = None
        self._sample_levels: _SampleLevels | None = None
        self._static_workspace_slots: set[str] = set()
        self._workspaces = _WorkspaceCache()
        self._staging_up_by_k: dict[int, _DirectionStaging] = {}
        self._staging_down_by_k: dict[int, _DirectionStaging] = {}
        self._nvtx = make_cupy_tracer("grg.cusparse", self._cp) if self._instrumentation else None
        self._install_memory(
            retained=CusparseRetained(
                alpha=None,
                beta_zero=None,
                beta_one=None,
                shared_values_materialized=None,
                shared_values_vmm=None,
                blocks_up=[],
                blocks_down=[],
                mut_selector_rows=None,
                mut_selector_cols=None,
                miss_selector_rows=None,
                miss_selector_cols=None,
                sample_level_ids=None,
                sample_level_rows=None,
                workspaces=self._workspaces,
                staging_up_by_k=self._staging_up_by_k,
                staging_down_by_k=self._staging_down_by_k,
            ),
            call_type=CusparseCall,
        )

        self._logger.info(
            "cuSPARSE runtime levels: CUDA runtime=%s cuSPARSE=%s",
            runtime_token,
            self._cslib.version,
        )
        if runtime_version != cusparse_plan._DOC_CUDA_VERSION:
            self._logger.warning(
                "CUDA runtime %s detected; CusparsePlan semantics are grounded in the CUDA 12.9.0 cuSPARSE SpMM docs.",
                runtime_token,
            )

    def _grid_for(self, direction: Direction) -> list[list[_CuBlock | None]]:
        return self._blocks_up if direction == Direction.UP else self._blocks_down

    def _operator_matrix(self, direction: Direction, *, dst_level: int, src_level: int) -> sp.spmatrix:
        if direction == Direction.UP:
            return self._A_blocks[dst_level][src_level]
        return self._A_blocks[src_level][dst_level]

    def _destroy_shared_ones(self) -> None:
        values = self._shared_ones
        if values is None:
            return
        try:
            values.destroy()
        finally:
            self._shared_ones = None
            self._sync_retained_root()
            self._bump_retained_epoch()

    def _sync_retained_root(self) -> None:
        retained = self._retained_mem
        retained.alpha = self._alpha
        retained.beta_zero = self._beta_zero
        retained.beta_one = self._beta_one
        if self._shared_ones is None:
            retained.shared_values_materialized = None
            retained.shared_values_vmm = None
        elif self._shared_ones.vmm:
            retained.shared_values_materialized = None
            retained.shared_values_vmm = VmmAliasedAlloc(
                ptr=int(self._shared_ones.ptr),
                physical_nbytes=int(self._shared_ones.physical_nbytes),
                logical_nbytes=int(self._shared_ones.logical_nbytes),
            )
        else:
            retained.shared_values_materialized = self._shared_ones._materialized
            retained.shared_values_vmm = None
        retained.blocks_up = [
            block.index_buffers
            for row in self._blocks_up
            for block in row
            if block is not None
        ]
        retained.blocks_down = [
            block.index_buffers
            for row in self._blocks_down
            for block in row
            if block is not None
        ]
        retained.mut_selector_rows = None if self._mut_selector is None else self._mut_selector.rows_by_level
        retained.mut_selector_cols = None if self._mut_selector is None else self._mut_selector.cols_by_level
        retained.miss_selector_rows = None if self._miss_selector is None else self._miss_selector.rows_by_level
        retained.miss_selector_cols = None if self._miss_selector is None else self._miss_selector.cols_by_level
        retained.sample_level_ids = None if self._sample_levels is None else self._sample_levels.sample_ids_by_level
        retained.sample_level_rows = None if self._sample_levels is None else self._sample_levels.row_ids_by_level
        retained.workspaces = self._workspaces
        retained.staging_up_by_k = self._staging_up_by_k
        retained.staging_down_by_k = self._staging_down_by_k

    def _max_block_nnz(self) -> int:
        return max((int(block.nnz) for row in self._A_blocks for block in row), default=0)

    def _build_materialized_shared_ones(self, max_nnz: int) -> _SharedOnes:
        with self._capture_stream:
            arr = self._cp.ones((int(max_nnz),), dtype=self._dtype)
        self._capture_stream.synchronize()
        return _SharedOnes(
            ptr=int(arr.data.ptr),
            logical_nbytes=int(arr.nbytes),
            physical_nbytes=int(arr.nbytes),
            vmm=False,
            _materialized=arr,
        )

    def _build_vmm_shared_ones(
        self,
        *,
        driver: CudaVmmDriver,
        device_id: int,
        logical_nbytes: int,
        tile_nbytes: int,
    ) -> _SharedOnes:
        itemsize = int(self._dtype.itemsize)
        reserved_nbytes = _round_up(logical_nbytes, tile_nbytes)
        handle = 0
        vaddr = 0
        try:
            handle = int(driver.mem_create(device_id, tile_nbytes))
            vaddr = int(driver.address_reserve(reserved_nbytes, alignment_bytes=tile_nbytes))
            for offset in range(0, reserved_nbytes, tile_nbytes):
                driver.mem_map(vaddr + offset, tile_nbytes, handle)
            driver.mem_set_access(vaddr, reserved_nbytes, device_id)
            init_owner = object()
            init_mem = self._cp.cuda.UnownedMemory(vaddr, tile_nbytes, init_owner, device_id)
            init_ptr = self._cp.cuda.MemoryPointer(init_mem, 0)
            init_tile = self._cp.ndarray((tile_nbytes // itemsize,), dtype=self._dtype, memptr=init_ptr)
            with self._capture_stream:
                init_tile.fill(1)
            self._capture_stream.synchronize()
            return _SharedOnes(
                ptr=vaddr,
                logical_nbytes=logical_nbytes,
                physical_nbytes=tile_nbytes,
                vmm=True,
                _driver=driver,
                _vaddr=vaddr,
                _reserved_nbytes=reserved_nbytes,
                _handle=handle,
            )
        except Exception:
            if vaddr and reserved_nbytes:
                try:
                    driver.mem_unmap(vaddr, reserved_nbytes)
                except Exception:
                    pass
            if handle:
                try:
                    driver.mem_release(handle)
                except Exception:
                    pass
            if vaddr and reserved_nbytes:
                try:
                    driver.address_free(vaddr, reserved_nbytes)
                except Exception:
                    pass
            raise

    def _build_shared_ones(self) -> _SharedOnes | None:
        max_nnz = self._max_block_nnz()
        if max_nnz <= 0:
            self._logger.info("cuSPARSE shared ones: mode=disabled reason=no-nonzero-blocks")
            return None

        itemsize = int(self._dtype.itemsize)
        logical_nbytes = int(max_nnz) * itemsize
        try:
            driver = CudaVmmDriver()
            device_id = int(self._cp.cuda.Device().id)
            current_context = driver.current_context()
            if current_context is None:
                values = self._build_materialized_shared_ones(max_nnz)
                self._logger.info(
                    "cuSPARSE shared ones: mode=materialized reason=no-current-cuda-context max_nnz=%d logical_bytes=%d physical_bytes=%d",
                    max_nnz,
                    logical_nbytes,
                    values.physical_nbytes,
                )
                return values
            if not driver.vmm_supported(device_id):
                values = self._build_materialized_shared_ones(max_nnz)
                self._logger.info(
                    "cuSPARSE shared ones: mode=materialized reason=vmm-unsupported max_nnz=%d logical_bytes=%d physical_bytes=%d",
                    max_nnz,
                    logical_nbytes,
                    values.physical_nbytes,
                )
                return values
            granularity_min = int(driver.allocation_granularity(device_id, recommended=False))
        except Exception as exc:
            warnings.warn(
                f"cuSPARSE shared ones falling back to one materialized all-ones array: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            values = self._build_materialized_shared_ones(max_nnz)
            self._logger.info(
                "cuSPARSE shared ones: mode=materialized reason=vmm-query-failed max_nnz=%d logical_bytes=%d physical_bytes=%d",
                max_nnz,
                logical_nbytes,
                values.physical_nbytes,
            )
            return values

        try:
            granularity_rec = int(driver.allocation_granularity(device_id, recommended=True))
        except Exception:
            granularity_rec = None

        tile_nbytes = granularity_min
        if tile_nbytes <= 0 or tile_nbytes % itemsize != 0:
            warnings.warn(
                (
                    "cuSPARSE shared ones falling back to one materialized all-ones array: "
                    f"minimum granularity {tile_nbytes} is not aligned to dtype itemsize {itemsize}"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            values = self._build_materialized_shared_ones(max_nnz)
            self._logger.info(
                "cuSPARSE shared ones: mode=materialized reason=invalid-vmm-granularity max_nnz=%d logical_bytes=%d physical_bytes=%d tile_bytes=%d granularity_min=%d granularity_rec=%s",
                max_nnz,
                logical_nbytes,
                values.physical_nbytes,
                tile_nbytes,
                granularity_min,
                granularity_rec,
            )
            return values

        if tile_nbytes >= logical_nbytes:
            values = self._build_materialized_shared_ones(max_nnz)
            self._logger.info(
                "cuSPARSE shared ones: mode=materialized reason=no-memory-savings max_nnz=%d logical_bytes=%d physical_bytes=%d tile_bytes=%d granularity_min=%d granularity_rec=%s",
                max_nnz,
                logical_nbytes,
                values.physical_nbytes,
                tile_nbytes,
                granularity_min,
                granularity_rec,
            )
            return values

        try:
            values = self._build_vmm_shared_ones(
                driver=driver,
                device_id=device_id,
                logical_nbytes=logical_nbytes,
                tile_nbytes=tile_nbytes,
            )
            self._logger.info(
                "cuSPARSE shared ones: mode=vmm reason=physical-bytes-reduced max_nnz=%d logical_bytes=%d physical_bytes=%d tile_bytes=%d granularity_min=%d granularity_rec=%s reserved_bytes=%d reserve_alignment=%d",
                max_nnz,
                values.logical_nbytes,
                values.physical_nbytes,
                tile_nbytes,
                granularity_min,
                granularity_rec,
                values._reserved_nbytes,
                tile_nbytes,
            )
            return values
        except Exception as exc:
            warnings.warn(
                f"cuSPARSE shared ones falling back to one materialized all-ones array: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            values = self._build_materialized_shared_ones(max_nnz)
            self._logger.info(
                "cuSPARSE shared ones: mode=materialized reason=vmm-build-failed max_nnz=%d logical_bytes=%d physical_bytes=%d tile_bytes=%d granularity_min=%d granularity_rec=%s",
                max_nnz,
                values.logical_nbytes,
                values.physical_nbytes,
                tile_nbytes,
                granularity_min,
                granularity_rec,
            )
            return values

    def setup(
        self,
        setup: BackendSetup,
    ) -> None:
        self._destroy_workspace_cache()
        self._clear_staging()
        _destroy_block_grid(self._blocks_up, cslib=self._cslib)
        _destroy_block_grid(self._blocks_down, cslib=self._cslib)
        self._destroy_shared_ones()

        self._apply_setup_state(setup)
        self._dtype = np.dtype(setup.dtype)
        self._cuda_dtype = cuda_dtype(self._dtype)
        self._H = len(self._level_offsets) - 1

        if self._cuda_dtype is None:
            raise RuntimeError("CUDA dtype not initialized")
        self._alpha = self._cp.ones(1, dtype=self._dtype)
        self._beta_zero = self._cp.zeros(1, dtype=self._dtype)
        self._beta_one = self._cp.ones(1, dtype=self._dtype)
        self._shared_ones = self._build_shared_ones()

        self._blocks_up = self._build_direction_blocks(direction=Direction.UP)
        self._blocks_down = self._build_direction_blocks(direction=Direction.DOWN)
        self._ops_up = self._build_direction_ops(Direction.UP)
        self._ops_down = self._build_direction_ops(Direction.DOWN)
        self._scratch_plan_up = self._build_scratch_level_plans(Direction.UP) if self._plan_up is not None else []
        self._scratch_plan_down = self._build_scratch_level_plans(Direction.DOWN) if self._plan_down is not None else []

        self._mut_selector = _SelectorLevels.from_csr(
            cp=self._cp,
            selector=self._sel_mut,
            level_offsets=self._level_offsets,
            H=self._H,
        )
        self._miss_selector = _SelectorLevels.from_csr(
            cp=self._cp,
            selector=self._sel_miss,
            level_offsets=self._level_offsets,
            H=self._H,
        )
        self._sample_levels = _SampleLevels.from_sample_rows(
            cp=self._cp,
            sample_rows=self._sample_rows,
            level_offsets=self._level_offsets,
            H=self._H,
        )
        self._level_streams = [self._cp.cuda.Stream(non_blocking=True) for _ in range(self._H)]
        self._scratch_streams_up_by_level = (
            self._build_scratch_streams(Direction.UP) if self._plan_up is not None else [[] for _ in range(self._H)]
        )
        self._scratch_streams_down_by_level = (
            self._build_scratch_streams(Direction.DOWN) if self._plan_down is not None else [[] for _ in range(self._H)]
        )

        for direction in self._configured_directions():
            self._log_block_memory(direction)
        self._xtx_host = None

        self._A_blocks = []
        self._sel_mut = sp.csr_matrix((0, 0))
        self._sel_miss = sp.csr_matrix((0, 0))

        self._logger.info(
            "CusparseBackend setup: H=%d K=%d n=%d m=%d plan_up=%s plan_down=%s",
            self._H,
            self._num_nodes,
            self._num_samples,
            self._num_mutations,
            "<unspecified>" if self._plan_up is None else str(self._plan_up),
            "<unspecified>" if self._plan_down is None else str(self._plan_down),
        )

        self._workspaces = _WorkspaceCache()
        self._static_workspace_slots = set()
        for direction in self._configured_directions():
            plan = self._require_plan(direction)
            hint = effective_k_hint(instrumentation=self._instrumentation, k_hint=plan.k_hint)
            if self._instrumentation and plan.k_hint is not None:
                warn_instrumentation_ignores_k_hint(backend="cuSPARSE", direction=direction, k_hint=int(plan.k_hint))
            if hint is None:
                continue
            slot = self._workspace_slot(direction, graph=True)
            ws = self._build_direction_workspace(direction, int(hint), use_graph_descs=True)
            ws.graph = self._capture_wavefront_graph(ws)
            setattr(self._workspaces, slot, ws)
            self._static_workspace_slots.add(slot)
        self._sync_retained_root()
        self._bump_retained_epoch()
        self._assert_setup_memory_contract()

    def _build_direction_blocks(self, *, direction: Direction) -> list[list[_CuBlock | None]]:
        if self._cuda_dtype is None:
            raise RuntimeError("CUDA dtype not initialized")
        if self._shared_ones is None and any(int(mat.nnz) > 0 for row in self._A_blocks for mat in row):
            raise RuntimeError("shared ones are not initialized")
        plan = self._plan_for(direction)
        if plan is None:
            return [[] for _ in range(self._H)]
        rows = [
            [None] * (dst_level if direction == Direction.UP else max(self._H - dst_level - 1, 0))
            for dst_level in range(self._H)
        ]
        store_actual = self._store_blocks_up if direction == Direction.UP else self._store_blocks_down
        if store_actual:
            for dst_level, src_level, row_index in iter_direction_level_pairs(direction, self._H):
                matrix = self._operator_matrix(direction, dst_level=dst_level, src_level=src_level)
                if matrix.nnz == 0:
                    continue
                stored = matrix if plan.store == plan.store.N else matrix.T.tocsr()
                rows[dst_level][row_index] = _CuBlock.from_scipy(
                    stored,
                    fmt=plan.fmt.value.lower(),
                    cp=self._cp,
                    data_ptr=int(self._shared_ones.ptr),
                    cslib=self._cslib,
                    cuda_dtype_id=self._cuda_dtype,
                )
            return rows

        owner_direction = (
            Direction.UP
            if (self._up_ops_owner if direction == Direction.UP else self._down_ops_owner) == "up"
            else Direction.DOWN
        )
        owner_grid = self._grid_for(owner_direction)
        for dst_level, src_level, row_index in iter_direction_level_pairs(direction, self._H):
            owner_dst = dst_level if owner_direction == direction else src_level
            owner_src = src_level if owner_direction == direction else dst_level
            owner_row_index = owner_src if owner_direction == Direction.UP else owner_src - owner_dst - 1
            owner_block = owner_grid[owner_dst][owner_row_index]
            if owner_block is None:
                continue
            matrix = self._operator_matrix(direction, dst_level=dst_level, src_level=src_level)
            if matrix.nnz == 0:
                continue
            nrows, ncols = matrix.shape
            if plan.store == plan.store.T:
                nrows, ncols = ncols, nrows
            rows[dst_level][row_index] = _CuBlock.from_buffers(
                fmt=plan.fmt.value.lower(),
                nrows=nrows,
                ncols=ncols,
                nnz=owner_block.nnz,
                index_buffers=owner_block.index_buffers,
                data_ptr=owner_block.data_ptr,
                payload_key=owner_block.payload_key,
                cslib=self._cslib,
                cuda_dtype_id=self._cuda_dtype,
            )
        return rows

    def _build_direction_ops(self, direction: Direction) -> list[list[_CuOp]]:
        ops: list[list[_CuOp]] = [[] for _ in range(self._H)]
        if self._plan_for(direction) is None:
            return ops
        grid = self._grid_for(direction)
        for dst, src, row_index in iter_direction_level_pairs(direction, self._H):
            block = grid[dst][row_index]
            if block is None:
                continue
            ops[dst].append(_CuOp(src_level=src, block=block, nnz=block.nnz))
        return ops

    def _log_block_memory(self, direction: Direction) -> None:
        if not self._logger.isEnabledFor(logging.DEBUG):
            return
        plan = self._plan_for(direction)
        if plan is None:
            return
        grid = self._grid_for(direction)
        store_actual = self._store_blocks_up if direction == Direction.UP else self._store_blocks_down
        bucket = "blocks_up" if direction == Direction.UP else "blocks_down"
        shared_logical = 0
        shared_physical = 0
        shared_bucket = "blocks_up" if self._plan_up is not None else "blocks_down"
        if store_actual and self._shared_ones is not None and bucket == shared_bucket:
            shared_logical = int(self._shared_ones.logical_nbytes)
            shared_physical = int(self._shared_ones.physical_nbytes)

        stored_blocks = 0
        alias_blocks = 0
        empty_blocks = 0
        total_rows = 0
        total_cols = 0
        total_nnz = 0
        indices_bytes = 0
        indptr_bytes = 0
        row_bytes = 0
        col_bytes = 0

        for dst_level, src_level, row_index in iter_direction_level_pairs(direction, self._H):
            block = grid[dst_level][row_index]
            if block is None:
                empty_blocks += 1
                continue
            alias = not store_actual
            total_rows += int(block.nrows)
            total_cols += int(block.ncols)
            total_nnz += int(block.nnz)

            block_indices = 0
            block_indptr = 0
            block_row = 0
            block_col = 0
            if not alias:
                if block.fmt == "csr":
                    block_indptr = int(block.index_buffers[0].nbytes)
                    block_indices = int(block.index_buffers[1].nbytes)
                elif block.fmt == "csc":
                    block_indptr = int(block.index_buffers[0].nbytes)
                    block_indices = int(block.index_buffers[1].nbytes)
                else:
                    block_row = int(block.index_buffers[0].nbytes)
                    block_col = int(block.index_buffers[1].nbytes)
                stored_blocks += 1
                indices_bytes += block_indices
                indptr_bytes += block_indptr
                row_bytes += block_row
                col_bytes += block_col
            else:
                alias_blocks += 1

            self._logger.debug(
                "cuSPARSE block dir=%s dst=%d src=%d fmt=%s rows=%d cols=%d nnz=%d alias=%s indices_bytes=%d indptr_bytes=%d row_bytes=%d col_bytes=%d",
                direction.value,
                dst_level,
                src_level,
                block.fmt,
                block.nrows,
                block.ncols,
                block.nnz,
                alias,
                block_indices,
                block_indptr,
                block_row,
                block_col,
            )

        self._logger.debug(
            "cuSPARSE blocks_%s rows=%d cols=%d nnz=%d stored_blocks=%d alias_blocks=%d empty_blocks=%d shared_data_physical=%d shared_data_logical=%d indices_bytes=%d indptr_bytes=%d row_bytes=%d col_bytes=%d",
            direction.value,
            total_rows,
            total_cols,
            total_nnz,
            stored_blocks,
            alias_blocks,
            empty_blocks,
            shared_physical,
            shared_logical,
            indices_bytes,
            indptr_bytes,
            row_bytes,
            col_bytes,
        )

    def _ops_for(self, direction: Direction) -> list[list[_CuOp]]:
        return self._ops_up if direction == Direction.UP else self._ops_down

    def _scratch_plans_for(self, direction: Direction) -> list[_ScratchLevelPlan]:
        return self._scratch_plan_up if direction == Direction.UP else self._scratch_plan_down

    def _scratch_streams_for(self, direction: Direction) -> list[list[CupyStream]]:
        return self._scratch_streams_up_by_level if direction == Direction.UP else self._scratch_streams_down_by_level

    def _await_direction_streams(self, direction: Direction) -> None:
        # Host-visible output collection must wait for all streams participating in a traversal,
        # not just the capture stream that launches the final gather/copy.
        self._capture_stream.synchronize()
        for stream in self._level_streams:
            stream.synchronize()
        for scratch_streams in self._scratch_streams_for(direction):
            for stream in scratch_streams:
                stream.synchronize()
        self._capture_stream.synchronize()

    def _resolve_scratch_levels(self, direction: Direction) -> frozenset[int]:
        plan = self._require_plan(direction)
        token = str(plan.scratch)
        if token == "none":
            return frozenset()
        if token == "all":
            return frozenset(range(self._H))
        levels = {int(piece) for piece in token.split("|")}
        invalid = sorted(level for level in levels if level < 0 or level >= self._H)
        if invalid:
            raise ValueError(
                f"cuSPARSE scratch levels out of range for {direction.value}: {invalid}; valid range is [0, {self._H})"
            )
        return frozenset(levels)

    def _build_scratch_level_plans(self, direction: Direction) -> list[_ScratchLevelPlan]:
        enabled_levels = self._resolve_scratch_levels(direction)
        plans: list[_ScratchLevelPlan] = []
        for dst_level, ops in enumerate(self._ops_for(direction)):
            if dst_level not in enabled_levels or not ops:
                plans.append(_ScratchLevelPlan(enabled=False, reduce_order=()))
                continue
            reduce_order = tuple(sorted(range(len(ops)), key=lambda idx: (int(ops[idx].nnz), int(ops[idx].src_level))))
            plans.append(_ScratchLevelPlan(enabled=True, reduce_order=reduce_order))
        return plans

    def _build_scratch_streams(self, direction: Direction) -> list[list[CupyStream]]:
        return [
            [self._cp.cuda.Stream(non_blocking=True) for _ in self._ops_for(direction)[dst_level]]
            if self._scratch_plans_for(direction)[dst_level].enabled
            else []
            for dst_level in range(self._H)
        ]

    def _destroy_workspace_cache(self) -> None:
        had_workspace = any(
            ws is not None
            for ws in (
                self._workspaces.graph_up,
                self._workspaces.dynamic_up,
                self._workspaces.graph_down,
                self._workspaces.dynamic_down,
            )
        )
        for ws in (
            self._workspaces.graph_up,
            self._workspaces.dynamic_up,
            self._workspaces.graph_down,
            self._workspaces.dynamic_down,
        ):
            if ws is None:
                continue
            try:
                ws.destroy(cslib=self._cslib)
            except Exception:
                pass
        self._workspaces = _WorkspaceCache()
        self._static_workspace_slots.clear()
        self._sync_retained_root()
        if had_workspace:
            self._bump_retained_epoch()

    def _clear_staging(self) -> None:
        had_staging = bool(self._staging_up_by_k or self._staging_down_by_k)
        self._staging_up_by_k.clear()
        self._staging_down_by_k.clear()
        self._sync_retained_root()
        if had_staging:
            self._bump_retained_epoch()

    def _staging_for(self, direction: Direction, k: int) -> _DirectionStaging:
        mapping = self._staging_up_by_k if direction == Direction.UP else self._staging_down_by_k
        staging = mapping.get(int(k))
        if staging is None:
            staging = _DirectionStaging(k=int(k))
            mapping[int(k)] = staging
            self._sync_retained_root()
            self._bump_retained_epoch()
        return staging

    def _ensure_staging_array(
        self,
        staging: _DirectionStaging,
        attr: str,
        shape: tuple[int, ...],
        *,
        order: str = "C",
    ):
        value = getattr(staging, attr)
        if value is None:
            value = self._cp.zeros(shape, dtype=self._dtype, order=order)
            setattr(staging, attr, value)
            self._bump_retained_epoch()
        return value

    def _workspace_slot(self, direction: Direction, *, graph: bool) -> str:
        return f"{'graph' if graph else 'dynamic'}_{direction.value}"

    def _build_direction_workspace(self, direction: Direction, k: int, *, use_graph_descs: bool) -> _DirectionWorkspace:
        if self._cuda_dtype is None or self._alpha is None or self._beta_zero is None or self._beta_one is None:
            raise RuntimeError("cuSPARSE runtime constants are uninitialized")
        if self._sample_levels is None:
            raise RuntimeError("Sample levels are not initialized")
        plan = self._require_plan(direction)
        level_sizes = [int(self._level_offsets[h + 1]) - int(self._level_offsets[h]) for h in range(self._H)]
        dense = _build_dense_state(
            cp=self._cp,
            cslib=self._cslib,
            plan=plan,
            level_sizes=level_sizes,
            k=k,
            dtype=self._dtype,
            cuda_dtype_id=self._cuda_dtype,
        )
        scratch_plans = self._scratch_plans_for(direction)
        ops_by_level = self._ops_for(direction)
        scratch_views_by_level: list[list[CupyArray]] = []
        scratch_dst_descs_by_level: list[list[c_void_p]] = []
        scratch_done_events_by_level: list[list[CupyEvent]] = []
        for dst_level in range(self._H):
            if not scratch_plans[dst_level].enabled:
                scratch_views_by_level.append([])
                scratch_dst_descs_by_level.append([])
                scratch_done_events_by_level.append([])
                continue
            views: list[CupyArray] = []
            descs: list[c_void_p] = []
            done_events: list[CupyEvent] = []
            rows = level_sizes[dst_level]
            for _ in ops_by_level[dst_level]:
                view = self._cp.zeros((rows, k), dtype=self._dtype, order=_dense_order_char(plan.order_c))
                views.append(view)
                descs.append(
                    _create_dense_desc(
                        cslib=self._cslib,
                        buf=view,
                        rows=view.shape[0],
                        cols=view.shape[1],
                        order=plan.order_c,
                        cuda_dtype_id=self._cuda_dtype,
                    )
                )
                done_events.append(self._cp.cuda.Event())
            scratch_views_by_level.append(views)
            scratch_dst_descs_by_level.append(descs)
            scratch_done_events_by_level.append(done_events)

        spmm_ext_by_level: list[list[CupyArray | None]] = []
        buffer_mismatches: list[str] = []
        preprocess_null_fallbacks = 0
        for dst_level, ops in enumerate(ops_by_level):
            row: list[CupyArray | None] = []
            scratch_enabled = scratch_plans[dst_level].enabled
            for op_idx, op in enumerate(ops):
                sp_desc = op.block.graph_desc if use_graph_descs else op.block.dynamic_desc
                dst_desc = (
                    scratch_dst_descs_by_level[dst_level][op_idx]
                    if scratch_enabled
                    else dense.dst_descs[dst_level]
                )
                beta = self._beta_zero if scratch_enabled else self._beta_one
                buffer_size = self._cslib.spmm_buffer_size(
                    int(plan.algo),
                    int(plan.op_a),
                    int(plan.op_b),
                    self._alpha.data.ptr,
                    sp_desc,
                    dense.src_descs[op.src_level],
                    beta.data.ptr,
                    dst_desc,
                    self._cuda_dtype,
                )
                needs_buffer = bool(buffer_size > 0)
                if needs_buffer != plan.need_buffer:
                    buffer_mismatches.append(
                        f"dst={dst_level} src={op.src_level} fmt={op.block.fmt} algo={plan.algo.value} size={buffer_size}"
                    )
                ext = (
                    self._cp.zeros((int(buffer_size),), dtype=self._cp.uint8)
                    if needs_buffer
                    else None
                )
                if plan.need_preprocess:
                    ext_ptr = 0 if ext is None else int(ext.data.ptr)
                    try:
                        self._cslib.spmm_preprocess(
                            int(plan.algo),
                            int(plan.op_a),
                            int(plan.op_b),
                            self._alpha.data.ptr,
                            sp_desc,
                            dense.src_descs[op.src_level],
                            beta.data.ptr,
                            dst_desc,
                            self._cuda_dtype,
                            ext_ptr,
                        )
                    except Exception:
                        if ext is not None:
                            raise
                        ext = self._cp.zeros((4,), dtype=self._cp.uint8)
                        preprocess_null_fallbacks += 1
                        self._cslib.spmm_preprocess(
                            int(plan.algo),
                            int(plan.op_a),
                            int(plan.op_b),
                            self._alpha.data.ptr,
                            sp_desc,
                            dense.src_descs[op.src_level],
                            beta.data.ptr,
                            dst_desc,
                            self._cuda_dtype,
                            int(ext.data.ptr),
                        )
                row.append(ext)
            spmm_ext_by_level.append(row)

        if buffer_mismatches:
            self._logger.warning(
                "cuSPARSE bufferSize disagrees with plan.need_buffer dir=%s k=%d graph_descs=%s mismatches=%d first=%s",
                direction.value,
                k,
                use_graph_descs,
                len(buffer_mismatches),
                buffer_mismatches[0],
            )
        if preprocess_null_fallbacks:
            self._logger.warning(
                "cuSPARSE preprocess rejected null ext buffer dir=%s k=%d graph_descs=%s fallbacks=%d",
                direction.value,
                k,
                use_graph_descs,
                preprocess_null_fallbacks,
            )

        input_len = self._num_samples if direction == Direction.UP else self._num_mutations
        ws = _DirectionWorkspace(
            direction=direction,
            k=int(k),
            use_graph_descs=bool(use_graph_descs),
            dense=dense,
            fork_event=self._cp.cuda.Event(),
            ready_events=[self._cp.cuda.Event() for _ in range(self._H)],
            spmm_ext_by_level=spmm_ext_by_level,
            scratch_views_by_level=scratch_views_by_level,
            scratch_dst_descs_by_level=scratch_dst_descs_by_level,
            scratch_done_events_by_level=scratch_done_events_by_level,
            input_primary=self._cp.zeros((input_len, k), dtype=self._dtype, order="C"),
        )
        self._logger.debug("workspace ready dir=%s k=%d graph_descs=%s", direction.value, k, use_graph_descs)
        return ws

    def _workspace_for(self, direction: Direction, k: int, *, graph: bool) -> _DirectionWorkspace:
        slot = self._workspace_slot(direction, graph=graph)
        ws = getattr(self._workspaces, slot)
        if graph:
            if ws is None or int(ws.k) != int(k):
                raise RuntimeError(f"Missing prebuilt cuSPARSE {slot} workspace for k={k}")
            return ws
        if ws is None or int(ws.k) != int(k):
            if ws is not None:
                self._static_workspace_slots.discard(slot)
                ws.destroy(cslib=self._cslib)
            ws = self._build_direction_workspace(direction, int(k), use_graph_descs=False)
            setattr(self._workspaces, slot, ws)
            self._sync_retained_root()
            self._bump_retained_epoch()
        return ws

    def _stage_inputs(
        self,
        ws: _DirectionWorkspace,
        staging: _DirectionStaging,
        x: np.ndarray,
        *,
        miss_arr: np.ndarray | None,
        init_mode: InitMode,
        init_payload: np.ndarray | None,
    ) -> None:
        ws.input_primary.set(x, stream=self._capture_stream)
        if miss_arr is not None:
            if staging.input_miss is None:
                staging.input_miss = self._ensure_staging_array(
                    staging,
                    "input_miss",
                    (self._num_mutations, ws.k),
                    order="C",
                )
            staging.input_miss.set(miss_arr, stream=self._capture_stream)

        match init_mode:
            case InitMode.NONE | InitMode.XTX:
                return
            case InitMode.VECTOR:
                if init_payload is None:
                    raise ValueError("init vector payload is required for init_mode=vector")
                if staging.init_vector is None:
                    staging.init_vector = self._ensure_staging_array(staging, "init_vector", (1, ws.k), order="C")
                staging.init_vector[0].set(init_payload, stream=self._capture_stream)
            case InitMode.MATRIX:
                if init_payload is None:
                    raise ValueError("init matrix payload is required for init_mode=matrix")
                if staging.init_matrix is None:
                    staging.init_matrix = self._ensure_staging_array(
                        staging,
                        "init_matrix",
                        (self._num_nodes, ws.k),
                        order="C",
                    )
                staging.init_matrix.set(init_payload, stream=self._capture_stream)
            case _:
                raise ValueError(f"Unknown init mode: {init_mode!r}")

    def _seed_workspace(
        self,
        ws: _DirectionWorkspace,
        staging: _DirectionStaging,
        *,
        init_mode: InitMode,
        has_miss_input: bool,
    ) -> None:
        if self._mut_selector is None or self._miss_selector is None or self._sample_levels is None:
            raise RuntimeError("cuSPARSE backend is not initialized")

        with self._capture_stream:
            for buf in ws.dense.level_bufs:
                buf.fill(0)

        if ws.direction == Direction.UP:
            self._sample_levels.scatter(
                cp=self._cp,
                stream=self._capture_stream,
                x_gpu=ws.input_primary,
                level_buffers=ws.dense.level_bufs,
            )
        else:
            self._mut_selector.scatter_add(
                cp=self._cp,
                stream=self._capture_stream,
                level_buffers=ws.dense.level_bufs,
                x_gpu=ws.input_primary,
            )
            if has_miss_input:
                if staging.input_miss is None:
                    raise RuntimeError("Missing DOWN miss buffer while has_miss_input=True")
                self._miss_selector.scatter_add(
                    cp=self._cp,
                    stream=self._capture_stream,
                    level_buffers=ws.dense.level_bufs,
                    x_gpu=staging.input_miss,
                )

        if init_mode == InitMode.NONE:
            return

        with self._capture_stream:
            match init_mode:
                case InitMode.XTX:
                    if staging.xtx_bias is None:
                        if self._coalescence_counts is None:
                            raise ValueError("init_mode=xtx requires GRG coalescence counts")
                        staging.xtx_bias = self._cp.asarray(
                            2.0 * self._coalescence_counts.astype(self._dtype, copy=False),
                            dtype=self._dtype,
                        ).reshape(self._num_nodes)
                        self._bump_retained_epoch()
                    for h in range(self._H):
                        lo = int(self._level_offsets[h])
                        hi = int(self._level_offsets[h + 1])
                        ws.dense.level_bufs[h] += staging.xtx_bias[lo:hi, None]
                case InitMode.VECTOR:
                    if staging.init_vector is None:
                        raise RuntimeError("Missing init vector buffer while init_mode=vector")
                    for buf in ws.dense.level_bufs:
                        buf += staging.init_vector
                case InitMode.MATRIX:
                    if staging.init_matrix is None:
                        raise RuntimeError("Missing init matrix buffer while init_mode=matrix")
                    for h in range(self._H):
                        lo = int(self._level_offsets[h])
                        hi = int(self._level_offsets[h + 1])
                        ws.dense.level_bufs[h] += staging.init_matrix[lo:hi]
                case _:
                    raise ValueError(f"Unknown init mode: {init_mode!r}")

    def _collect_outputs(
        self,
        ws: _DirectionWorkspace,
        staging: _DirectionStaging,
        *,
        need_miss_output: bool,
    ) -> tuple[np.ndarray, np.ndarray | None] | np.ndarray:
        if self._mut_selector is None or self._miss_selector is None or self._sample_levels is None:
            raise RuntimeError("cuSPARSE backend is not initialized")

        if ws.direction == Direction.UP:
            if staging.output_main is None:
                staging.output_main = self._ensure_staging_array(
                    staging,
                    "output_main",
                    (self._num_mutations, ws.k),
                    order="C",
                )
            self._mut_selector.gather(
                cp=self._cp,
                stream=self._capture_stream,
                level_buffers=ws.dense.level_bufs,
                out_gpu=staging.output_main,
            )
            if need_miss_output:
                if staging.output_miss is None:
                    staging.output_miss = self._ensure_staging_array(
                        staging,
                        "output_miss",
                        (self._num_mutations, ws.k),
                        order="C",
                    )
                self._miss_selector.gather(
                    cp=self._cp,
                    stream=self._capture_stream,
                    level_buffers=ws.dense.level_bufs,
                    out_gpu=staging.output_miss,
                )
            tracer = self._nvtx
            if tracer is not None:
                with tracer.range("await_outputs", dir=ws.direction.value):
                    self._await_direction_streams(ws.direction)
            else:
                self._await_direction_streams(ws.direction)
            out_main = staging.output_main.get()
            out_aux = staging.output_miss.get() if need_miss_output and staging.output_miss is not None else None
            return out_main, out_aux

        if staging.output_main is None:
            staging.output_main = self._ensure_staging_array(
                staging,
                "output_main",
                (self._num_samples, ws.k),
                order="C",
            )
        self._sample_levels.gather(
            cp=self._cp,
            stream=self._capture_stream,
            level_buffers=ws.dense.level_bufs,
            out_gpu=staging.output_main,
        )
        tracer = self._nvtx
        if tracer is not None:
            with tracer.range("await_outputs", dir=ws.direction.value):
                self._await_direction_streams(ws.direction)
        else:
            self._await_direction_streams(ws.direction)
        return staging.output_main.get()

    def _copy_node_outputs_to_host(self, ws: _DirectionWorkspace) -> np.ndarray:
        tracer = self._nvtx
        if tracer is not None:
            with tracer.range("await_outputs", dir=ws.direction.value):
                self._await_direction_streams(ws.direction)
        else:
            self._await_direction_streams(ws.direction)
        out = np.empty((self._num_nodes, ws.k), dtype=self._dtype)
        offset = 0
        for buf in ws.dense.level_bufs:
            rows = int(buf.shape[0])
            out[offset : offset + rows] = buf.get()
            offset += rows
        return out

    def _enqueue_wavefront(self, ws: _DirectionWorkspace) -> None:
        if self._alpha is None or self._beta_zero is None or self._beta_one is None or self._cuda_dtype is None:
            raise RuntimeError("cuSPARSE runtime constants are uninitialized")

        plan = self._require_plan(ws.direction)
        ops_by_level = self._ops_for(ws.direction)
        scratch_plans = self._scratch_plans_for(ws.direction)
        scratch_streams_by_level = self._scratch_streams_for(ws.direction)
        if ws.direction == Direction.UP:
            seed_level = 0
            level_iter = range(1, self._H)
        else:
            seed_level = self._H - 1
            level_iter = range(self._H - 2, -1, -1)

        with self._capture_stream:
            ws.fork_event.record(self._capture_stream)
        for stream in self._level_streams:
            stream.wait_event(ws.fork_event)
        for scratch_streams in scratch_streams_by_level:
            for stream in scratch_streams:
                stream.wait_event(ws.fork_event)

        if self._H > 0:
            seed_stream = self._level_streams[seed_level]
            with seed_stream:
                _publish_level_source_view(cp=self._cp, dense=ws.dense, plan=plan, level=seed_level)
                ws.ready_events[seed_level].record(seed_stream)

        for dst_level in level_iter:
            stream = self._level_streams[dst_level]
            ops = ops_by_level[dst_level]
            scratch_plan = scratch_plans[dst_level]
            if scratch_plan.enabled:
                scratch_views = ws.scratch_views_by_level[dst_level]
                scratch_done_events = ws.scratch_done_events_by_level[dst_level]
                scratch_streams = scratch_streams_by_level[dst_level]
                scratch_dst_descs = ws.scratch_dst_descs_by_level[dst_level]
                for op_idx, op in enumerate(ops):
                    helper_stream = scratch_streams[op_idx]
                    with helper_stream:
                        helper_stream.wait_event(ws.ready_events[op.src_level])
                        sp_desc = op.block.graph_desc if ws.use_graph_descs else op.block.dynamic_desc
                        ext = ws.spmm_ext_by_level[dst_level][op_idx]
                        self._cslib.set_stream(helper_stream.ptr)
                        self._cslib.spmm(
                            int(plan.algo),
                            int(plan.op_a),
                            int(plan.op_b),
                            self._alpha.data.ptr,
                            sp_desc,
                            ws.dense.src_descs[op.src_level],
                            self._beta_zero.data.ptr,
                            scratch_dst_descs[op_idx],
                            self._cuda_dtype,
                            0 if ext is None else ext.data.ptr,
                        )
                        scratch_done_events[op_idx].record(helper_stream)
                with stream:
                    for op_idx in scratch_plan.reduce_order:
                        stream.wait_event(scratch_done_events[op_idx])
                        ws.dense.level_bufs[dst_level] += scratch_views[op_idx]
                    _publish_level_source_view(cp=self._cp, dense=ws.dense, plan=plan, level=dst_level)
                    ws.ready_events[dst_level].record(stream)
                continue

            with stream:
                for op_idx, op in enumerate(ops):
                    stream.wait_event(ws.ready_events[op.src_level])
                    sp_desc = op.block.graph_desc if ws.use_graph_descs else op.block.dynamic_desc
                    ext = ws.spmm_ext_by_level[dst_level][op_idx]
                    self._cslib.set_stream(stream.ptr)
                    self._cslib.spmm(
                        int(plan.algo),
                        int(plan.op_a),
                        int(plan.op_b),
                        self._alpha.data.ptr,
                        sp_desc,
                        ws.dense.src_descs[op.src_level],
                        self._beta_one.data.ptr,
                        ws.dense.dst_descs[dst_level],
                        self._cuda_dtype,
                        0 if ext is None else ext.data.ptr,
                    )
                _publish_level_source_view(cp=self._cp, dense=ws.dense, plan=plan, level=dst_level)
                ws.ready_events[dst_level].record(stream)

        with self._capture_stream:
            for event in ws.ready_events:
                self._capture_stream.wait_event(event)

    def _enqueue_wavefront_nvtx(self, ws: _DirectionWorkspace) -> None:
        tracer = self._nvtx
        if tracer is None:
            raise RuntimeError("cuSPARSE NVTX tracer is not initialized")
        if self._alpha is None or self._beta_zero is None or self._beta_one is None or self._cuda_dtype is None:
            raise RuntimeError("cuSPARSE runtime constants are uninitialized")

        plan = self._require_plan(ws.direction)
        ops_by_level = self._ops_for(ws.direction)
        scratch_plans = self._scratch_plans_for(ws.direction)
        scratch_streams_by_level = self._scratch_streams_for(ws.direction)
        if ws.direction == Direction.UP:
            seed_level = 0
            level_iter = range(1, self._H)
        else:
            seed_level = self._H - 1
            level_iter = range(self._H - 2, -1, -1)

        with tracer.range("wavefront", dir=ws.direction.value):
            with self._capture_stream:
                ws.fork_event.record(self._capture_stream)
                tracer.mark("event.record_fork", dir=ws.direction.value)
            for stream in self._level_streams:
                stream.wait_event(ws.fork_event)
            for scratch_streams in scratch_streams_by_level:
                for stream in scratch_streams:
                    stream.wait_event(ws.fork_event)

            if self._H > 0:
                seed_stream = self._level_streams[seed_level]
                with seed_stream:
                    with tracer.range("seed", dir=ws.direction.value, level=seed_level):
                        _publish_level_source_view(cp=self._cp, dense=ws.dense, plan=plan, level=seed_level)
                        ws.ready_events[seed_level].record(seed_stream)
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
                        scratch_views = ws.scratch_views_by_level[dst_level]
                        scratch_done_events = ws.scratch_done_events_by_level[dst_level]
                        scratch_streams = scratch_streams_by_level[dst_level]
                        scratch_dst_descs = ws.scratch_dst_descs_by_level[dst_level]
                        for op_idx, op in enumerate(ops):
                            helper_stream = scratch_streams[op_idx]
                            with helper_stream:
                                tracer.mark("wait_ready", dir=ws.direction.value, dst=dst_level, src=op.src_level)
                                helper_stream.wait_event(ws.ready_events[op.src_level])
                                with tracer.range(
                                    "helper_launch",
                                    dir=ws.direction.value,
                                    dst=dst_level,
                                    src=op.src_level,
                                    helper=op_idx,
                                ):
                                    sp_desc = op.block.graph_desc if ws.use_graph_descs else op.block.dynamic_desc
                                    self._cslib.set_stream(helper_stream.ptr)
                                    with tracer.range(
                                        "launch",
                                        dir=ws.direction.value,
                                        dst=dst_level,
                                        src=op.src_level,
                                        helper=op_idx,
                                        fmt=op.block.fmt,
                                        rows=op.block.nrows,
                                        cols=op.block.ncols,
                                        nnz=op.nnz,
                                    ):
                                        ext = ws.spmm_ext_by_level[dst_level][op_idx]
                                        self._cslib.spmm(
                                            int(plan.algo),
                                            int(plan.op_a),
                                            int(plan.op_b),
                                            self._alpha.data.ptr,
                                            sp_desc,
                                            ws.dense.src_descs[op.src_level],
                                            self._beta_zero.data.ptr,
                                            scratch_dst_descs[op_idx],
                                            self._cuda_dtype,
                                            0 if ext is None else ext.data.ptr,
                                        )
                                    scratch_done_events[op_idx].record(helper_stream)
                                    tracer.mark(
                                        "event.record_scratch_done",
                                        dir=ws.direction.value,
                                        dst=dst_level,
                                        src=op.src_level,
                                        helper=op_idx,
                                    )
                        with stream:
                            for op_idx in scratch_plan.reduce_order:
                                src_level = ops[op_idx].src_level
                                tracer.mark(
                                    "wait_scratch_done",
                                    dir=ws.direction.value,
                                    dst=dst_level,
                                    src=src_level,
                                    helper=op_idx,
                                )
                                stream.wait_event(scratch_done_events[op_idx])
                                with tracer.range(
                                    "reduce_add",
                                    dir=ws.direction.value,
                                    dst=dst_level,
                                    src=src_level,
                                    helper=op_idx,
                                ):
                                    ws.dense.level_bufs[dst_level] += scratch_views[op_idx]
                            with tracer.range("publish_level", dir=ws.direction.value, level=dst_level):
                                _publish_level_source_view(cp=self._cp, dense=ws.dense, plan=plan, level=dst_level)
                            ws.ready_events[dst_level].record(stream)
                            tracer.mark("event.record_ready", dir=ws.direction.value, level=dst_level)
                        continue

                    with stream:
                        for op_idx, op in enumerate(ops):
                            tracer.mark("wait_ready", dir=ws.direction.value, dst=dst_level, src=op.src_level)
                            stream.wait_event(ws.ready_events[op.src_level])
                            sp_desc = op.block.graph_desc if ws.use_graph_descs else op.block.dynamic_desc
                            ext = ws.spmm_ext_by_level[dst_level][op_idx]
                            self._cslib.set_stream(stream.ptr)
                            with tracer.range(
                                "launch",
                                dir=ws.direction.value,
                                dst=dst_level,
                                src=op.src_level,
                                fmt=op.block.fmt,
                                rows=op.block.nrows,
                                cols=op.block.ncols,
                                nnz=op.nnz,
                            ):
                                self._cslib.spmm(
                                    int(plan.algo),
                                    int(plan.op_a),
                                    int(plan.op_b),
                                    self._alpha.data.ptr,
                                    sp_desc,
                                    ws.dense.src_descs[op.src_level],
                                    self._beta_one.data.ptr,
                                    ws.dense.dst_descs[dst_level],
                                    self._cuda_dtype,
                                    0 if ext is None else ext.data.ptr,
                                )
                        with tracer.range("publish_level", dir=ws.direction.value, level=dst_level):
                            _publish_level_source_view(cp=self._cp, dense=ws.dense, plan=plan, level=dst_level)
                        ws.ready_events[dst_level].record(stream)
                        tracer.mark("event.record_ready", dir=ws.direction.value, level=dst_level)

            with tracer.range("join_ready", dir=ws.direction.value):
                with self._capture_stream:
                    for event in ws.ready_events:
                        self._capture_stream.wait_event(event)

    def _run_wavefront(self, ws: _DirectionWorkspace) -> None:
        if self._instrumentation:
            self._enqueue_wavefront_nvtx(ws)
            return
        self._enqueue_wavefront(ws)

    def _capture_wavefront_graph(self, ws: _DirectionWorkspace) -> CupyGraph:
        with self._capture_stream:
            for buf in ws.dense.level_bufs:
                buf.fill(0)
        self._enqueue_wavefront(ws)
        self._await_direction_streams(ws.direction)

        self._capture_stream.begin_capture()
        try:
            self._enqueue_wavefront(ws)
            graph = self._capture_stream.end_capture()
        except Exception:
            try:
                self._capture_stream.end_capture()
            except Exception:
                pass
            raise

        graph.upload(self._capture_stream)
        self._await_direction_streams(ws.direction)
        return graph

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
        plan = self._require_plan(direction)
        x, k = self._normalize_primary_input(direction=direction, primary=primary)
        miss_arr = self._normalize_down_miss_input(miss, k=k) if direction == Direction.DOWN else None
        mode = parse_init_mode(init_mode)
        init_payload = self._validate_init(mode, init, k)
        hint = effective_k_hint(instrumentation=self._instrumentation, k_hint=plan.k_hint)
        use_graph = bool(not self._instrumentation and hint is not None and int(k) == int(hint))
        if hint is not None and int(k) != int(hint):
            warn_k_hint_mismatch(
                backend="cuSPARSE",
                direction=direction,
                runtime_k=int(k),
                k_hint=int(hint),
            )
        ws = self._workspace_for(direction, int(k), graph=use_graph)
        staging = self._staging_for(direction, int(k))
        self._stage_inputs(
            ws,
            staging,
            x,
            miss_arr=miss_arr,
            init_mode=mode,
            init_payload=init_payload,
        )

        tracer = self._nvtx
        if tracer is not None:
            exec_mode = "instrumented"
            with tracer.range("run_direction", dir=direction.value):
                with tracer.range("seed_direction", dir=direction.value):
                    self._seed_workspace(ws, staging, init_mode=mode, has_miss_input=miss_arr is not None)
                self._run_wavefront(ws)
                with tracer.range("collect_outputs", dir=direction.value):
                    outputs = (
                        self._copy_node_outputs_to_host(ws)
                        if emit_all_nodes
                        else self._collect_outputs(ws, staging, need_miss_output=need_miss_output)
                    )
        else:
            self._seed_workspace(ws, staging, init_mode=mode, has_miss_input=miss_arr is not None)
            if use_graph:
                exec_mode = "graph"
                if ws.graph is None:
                    raise RuntimeError(f"Missing captured {direction.value.upper()} CUDA graph for configured k_hint")
                with self._capture_stream:
                    ws.graph.launch(self._capture_stream)
            else:
                exec_mode = "dynamic"
                self._run_wavefront(ws)
            outputs = (
                self._copy_node_outputs_to_host(ws)
                if emit_all_nodes
                else self._collect_outputs(ws, staging, need_miss_output=need_miss_output)
            )

        if self._capture_active:
            call = self._call_mem
            assert isinstance(call, CusparseCall)
            call.node_values_host = outputs if emit_all_nodes else None
            call.miss_output_host = outputs[1] if (direction == Direction.UP and not emit_all_nodes) else None
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
                    runtime_k=int(k),
                    active_alloc_keys=self._alloc_keys(*active_values),
                    meta={
                        "emit_all_nodes": bool(emit_all_nodes),
                        "need_miss_output": bool(need_miss_output) if direction == Direction.UP else False,
                        "has_miss_input": bool(miss_arr is not None) if direction == Direction.DOWN else False,
                        "mode": exec_mode,
                    },
                )
            )
        return outputs

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
        out = self._run_direction(
            Direction.DOWN,
            primary,
            miss=miss,
            init_mode=init_mode,
            init=init,
            need_miss_output=False,
            emit_all_nodes=False,
        )
        return out

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

    def __del__(self):
        cslib = getattr(self, "_cslib", None)
        if cslib is None:
            return
        try:
            self._destroy_workspace_cache()
        except Exception:
            pass
        try:
            _destroy_block_grid(getattr(self, "_blocks_up", []), cslib=cslib)
            _destroy_block_grid(getattr(self, "_blocks_down", []), cslib=cslib)
        except Exception:
            pass
        try:
            self._destroy_shared_ones()
        except Exception:
            pass
        try:
            cslib.destroy()
        except Exception:
            pass


def _gpu_nbytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return int(sum(_gpu_nbytes(v) for v in value))
    if isinstance(value, tuple):
        return int(sum(_gpu_nbytes(v) for v in value))
    nbytes = getattr(value, "nbytes", None)
    if nbytes is None:
        return 0
    return int(nbytes)


def _workspace_nbytes(ws: _DirectionWorkspace | None) -> int:
    if ws is None:
        return 0
    return int(
        _dense_state_nbytes(ws.dense)
        + _gpu_nbytes(ws.scratch_views_by_level)
        + int(ws.input_primary.nbytes)
        + _gpu_nbytes(ws.spmm_ext_by_level)
    )


__all__ = ["CusparseBackend", "CusparsePlan", "DenseOrder", "Operation", "SparseFormat", "SpMMAlgorithm", "is_valid_combo"]
