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
- ``_BlockOp`` is one logical wavefront contribution.
- ``_SelectorLevels`` and ``_SampleRouting`` hold the only two scatter/gather
  schemes needed at the GRG boundary.
- ``_Workspace`` is the reusable device-state cache for one runtime ``k``.

Everything else is plain lists and helper methods so the hot path stays direct.
"""

from __future__ import annotations

import logging
import warnings
from ctypes import c_void_p
from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends import (
    Backend,
    build_wavefront_level_stats,
    estimate_common_host_static_bytes,
    estimate_sparse_payload_bytes,
    log_wavefront_profile,
    selector_rows_unique_from_csr_indptr,
    warn_k_hint_mismatch,
)
from pygrgl_spmv.backends.cuda_utils import (
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_TRANSPOSE,
    CUSPARSE_ORDER_ROW,
    CuSparseLib,
    cuda_dtype,
)
from pygrgl_spmv.backends.memory import RuntimeBytes, StaticBytes
from pygrgl_spmv.backends.types import Direction, InitMode, SparseFormat, parse_init_mode
from . import plan as cusparse_plan
from .plan import CusparsePlan, DenseOrder, Operation, SpMMAlgorithm

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

_OP_N = CUSPARSE_OPERATION_NON_TRANSPOSE
_OP_T = CUSPARSE_OPERATION_TRANSPOSE


def _runtime_supported(plan: CusparsePlan) -> bool:
    return plan.supported and not (plan.fmt == plan.fmt.CSC and plan.algo == plan.algo.CSR_ALG3)


def is_valid_combo(fmt: str, transpose_bool: bool, algo: str) -> bool:
    """Return whether a combo is executable on the current backend/runtime path."""
    plan = CusparsePlan.from_mapping(
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
    return _runtime_supported(plan)


def _validate_plan_slot(*, slot: str, expected: Direction, plan: CusparsePlan) -> None:
    actual = str(getattr(plan.direction, "name", plan.direction)).lower()
    if actual != expected.value:
        raise ValueError(
            f"cuSPARSE {slot} expects a {expected.value.upper()} plan, got {actual.upper()}: {plan}"
        )


@dataclass
class _CuBlock:
    """One physical sparse block plus its cuSPARSE descriptors."""

    fmt: str
    nrows: int
    ncols: int
    nnz: int
    buffers: tuple[CupyArray, CupyArray, CupyArray]
    desc_static: c_void_p
    desc_dynamic: c_void_p
    payload_key: tuple[int, int, int]

    @classmethod
    def from_scipy(
        cls,
        matrix: sp.spmatrix,
        *,
        fmt: str,
        cp: Any,
        dtype: np.dtype,
        cslib: CuSparseLib,
        cuda_dtype_id: int,
    ) -> _CuBlock:
        if fmt == "csr":
            mat = sp.csr_matrix(matrix).astype(dtype)
            buffers = (
                cp.asarray(mat.indptr.astype(np.int32, copy=False)),
                cp.asarray(mat.indices.astype(np.int32, copy=False)),
                cp.asarray(mat.data),
            )
        elif fmt == "csc":
            mat = matrix.tocsc().astype(dtype)
            buffers = (
                cp.asarray(mat.indptr.astype(np.int32, copy=False)),
                cp.asarray(mat.indices.astype(np.int32, copy=False)),
                cp.asarray(mat.data),
            )
        elif fmt == "coo":
            mat = matrix.tocoo().astype(dtype)
            buffers = (
                cp.asarray(mat.row.astype(np.int32, copy=False)),
                cp.asarray(mat.col.astype(np.int32, copy=False)),
                cp.asarray(mat.data),
            )
        else:
            raise ValueError(f"Unknown sparse format: {fmt!r}")

        block = cls(
            fmt=fmt,
            nrows=int(mat.shape[0]),
            ncols=int(mat.shape[1]),
            nnz=int(mat.nnz),
            buffers=buffers,
            desc_static=c_void_p(),
            desc_dynamic=c_void_p(),
            payload_key=tuple(int(buf.data.ptr) for buf in buffers),
        )
        block.desc_static = block._create_desc(cslib=cslib, cuda_dtype_id=cuda_dtype_id)
        block.desc_dynamic = block._create_desc(cslib=cslib, cuda_dtype_id=cuda_dtype_id)
        return block

    @classmethod
    def from_buffers(
        cls,
        *,
        fmt: str,
        nrows: int,
        ncols: int,
        nnz: int,
        buffers: tuple[CupyArray, CupyArray, CupyArray],
        payload_key: tuple[int, int, int],
        cslib: CuSparseLib,
        cuda_dtype_id: int,
    ) -> _CuBlock:
        block = cls(
            fmt=fmt,
            nrows=int(nrows),
            ncols=int(ncols),
            nnz=int(nnz),
            buffers=buffers,
            desc_static=c_void_p(),
            desc_dynamic=c_void_p(),
            payload_key=payload_key,
        )
        block.desc_static = block._create_desc(cslib=cslib, cuda_dtype_id=cuda_dtype_id)
        block.desc_dynamic = block._create_desc(cslib=cslib, cuda_dtype_id=cuda_dtype_id)
        return block

    def _create_desc(self, *, cslib: CuSparseLib, cuda_dtype_id: int) -> c_void_p:
        b0, b1, b2 = self.buffers
        if self.fmt == "csr":
            return cslib.create_csr(
                self.nrows,
                self.ncols,
                self.nnz,
                b0.data.ptr,
                b1.data.ptr,
                b2.data.ptr,
                cuda_dtype_id,
            )
        if self.fmt == "csc":
            return cslib.create_csc(
                self.nrows,
                self.ncols,
                self.nnz,
                b0.data.ptr,
                b1.data.ptr,
                b2.data.ptr,
                cuda_dtype_id,
            )
        if self.fmt == "coo":
            return cslib.create_coo(
                self.nrows,
                self.ncols,
                self.nnz,
                b0.data.ptr,
                b1.data.ptr,
                b2.data.ptr,
                cuda_dtype_id,
            )
        raise ValueError(f"Unknown sparse format: {self.fmt!r}")

    def desc(self, *, use_static: bool) -> c_void_p:
        return self.desc_static if use_static else self.desc_dynamic

    def nbytes(self) -> int:
        return int(sum(int(buf.nbytes) for buf in self.buffers))

    def estimate_nbytes(self, *, data_itemsize: int, index_itemsize: int) -> int:
        return estimate_sparse_payload_bytes(
            fmt=self.fmt,
            nrows=self.nrows,
            ncols=self.ncols,
            nnz=self.nnz,
            data_itemsize=data_itemsize,
            index_itemsize=index_itemsize,
        )

    def destroy(self, *, cslib: CuSparseLib) -> None:
        for desc in (self.desc_static, self.desc_dynamic):
            cslib.destroy_sp_mat(desc)


@dataclass(frozen=True)
class _BlockOp:
    """One logical block application inside a level wavefront."""

    src_level: int
    block: _CuBlock
    nnz: int


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
class _SampleRouting:
    """Forward sample scatter and backward sample gather split by level."""

    fwd_ns: list[int]
    fwd_src: list[CupyArray | None]
    bwd_dst: list[CupyArray]
    bwd_src: list[CupyArray]

    @classmethod
    def from_permutations(
        cls,
        *,
        cp: Any,
        sample_perm: np.ndarray,
        inv_sample_perm: np.ndarray,
        level_offsets: np.ndarray,
        H: int,
        n: int,
    ) -> _SampleRouting:
        fwd_ns: list[int] = []
        fwd_src: list[CupyArray | None] = []
        bwd_dst: list[CupyArray] = []
        bwd_src: list[CupyArray] = []

        for h in range(H):
            lo = int(level_offsets[h])
            hi = int(level_offsets[h + 1])

            ns = max(0, min(hi, n) - lo)
            fwd_ns.append(int(ns))
            fwd_src.append(cp.asarray(sample_perm[lo : lo + ns], dtype=np.int32) if ns > 0 else None)

            mask = (inv_sample_perm >= lo) & (inv_sample_perm < hi)
            bwd_dst.append(cp.asarray(np.where(mask)[0], dtype=np.int32))
            bwd_src.append(cp.asarray(inv_sample_perm[mask] - lo, dtype=np.int32))

        return cls(fwd_ns=fwd_ns, fwd_src=fwd_src, bwd_dst=bwd_dst, bwd_src=bwd_src)

    def scatter(
        self,
        *,
        cp: Any,
        stream: Any,
        x_gpu: CupyArray,
        level_buffers: list[CupyArray],
    ) -> None:
        with stream:
            for h, ns in enumerate(self.fwd_ns):
                if ns <= 0:
                    continue
                src = self.fwd_src[h]
                if src is None:
                    continue
                cp.take(x_gpu, src, axis=0, out=level_buffers[h][:ns])

    def gather(
        self,
        *,
        cp: Any,
        stream: Any,
        level_buffers: list[CupyArray],
        gather_temp: list[CupyArray | None],
        out_gpu: CupyArray,
    ) -> None:
        with stream:
            out_gpu.fill(0)
            for h, dst in enumerate(self.bwd_dst):
                if dst.size == 0:
                    continue
                temp = gather_temp[h]
                if temp is None:
                    continue
                cp.take(level_buffers[h], self.bwd_src[h], axis=0, out=temp)
                out_gpu[dst] = temp


@dataclass
class _DenseViews:
    """Per-direction dense state, destination views, and source views."""

    state_bufs: list[CupyArray]
    dst_descs: list[c_void_p]
    src_descs: list[c_void_p]
    src_bufs: list[CupyArray] | None


@dataclass
class _Workspace:
    """Reusable device state for one runtime ``k``."""

    k: int
    use_static_descs: bool
    up_dense: _DenseViews
    down_dense: _DenseViews
    gather_temp: list[CupyArray | None]
    fwd_input: CupyArray
    bwd_input_mut: CupyArray
    bwd_input_miss: CupyArray | None
    mut_out: CupyArray
    miss_out: CupyArray
    sample_out: CupyArray
    init_vec: CupyArray
    init_matrix: CupyArray | None
    up: Any
    down: Any

    def ensure_init_matrix(self, *, cp: Any, K: int, dtype: np.dtype) -> CupyArray:
        if self.init_matrix is None:
            self.init_matrix = cp.zeros((K, self.k), dtype=dtype, order="C")
        return self.init_matrix

    def ensure_miss_input(self, *, cp: Any, m: int, dtype: np.dtype) -> CupyArray:
        if self.bwd_input_miss is None:
            self.bwd_input_miss = cp.zeros((m, self.k), dtype=dtype, order="C")
        return self.bwd_input_miss

    def dense(self, direction: Direction) -> _DenseViews:
        return self.up_dense if direction == Direction.UP else self.down_dense

    def destroy(self, *, cslib: CuSparseLib) -> None:
        seen: set[int] = set()
        for dense in (self.up_dense, self.down_dense):
            for desc in [*dense.dst_descs, *dense.src_descs]:
                key = id(desc)
                if key in seen:
                    continue
                seen.add(key)
                cslib.destroy_dn_mat(desc)


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


def _build_dense_views(
    *,
    cp: Any,
    cslib: CuSparseLib,
    plan: CusparsePlan | None,
    level_sizes: list[int],
    k: int,
    dtype: np.dtype,
    cuda_dtype_id: int,
    shared_state_bufs: list[CupyArray] | None = None,
) -> _DenseViews:
    if plan is None:
        return _DenseViews(state_bufs=[], dst_descs=[], src_descs=[], src_bufs=None)

    if shared_state_bufs is None:
        state_bufs = [cp.zeros((nrows, k), dtype=dtype, order=_dense_order_char(plan.order_c)) for nrows in level_sizes]
    else:
        state_bufs = shared_state_bufs

    dst_descs = [
        _create_dense_desc(cslib=cslib, buf=buf, rows=buf.shape[0], cols=buf.shape[1], order=plan.order_c, cuda_dtype_id=cuda_dtype_id)
        for buf in state_bufs
    ]

    if plan.op_b == Operation.N and plan.order_b == plan.order_c:
        return _DenseViews(state_bufs=state_bufs, dst_descs=dst_descs, src_descs=dst_descs, src_bufs=None)

    if plan.op_b == Operation.T and plan.order_b != plan.order_c:
        src_descs = [
            _create_dense_desc(cslib=cslib, buf=buf, rows=k, cols=buf.shape[0], order=plan.order_b, cuda_dtype_id=cuda_dtype_id)
            for buf in state_bufs
        ]
        return _DenseViews(state_bufs=state_bufs, dst_descs=dst_descs, src_descs=src_descs, src_bufs=None)

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
    return _DenseViews(state_bufs=state_bufs, dst_descs=dst_descs, src_descs=src_descs, src_bufs=src_bufs)


def _dense_views_nbytes(dense: _DenseViews) -> int:
    total = int(sum(int(buf.nbytes) for buf in dense.state_bufs))
    if dense.src_bufs is not None:
        total += int(sum(int(buf.nbytes) for buf in dense.src_bufs))
    return total


def _build_direction_state(
    *,
    cp: Any,
    ops_by_level: list[list[_BlockOp]],
    src_descs: list[c_void_p],
    dst_descs: list[c_void_p],
    use_static_descs: bool,
    cslib: CuSparseLib,
    alpha: CupyArray,
    beta_one: CupyArray,
    cuda_dtype_id: int,
    algo_id: int,
    op_a: int,
    op_b: int,
) -> Any:
    ext_buffers: list[list[CupyArray]] = []
    for dst_level, ops in enumerate(ops_by_level):
        row: list[CupyArray] = []
        for op in ops:
            sp_desc = op.block.desc(use_static=use_static_descs)
            ext = cslib.spmm_buffer_size(
                cp,
                algo_id,
                op_a,
                op_b,
                alpha.data.ptr,
                sp_desc,
                src_descs[op.src_level],
                beta_one.data.ptr,
                dst_descs[dst_level],
                cuda_dtype_id,
            )
            cslib.spmm_preprocess(
                algo_id,
                op_a,
                op_b,
                alpha.data.ptr,
                sp_desc,
                src_descs[op.src_level],
                beta_one.data.ptr,
                dst_descs[dst_level],
                cuda_dtype_id,
                ext.data.ptr,
            )
            row.append(ext)
        ext_buffers.append(row)

    H = len(dst_descs)
    return SimpleNamespace(
        ext_buffers=ext_buffers,
        graph=None,
        fork_event=cp.cuda.Event(),
        ready_events=[cp.cuda.Event() for _ in range(H)],
        start_events=[cp.cuda.Event() for _ in range(H)],
        end_events=[cp.cuda.Event() for _ in range(H)],
    )


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


def _block_grid_bytes(grid: list[list[_CuBlock | None]]) -> int:
    return int(sum(block.nbytes() for block in _iter_unique_blocks(grid)))


def _estimate_block_grid_bytes(
    grid: list[list[_CuBlock | None]],
    *,
    data_itemsize: int,
    index_itemsize: int,
) -> int:
    return int(
        sum(
            block.estimate_nbytes(data_itemsize=data_itemsize, index_itemsize=index_itemsize)
            for block in _iter_unique_blocks(grid)
        )
    )


def _publish_level_source(*, cp: Any, dense: _DenseViews, plan: CusparsePlan, level: int) -> None:
    if dense.src_bufs is None:
        return
    src = dense.src_bufs[level]
    state = dense.state_bufs[level]
    if plan.op_b == Operation.N:
        cp.copyto(src, state)
    else:
        cp.copyto(src, state.T)


def _destroy_block_grid(grid: list[list[_CuBlock | None]], *, cslib: CuSparseLib) -> None:
    for block in _iter_unique_blocks(grid):
        block.destroy(cslib=cslib)


class CusparseBackend(Backend):
    """GPU backend using cuSPARSE for block-wise level traversal."""

    def __init__(
        self,
        *,
        plan_up: CusparsePlan | None,
        plan_down: CusparsePlan | None,
        log_level: str = "WARNING",
    ):
        up = None if plan_up is None else CusparsePlan.from_any(plan_up)
        down = None if plan_down is None else CusparsePlan.from_any(plan_down)
        if up is not None:
            _validate_plan_slot(slot="plan_up", expected=Direction.UP, plan=up)
        if down is not None:
            _validate_plan_slot(slot="plan_down", expected=Direction.DOWN, plan=down)
        for name, plan in (("UP", up), ("DOWN", down)):
            if plan is not None and not _runtime_supported(plan):
                raise ValueError(f"Unsupported cuSPARSE {name} plan: {plan}")

        try:
            import cupy as cp

            self._cp = cp
        except ImportError as exc:
            raise ImportError("CuPy required: pip install cupy-cuda12x") from exc

        super().__init__(plan_up=up, plan_down=down, log_level=log_level)

        self._cslib = CuSparseLib()
        runtime_version = cusparse_plan._runtime_cuda_version()
        runtime_token = ".".join(str(part) for part in runtime_version)
        self._stream: CupyStream = self._cp.cuda.Stream(non_blocking=True)
        self._level_streams: list[CupyStream] = []

        self._dtype = np.float64
        self._cuda_dtype: int | None = None
        self._alpha: CupyArray | None = None
        self._beta_one: CupyArray | None = None

        self._H = 0
        self._n = 0
        self._m = 0
        self._K = 0
        self._level_offsets = np.empty(0, dtype=np.int64)

        self._sample_perm_host = np.empty(0, dtype=np.int64)
        self._inv_sample_perm_host = np.empty(0, dtype=np.int64)
        self._coalescence_counts: np.ndarray | None = None
        self._xtx_levels: list[CupyArray] = []

        self._blocks_up: list[list[_CuBlock | None]] = []
        self._blocks_down: list[list[_CuBlock | None]] = []
        self._ops_up: list[list[_BlockOp]] = []
        self._ops_down: list[list[_BlockOp]] = []
        self._up_calls = np.empty(0, dtype=np.int32)
        self._down_calls = np.empty(0, dtype=np.int32)
        self._up_nnz = np.empty(0, dtype=np.int64)
        self._down_nnz = np.empty(0, dtype=np.int64)

        self._mut_selector: _SelectorLevels | None = None
        self._miss_selector: _SelectorLevels | None = None
        self._sample_routing: _SampleRouting | None = None
        self._workspaces = SimpleNamespace(static_up=None, static_down=None, dynamic=None)

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

    def setup(
        self,
        A_blocks: list[list[sp.csr_matrix]],
        level_offsets: np.ndarray,
        n: int,
        K: int,
        sel_mut: sp.csr_matrix,
        sel_miss: sp.csr_matrix,
        sample_perm: np.ndarray,
        inv_sample_perm: np.ndarray,
        coalescence_counts: np.ndarray | None,
        dtype: np.dtype,
    ) -> None:
        self._destroy_workspace_cache()
        _destroy_block_grid(self._blocks_up, cslib=self._cslib)
        _destroy_block_grid(self._blocks_down, cslib=self._cslib)

        self._dtype = np.dtype(dtype)
        self._cuda_dtype = cuda_dtype(self._dtype)
        self._level_offsets = np.asarray(level_offsets)
        self._H = len(self._level_offsets) - 1
        self._n = int(n)
        self._K = int(K)
        self._m = int(sel_mut.shape[0])
        self._sample_perm_host = np.asarray(sample_perm, dtype=np.int64)
        self._inv_sample_perm_host = np.asarray(inv_sample_perm, dtype=np.int64)
        self._coalescence_counts = (
            None if coalescence_counts is None else np.asarray(coalescence_counts, dtype=np.int64)
        )

        if self._cuda_dtype is None:
            raise RuntimeError("CUDA dtype not initialized")
        self._alpha = self._cp.ones(1, dtype=self._dtype)
        self._beta_one = self._cp.ones(1, dtype=self._dtype)

        self._blocks_up = self._build_direction_blocks(A_blocks, direction=Direction.UP)
        self._blocks_down = self._build_direction_blocks(A_blocks, direction=Direction.DOWN)
        self._ops_up = self._build_up_ops()
        self._ops_down = self._build_down_ops()
        self._up_calls, self._up_nnz = build_wavefront_level_stats(self._ops_up)
        self._down_calls, self._down_nnz = build_wavefront_level_stats(self._ops_down)

        self._mut_selector = _SelectorLevels.from_csr(
            cp=self._cp,
            selector=sel_mut,
            level_offsets=self._level_offsets,
            H=self._H,
        )
        self._miss_selector = _SelectorLevels.from_csr(
            cp=self._cp,
            selector=sel_miss,
            level_offsets=self._level_offsets,
            H=self._H,
        )
        self._sample_routing = _SampleRouting.from_permutations(
            cp=self._cp,
            sample_perm=self._sample_perm_host,
            inv_sample_perm=self._inv_sample_perm_host,
            level_offsets=self._level_offsets,
            H=self._H,
            n=self._n,
        )
        self._level_streams = [self._cp.cuda.Stream(non_blocking=True) for _ in range(self._H)]

        self._xtx_levels = []
        if self._coalescence_counts is not None:
            xtx = (2.0 * self._coalescence_counts).astype(self._dtype, copy=False)
            for h in range(self._H):
                lo = int(self._level_offsets[h])
                hi = int(self._level_offsets[h + 1])
                self._xtx_levels.append(self._cp.asarray(xtx[lo:hi, None], dtype=self._dtype))

        self.mem_usage.reset()
        common_host = estimate_common_host_static_bytes(
            level_offsets=self._level_offsets,
            sample_perm=self._sample_perm_host,
            inv_sample_perm=self._inv_sample_perm_host,
            coalescence_counts=self._coalescence_counts,
            xtx_init=None,
        )
        self.mem_usage.host_static = common_host
        share_storage = self._plan_up is not None and self._plan_down is not None and self._plan_up.can_share_storage_with(self._plan_down)
        self.mem_usage.device_static.blocks_up = _block_grid_bytes(self._blocks_up)
        self.mem_usage.device_static.blocks_down = 0 if share_storage else _block_grid_bytes(self._blocks_down)
        self.mem_usage.device_static.selector_mut = 0 if self._mut_selector is None else self._mut_selector.nbytes()
        self.mem_usage.device_static.selector_miss = 0 if self._miss_selector is None else self._miss_selector.nbytes()
        self.mem_usage.device_static.xtx_init = _gpu_nbytes(self._xtx_levels)

        self._logger.info(
            "CusparseBackend setup: H=%d K=%d n=%d m=%d plan_up=%s plan_down=%s",
            self._H,
            self._K,
            self._n,
            self._m,
            "<unspecified>" if self._plan_up is None else str(self._plan_up),
            "<unspecified>" if self._plan_down is None else str(self._plan_down),
        )

        self._workspaces = SimpleNamespace(static_up=None, static_down=None, dynamic=None)

    def _build_direction_blocks(self, A_blocks: list[list[sp.csr_matrix]], *, direction: Direction) -> list[list[_CuBlock | None]]:
        rows: list[list[_CuBlock | None]] = [[] for _ in range(self._H)]
        if self._cuda_dtype is None:
            raise RuntimeError("CUDA dtype not initialized")
        plan = self._plan_up if direction == Direction.UP else self._plan_down
        if plan is None:
            return rows
        share = direction == Direction.DOWN and self._plan_up is not None and self._plan_down is not None and self._plan_up.can_share_storage_with(self._plan_down)

        for dst in range(self._H):
            row: list[_CuBlock | None] = []
            sources = range(dst) if direction == Direction.UP else range(dst + 1, self._H)
            for src in sources:
                matrix = A_blocks[dst][src] if direction == Direction.UP else A_blocks[src][dst]
                if matrix.nnz == 0:
                    row.append(None)
                    continue
                if share:
                    source = self._blocks_up[src][dst]
                    if source is None:
                        row.append(None)
                        continue
                    nrows, ncols = matrix.shape
                    if plan.store == plan.store.T:
                        nrows, ncols = ncols, nrows
                    row.append(
                        _CuBlock.from_buffers(
                            fmt=plan.fmt.value.lower(),
                            nrows=nrows,
                            ncols=ncols,
                            nnz=source.nnz,
                            buffers=source.buffers,
                            payload_key=source.payload_key,
                            cslib=self._cslib,
                            cuda_dtype_id=self._cuda_dtype,
                        )
                    )
                    continue
                stored = matrix if plan.store == plan.store.N else matrix.T.tocsr()
                row.append(
                    _CuBlock.from_scipy(
                        stored,
                        fmt=plan.fmt.value.lower(),
                        cp=self._cp,
                        dtype=self._dtype,
                        cslib=self._cslib,
                        cuda_dtype_id=self._cuda_dtype,
                    )
                )
            rows[dst] = row
        return rows

    def _build_up_ops(self) -> list[list[_BlockOp]]:
        ops: list[list[_BlockOp]] = [[] for _ in range(self._H)]
        if self._plan_up is None:
            return ops
        for dst in range(1, self._H):
            row: list[_BlockOp] = []
            for src in range(dst):
                block = self._blocks_up[dst][src]
                if block is None:
                    continue
                row.append(_BlockOp(src_level=src, block=block, nnz=block.nnz))
            ops[dst] = row
        return ops

    def _build_down_ops(self) -> list[list[_BlockOp]]:
        ops: list[list[_BlockOp]] = [[] for _ in range(self._H)]
        if self._plan_down is None:
            return ops
        for dst in range(self._H - 1):
            row: list[_BlockOp] = []
            for src in range(self._H - 1, dst, -1):
                block = self._blocks_down[dst][src - dst - 1]
                if block is None:
                    continue
                row.append(_BlockOp(src_level=src, block=block, nnz=block.nnz))
            ops[dst] = row
        return ops

    def _destroy_workspace_cache(self) -> None:
        for ws in (
            getattr(self._workspaces, "static_up", None),
            getattr(self._workspaces, "static_down", None),
            getattr(self._workspaces, "dynamic", None),
        ):
            if ws is None:
                continue
            try:
                ws.destroy(cslib=self._cslib)
            except Exception:
                pass
        self._workspaces = SimpleNamespace(static_up=None, static_down=None, dynamic=None)

    def _create_workspace(self, k: int, *, use_static_descs: bool) -> _Workspace:
        if self._cuda_dtype is None or self._alpha is None or self._beta_one is None:
            raise RuntimeError("cuSPARSE runtime constants are uninitialized")
        if self._sample_routing is None:
            raise RuntimeError("Sample routing is not initialized")

        level_sizes = [int(self._level_offsets[h + 1]) - int(self._level_offsets[h]) for h in range(self._H)]
        up_dense = _build_dense_views(
            cp=self._cp,
            cslib=self._cslib,
            plan=self._plan_up,
            level_sizes=level_sizes,
            k=k,
            dtype=self._dtype,
            cuda_dtype_id=self._cuda_dtype,
        )
        share_state = self._plan_up is not None and self._plan_down is not None and self._plan_up.order_c == self._plan_down.order_c
        down_dense = _build_dense_views(
            cp=self._cp,
            cslib=self._cslib,
            plan=self._plan_down,
            level_sizes=level_sizes,
            k=k,
            dtype=self._dtype,
            cuda_dtype_id=self._cuda_dtype,
            shared_state_bufs=up_dense.state_bufs if share_state else None,
        )

        gather_temp: list[CupyArray | None] = []
        for h in range(self._H):
            count = int(self._sample_routing.bwd_dst[h].size)
            gather_temp.append(self._cp.zeros((count, k), dtype=self._dtype, order="C") if count > 0 else None)

        ws = _Workspace(
            k=int(k),
            use_static_descs=bool(use_static_descs),
            up_dense=up_dense,
            down_dense=down_dense,
            gather_temp=gather_temp,
            fwd_input=self._cp.zeros((self._n, k), dtype=self._dtype, order="C"),
            bwd_input_mut=self._cp.zeros((self._m, k), dtype=self._dtype, order="C"),
            bwd_input_miss=None,
            mut_out=self._cp.zeros((self._m, k), dtype=self._dtype, order="C"),
            miss_out=self._cp.zeros((self._m, k), dtype=self._dtype, order="C"),
            sample_out=self._cp.zeros((self._n, k), dtype=self._dtype, order="C"),
            init_vec=self._cp.zeros((1, k), dtype=self._dtype, order="C"),
            init_matrix=None,
            up=None,
            down=None,
        )
        ws.up = _build_direction_state(
            cp=self._cp,
            ops_by_level=self._ops_up,
            src_descs=ws.up_dense.src_descs,
            dst_descs=ws.up_dense.dst_descs,
            use_static_descs=use_static_descs,
            cslib=self._cslib,
            alpha=self._alpha,
            beta_one=self._beta_one,
            cuda_dtype_id=self._cuda_dtype,
            algo_id=int(self._plan_up.algo) if self._plan_up is not None else int(SpMMAlgorithm.DEFAULT),
            op_a=int(self._plan_up.op_a) if self._plan_up is not None else int(Operation.N),
            op_b=int(self._plan_up.op_b) if self._plan_up is not None else int(Operation.N),
        )
        ws.down = _build_direction_state(
            cp=self._cp,
            ops_by_level=self._ops_down,
            src_descs=ws.down_dense.src_descs,
            dst_descs=ws.down_dense.dst_descs,
            use_static_descs=use_static_descs,
            cslib=self._cslib,
            alpha=self._alpha,
            beta_one=self._beta_one,
            cuda_dtype_id=self._cuda_dtype,
            algo_id=int(self._plan_down.algo) if self._plan_down is not None else int(SpMMAlgorithm.DEFAULT),
            op_a=int(self._plan_down.op_a) if self._plan_down is not None else int(Operation.N),
            op_b=int(self._plan_down.op_b) if self._plan_down is not None else int(Operation.N),
        )
        self._logger.debug("workspace ready k=%d static_descs=%s", k, use_static_descs)
        return ws

    def _run_wavefront(self, ws: _Workspace, direction: Direction, *, track_timing: bool) -> np.ndarray | None:
        if self._alpha is None or self._beta_one is None or self._cuda_dtype is None:
            raise RuntimeError("cuSPARSE runtime constants are uninitialized")

        dense = ws.dense(direction)
        plan = self._require_plan(direction)
        if direction == Direction.UP:
            ops_by_level = self._ops_up
            state = ws.up
            seed_level = 0
            level_iter = range(1, self._H)
        else:
            ops_by_level = self._ops_down
            state = ws.down
            seed_level = self._H - 1
            level_iter = range(self._H - 2, -1, -1)

        with self._stream:
            state.fork_event.record(self._stream)
        for stream in self._level_streams:
            stream.wait_event(state.fork_event)

        seed_stream = self._level_streams[seed_level]
        with seed_stream:
            _publish_level_source(cp=self._cp, dense=dense, plan=plan, level=seed_level)
            state.ready_events[seed_level].record(seed_stream)

        for dst_level in level_iter:
            stream = self._level_streams[dst_level]
            ops = ops_by_level[dst_level]
            with stream:
                if track_timing and ops:
                    state.start_events[dst_level].record(stream)
                for idx, op in enumerate(ops):
                    stream.wait_event(state.ready_events[op.src_level])
                    sp_desc = op.block.desc(use_static=ws.use_static_descs)
                    self._cslib.set_stream(stream.ptr)
                    self._cslib.spmm(
                        int(plan.algo),
                        int(plan.op_a),
                        int(plan.op_b),
                        self._alpha.data.ptr,
                        sp_desc,
                        dense.src_descs[op.src_level],
                        self._beta_one.data.ptr,
                        dense.dst_descs[dst_level],
                        self._cuda_dtype,
                        state.ext_buffers[dst_level][idx].data.ptr,
                    )
                _publish_level_source(cp=self._cp, dense=dense, plan=plan, level=dst_level)
                if track_timing and ops:
                    state.end_events[dst_level].record(stream)
                state.ready_events[dst_level].record(stream)

        with self._stream:
            for event in state.ready_events:
                self._stream.wait_event(event)

        if not track_timing:
            return None

        level_ms = np.zeros(self._H, dtype=np.float64)
        self._stream.synchronize()
        idx_iter = range(1, self._H) if direction == Direction.UP else range(self._H - 1)
        for h in idx_iter:
            if ops_by_level[h]:
                level_ms[h] = self._cp.cuda.get_elapsed_time(state.start_events[h], state.end_events[h])
        return level_ms

    def _zero_level_buffers(self, ws: _Workspace, direction: Direction) -> None:
        dense = ws.dense(direction)
        with self._stream:
            for buf in dense.state_bufs:
                buf.fill(0)

    def _capture_graph(self, ws: _Workspace, direction: Direction) -> CupyGraph:
        self._zero_level_buffers(ws, direction)
        self._run_wavefront(ws, direction, track_timing=False)
        self._stream.synchronize()

        self._stream.begin_capture()
        try:
            self._run_wavefront(ws, direction, track_timing=False)
            graph = self._stream.end_capture()
        except Exception:
            try:
                self._stream.end_capture()
            except Exception:
                pass
            raise

        graph.upload(self._stream)
        self._stream.synchronize()
        return graph

    def _log_wavefront_profile(self, direction: Direction, level_ms: np.ndarray) -> None:
        if direction == Direction.UP:
            calls = self._up_calls
            nnz = self._up_nnz
        else:
            calls = self._down_calls
            nnz = self._down_nnz
        log_wavefront_profile(self._logger, direction=direction, calls=calls, nnz=nnz, level_ms=level_ms)

    def _validate_init(self, init_mode: InitMode, init: np.ndarray | None, k: int) -> np.ndarray | None:
        match init_mode:
            case InitMode.NONE:
                if init is not None:
                    raise ValueError("init payload provided with init_mode=none")
                return None
            case InitMode.XTX:
                if init is not None:
                    raise ValueError("init payload must be None when init_mode=xtx")
                if not self._xtx_levels:
                    raise ValueError("init_mode=xtx requires GRG coalescence counts")
                return None
            case InitMode.VECTOR:
                arr = np.asarray(init, dtype=self._dtype, order="C")
                if arr.ndim != 1 or arr.shape[0] != k:
                    raise ValueError(f"init vector must have shape ({k},), got {arr.shape}")
                return arr
            case InitMode.MATRIX:
                arr = np.asarray(init, dtype=self._dtype, order="C")
                if arr.ndim != 2 or arr.shape != (self._K, k):
                    raise ValueError(f"init matrix must have shape ({self._K}, {k}), got {arr.shape}")
                return arr
            case _:
                raise ValueError(f"Unknown init_mode {init_mode!r}")

    def _run_direction(
        self,
        direction: Direction,
        primary: np.ndarray,
        *,
        miss: np.ndarray | None,
        init_mode: InitMode,
        init: np.ndarray | None,
        need_miss_output: bool,
    ) -> tuple[np.ndarray, np.ndarray | None] | np.ndarray:
        total_t0 = perf_counter()
        t0 = perf_counter()
        x = np.asarray(primary, dtype=self._dtype, order="C")
        if x.ndim != 2:
            raise ValueError(f"primary input must be 2D, got shape {x.shape}")
        if direction == Direction.UP:
            if x.shape[0] != self._n:
                raise ValueError(f"UP primary input must have {self._n} rows, got {x.shape[0]}")
        else:
            if x.shape[0] != self._m:
                raise ValueError(f"DOWN primary input must have {self._m} rows, got {x.shape[0]}")
        k = int(x.shape[1])
        parse_ms = (perf_counter() - t0) * 1000.0

        miss_arr = None
        if direction == Direction.DOWN and miss is not None:
            miss_arr = np.asarray(miss, dtype=self._dtype, order="C")
            if miss_arr.shape != (self._m, k):
                raise ValueError(f"miss input must have shape ({self._m}, {k}), got {miss_arr.shape}")

        mode = parse_init_mode(init_mode)
        t0 = perf_counter()
        init_payload = self._validate_init(mode, init, k)
        init_parse_ms = (perf_counter() - t0) * 1000.0

        plan = self._require_plan(direction)
        static_attr = "static_up" if direction == Direction.UP else "static_down"
        hint_k = plan.k_hint
        if hint_k is not None and int(k) == int(hint_k):
            ws = getattr(self._workspaces, static_attr)
            if ws is None:
                ws = self._create_workspace(int(k), use_static_descs=True)
                if direction == Direction.UP:
                    ws.up.graph = self._capture_graph(ws, Direction.UP)
                else:
                    ws.down.graph = self._capture_graph(ws, Direction.DOWN)
                setattr(self._workspaces, static_attr, ws)
        else:
            ws = self._workspaces.dynamic
            if ws is None or int(ws.k) != int(k):
                if ws is not None:
                    ws.destroy(cslib=self._cslib)
                ws = self._create_workspace(int(k), use_static_descs=False)
                self._workspaces.dynamic = ws
        track_wave = self._logger.isEnabledFor(logging.DEBUG)

        t0 = perf_counter()
        if direction == Direction.UP:
            ws.fwd_input.set(x, stream=self._stream)
        else:
            ws.bwd_input_mut.set(x, stream=self._stream)
        self._stream.synchronize()
        h2d_primary_ms = (perf_counter() - t0) * 1000.0

        h2d_miss_ms = 0.0
        if miss_arr is not None:
            t0 = perf_counter()
            miss_buf = ws.ensure_miss_input(cp=self._cp, m=self._m, dtype=self._dtype)
            miss_buf.set(miss_arr, stream=self._stream)
            self._stream.synchronize()
            h2d_miss_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        match mode:
            case InitMode.NONE | InitMode.XTX:
                pass
            case InitMode.VECTOR:
                if init_payload is None:
                    raise ValueError("init vector payload is required for init_mode=vector")
                with self._stream:
                    ws.init_vec[0].set(init_payload, stream=self._stream)
            case InitMode.MATRIX:
                if init_payload is None:
                    raise ValueError("init matrix payload is required for init_mode=matrix")
                with self._stream:
                    ws.ensure_init_matrix(cp=self._cp, K=self._K, dtype=self._dtype).set(
                        init_payload,
                        stream=self._stream,
                    )
            case _:
                raise ValueError(f"Unknown init mode: {mode!r}")
        self._stream.synchronize()
        init_stage_ms = (perf_counter() - t0) * 1000.0

        if self._mut_selector is None or self._miss_selector is None or self._sample_routing is None:
            raise RuntimeError("cuSPARSE backend is not initialized")

        exec_mode = "dynamic"
        level_ms = None
        t0 = perf_counter()
        self._zero_level_buffers(ws, direction)
        if direction == Direction.UP:
            self._sample_routing.scatter(cp=self._cp, stream=self._stream, x_gpu=ws.fwd_input, level_buffers=ws.dense(direction).state_bufs)
        else:
            self._mut_selector.scatter_add(
                cp=self._cp,
                stream=self._stream,
                level_buffers=ws.dense(direction).state_bufs,
                x_gpu=ws.bwd_input_mut,
            )
            if miss_arr is not None:
                if ws.bwd_input_miss is None:
                    raise RuntimeError("Missing bwd_input_miss buffer while has_miss_input=True")
                self._miss_selector.scatter_add(
                    cp=self._cp,
                    stream=self._stream,
                    level_buffers=ws.dense(direction).state_bufs,
                    x_gpu=ws.bwd_input_miss,
                )
        if mode != InitMode.NONE:
            with self._stream:
                match mode:
                    case InitMode.XTX:
                        if not self._xtx_levels:
                            raise ValueError("init_mode=xtx requires GRG coalescence counts")
                        for h, xtx in enumerate(self._xtx_levels):
                            ws.dense(direction).state_bufs[h] += xtx
                    case InitMode.VECTOR:
                        for buf in ws.dense(direction).state_bufs:
                            buf += ws.init_vec
                    case InitMode.MATRIX:
                        init_matrix = ws.ensure_init_matrix(cp=self._cp, K=self._K, dtype=self._dtype)
                        for h in range(self._H):
                            lo = int(self._level_offsets[h])
                            hi = int(self._level_offsets[h + 1])
                            ws.dense(direction).state_bufs[h] += init_matrix[lo:hi]
                    case _:
                        raise ValueError(f"Unknown init mode: {mode!r}")
        use_graph = False
        if hint_k is not None and int(k) == int(hint_k) and not track_wave:
            use_graph = True
        elif hint_k is not None and int(k) != int(hint_k):
            warn_k_hint_mismatch(
                backend="cuSPARSE",
                direction=direction,
                runtime_k=int(k),
                k_hint=int(hint_k),
            )
        elif hint_k is not None and track_wave:
            warnings.warn(
                (
                    f"cuSPARSE {direction.value} graph disabled: per-level wavefront timing "
                    "requires dynamic execution."
                ),
                RuntimeWarning,
                stacklevel=3,
            )
        if use_graph:
            exec_mode = "graph"
            graph = ws.up.graph if direction == Direction.UP else ws.down.graph
            if graph is None:
                raise RuntimeError(f"Missing captured {direction.value.upper()} CUDA graph for configured k_hint")
            with self._stream:
                graph.launch(self._stream)
        else:
            level_ms = self._run_wavefront(ws, direction, track_timing=track_wave)

        if direction == Direction.UP:
            self._mut_selector.gather(
                cp=self._cp,
                stream=self._stream,
                level_buffers=ws.dense(direction).state_bufs,
                out_gpu=ws.mut_out,
            )
            if need_miss_output:
                self._miss_selector.gather(
                    cp=self._cp,
                    stream=self._stream,
                    level_buffers=ws.dense(direction).state_bufs,
                    out_gpu=ws.miss_out,
                )
        else:
            self._sample_routing.gather(
                cp=self._cp,
                stream=self._stream,
                level_buffers=ws.dense(direction).state_bufs,
                gather_temp=ws.gather_temp,
                out_gpu=ws.sample_out,
            )

        self._stream.synchronize()
        compute_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        if direction == Direction.UP:
            out_mut = ws.mut_out.get()
            out_miss = None
            d2h_ms = float((perf_counter() - t0) * 1000.0)
            d2h_miss_ms = 0.0
            if need_miss_output:
                t0 = perf_counter()
                out_miss = ws.miss_out.get()
                d2h_miss_ms = (perf_counter() - t0) * 1000.0
            host_runtime = RuntimeBytes(
                level_buffers=0,
                inputs=int(x.nbytes),
                outputs=int(out_mut.nbytes + (0 if out_miss is None else out_miss.nbytes)),
                aux=0 if init_payload is None else int(init_payload.nbytes),
            )
            device_runtime = RuntimeBytes(
                level_buffers=int(_dense_views_nbytes(ws.dense(direction))),
                inputs=int(ws.fwd_input.nbytes),
                outputs=int(ws.mut_out.nbytes + (ws.miss_out.nbytes if need_miss_output else 0)),
                aux=int(
                    _gpu_nbytes(ws.up.ext_buffers)
                    + int(ws.init_vec.nbytes)
                    + _gpu_nbytes(ws.init_matrix)
                    + (0 if level_ms is None else level_ms.nbytes)
                ),
            )
            self.mem_usage.record(
                stage="run_up",
                runtime_k=k,
                host_runtime=host_runtime,
                device_runtime=device_runtime,
                meta={"direction": "up", "need_miss_output": bool(need_miss_output), "mode": exec_mode},
            )
            if self._logger.isEnabledFor(logging.INFO):
                total_ms = (perf_counter() - total_t0) * 1000.0
                self._logger.info(
                    (
                        "cusparse.run_up k=%d mode=%s init=%s miss_out=%s: parse=%.3fms init_parse=%.3fms "
                        "H2D=%.3fms init_stage=%.3fms compute=%.3fms D2H_mut=%.3fms D2H_miss=%.3fms total=%.3fms"
                    ),
                    k,
                    exec_mode,
                    mode.value,
                    need_miss_output,
                    parse_ms,
                    init_parse_ms,
                    h2d_primary_ms,
                    init_stage_ms,
                    compute_ms,
                    d2h_ms,
                    d2h_miss_ms,
                    total_ms,
                )
                if track_wave and level_ms is not None:
                    self._log_wavefront_profile(Direction.UP, level_ms)
            return out_mut, out_miss

        out = ws.sample_out.get()
        d2h_ms = (perf_counter() - t0) * 1000.0
        host_runtime = RuntimeBytes(
            level_buffers=0,
            inputs=int(x.nbytes + (0 if miss_arr is None else miss_arr.nbytes)),
            outputs=int(out.nbytes),
            aux=0 if init_payload is None else int(init_payload.nbytes),
        )
        device_runtime = RuntimeBytes(
            level_buffers=int(_dense_views_nbytes(ws.dense(direction))),
            inputs=int(ws.bwd_input_mut.nbytes + (0 if ws.bwd_input_miss is None or miss_arr is None else ws.bwd_input_miss.nbytes)),
            outputs=int(ws.sample_out.nbytes),
            aux=int(
                _gpu_nbytes(ws.down.ext_buffers)
                + _gpu_nbytes(ws.gather_temp)
                + int(ws.init_vec.nbytes)
                + _gpu_nbytes(ws.init_matrix)
                + (0 if level_ms is None else level_ms.nbytes)
            ),
        )
        self.mem_usage.record(
            stage="run_down",
            runtime_k=k,
            host_runtime=host_runtime,
            device_runtime=device_runtime,
            meta={"direction": "down", "has_miss_input": bool(miss_arr is not None), "mode": exec_mode},
        )
        if self._logger.isEnabledFor(logging.INFO):
            total_ms = (perf_counter() - total_t0) * 1000.0
            self._logger.info(
                (
                    "cusparse.run_down k=%d mode=%s init=%s miss_in=%s: parse=%.3fms init_parse=%.3fms "
                    "H2D_mut=%.3fms H2D_miss=%.3fms init_stage=%.3fms compute=%.3fms D2H=%.3fms total=%.3fms"
                ),
                k,
                exec_mode,
                mode.value,
                miss_arr is not None,
                parse_ms,
                init_parse_ms,
                h2d_primary_ms,
                h2d_miss_ms,
                init_stage_ms,
                compute_ms,
                d2h_ms,
                total_ms,
            )
            if track_wave and level_ms is not None:
                self._log_wavefront_profile(Direction.DOWN, level_ms)
        return out

    def run_up(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
        need_miss_output: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self._require_plan(Direction.UP)
        out_mut, out_miss = self._run_direction(
            Direction.UP,
            primary,
            miss=None,
            init_mode=init_mode,
            init=init,
            need_miss_output=need_miss_output,
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
        out = self._run_direction(
            Direction.DOWN,
            primary,
            miss=miss,
            init_mode=init_mode,
            init=init,
            need_miss_output=False,
        )
        return out

    def estimate_static_bytes(self) -> tuple[StaticBytes, StaticBytes]:
        host = estimate_common_host_static_bytes(
            level_offsets=self._level_offsets,
            sample_perm=self._sample_perm_host,
            inv_sample_perm=self._inv_sample_perm_host,
            coalescence_counts=self._coalescence_counts,
            xtx_init=None,
        )
        device = StaticBytes()
        data_itemsize = int(np.dtype(self._dtype).itemsize)
        index_itemsize = int(np.dtype(np.int32).itemsize)
        share_storage = self._plan_up is not None and self._plan_down is not None and self._plan_up.can_share_storage_with(self._plan_down)
        device.blocks_up = _estimate_block_grid_bytes(
            self._blocks_up,
            data_itemsize=data_itemsize,
            index_itemsize=index_itemsize,
        )
        device.blocks_down = 0 if share_storage else _estimate_block_grid_bytes(
            self._blocks_down,
            data_itemsize=data_itemsize,
            index_itemsize=index_itemsize,
        )
        device.selector_mut = 0 if self._mut_selector is None else int(self._mut_selector.nnz() * 2 * index_itemsize)
        device.selector_miss = 0 if self._miss_selector is None else int(self._miss_selector.nnz() * 2 * index_itemsize)
        if self._coalescence_counts is not None:
            device.xtx_init = int(self._coalescence_counts.size * data_itemsize)
        return host, device

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


__all__ = ["CusparseBackend", "CusparsePlan", "DenseOrder", "Operation", "SparseFormat", "SpMMAlgorithm", "is_valid_combo"]
