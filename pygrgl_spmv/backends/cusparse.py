"""cuSPARSE backend for fused GRG sparse matmul traversal."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from time import perf_counter
from typing import List

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends import (
    Backend,
    estimate_common_host_static_bytes,
    estimate_sparse_payload_bytes,
)
from pygrgl_spmv.backends.memory import RuntimeBytes, StaticBytes
from pygrgl_spmv.backends.cuda_utils import (
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_TRANSPOSE,
    CuSparseLib,
    cuda_dtype,
    parse_algo,
)
from pygrgl_spmv.backends.types import Direction, InitMode, parse_init_mode

_OP_N = CUSPARSE_OPERATION_NON_TRANSPOSE
_OP_T = CUSPARSE_OPERATION_TRANSPOSE

# Whitelist of (fmt, algo) combinations that cuSPARSE supports.
VALID_FMT_ALGO_COMBOS = frozenset(
    {
        ("csr", "default"),
        ("csr", "csr_alg1"),
        ("csr", "csr_alg2"),
        ("csc", "default"),
        ("csc", "csr_alg1"),
        ("csc", "csr_alg2"),
        ("coo", "default"),
        ("coo", "coo_alg1"),
        ("coo", "coo_alg2"),
        ("coo", "coo_alg3"),
        ("coo", "coo_alg4"),
    }
)


def is_valid_combo(fmt: str, algo: str) -> bool:
    """Return True if (fmt, algo) is a supported cuSPARSE SpMM combination."""
    return (fmt, algo) in VALID_FMT_ALGO_COMBOS


@dataclass(frozen=True)
class _CuBlockOp:
    """One sparse block application in a level wavefront."""

    src_level: int
    sp_desc: object
    op_a: int
    algo: int
    nnz: int


def _count_gpu_bytes(obj) -> int:
    if obj is None:
        return 0
    if isinstance(obj, dict):
        return sum(_count_gpu_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_count_gpu_bytes(v) for v in obj)
    nbytes = getattr(obj, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return 0


class CusparseBackend(Backend):
    """GPU backend using cuSPARSE for block-wise SpMM."""

    def __init__(
        self,
        fmt_up: str | None = "csr",
        fmt_down: str | None = None,
        k_hint: int | None = None,
        algo_up: str = "default",
        algo_down: str = "default",
        log_level: str = "WARNING",
    ):
        if not isinstance(algo_up, str):
            raise TypeError(f"algo_up must be a string, got {type(algo_up).__name__}")
        if not isinstance(algo_down, str):
            raise TypeError(f"algo_down must be a string, got {type(algo_down).__name__}")

        try:
            import cupy as cp

            self._cp = cp
        except ImportError as exc:
            raise ImportError("CuPy required: pip install cupy-cuda12x") from exc

        super().__init__(
            fmt_up=fmt_up,
            fmt_down=fmt_down,
            k_hint=k_hint,
            log_level=log_level,
        )
        if not is_valid_combo(self._fmt_up, algo_up):
            raise ValueError(
                f"(fmt_up={self._fmt_up!r}, algo_up={algo_up!r}) is not a supported "
                f"cuSPARSE SpMM combination. Valid combinations: {sorted(VALID_FMT_ALGO_COMBOS)}"
            )
        if not is_valid_combo(self._fmt_down, algo_down):
            raise ValueError(
                f"(fmt_down={self._fmt_down!r}, algo_down={algo_down!r}) is not a supported "
                f"cuSPARSE SpMM combination. Valid combinations: {sorted(VALID_FMT_ALGO_COMBOS)}"
            )

        self._algo_up = parse_algo(algo_up)
        self._algo_down = parse_algo(algo_down)

        self._cslib = CuSparseLib()
        self._stream = self._cp.cuda.Stream(non_blocking=True)

        # Set in setup().
        self._dtype = np.float64
        self._cuda_dtype = None
        self._H = 0
        self._n = 0
        self._K = 0
        self._m = 0

        self._alpha = None
        self._beta_one = None
        self._xtx_levels = []
        self._sample_perm_host = np.empty(0, dtype=np.int64)
        self._inv_sample_perm_host = np.empty(0, dtype=np.int64)

        self._blocks_up: list[list[object | None]] = []
        self._blocks_up_dynamic: list[list[object | None]] = []
        self._blocks_up_bufs: list[list[object | None]] = []
        self._blocks_down: list[list[object | None]] = []
        self._blocks_down_dynamic: list[list[object | None]] = []
        self._blocks_down_bufs: list[list[object | None]] = []
        self._desc_nnz: dict[int, int] = {}
        self._desc_meta: dict[int, tuple[str, int, int, int]] = {}

        self._ops_up: list[list[_CuBlockOp]] = []
        self._ops_down: list[list[_CuBlockOp]] = []
        self._ops_up_dynamic: list[list[_CuBlockOp]] = []
        self._ops_down_dynamic: list[list[_CuBlockOp]] = []
        self._selector_rows = {"mut": [], "miss": []}
        self._selector_cols = {"mut": [], "miss": []}
        self._selector_row_unique = {"mut": True, "miss": True}
        self._fwd_scatter_ns = []
        self._fwd_scatter_src = []
        self._bwd_gather_dst = []
        self._bwd_gather_src = []
        self._level_streams = []
        self._wavefront_calls_up = None
        self._wavefront_nnz_up = None
        self._wavefront_calls_down = None
        self._wavefront_nnz_down = None

        # Static workspace (k_hint) and one reusable dynamic workspace.
        self._workspace_static = None
        self._workspace_dynamic = None
        self._workspace_dynamic_k: int | None = None

        self._logger.info("cuSPARSE version: %s", self._cslib.version)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(
        self,
        A_blocks: List[List[sp.csr_matrix]],
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
        cp = self._cp

        self._dtype = np.dtype(dtype)
        self._cuda_dtype = cuda_dtype(self._dtype)
        self._level_offsets = np.asarray(level_offsets)
        self._H = len(self._level_offsets) - 1
        self._n = int(n)
        self._K = int(K)
        self._m = int(sel_mut.shape[0])
        self._coalescence_counts = (
            None if coalescence_counts is None else np.asarray(coalescence_counts, dtype=np.int64)
        )

        self._alpha = cp.ones(1, dtype=self._dtype)
        self._beta_one = cp.ones(1, dtype=self._dtype)

        self._build_owned_block_sets(A_blocks)
        self._ops_up, self._ops_down = self._build_logical_ops_for(
            self._blocks_up,
            self._blocks_down,
        )
        self._ops_up_dynamic, self._ops_down_dynamic = self._build_logical_ops_for(
            self._blocks_up_dynamic,
            self._blocks_down_dynamic,
        )

        self._selector_rows["mut"], self._selector_cols["mut"] = self._build_selector_indices(sel_mut)
        self._selector_rows["miss"], self._selector_cols["miss"] = self._build_selector_indices(sel_miss)
        self._selector_row_unique["mut"] = bool(np.all(np.diff(sel_mut.indptr) <= 1))
        self._selector_row_unique["miss"] = bool(np.all(np.diff(sel_miss.indptr) <= 1))

        self._sample_perm_host = np.asarray(sample_perm, dtype=np.int64)
        self._inv_sample_perm_host = np.asarray(inv_sample_perm, dtype=np.int64)
        self._build_sample_permutations(sample_perm, inv_sample_perm)
        self._build_wavefront_level_stats()
        self._level_streams = [cp.cuda.Stream(non_blocking=True) for _ in range(self._H)]

        self._xtx_levels = []
        if self._coalescence_counts is not None:
            xtx = 2.0 * self._coalescence_counts.astype(self._dtype, copy=False)
            for h in range(self._H):
                lo, hi = int(self._level_offsets[h]), int(self._level_offsets[h + 1])
                self._xtx_levels.append(cp.asarray(xtx[lo:hi, None], dtype=self._dtype))

        self.mem_usage.reset()
        common_host = estimate_common_host_static_bytes(
            level_offsets=self._level_offsets,
            sample_perm=self._sample_perm_host,
            inv_sample_perm=self._inv_sample_perm_host,
            coalescence_counts=self._coalescence_counts,
            xtx_init=None,
        )
        self.mem_usage.host_static.level_offsets = common_host.level_offsets
        self.mem_usage.host_static.sample_perm = common_host.sample_perm
        self.mem_usage.host_static.inv_sample_perm = common_host.inv_sample_perm
        self.mem_usage.host_static.coalescence_counts = common_host.coalescence_counts
        self.mem_usage.host_static.xtx_init = common_host.xtx_init
        self.mem_usage.device_static.blocks_up = int(_count_gpu_bytes(self._blocks_up_bufs))
        self.mem_usage.device_static.blocks_down = int(_count_gpu_bytes(self._blocks_down_bufs))
        self.mem_usage.device_static.selector_mut = int(
            _count_gpu_bytes(self._selector_rows.get("mut")) + _count_gpu_bytes(self._selector_cols.get("mut"))
        )
        self.mem_usage.device_static.selector_miss = int(
            _count_gpu_bytes(self._selector_rows.get("miss")) + _count_gpu_bytes(self._selector_cols.get("miss"))
        )
        self.mem_usage.device_static.xtx_init = int(_count_gpu_bytes(self._xtx_levels))

        self._logger.info(
            (
                "CusparseBackend setup: H=%d K=%d n=%d m=%d fmt_up=%s fmt_down=%s "
                "k_hint=%s store_up=%s store_down=%s up_owner=%s down_owner=%s"
            ),
            self._H,
            self._K,
            self._n,
            self._m,
            self._fmt_up,
            self._fmt_down,
            self._k_hint,
            self._store_blocks_up,
            self._store_blocks_down,
            self._up_ops_owner,
            self._down_ops_owner,
        )
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(
                "up_ops=%d down_ops=%d",
                sum(len(ops) for ops in self._ops_up),
                sum(len(ops) for ops in self._ops_down),
            )

        self._workspace_static = None
        self._workspace_dynamic = None
        self._workspace_dynamic_k = None

        if self._k_hint is not None:
            ws = self._ensure_workspace_static()
            try:
                ws["graph_up"] = self._capture_graph(ws, Direction.UP)
                ws["graph_down"] = self._capture_graph(ws, Direction.DOWN)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to capture CUDA graphs for k_hint={self._k_hint}: {exc!r}"
                ) from exc

    def _build_selector_indices(self, selector: sp.csr_matrix):
        cp = self._cp
        rows_by_level = []
        cols_by_level = []
        off = self._level_offsets
        for h in range(self._H):
            lo, hi = int(off[h]), int(off[h + 1])
            block = selector[:, lo:hi].tocoo()
            rows_by_level.append(cp.asarray(block.row.astype(np.int32)))
            cols_by_level.append(cp.asarray(block.col.astype(np.int32)))
        return rows_by_level, cols_by_level

    def _build_sample_permutations(self, sample_perm: np.ndarray, inv_sample_perm: np.ndarray) -> None:
        cp = self._cp
        off = self._level_offsets

        # Forward scatter: level_buf[h][:ns] = X_gpu[src]
        self._fwd_scatter_ns = []
        self._fwd_scatter_src = []
        for h in range(self._H):
            lo, hi = int(off[h]), int(off[h + 1])
            ns = max(0, min(hi, self._n) - lo)
            src = cp.asarray(sample_perm[lo : lo + ns], dtype=np.int32) if ns > 0 else None
            self._fwd_scatter_ns.append(ns)
            self._fwd_scatter_src.append(src)

        # Backward gather: out[dst] = level_buf[h][src]
        self._bwd_gather_dst = []
        self._bwd_gather_src = []
        for h in range(self._H):
            lo, hi = int(off[h]), int(off[h + 1])
            mask = (inv_sample_perm >= lo) & (inv_sample_perm < hi)
            dst = cp.asarray(np.where(mask)[0], dtype=np.int32)
            src = cp.asarray(inv_sample_perm[mask] - lo, dtype=np.int32)
            self._bwd_gather_dst.append(dst)
            self._bwd_gather_src.append(src)

    def _build_owned_block_sets(self, A_blocks: List[List[sp.csr_matrix]]) -> None:
        self._desc_nnz = {}
        self._desc_meta = {}
        self._blocks_up = [[] for _ in range(self._H)]
        self._blocks_up_dynamic = [[] for _ in range(self._H)]
        self._blocks_up_bufs = [[] for _ in range(self._H)]
        self._blocks_down = [[] for _ in range(self._H)]
        self._blocks_down_dynamic = [[] for _ in range(self._H)]
        self._blocks_down_bufs = [[] for _ in range(self._H)]

        if self._store_blocks_up:
            for h in range(self._H):
                row_desc_static = []
                row_desc_dynamic = []
                row_bufs = []
                for j in range(h):
                    blk = A_blocks[h][j]
                    if blk.nnz == 0:
                        row_desc_static.append(None)
                        row_desc_dynamic.append(None)
                        row_bufs.append(None)
                        continue
                    desc_static, bufs = self._upload_sparse(blk, fmt=self._fmt_up)
                    desc_dynamic = self._make_sparse_desc_from_bufs(
                        fmt=self._fmt_up,
                        nrows=int(blk.shape[0]),
                        ncols=int(blk.shape[1]),
                        nnz=int(blk.nnz),
                        bufs=bufs,
                    )
                    row_desc_static.append(desc_static)
                    row_desc_dynamic.append(desc_dynamic)
                    row_bufs.append(bufs)
                    blk_nrows = int(blk.shape[0])
                    blk_ncols = int(blk.shape[1])
                    blk_nnz = int(blk.nnz)
                    self._desc_nnz[id(desc_static)] = blk_nnz
                    self._desc_meta[id(desc_static)] = (self._fmt_up, blk_nrows, blk_ncols, blk_nnz)
                    self._desc_nnz[id(desc_dynamic)] = blk_nnz
                    self._desc_meta[id(desc_dynamic)] = (self._fmt_up, blk_nrows, blk_ncols, blk_nnz)
                self._blocks_up[h] = row_desc_static
                self._blocks_up_dynamic[h] = row_desc_dynamic
                self._blocks_up_bufs[h] = row_bufs

        if self._store_blocks_down:
            for h in range(self._H):
                row_desc_static = []
                row_desc_dynamic = []
                row_bufs = []
                for src in range(h + 1, self._H):
                    blk = A_blocks[src][h]
                    if blk.nnz == 0:
                        row_desc_static.append(None)
                        row_desc_dynamic.append(None)
                        row_bufs.append(None)
                        continue
                    blk_t = blk.T.tocsr()
                    desc_static, bufs = self._upload_sparse(blk_t, fmt=self._fmt_down)
                    desc_dynamic = self._make_sparse_desc_from_bufs(
                        fmt=self._fmt_down,
                        nrows=int(blk_t.shape[0]),
                        ncols=int(blk_t.shape[1]),
                        nnz=int(blk_t.nnz),
                        bufs=bufs,
                    )
                    row_desc_static.append(desc_static)
                    row_desc_dynamic.append(desc_dynamic)
                    row_bufs.append(bufs)
                    blk_nrows = int(blk_t.shape[0])
                    blk_ncols = int(blk_t.shape[1])
                    blk_nnz = int(blk_t.nnz)
                    self._desc_nnz[id(desc_static)] = blk_nnz
                    self._desc_meta[id(desc_static)] = (self._fmt_down, blk_nrows, blk_ncols, blk_nnz)
                    self._desc_nnz[id(desc_dynamic)] = blk_nnz
                    self._desc_meta[id(desc_dynamic)] = (self._fmt_down, blk_nrows, blk_ncols, blk_nnz)
                self._blocks_down[h] = row_desc_static
                self._blocks_down_dynamic[h] = row_desc_dynamic
                self._blocks_down_bufs[h] = row_bufs

    def _build_logical_ops_for(
        self,
        blocks_up: list[list[object | None]],
        blocks_down: list[list[object | None]],
    ) -> tuple[list[list[_CuBlockOp]], list[list[_CuBlockOp]]]:
        ops_up = [[] for _ in range(self._H)]
        ops_down = [[] for _ in range(self._H)]

        for h in range(1, self._H):
            for j in range(h):
                if self._up_ops_owner == "up":
                    desc = blocks_up[h][j]
                    op_a = _OP_N
                else:
                    desc = blocks_down[j][h - j - 1]
                    op_a = _OP_T
                if desc is None:
                    continue
                ops_up[h].append(
                    _CuBlockOp(
                        src_level=j,
                        sp_desc=desc,
                        op_a=op_a,
                        algo=self._algo_up,
                        nnz=int(self._desc_nnz.get(id(desc), 0)),
                    )
                )

        for h in range(self._H - 1):
            for src in range(self._H - 1, h, -1):
                if self._down_ops_owner == "down":
                    desc = blocks_down[h][src - h - 1]
                    op_a = _OP_N
                else:
                    desc = blocks_up[src][h]
                    op_a = _OP_T
                if desc is None:
                    continue
                ops_down[h].append(
                    _CuBlockOp(
                        src_level=src,
                        sp_desc=desc,
                        op_a=op_a,
                        algo=self._algo_down,
                        nnz=int(self._desc_nnz.get(id(desc), 0)),
                    )
                )

        return ops_up, ops_down

    def _build_wavefront_level_stats(self) -> None:
        self._wavefront_calls_up = np.zeros(self._H, dtype=np.int32)
        self._wavefront_nnz_up = np.zeros(self._H, dtype=np.int64)
        self._wavefront_calls_down = np.zeros(self._H, dtype=np.int32)
        self._wavefront_nnz_down = np.zeros(self._H, dtype=np.int64)

        for h in range(self._H):
            self._wavefront_calls_up[h] = len(self._ops_up[h])
            self._wavefront_nnz_up[h] = int(sum(op.nnz for op in self._ops_up[h]))
            self._wavefront_calls_down[h] = len(self._ops_down[h])
            self._wavefront_nnz_down[h] = int(sum(op.nnz for op in self._ops_down[h]))

    def _print_wavefront_profile(self, direction: Direction, level_ms: np.ndarray) -> None:
        match direction:
            case Direction.UP:
                calls = self._wavefront_calls_up
                nnz = self._wavefront_nnz_up
            case Direction.DOWN:
                calls = self._wavefront_calls_down
                nnz = self._wavefront_nnz_down
            case _:
                raise ValueError(f"Unknown direction for wavefront profile: {direction!r}")

        records = [
            (h, float(level_ms[h]), int(calls[h]), int(nnz[h]))
            for h in range(self._H)
            if calls[h] > 0
        ]
        if not records:
            return

        total_ms = sum(ms for _, ms, _, _ in records)
        self._logger.debug(
            "wavefront[%s] levels=%d total=%.3fms",
            direction.value,
            len(records),
            total_ms,
        )
        for h, ms, call_count, nnz_count in records:
            pct = (100.0 * ms / total_ms) if total_ms > 0.0 else 0.0
            self._logger.debug(
                "  level=%2d ms=%.3f (%5.1f%%) calls=%3d nnz=%d",
                h,
                ms,
                pct,
                call_count,
                nnz_count,
            )

    # ------------------------------------------------------------------
    # sparse upload
    # ------------------------------------------------------------------

    def _upload_sparse(self, A_scipy, *, fmt: str):
        cp = self._cp
        cslib = self._cslib
        cdt = self._cuda_dtype

        match fmt:
            case "csr":
                A = sp.csr_matrix(A_scipy).astype(self._dtype)
                indptr = cp.asarray(A.indptr.astype(np.int32))
                indices = cp.asarray(A.indices.astype(np.int32))
                data = cp.asarray(A.data)
                desc = cslib.create_csr(
                    A.shape[0],
                    A.shape[1],
                    A.nnz,
                    indptr.data.ptr,
                    indices.data.ptr,
                    data.data.ptr,
                    cdt,
                )
                return desc, (indptr, indices, data)
            case "csc":
                A = A_scipy.tocsc().astype(self._dtype)
                indptr = cp.asarray(A.indptr.astype(np.int32))
                indices = cp.asarray(A.indices.astype(np.int32))
                data = cp.asarray(A.data)
                desc = cslib.create_csc(
                    A.shape[0],
                    A.shape[1],
                    A.nnz,
                    indptr.data.ptr,
                    indices.data.ptr,
                    data.data.ptr,
                    cdt,
                )
                return desc, (indptr, indices, data)
            case "coo":
                A = A_scipy.tocoo().astype(self._dtype)
                row_idx = cp.asarray(A.row.astype(np.int32))
                col_idx = cp.asarray(A.col.astype(np.int32))
                data = cp.asarray(A.data)
                desc = cslib.create_coo(
                    A.shape[0],
                    A.shape[1],
                    A.nnz,
                    row_idx.data.ptr,
                    col_idx.data.ptr,
                    data.data.ptr,
                    cdt,
                )
                return desc, (row_idx, col_idx, data)
            case _:
                raise ValueError(f"Unknown sparse format: {fmt!r}")

    def _make_sparse_desc_from_bufs(
        self,
        *,
        fmt: str,
        nrows: int,
        ncols: int,
        nnz: int,
        bufs: tuple[object, ...],
    ):
        cslib = self._cslib
        cdt = self._cuda_dtype
        match fmt:
            case "csr":
                indptr, indices, data = bufs
                return cslib.create_csr(
                    nrows,
                    ncols,
                    nnz,
                    indptr.data.ptr,
                    indices.data.ptr,
                    data.data.ptr,
                    cdt,
                )
            case "csc":
                indptr, indices, data = bufs
                return cslib.create_csc(
                    nrows,
                    ncols,
                    nnz,
                    indptr.data.ptr,
                    indices.data.ptr,
                    data.data.ptr,
                    cdt,
                )
            case "coo":
                row_idx, col_idx, data = bufs
                return cslib.create_coo(
                    nrows,
                    ncols,
                    nnz,
                    row_idx.data.ptr,
                    col_idx.data.ptr,
                    data.data.ptr,
                    cdt,
                )
            case _:
                raise ValueError(f"Unknown sparse format for descriptor clone: {fmt!r}")

    # ------------------------------------------------------------------
    # workspace and kernels
    # ------------------------------------------------------------------

    def _preprocess_one(self, sp_desc, B_desc, C_desc, op_a, algo):
        a1 = self._alpha.data.ptr
        b1 = self._beta_one.data.ptr
        ext = self._cslib.spmm_buffer_size(
            self._cp,
            algo,
            op_a,
            _OP_N,
            a1,
            sp_desc,
            B_desc,
            b1,
            C_desc,
            self._cuda_dtype,
        )
        self._cslib.spmm_preprocess(
            algo,
            op_a,
            _OP_N,
            a1,
            sp_desc,
            B_desc,
            b1,
            C_desc,
            self._cuda_dtype,
            ext.data.ptr,
        )
        return ext

    def _create_workspace(
        self,
        k: int,
        *,
        ops_up: list[list[_CuBlockOp]],
        ops_down: list[list[_CuBlockOp]],
    ):
        cp = self._cp
        off = self._level_offsets

        level_bufs = []
        for h in range(self._H):
            sz = int(off[h + 1]) - int(off[h])
            level_bufs.append(cp.zeros((sz, k), dtype=self._dtype, order="C"))
        level_dn = [
            self._cslib.create_dnmat(buf.shape[0], k, buf.shape[1], buf.data.ptr, self._cuda_dtype)
            for buf in level_bufs
        ]

        # UP/DOWN preprocess ext buffers aligned to logical ops.
        ops_up_ext = []
        for h in range(self._H):
            row = []
            for op in ops_up[h]:
                row.append(
                    self._preprocess_one(
                        op.sp_desc,
                        level_dn[op.src_level],
                        level_dn[h],
                        op.op_a,
                        op.algo,
                    )
                )
            ops_up_ext.append(row)

        ops_down_ext = []
        for h in range(self._H):
            row = []
            for op in ops_down[h]:
                row.append(
                    self._preprocess_one(
                        op.sp_desc,
                        level_dn[op.src_level],
                        level_dn[h],
                        op.op_a,
                        op.algo,
                    )
                )
            ops_down_ext.append(row)

        gather_temp = []
        for h in range(self._H):
            ng = int(self._bwd_gather_dst[h].size)
            gather_temp.append(cp.zeros((ng, k), dtype=self._dtype, order="C") if ng > 0 else None)

        ws = {
            "k": k,
            "ops_up": ops_up,
            "ops_down": ops_down,
            "level_bufs": level_bufs,
            "level_dn": level_dn,
            "ops_up_ext": ops_up_ext,
            "ops_down_ext": ops_down_ext,
            "fwd_input": cp.zeros((self._n, k), dtype=self._dtype, order="C"),
            "bwd_input_mut": cp.zeros((self._m, k), dtype=self._dtype, order="C"),
            "bwd_input_miss": None,
            "mut_out": cp.zeros((self._m, k), dtype=self._dtype, order="C"),
            "miss_out": cp.zeros((self._m, k), dtype=self._dtype, order="C"),
            "sample_out": cp.zeros((self._n, k), dtype=self._dtype, order="C"),
            "gather_temp": gather_temp,
            "init_vec": cp.zeros((1, k), dtype=self._dtype, order="C"),
            "init_matrix": None,
            "up_fork_event": cp.cuda.Event(),
            "down_fork_event": cp.cuda.Event(),
            "up_ready_events": [cp.cuda.Event() for _ in range(self._H)],
            "down_ready_events": [cp.cuda.Event() for _ in range(self._H)],
            "up_start_events": [cp.cuda.Event() for _ in range(self._H)],
            "up_end_events": [cp.cuda.Event() for _ in range(self._H)],
            "down_start_events": [cp.cuda.Event() for _ in range(self._H)],
            "down_end_events": [cp.cuda.Event() for _ in range(self._H)],
            "graph_up": None,
            "graph_down": None,
        }
        self._initialize_workspace_memory_bytes(ws)
        self._logger.debug("workspace ready for k=%d", k)
        return ws

    def _ensure_workspace_static(self):
        if self._k_hint is None:
            raise RuntimeError("Static workspace requested with k_hint=None")
        if self._workspace_static is None:
            self._workspace_static = self._create_workspace(
                self._k_hint,
                ops_up=self._ops_up,
                ops_down=self._ops_down,
            )
        return self._workspace_static

    def _destroy_workspace(self, ws) -> None:
        if ws is None:
            return
        for desc in ws.get("level_dn", []):
            if desc is not None:
                self._cslib.destroy_dn_mat(desc)

    def _ensure_workspace_dynamic(self, k: int):
        if self._workspace_dynamic is not None and self._workspace_dynamic_k == int(k):
            return self._workspace_dynamic

        if self._workspace_dynamic is not None:
            self._destroy_workspace(self._workspace_dynamic)
            self._workspace_dynamic = None
            self._workspace_dynamic_k = None

        self._workspace_dynamic = self._create_workspace(
            int(k),
            ops_up=self._ops_up_dynamic,
            ops_down=self._ops_down_dynamic,
        )
        self._workspace_dynamic_k = int(k)
        return self._workspace_dynamic

    def _initialize_workspace_memory_bytes(self, ws) -> None:
        ws["_mem_level_buffers"] = int(sum(int(buf.nbytes) for buf in ws["level_bufs"]))
        ws["_mem_inputs_up"] = int(ws["fwd_input"].nbytes)
        ws["_mem_inputs_down"] = int(ws["bwd_input_mut"].nbytes)
        ws["_mem_inputs_down_miss"] = 0
        ws["_mem_outputs_up_mut"] = int(ws["mut_out"].nbytes)
        ws["_mem_outputs_up_miss"] = int(ws["miss_out"].nbytes)
        ws["_mem_outputs_down"] = int(ws["sample_out"].nbytes)
        ws["_mem_init_matrix"] = 0
        ws["_mem_aux_up"] = int(_count_gpu_bytes(ws["ops_up_ext"]) + ws["init_vec"].nbytes)
        ws["_mem_aux_down"] = int(
            _count_gpu_bytes(ws["ops_down_ext"])
            + _count_gpu_bytes(ws["gather_temp"])
            + ws["init_vec"].nbytes
        )

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

    def _spmm(self, sp_desc, B_desc, C_desc, ext_buf, op_a, algo):
        self._cslib.spmm(
            algo,
            op_a,
            _OP_N,
            self._alpha.data.ptr,
            sp_desc,
            B_desc,
            self._beta_one.data.ptr,
            C_desc,
            self._cuda_dtype,
            ext_buf.data.ptr,
        )

    def _zero_level_buffers(self, ws) -> None:
        with self._stream:
            for buf in ws["level_bufs"]:
                buf.fill(0)

    def _scatter_samples_to_levels(self, X_gpu, ws) -> None:
        cp = self._cp
        with self._stream:
            for h in range(self._H):
                ns = self._fwd_scatter_ns[h]
                if ns > 0:
                    cp.take(X_gpu, self._fwd_scatter_src[h], axis=0, out=ws["level_bufs"][h][:ns])

    def _selector_scatter_add(self, selector: str, X_gpu, ws) -> None:
        cp = self._cp
        rows_by_level = self._selector_rows[selector]
        cols_by_level = self._selector_cols[selector]
        with self._stream:
            for h in range(self._H):
                rows = rows_by_level[h]
                if rows.size > 0:
                    cp.add.at(ws["level_bufs"][h], cols_by_level[h], X_gpu[rows])

    def _selector_gather(self, selector: str, ws, out_gpu) -> None:
        cp = self._cp
        rows_by_level = self._selector_rows[selector]
        cols_by_level = self._selector_cols[selector]
        unique_rows = self._selector_row_unique[selector]
        with self._stream:
            out_gpu.fill(0)
            for h in range(self._H):
                rows = rows_by_level[h]
                if rows.size > 0:
                    values = ws["level_bufs"][h][cols_by_level[h]]
                    if unique_rows:
                        out_gpu[rows] = values
                    else:
                        cp.add.at(out_gpu, rows, values)

    def _ensure_init_matrix_buffer(self, ws):
        if ws["init_matrix"] is None:
            ws["init_matrix"] = self._cp.zeros((self._K, ws["k"]), dtype=self._dtype, order="C")
            ws["_mem_init_matrix"] = int(ws["init_matrix"].nbytes)
        return ws["init_matrix"]

    def _stage_init_payload(self, ws, init_mode: InitMode, init_payload: np.ndarray | None) -> None:
        match init_mode:
            case InitMode.VECTOR:
                if init_payload is None:
                    raise ValueError("init vector payload is required for init_mode=vector")
                with self._stream:
                    ws["init_vec"][0].set(init_payload, stream=self._stream)
            case InitMode.MATRIX:
                if init_payload is None:
                    raise ValueError("init matrix payload is required for init_mode=matrix")
                with self._stream:
                    self._ensure_init_matrix_buffer(ws).set(init_payload, stream=self._stream)
            case InitMode.NONE | InitMode.XTX:
                return
            case _:
                raise ValueError(f"Unknown init mode: {init_mode!r}")

    def _apply_init(self, ws, init_mode: InitMode) -> None:
        if init_mode == InitMode.NONE:
            return

        with self._stream:
            match init_mode:
                case InitMode.XTX:
                    if not self._xtx_levels:
                        raise ValueError("init_mode=xtx requires GRG coalescence counts")
                    for h in range(self._H):
                        ws["level_bufs"][h] += self._xtx_levels[h]
                case InitMode.VECTOR:
                    vec = ws["init_vec"]
                    for h in range(self._H):
                        ws["level_bufs"][h] += vec
                case InitMode.MATRIX:
                    init_matrix = self._ensure_init_matrix_buffer(ws)
                    off = self._level_offsets
                    for h in range(self._H):
                        lo, hi = int(off[h]), int(off[h + 1])
                        ws["level_bufs"][h] += init_matrix[lo:hi]
                case _:
                    raise ValueError(f"Unknown init_mode: {init_mode!r}")

    def _ensure_miss_input_buffer(self, ws):
        if ws["bwd_input_miss"] is None:
            ws["bwd_input_miss"] = self._cp.zeros((self._m, ws["k"]), dtype=self._dtype, order="C")
            ws["_mem_inputs_down_miss"] = int(ws["bwd_input_miss"].nbytes)
        return ws["bwd_input_miss"]

    def _wavefront_up(self, ws) -> None:
        cslib = self._cslib
        streams = self._level_streams
        ready = ws["up_ready_events"]
        ops_up = ws["ops_up"]

        with self._stream:
            ws["up_fork_event"].record(self._stream)
        for h in range(self._H):
            streams[h].wait_event(ws["up_fork_event"])

        ready[0].record(streams[0])
        for h in range(1, self._H):
            s = streams[h]
            with s:
                for idx, op in enumerate(ops_up[h]):
                    s.wait_event(ready[op.src_level])
                    cslib.set_stream(s.ptr)
                    self._spmm(
                        op.sp_desc,
                        ws["level_dn"][op.src_level],
                        ws["level_dn"][h],
                        ws["ops_up_ext"][h][idx],
                        op_a=op.op_a,
                        algo=op.algo,
                    )
                ready[h].record(s)

        with self._stream:
            for h in range(self._H):
                self._stream.wait_event(ready[h])

    def _wavefront_up_timed(self, ws) -> np.ndarray:
        cp = self._cp
        cslib = self._cslib
        streams = self._level_streams
        ready = ws["up_ready_events"]
        start_events = ws["up_start_events"]
        end_events = ws["up_end_events"]
        ops_up = ws["ops_up"]
        level_ms = np.zeros(self._H, dtype=np.float64)

        with self._stream:
            ws["up_fork_event"].record(self._stream)
        for h in range(self._H):
            streams[h].wait_event(ws["up_fork_event"])

        ready[0].record(streams[0])
        for h in range(1, self._H):
            s = streams[h]
            with s:
                if len(ops_up[h]) > 0:
                    start_events[h].record(s)
                for idx, op in enumerate(ops_up[h]):
                    s.wait_event(ready[op.src_level])
                    cslib.set_stream(s.ptr)
                    self._spmm(
                        op.sp_desc,
                        ws["level_dn"][op.src_level],
                        ws["level_dn"][h],
                        ws["ops_up_ext"][h][idx],
                        op_a=op.op_a,
                        algo=op.algo,
                    )
                if len(ops_up[h]) > 0:
                    end_events[h].record(s)
                ready[h].record(s)

        with self._stream:
            for h in range(self._H):
                self._stream.wait_event(ready[h])
        self._stream.synchronize()
        for h in range(1, self._H):
            if len(ops_up[h]) > 0:
                level_ms[h] = cp.cuda.get_elapsed_time(start_events[h], end_events[h])
        return level_ms

    def _wavefront_down(self, ws) -> None:
        cslib = self._cslib
        streams = self._level_streams
        ready = ws["down_ready_events"]
        ops_down = ws["ops_down"]

        with self._stream:
            ws["down_fork_event"].record(self._stream)
        for h in range(self._H):
            streams[h].wait_event(ws["down_fork_event"])

        ready[self._H - 1].record(streams[self._H - 1])
        for h in range(self._H - 2, -1, -1):
            s = streams[h]
            with s:
                for idx, op in enumerate(ops_down[h]):
                    s.wait_event(ready[op.src_level])
                    cslib.set_stream(s.ptr)
                    self._spmm(
                        op.sp_desc,
                        ws["level_dn"][op.src_level],
                        ws["level_dn"][h],
                        ws["ops_down_ext"][h][idx],
                        op_a=op.op_a,
                        algo=op.algo,
                    )
                ready[h].record(s)

        with self._stream:
            for h in range(self._H):
                self._stream.wait_event(ready[h])

    def _wavefront_down_timed(self, ws) -> np.ndarray:
        cp = self._cp
        cslib = self._cslib
        streams = self._level_streams
        ready = ws["down_ready_events"]
        start_events = ws["down_start_events"]
        end_events = ws["down_end_events"]
        ops_down = ws["ops_down"]
        level_ms = np.zeros(self._H, dtype=np.float64)

        with self._stream:
            ws["down_fork_event"].record(self._stream)
        for h in range(self._H):
            streams[h].wait_event(ws["down_fork_event"])

        ready[self._H - 1].record(streams[self._H - 1])
        for h in range(self._H - 2, -1, -1):
            s = streams[h]
            with s:
                if len(ops_down[h]) > 0:
                    start_events[h].record(s)
                for idx, op in enumerate(ops_down[h]):
                    s.wait_event(ready[op.src_level])
                    cslib.set_stream(s.ptr)
                    self._spmm(
                        op.sp_desc,
                        ws["level_dn"][op.src_level],
                        ws["level_dn"][h],
                        ws["ops_down_ext"][h][idx],
                        op_a=op.op_a,
                        algo=op.algo,
                    )
                if len(ops_down[h]) > 0:
                    end_events[h].record(s)
                ready[h].record(s)

        with self._stream:
            for h in range(self._H):
                self._stream.wait_event(ready[h])
        self._stream.synchronize()
        for h in range(self._H - 1):
            if len(ops_down[h]) > 0:
                level_ms[h] = cp.cuda.get_elapsed_time(start_events[h], end_events[h])
        return level_ms

    def _gather_samples(self, ws, out_gpu) -> None:
        cp = self._cp
        with self._stream:
            out_gpu.fill(0)
            for h in range(self._H):
                dst = self._bwd_gather_dst[h]
                if dst.size == 0:
                    continue
                cp.take(ws["level_bufs"][h], self._bwd_gather_src[h], axis=0, out=ws["gather_temp"][h])
                out_gpu[dst] = ws["gather_temp"][h]

    def _execute_up_dynamic(
        self,
        ws,
        *,
        init_mode: InitMode,
        need_miss_output: bool,
        track_wave: bool,
    ) -> np.ndarray | None:
        self._zero_level_buffers(ws)
        self._scatter_samples_to_levels(ws["fwd_input"], ws)
        self._apply_init(ws, init_mode)
        level_ms = self._wavefront_up_timed(ws) if track_wave else None
        if level_ms is None:
            self._wavefront_up(ws)
        self._selector_gather("mut", ws, ws["mut_out"])
        if need_miss_output:
            self._selector_gather("miss", ws, ws["miss_out"])
        return level_ms

    def _execute_down_dynamic(
        self,
        ws,
        *,
        init_mode: InitMode,
        has_miss_input: bool,
        track_wave: bool,
    ) -> np.ndarray | None:
        self._zero_level_buffers(ws)
        self._selector_scatter_add("mut", ws["bwd_input_mut"], ws)
        if has_miss_input:
            self._selector_scatter_add("miss", ws["bwd_input_miss"], ws)
        self._apply_init(ws, init_mode)
        level_ms = self._wavefront_down_timed(ws) if track_wave else None
        if level_ms is None:
            self._wavefront_down(ws)
        self._gather_samples(ws, ws["sample_out"])
        return level_ms

    def _capture_graph(self, ws, direction: Direction):
        # Prime kernels before capture.
        self._zero_level_buffers(ws)
        match direction:
            case Direction.UP:
                self._wavefront_up(ws)
            case Direction.DOWN:
                self._wavefront_down(ws)
            case _:
                raise ValueError(f"Unknown capture direction: {direction!r}")
        self._stream.synchronize()

        self._stream.begin_capture()
        try:
            match direction:
                case Direction.UP:
                    self._wavefront_up(ws)
                case Direction.DOWN:
                    self._wavefront_down(ws)
                case _:
                    raise ValueError(f"Unknown capture direction: {direction!r}")
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

    def _can_use_graph(self, runtime_k: int, track_wave: bool, direction: Direction) -> bool:
        if self._k_hint is None:
            return False
        if runtime_k != self._k_hint:
            warnings.warn(
                (
                    f"cuSPARSE {direction.value} graph disabled: runtime k={runtime_k} "
                    f"does not match k_hint={self._k_hint}."
                ),
                RuntimeWarning,
                stacklevel=3,
            )
            return False
        if track_wave:
            warnings.warn(
                (
                    f"cuSPARSE {direction.value} graph disabled: per-level wavefront timing "
                    "requires dynamic execution."
                ),
                RuntimeWarning,
                stacklevel=3,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run_up(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
        need_miss_output: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        total_t0 = perf_counter()

        t0 = perf_counter()
        X = np.asarray(primary, dtype=self._dtype, order="C")
        if X.ndim != 2:
            raise ValueError(f"primary input must be 2D, got shape {X.shape}")
        if X.shape[0] != self._n:
            raise ValueError(f"UP primary input must have {self._n} rows, got {X.shape[0]}")
        k = int(X.shape[1])
        parse_ms = (perf_counter() - t0) * 1000.0

        init_mode = parse_init_mode(init_mode)
        t0 = perf_counter()
        init_payload = self._validate_init(init_mode, init, k)
        init_parse_ms = (perf_counter() - t0) * 1000.0

        if self._k_hint is not None and k == self._k_hint:
            ws = self._ensure_workspace_static()
        else:
            ws = self._ensure_workspace_dynamic(k)
        track_wave = self._logger.isEnabledFor(logging.DEBUG)

        t0 = perf_counter()
        ws["fwd_input"].set(X, stream=self._stream)
        self._stream.synchronize()
        h2d_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        self._stage_init_payload(ws, init_mode, init_payload)
        self._stream.synchronize()
        init_stage_ms = (perf_counter() - t0) * 1000.0

        mode = "dynamic"
        t0 = perf_counter()
        level_ms = None
        if self._can_use_graph(k, track_wave, Direction.UP):
            graph = ws.get("graph_up")
            if graph is None:
                raise RuntimeError("Missing captured UP CUDA graph for configured k_hint")
            mode = "graph"
            self._zero_level_buffers(ws)
            self._scatter_samples_to_levels(ws["fwd_input"], ws)
            self._apply_init(ws, init_mode)
            with self._stream:
                graph.launch(self._stream)
            self._selector_gather("mut", ws, ws["mut_out"])
            if need_miss_output:
                self._selector_gather("miss", ws, ws["miss_out"])
        else:
            # Graphs are shape-specialized; mismatched k must run dynamic kernels.
            level_ms = self._execute_up_dynamic(
                ws,
                init_mode=init_mode,
                need_miss_output=need_miss_output,
                track_wave=track_wave,
            )

        self._stream.synchronize()
        compute_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        out_mut = ws["mut_out"].get()
        d2h_mut_ms = (perf_counter() - t0) * 1000.0

        out_miss = None
        d2h_miss_ms = 0.0
        if need_miss_output:
            t0 = perf_counter()
            out_miss = ws["miss_out"].get()
            d2h_miss_ms = (perf_counter() - t0) * 1000.0

        if self._logger.isEnabledFor(logging.INFO):
            total_ms = (perf_counter() - total_t0) * 1000.0
            self._logger.info(
                (
                    "cusparse.run_up k=%d mode=%s init=%s miss_out=%s: parse=%.3fms init_parse=%.3fms "
                    "H2D=%.3fms init_stage=%.3fms compute=%.3fms D2H_mut=%.3fms D2H_miss=%.3fms total=%.3fms"
                ),
                k,
                mode,
                init_mode.value,
                need_miss_output,
                parse_ms,
                init_parse_ms,
                h2d_ms,
                init_stage_ms,
                compute_ms,
                d2h_mut_ms,
                d2h_miss_ms,
                total_ms,
            )
            if track_wave and level_ms is not None:
                self._print_wavefront_profile(Direction.UP, level_ms)
        self.mem_usage.record(
            stage="run_up",
            runtime_k=k,
            host_runtime=RuntimeBytes(
                level_buffers=0,
                inputs=int(X.nbytes),
                outputs=int(out_mut.nbytes + (0 if out_miss is None else out_miss.nbytes)),
                aux=0 if init_payload is None else int(init_payload.nbytes),
            ),
            device_runtime=RuntimeBytes(
                level_buffers=int(ws["_mem_level_buffers"]),
                inputs=int(ws["_mem_inputs_up"]),
                outputs=int(ws["_mem_outputs_up_mut"] + (ws["_mem_outputs_up_miss"] if need_miss_output else 0)),
                aux=int(ws["_mem_aux_up"] + ws["_mem_init_matrix"] + (0 if level_ms is None else level_ms.nbytes)),
            ),
            meta={"direction": "up", "need_miss_output": bool(need_miss_output), "mode": mode},
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
        total_t0 = perf_counter()

        t0 = perf_counter()
        X = np.asarray(primary, dtype=self._dtype, order="C")
        if X.ndim != 2:
            raise ValueError(f"primary input must be 2D, got shape {X.shape}")
        if X.shape[0] != self._m:
            raise ValueError(f"DOWN primary input must have {self._m} rows, got {X.shape[0]}")
        k = int(X.shape[1])
        parse_ms = (perf_counter() - t0) * 1000.0

        miss_arr = None
        if miss is not None:
            miss_arr = np.asarray(miss, dtype=self._dtype, order="C")
            if miss_arr.shape != (self._m, k):
                raise ValueError(f"miss input must have shape ({self._m}, {k}), got {miss_arr.shape}")

        init_mode = parse_init_mode(init_mode)
        t0 = perf_counter()
        init_payload = self._validate_init(init_mode, init, k)
        init_parse_ms = (perf_counter() - t0) * 1000.0

        if self._k_hint is not None and k == self._k_hint:
            ws = self._ensure_workspace_static()
        else:
            ws = self._ensure_workspace_dynamic(k)
        track_wave = self._logger.isEnabledFor(logging.DEBUG)

        t0 = perf_counter()
        ws["bwd_input_mut"].set(X, stream=self._stream)
        self._stream.synchronize()
        h2d_mut_ms = (perf_counter() - t0) * 1000.0

        h2d_miss_ms = 0.0
        if miss_arr is not None:
            t0 = perf_counter()
            miss_buf = self._ensure_miss_input_buffer(ws)
            miss_buf.set(miss_arr, stream=self._stream)
            self._stream.synchronize()
            h2d_miss_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        self._stage_init_payload(ws, init_mode, init_payload)
        self._stream.synchronize()
        init_stage_ms = (perf_counter() - t0) * 1000.0

        mode = "dynamic"
        t0 = perf_counter()
        level_ms = None
        if self._can_use_graph(k, track_wave, Direction.DOWN):
            graph = ws.get("graph_down")
            if graph is None:
                raise RuntimeError("Missing captured DOWN CUDA graph for configured k_hint")
            mode = "graph"
            self._zero_level_buffers(ws)
            self._selector_scatter_add("mut", ws["bwd_input_mut"], ws)
            if miss_arr is not None:
                self._selector_scatter_add("miss", ws["bwd_input_miss"], ws)
            self._apply_init(ws, init_mode)
            with self._stream:
                graph.launch(self._stream)
            self._gather_samples(ws, ws["sample_out"])
        else:
            # Graphs are shape-specialized; mismatched k must run dynamic kernels.
            level_ms = self._execute_down_dynamic(
                ws,
                init_mode=init_mode,
                has_miss_input=(miss_arr is not None),
                track_wave=track_wave,
            )

        self._stream.synchronize()
        compute_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        out = ws["sample_out"].get()
        d2h_ms = (perf_counter() - t0) * 1000.0

        if self._logger.isEnabledFor(logging.INFO):
            total_ms = (perf_counter() - total_t0) * 1000.0
            self._logger.info(
                (
                    "cusparse.run_down k=%d mode=%s init=%s miss_in=%s: parse=%.3fms init_parse=%.3fms "
                    "H2D_mut=%.3fms H2D_miss=%.3fms init_stage=%.3fms compute=%.3fms D2H=%.3fms total=%.3fms"
                ),
                k,
                mode,
                init_mode.value,
                miss_arr is not None,
                parse_ms,
                init_parse_ms,
                h2d_mut_ms,
                h2d_miss_ms,
                init_stage_ms,
                compute_ms,
                d2h_ms,
                total_ms,
            )
            if track_wave and level_ms is not None:
                self._print_wavefront_profile(Direction.DOWN, level_ms)
        self.mem_usage.record(
            stage="run_down",
            runtime_k=k,
            host_runtime=RuntimeBytes(
                level_buffers=0,
                inputs=int(X.nbytes + (0 if miss_arr is None else miss_arr.nbytes)),
                outputs=int(out.nbytes),
                aux=0 if init_payload is None else int(init_payload.nbytes),
            ),
            device_runtime=RuntimeBytes(
                level_buffers=int(ws["_mem_level_buffers"]),
                inputs=int(ws["_mem_inputs_down"] + (ws["_mem_inputs_down_miss"] if miss_arr is not None else 0)),
                outputs=int(ws["_mem_outputs_down"]),
                aux=int(ws["_mem_aux_down"] + ws["_mem_init_matrix"] + (0 if level_ms is None else level_ms.nbytes)),
            ),
            meta={"direction": "down", "has_miss_input": bool(miss_arr is not None), "mode": mode},
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

        for row in self._blocks_up:
            for desc in row:
                if desc is None:
                    continue
                meta = self._desc_meta.get(id(desc))
                if meta is None:
                    raise ValueError("Missing descriptor metadata for UP block estimate")
                fmt, nrows, ncols, nnz = meta
                device.blocks_up += estimate_sparse_payload_bytes(
                    fmt=fmt,
                    nrows=nrows,
                    ncols=ncols,
                    nnz=nnz,
                    data_itemsize=data_itemsize,
                    index_itemsize=index_itemsize,
                )

        for row in self._blocks_down:
            for desc in row:
                if desc is None:
                    continue
                meta = self._desc_meta.get(id(desc))
                if meta is None:
                    raise ValueError("Missing descriptor metadata for DOWN block estimate")
                fmt, nrows, ncols, nnz = meta
                device.blocks_down += estimate_sparse_payload_bytes(
                    fmt=fmt,
                    nrows=nrows,
                    ncols=ncols,
                    nnz=nnz,
                    data_itemsize=data_itemsize,
                    index_itemsize=index_itemsize,
                )

        mut_nnz = sum(int(rows.size) for rows in self._selector_rows["mut"])
        miss_nnz = sum(int(rows.size) for rows in self._selector_rows["miss"])
        device.selector_mut = int(mut_nnz * 2 * index_itemsize)
        device.selector_miss = int(miss_nnz * 2 * index_itemsize)
        if self._coalescence_counts is not None:
            device.xtx_init = int(self._coalescence_counts.size * data_itemsize)
        return host, device

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------

    def __del__(self):
        cslib = getattr(self, "_cslib", None)
        if cslib is None:
            return

        # Dense descriptors.
        self._destroy_workspace(getattr(self, "_workspace_static", None))
        self._destroy_workspace(getattr(self, "_workspace_dynamic", None))

        # Sparse descriptors (both static and dynamic descriptor sets).
        for row in getattr(self, "_blocks_up", []):
            for desc in row:
                if desc is not None:
                    cslib.destroy_sp_mat(desc)
        for row in getattr(self, "_blocks_up_dynamic", []):
            for desc in row:
                if desc is not None:
                    cslib.destroy_sp_mat(desc)
        for row in getattr(self, "_blocks_down", []):
            for desc in row:
                if desc is not None:
                    cslib.destroy_sp_mat(desc)
        for row in getattr(self, "_blocks_down_dynamic", []):
            for desc in row:
                if desc is not None:
                    cslib.destroy_sp_mat(desc)

        cslib.destroy()
