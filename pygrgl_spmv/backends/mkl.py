"""MKL backend — Intel MKL-accelerated fused GRG matmul traversal."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from time import perf_counter
from typing import List

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends import Backend, estimate_common_host_static_bytes, estimate_sparse_payload_bytes
from pygrgl_spmv.backends.memory import RuntimeBytes, StaticBytes
from pygrgl_spmv.backends.mkl_utils import (
    MklSparseHandle,
    mkl_get_max_threads,
    mkl_set_num_threads,
)
from pygrgl_spmv.backends.types import Direction, InitMode, parse_init_mode


@dataclass(frozen=True)
class _MklBlockOp:
    """One sparse block application in a level wavefront."""

    src_level: int
    handle: MklSparseHandle
    transpose: bool
    nnz: int


def _handle_bytes(handle: MklSparseHandle | None) -> int:
    if handle is None:
        return 0
    mat = getattr(handle, "_mat", None)
    if mat is None:
        return 0
    total = int(getattr(mat, "data").nbytes)
    if hasattr(mat, "indices"):
        total += int(mat.indices.nbytes)
    if hasattr(mat, "indptr"):
        total += int(mat.indptr.nbytes)
    if hasattr(mat, "row"):
        total += int(mat.row.nbytes)
    if hasattr(mat, "col"):
        total += int(mat.col.nbytes)
    return total


def _estimate_handle_payload_bytes(handle: MklSparseHandle) -> int:
    mat = getattr(handle, "_mat", None)
    if mat is None:
        return 0
    fmt = str(getattr(handle, "_fmt", "csr"))
    nrows, ncols = mat.shape
    nnz = int(mat.nnz)
    data_itemsize = int(np.dtype(mat.data.dtype).itemsize)

    if fmt in {"csr", "csc"}:
        if hasattr(mat, "indices"):
            index_itemsize = int(np.dtype(mat.indices.dtype).itemsize)
        elif hasattr(mat, "indptr"):
            index_itemsize = int(np.dtype(mat.indptr.dtype).itemsize)
        else:
            raise ValueError(f"MKL sparse matrix in {fmt} is missing indices/indptr arrays")
    elif fmt == "coo":
        if hasattr(mat, "row"):
            index_itemsize = int(np.dtype(mat.row.dtype).itemsize)
        elif hasattr(mat, "col"):
            index_itemsize = int(np.dtype(mat.col.dtype).itemsize)
        else:
            raise ValueError("MKL COO matrix is missing row/col arrays")
    else:
        raise ValueError(f"Unknown MKL sparse format {fmt!r} for static estimate")

    return estimate_sparse_payload_bytes(
        fmt=fmt,
        nrows=int(nrows),
        ncols=int(ncols),
        nnz=nnz,
        data_itemsize=data_itemsize,
        index_itemsize=index_itemsize,
    )


class MklBackend(Backend):
    """
    MKL-accelerated backend using the Inspector-Executor Sparse BLAS API.

    Persistent MKL handles are created once in setup().
    Each run_up()/run_down() call performs one fused traversal.
    """

    def __init__(
        self,
        n_threads: int = 0,
        fmt_up: str | None = "csr",
        fmt_down: str | None = None,
        k_hint: int | None = None,
        log_level: str = "WARNING",
    ):
        super().__init__(
            fmt_up=fmt_up,
            fmt_down=fmt_down,
            k_hint=k_hint,
            log_level=log_level,
        )
        self._n_threads = os.cpu_count() if n_threads == 0 else n_threads
        if self._fmt_up not in {"csr", "csc", "coo"}:
            raise ValueError(f"Unsupported MKL fmt_up={self._fmt_up!r}; expected csr/csc/coo")
        if self._fmt_down not in {"csr", "csc", "coo"}:
            raise ValueError(f"Unsupported MKL fmt_down={self._fmt_down!r}; expected csr/csc/coo")
        self._coalescence_counts = None

    def _build_owned_block_handles(self, A_blocks: List[List[sp.csr_matrix]]) -> None:
        num_levels = len(self._level_offsets) - 1
        self._blocks_up = [[] for _ in range(num_levels)]
        self._blocks_down = [[] for _ in range(num_levels)]

        if self._store_blocks_up:
            for h in range(num_levels):
                row: list[MklSparseHandle | None] = []
                for j in range(h):
                    blk = A_blocks[h][j]
                    row.append(None if blk.nnz == 0 else MklSparseHandle(blk, self._fmt_up))
                self._blocks_up[h] = row

        if self._store_blocks_down:
            for h in range(num_levels):
                row = []
                for src in range(h + 1, num_levels):
                    blk_t = A_blocks[src][h].T.tocsr()
                    row.append(None if blk_t.nnz == 0 else MklSparseHandle(blk_t, self._fmt_down))
                self._blocks_down[h] = row

    def _build_logical_ops(self) -> None:
        num_levels = len(self._level_offsets) - 1
        self._ops_up: list[list[_MklBlockOp]] = [[] for _ in range(num_levels)]
        self._ops_down: list[list[_MklBlockOp]] = [[] for _ in range(num_levels)]

        for h in range(1, num_levels):
            for j in range(h):
                if self._up_ops_owner == "up":
                    handle = self._blocks_up[h][j]
                    transpose = False
                else:
                    handle = self._blocks_down[j][h - j - 1]
                    transpose = True
                if handle is None:
                    continue
                self._ops_up[h].append(
                    _MklBlockOp(
                        src_level=j,
                        handle=handle,
                        transpose=transpose,
                        nnz=int(handle.nnz),
                    )
                )

        for h in range(num_levels - 1):
            for src in range(num_levels - 1, h, -1):
                if self._down_ops_owner == "down":
                    handle = self._blocks_down[h][src - h - 1]
                    transpose = False
                else:
                    handle = self._blocks_up[src][h]
                    transpose = True
                if handle is None:
                    continue
                self._ops_down[h].append(
                    _MklBlockOp(
                        src_level=src,
                        handle=handle,
                        transpose=transpose,
                        nnz=int(handle.nnz),
                    )
                )

    def _build_selector_indices(self, selector: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, bool]:
        coo = selector.tocoo()
        rows = np.asarray(coo.row, dtype=np.int64)
        cols = np.asarray(coo.col, dtype=np.int64)
        row_unique = bool(np.all(np.diff(selector.indptr) <= 1))
        return rows, cols, row_unique

    def _configure_handle_hints(self) -> None:
        usage: dict[int, dict[str, object]] = {}
        for ops_by_level in (self._ops_up, self._ops_down):
            for ops in ops_by_level:
                for op in ops:
                    key = id(op.handle)
                    entry = usage.get(key)
                    if entry is None:
                        entry = {"handle": op.handle, "non_transpose": False, "transpose": False}
                        usage[key] = entry
                    if op.transpose:
                        entry["transpose"] = True
                    else:
                        entry["non_transpose"] = True

        hint_k = self._k_hint
        expected = 1000
        for entry in usage.values():
            handle = entry["handle"]
            assert isinstance(handle, MklSparseHandle)
            if bool(entry["non_transpose"]):
                handle.set_mv_hint(transpose=False, expected_calls=expected)
                if hint_k is not None and hint_k > 1:
                    handle.set_mm_hint(hint_k, transpose=False, expected_calls=expected)
            if bool(entry["transpose"]):
                handle.set_mv_hint(transpose=True, expected_calls=expected)
                if hint_k is not None and hint_k > 1:
                    handle.set_mm_hint(hint_k, transpose=True, expected_calls=expected)
            handle.optimize()

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
        mkl_set_num_threads(self._n_threads)

        self._level_offsets = np.asarray(level_offsets)
        self._n = int(n)
        self._K = int(K)
        self._sample_perm = np.asarray(sample_perm)
        self._inv_sample_perm = np.asarray(inv_sample_perm)

        self._dtype = np.float64
        self._m = int(sel_mut.shape[0])
        self._coalescence_counts = (
            None if coalescence_counts is None else np.asarray(coalescence_counts, dtype=np.int64)
        )
        self._xtx_init = None
        if self._coalescence_counts is not None:
            self._xtx_init = (2.0 * self._coalescence_counts.astype(self._dtype, copy=False)).reshape(self._K)

        requested_dtype = np.dtype(dtype)
        if requested_dtype != np.float64:
            self._logger.info(
                "MKL backend uses float64 kernels; requested dtype %s is cast to float64 internally",
                requested_dtype,
            )

        self._build_owned_block_handles(A_blocks)
        self._build_logical_ops()
        self._selector_rows = {}
        self._selector_cols = {}
        self._selector_row_unique = {}
        self._selector_rows["mut"], self._selector_cols["mut"], self._selector_row_unique["mut"] = (
            self._build_selector_indices(sel_mut)
        )
        self._selector_rows["miss"], self._selector_cols["miss"], self._selector_row_unique["miss"] = (
            self._build_selector_indices(sel_miss)
        )
        self._configure_handle_hints()

        self.mem_usage.reset()
        common_host = estimate_common_host_static_bytes(
            level_offsets=self._level_offsets,
            sample_perm=self._sample_perm,
            inv_sample_perm=self._inv_sample_perm,
            coalescence_counts=self._coalescence_counts,
            xtx_init=self._xtx_init,
        )
        self.mem_usage.host_static.level_offsets = common_host.level_offsets
        self.mem_usage.host_static.sample_perm = common_host.sample_perm
        self.mem_usage.host_static.inv_sample_perm = common_host.inv_sample_perm
        self.mem_usage.host_static.coalescence_counts = common_host.coalescence_counts
        self.mem_usage.host_static.xtx_init = common_host.xtx_init
        self.mem_usage.host_static.blocks_up = int(
            sum(_handle_bytes(h) for row in self._blocks_up for h in row if h is not None)
        )
        self.mem_usage.host_static.blocks_down = int(
            sum(_handle_bytes(h) for row in self._blocks_down for h in row if h is not None)
        )
        self.mem_usage.host_static.selector_mut = int(
            self._selector_rows["mut"].nbytes + self._selector_cols["mut"].nbytes
        )
        self.mem_usage.host_static.selector_miss = int(
            self._selector_rows["miss"].nbytes + self._selector_cols["miss"].nbytes
        )

        self._logger.info(
            (
                "MklBackend setup: fmt_up=%s fmt_down=%s k_hint=%s n_threads=%d (actual=%d) "
                "store_up=%s store_down=%s up_owner=%s down_owner=%s"
            ),
            self._fmt_up,
            self._fmt_down,
            self._k_hint,
            self._n_threads,
            mkl_get_max_threads(),
            self._store_blocks_up,
            self._store_blocks_down,
            self._up_ops_owner,
            self._down_ops_owner,
        )
        if self._logger.isEnabledFor(logging.DEBUG):
            total_nnz_up = sum(op.nnz for ops in self._ops_up for op in ops)
            total_nnz_down = sum(op.nnz for ops in self._ops_down for op in ops)
            self._logger.debug(
                "levels=%d up_ops_nnz=%d down_ops_nnz=%d",
                len(self._level_offsets) - 1,
                total_nnz_up,
                total_nnz_down,
            )

        self._build_wavefront_level_stats()

    def _build_wavefront_level_stats(self) -> None:
        num_levels = len(self._level_offsets) - 1
        self._ops_up_calls = np.zeros(num_levels, dtype=np.int32)
        self._ops_up_nnz = np.zeros(num_levels, dtype=np.int64)
        self._ops_down_calls = np.zeros(num_levels, dtype=np.int32)
        self._ops_down_nnz = np.zeros(num_levels, dtype=np.int64)

        for h in range(1, num_levels):
            self._ops_up_calls[h] = len(self._ops_up[h])
            self._ops_up_nnz[h] = int(sum(op.nnz for op in self._ops_up[h]))

        for h in range(num_levels - 1):
            self._ops_down_calls[h] = len(self._ops_down[h])
            self._ops_down_nnz[h] = int(sum(op.nnz for op in self._ops_down[h]))

    def _print_wavefront_profile(self, direction: Direction, level_ms: np.ndarray) -> None:
        match direction:
            case Direction.UP:
                calls = self._ops_up_calls
                nnz = self._ops_up_nnz
            case Direction.DOWN:
                calls = self._ops_down_calls
                nnz = self._ops_down_nnz
            case _:
                raise ValueError(f"Unknown direction for wavefront profile: {direction!r}")

        records = [
            (h, float(level_ms[h]), int(calls[h]), int(nnz[h]))
            for h in range(len(level_ms))
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

    def _propagate_up_inplace(self, U: np.ndarray, level_ms: np.ndarray | None = None) -> None:
        off = self._level_offsets
        k = U.shape[1]
        if level_ms is None:
            for h in range(1, len(off) - 1):
                lo, hi = int(off[h]), int(off[h + 1])
                for op in self._ops_up[h]:
                    src_lo, src_hi = int(off[op.src_level]), int(off[op.src_level + 1])
                    if k == 1:
                        op.handle.mv(
                            U[src_lo:src_hi, 0],
                            U[lo:hi, 0],
                            alpha=1.0,
                            beta=1.0,
                            transpose=op.transpose,
                        )
                    else:
                        op.handle.mm(
                            U[src_lo:src_hi],
                            U[lo:hi],
                            alpha=1.0,
                            beta=1.0,
                            transpose=op.transpose,
                        )
            return

        for h in range(1, len(off) - 1):
            lo, hi = int(off[h]), int(off[h + 1])
            t_level = perf_counter()
            for op in self._ops_up[h]:
                src_lo, src_hi = int(off[op.src_level]), int(off[op.src_level + 1])
                if k == 1:
                    op.handle.mv(
                        U[src_lo:src_hi, 0],
                        U[lo:hi, 0],
                        alpha=1.0,
                        beta=1.0,
                        transpose=op.transpose,
                    )
                else:
                    op.handle.mm(
                        U[src_lo:src_hi],
                        U[lo:hi],
                        alpha=1.0,
                        beta=1.0,
                        transpose=op.transpose,
                    )
            level_ms[h] = (perf_counter() - t_level) * 1000.0

    def _propagate_down_inplace(self, V: np.ndarray, level_ms: np.ndarray | None = None) -> None:
        off = self._level_offsets
        k = V.shape[1]
        if level_ms is None:
            for h in range(len(off) - 2, -1, -1):
                lo, hi = int(off[h]), int(off[h + 1])
                for op in self._ops_down[h]:
                    src_lo, src_hi = int(off[op.src_level]), int(off[op.src_level + 1])
                    if k == 1:
                        op.handle.mv(
                            V[src_lo:src_hi, 0],
                            V[lo:hi, 0],
                            alpha=1.0,
                            beta=1.0,
                            transpose=op.transpose,
                        )
                    else:
                        op.handle.mm(
                            V[src_lo:src_hi],
                            V[lo:hi],
                            alpha=1.0,
                            beta=1.0,
                            transpose=op.transpose,
                        )
            return

        for h in range(len(off) - 2, -1, -1):
            lo, hi = int(off[h]), int(off[h + 1])
            t_level = perf_counter()
            for op in self._ops_down[h]:
                src_lo, src_hi = int(off[op.src_level]), int(off[op.src_level + 1])
                if k == 1:
                    op.handle.mv(
                        V[src_lo:src_hi, 0],
                        V[lo:hi, 0],
                        alpha=1.0,
                        beta=1.0,
                        transpose=op.transpose,
                    )
                else:
                    op.handle.mm(
                        V[src_lo:src_hi],
                        V[lo:hi],
                        alpha=1.0,
                        beta=1.0,
                        transpose=op.transpose,
                    )
            level_ms[h] = (perf_counter() - t_level) * 1000.0

    def _selector_forward(self, node_values: np.ndarray, selector: str) -> np.ndarray:
        rows = self._selector_rows[selector]
        cols = self._selector_cols[selector]
        unique_rows = self._selector_row_unique[selector]
        k = node_values.shape[1]
        out = np.zeros((self._m, k), dtype=self._dtype)
        if rows.size == 0:
            return out
        values = node_values[cols]
        if unique_rows:
            out[rows] = values
        else:
            np.add.at(out, rows, values)
        return out

    def _selector_backward_add(self, source: np.ndarray, selector: str, node_values: np.ndarray) -> None:
        rows = self._selector_rows[selector]
        cols = self._selector_cols[selector]
        if rows.size == 0:
            return
        np.add.at(node_values, cols, source[rows])

    def run_up(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
        need_miss_output: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        mkl_set_num_threads(self._n_threads)
        total_t0 = perf_counter()

        t0 = perf_counter()
        X = np.asarray(primary, dtype=self._dtype, order="C")
        if X.ndim != 2:
            raise ValueError(f"primary input must be 2D, got shape {X.shape}")
        if X.shape[0] != self._n:
            raise ValueError(f"UP primary input must have {self._n} rows, got {X.shape[0]}")
        k = int(X.shape[1])
        if self._k_hint is not None and k != self._k_hint:
            warnings.warn(
                (
                    f"MKL backend runtime k={k} does not match k_hint={self._k_hint}; "
                    "continuing with non-hinted execution."
                ),
                RuntimeWarning,
                stacklevel=3,
            )
        parse_ms = (perf_counter() - t0) * 1000.0

        init_mode = parse_init_mode(init_mode)
        t0 = perf_counter()
        init_payload = self._validate_init(init_mode, init, k)
        init_parse_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        node_values = np.zeros((self._K, k), dtype=self._dtype)
        alloc_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        self._apply_init_inplace(node_values, init_mode, init_payload)
        init_apply_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        np.add(node_values[: self._n], X[self._sample_perm], out=node_values[: self._n])
        seed_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        track_wave = self._logger.isEnabledFor(logging.DEBUG)
        level_ms = np.zeros(len(self._level_offsets) - 1, dtype=np.float64) if track_wave else None
        self._propagate_up_inplace(node_values, level_ms=level_ms)
        wave_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        out_mut = self._selector_forward(node_values, "mut")
        select_mut_ms = (perf_counter() - t0) * 1000.0

        out_miss = None
        select_miss_ms = 0.0
        if need_miss_output:
            t0 = perf_counter()
            out_miss = self._selector_forward(node_values, "miss")
            select_miss_ms = (perf_counter() - t0) * 1000.0

        if self._logger.isEnabledFor(logging.INFO):
            total_ms = (perf_counter() - total_t0) * 1000.0
            self._logger.info(
                (
                    "mkl.run_up k=%d init=%s miss_out=%s: parse=%.3fms init_parse=%.3fms "
                    "alloc=%.3fms init_apply=%.3fms seed=%.3fms wavefront=%.3fms "
                    "select_mut=%.3fms select_miss=%.3fms total=%.3fms"
                ),
                k,
                init_mode.value,
                need_miss_output,
                parse_ms,
                init_parse_ms,
                alloc_ms,
                init_apply_ms,
                seed_ms,
                wave_ms,
                select_mut_ms,
                select_miss_ms,
                total_ms,
            )
            if track_wave and level_ms is not None:
                self._print_wavefront_profile(Direction.UP, level_ms)
        self.mem_usage.record(
            stage="run_up",
            runtime_k=k,
            host_runtime=RuntimeBytes(
                level_buffers=int(node_values.nbytes),
                inputs=int(X.nbytes),
                outputs=int(out_mut.nbytes + (0 if out_miss is None else out_miss.nbytes)),
                aux=int((0 if init_payload is None else init_payload.nbytes) + (0 if level_ms is None else level_ms.nbytes)),
            ),
            meta={"direction": "up", "need_miss_output": bool(need_miss_output)},
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
        mkl_set_num_threads(self._n_threads)
        total_t0 = perf_counter()

        t0 = perf_counter()
        X = np.asarray(primary, dtype=self._dtype, order="C")
        if X.ndim != 2:
            raise ValueError(f"primary input must be 2D, got shape {X.shape}")
        if X.shape[0] != self._m:
            raise ValueError(f"DOWN primary input must have {self._m} rows, got {X.shape[0]}")
        k = int(X.shape[1])
        if self._k_hint is not None and k != self._k_hint:
            warnings.warn(
                (
                    f"MKL backend runtime k={k} does not match k_hint={self._k_hint}; "
                    "continuing with non-hinted execution."
                ),
                RuntimeWarning,
                stacklevel=3,
            )
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

        t0 = perf_counter()
        node_values = np.zeros((self._K, k), dtype=self._dtype)
        alloc_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        self._apply_init_inplace(node_values, init_mode, init_payload)
        init_apply_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        self._selector_backward_add(X, "mut", node_values)
        seed_mut_ms = (perf_counter() - t0) * 1000.0

        seed_miss_ms = 0.0
        if miss_arr is not None:
            t0 = perf_counter()
            self._selector_backward_add(miss_arr, "miss", node_values)
            seed_miss_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        track_wave = self._logger.isEnabledFor(logging.DEBUG)
        level_ms = np.zeros(len(self._level_offsets) - 1, dtype=np.float64) if track_wave else None
        self._propagate_down_inplace(node_values, level_ms=level_ms)
        wave_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        out = node_values[self._inv_sample_perm]
        gather_ms = (perf_counter() - t0) * 1000.0

        if self._logger.isEnabledFor(logging.INFO):
            total_ms = (perf_counter() - total_t0) * 1000.0
            self._logger.info(
                (
                    "mkl.run_down k=%d init=%s miss_in=%s: parse=%.3fms init_parse=%.3fms "
                    "alloc=%.3fms init_apply=%.3fms seed_mut=%.3fms seed_miss=%.3fms "
                    "wavefront=%.3fms gather=%.3fms total=%.3fms"
                ),
                k,
                init_mode.value,
                miss_arr is not None,
                parse_ms,
                init_parse_ms,
                alloc_ms,
                init_apply_ms,
                seed_mut_ms,
                seed_miss_ms,
                wave_ms,
                gather_ms,
                total_ms,
            )
            if track_wave and level_ms is not None:
                self._print_wavefront_profile(Direction.DOWN, level_ms)
        self.mem_usage.record(
            stage="run_down",
            runtime_k=k,
            host_runtime=RuntimeBytes(
                level_buffers=int(node_values.nbytes),
                inputs=int(X.nbytes + (0 if miss_arr is None else miss_arr.nbytes)),
                outputs=int(out.nbytes),
                aux=int((0 if init_payload is None else init_payload.nbytes) + (0 if level_ms is None else level_ms.nbytes)),
            ),
            meta={"direction": "down", "has_miss_input": bool(miss_arr is not None)},
        )
        return out

    def estimate_static_bytes(self) -> tuple[StaticBytes, StaticBytes]:
        host = estimate_common_host_static_bytes(
            level_offsets=self._level_offsets,
            sample_perm=self._sample_perm,
            inv_sample_perm=self._inv_sample_perm,
            coalescence_counts=self._coalescence_counts,
            xtx_init=self._xtx_init,
        )
        device = StaticBytes()

        host.blocks_up = int(sum(_estimate_handle_payload_bytes(h) for row in self._blocks_up for h in row if h is not None))
        host.blocks_down = int(
            sum(_estimate_handle_payload_bytes(h) for row in self._blocks_down for h in row if h is not None)
        )
        host.selector_mut = self._estimate_selector_payload_bytes("mut")
        host.selector_miss = self._estimate_selector_payload_bytes("miss")
        return host, device

    def _estimate_selector_payload_bytes(self, selector: str) -> int:
        rows = self._selector_rows[selector]
        cols = self._selector_cols[selector]
        if rows.shape != cols.shape:
            raise ValueError(f"Selector {selector!r} row/col shapes mismatch: {rows.shape} vs {cols.shape}")
        itemsize = int(np.dtype(rows.dtype).itemsize)
        return int(rows.size * 2 * itemsize)
