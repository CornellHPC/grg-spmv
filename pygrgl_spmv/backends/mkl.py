"""MKL backend — Intel MKL-accelerated fused GRG matmul traversal."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from time import perf_counter
from typing import List

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends import (
    _parse_optional_k_hint,
    Backend,
    build_wavefront_level_stats,
    estimate_common_host_static_bytes,
    estimate_sparse_payload_bytes,
    log_wavefront_profile,
    selector_rows_unique_from_csr_indptr,
    warn_k_hint_mismatch,
)
from pygrgl_spmv.backends.memory import RuntimeBytes, StaticBytes
from pygrgl_spmv.backends.mkl_utils import (
    MklSparseHandle,
    mkl_get_max_threads,
    mkl_set_num_threads,
)
from pygrgl_spmv.backends.types import Direction, InitMode, SparseFormat, StoredMatrix, parse_init_mode, parse_sparse_format, parse_store, transpose_compatible_format


@dataclass(frozen=True)
class MklPlan:
    """Explicit MKL traversal/storage plan."""

    store: StoredMatrix
    fmt: SparseFormat
    n_threads: int = 0
    k_hint: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_hint", _parse_optional_k_hint(self.k_hint))

    @classmethod
    def from_any(cls, value):
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            allowed_keys = {"store", "fmt", "n_threads", "k_hint"}
            extra = sorted(set(value) - allowed_keys)
            if extra:
                raise ValueError(f"Unknown MklPlan field(s): {extra}")
            n_threads = int(value.get("n_threads", 0))
            return cls(
                store=parse_store(value["store"]),
                fmt=parse_sparse_format(value["fmt"]),
                n_threads=n_threads,
                k_hint=_parse_optional_k_hint(value.get("k_hint")),
            )
        if hasattr(value, "store") and hasattr(value, "fmt"):
            n_threads = int(getattr(value, "n_threads", 0))
            return cls(
                store=parse_store(getattr(value, "store")),
                fmt=parse_sparse_format(getattr(value, "fmt")),
                n_threads=n_threads,
                k_hint=_parse_optional_k_hint(getattr(value, "k_hint", None)),
            )
        raise TypeError(f"Cannot construct MklPlan from {type(value).__name__}")

    def can_share_storage_with(self, other: "MklPlan") -> bool:
        if self.store == other.store:
            return self.fmt == other.fmt
        return transpose_compatible_format(self.fmt) == other.fmt

    def __str__(self) -> str:
        return (
            "["
            f"k_hint={'none' if self.k_hint is None else self.k_hint},"
            f"store={self.store.value},fmt={self.fmt.value},"
            f"n_threads={self.n_threads}"
            "]"
        )


@dataclass(frozen=True)
class _MklBlockOp:
    """One sparse block application in a level wavefront."""

    src_level: int
    handle: MklSparseHandle
    transpose: bool
    nnz: int


def _stored_block(base_block: sp.spmatrix, store: StoredMatrix) -> sp.spmatrix:
    return base_block if store == StoredMatrix.N else base_block.T


def _needs_transpose(direction: Direction, store: StoredMatrix) -> bool:
    return (direction == Direction.DOWN) != (store == StoredMatrix.T)


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
        *,
        plan_up: MklPlan,
        plan_down: MklPlan,
        log_level: str = "WARNING",
    ):
        up = None if plan_up is None else MklPlan.from_any(plan_up)
        down = None if plan_down is None else MklPlan.from_any(plan_down)
        super().__init__(
            plan_up=up,
            plan_down=down,
            log_level=log_level,
        )
        self._n_threads_up = self._resolve_thread_count(self._plan_up)
        self._n_threads_down = self._resolve_thread_count(self._plan_down)
        configured_threads = [
            count for count in (self._n_threads_up, self._n_threads_down) if count is not None
        ]
        self._n_threads_setup = max(configured_threads)
        if self._plan_up is not None and self._fmt_up not in {"csr", "csc", "coo"}:
            raise ValueError(f"Unsupported MKL fmt_up={self._fmt_up!r}; expected csr/csc/coo")
        if self._plan_down is not None and self._fmt_down not in {"csr", "csc", "coo"}:
            raise ValueError(f"Unsupported MKL fmt_down={self._fmt_down!r}; expected csr/csc/coo")
        self._coalescence_counts = None

    def _resolve_thread_count(self, plan: MklPlan | None) -> int | None:
        if plan is None:
            return None
        cpu_count = os.cpu_count()
        if cpu_count is None:
            cpu_count = 1
        return cpu_count if plan.n_threads == 0 else int(plan.n_threads)

    def _thread_count_for(self, direction: Direction) -> int:
        count = self._n_threads_up if direction == Direction.UP else self._n_threads_down
        if count is None:
            raise ValueError(f"{direction.value.upper()} plan is not configured")
        return count

    def _build_owned_block_handles(self, A_blocks: List[List[sp.csr_matrix]]) -> None:
        num_levels = len(self._level_offsets) - 1
        self._blocks_up = [[] for _ in range(num_levels)]
        self._blocks_down = [[] for _ in range(num_levels)]

        if self._store_blocks_up:
            if self._plan_up is None:
                raise RuntimeError("UP storage requested without an UP plan")
            for h in range(num_levels):
                row: list[MklSparseHandle | None] = []
                for j in range(h):
                    blk = _stored_block(A_blocks[h][j], self._plan_up.store)
                    row.append(None if blk.nnz == 0 else MklSparseHandle(blk, self._fmt_up))
                self._blocks_up[h] = row

        if self._store_blocks_down:
            if self._plan_down is None:
                raise RuntimeError("DOWN storage requested without a DOWN plan")
            for h in range(num_levels):
                row = []
                for src in range(h + 1, num_levels):
                    blk = _stored_block(A_blocks[src][h], self._plan_down.store)
                    row.append(None if blk.nnz == 0 else MklSparseHandle(blk, self._fmt_down))
                self._blocks_down[h] = row

    def _build_logical_ops(self) -> None:
        num_levels = len(self._level_offsets) - 1
        self._ops_up: list[list[_MklBlockOp]] = [[] for _ in range(num_levels)]
        self._ops_down: list[list[_MklBlockOp]] = [[] for _ in range(num_levels)]

        if self._plan_up is not None:
            for h in range(1, num_levels):
                for j in range(h):
                    if self._up_ops_owner == "up":
                        handle = self._blocks_up[h][j]
                        owner_store = self._plan_up.store
                    else:
                        handle = self._blocks_down[j][h - j - 1]
                        if self._plan_down is None:
                            raise RuntimeError("UP ops requested shared DOWN storage without a DOWN plan")
                        owner_store = self._plan_down.store
                    if handle is None:
                        continue
                    self._ops_up[h].append(
                        _MklBlockOp(
                            src_level=j,
                            handle=handle,
                            transpose=_needs_transpose(Direction.UP, owner_store),
                            nnz=int(handle.nnz),
                        )
                    )

        if self._plan_down is not None:
            for h in range(num_levels - 1):
                for src in range(num_levels - 1, h, -1):
                    if self._down_ops_owner == "down":
                        handle = self._blocks_down[h][src - h - 1]
                        owner_store = self._plan_down.store
                    else:
                        handle = self._blocks_up[src][h]
                        if self._plan_up is None:
                            raise RuntimeError("DOWN ops requested shared UP storage without an UP plan")
                        owner_store = self._plan_up.store
                    if handle is None:
                        continue
                    self._ops_down[h].append(
                        _MklBlockOp(
                            src_level=src,
                            handle=handle,
                            transpose=_needs_transpose(Direction.DOWN, owner_store),
                            nnz=int(handle.nnz),
                        )
                    )

    def _build_selector_indices(self, selector: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, bool]:
        coo = selector.tocoo()
        rows = np.asarray(coo.row, dtype=np.int64)
        cols = np.asarray(coo.col, dtype=np.int64)
        row_unique = selector_rows_unique_from_csr_indptr(selector.indptr)
        return rows, cols, row_unique

    def _configure_handle_hints(self) -> None:
        usage: dict[tuple[int, bool], dict[str, object]] = {}
        for direction, ops_by_level, plan in (
            (Direction.UP, self._ops_up, self._plan_up),
            (Direction.DOWN, self._ops_down, self._plan_down),
        ):
            if plan is None:
                continue
            for ops in ops_by_level:
                for op in ops:
                    key = (id(op.handle), bool(op.transpose))
                    entry = usage.get(key)
                    if entry is None:
                        entry = {
                            "handle": op.handle,
                            "transpose": bool(op.transpose),
                            "k_hint": plan.k_hint,
                        }
                        usage[key] = entry
                        continue
                    existing_hint = entry["k_hint"]
                    if existing_hint is None:
                        entry["k_hint"] = plan.k_hint
                    elif plan.k_hint is not None and int(existing_hint) != int(plan.k_hint):
                        raise ValueError(
                            "Incompatible MKL k_hint values share the same handle and transpose mode: "
                            f"{direction.value} requested {plan.k_hint}, existing {existing_hint}"
                        )
        expected = 1000
        for entry in usage.values():
            handle = entry["handle"]
            assert isinstance(handle, MklSparseHandle)
            transpose = bool(entry["transpose"])
            handle.set_mv_hint(transpose=transpose, expected_calls=expected)
            hint_k = entry["k_hint"]
            if hint_k is not None and int(hint_k) > 1:
                handle.set_mm_hint(int(hint_k), transpose=transpose, expected_calls=expected)
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
        mkl_set_num_threads(self._n_threads_setup)

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
                "MklBackend setup: fmt_up=%s fmt_down=%s k_hint=%s n_threads=%s (actual=%d) "
                "store_up=%s store_down=%s up_owner=%s down_owner=%s"
            ),
            "<unspecified>" if self._plan_up is None else self._fmt_up,
            "<unspecified>" if self._plan_down is None else self._fmt_down,
            (None if self._plan_up is None else self._plan_up.k_hint, None if self._plan_down is None else self._plan_down.k_hint),
            (self._n_threads_up, self._n_threads_down),
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
        self._ops_up_calls, self._ops_up_nnz = build_wavefront_level_stats(self._ops_up)
        self._ops_down_calls, self._ops_down_nnz = build_wavefront_level_stats(self._ops_down)

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
        log_wavefront_profile(self._logger, direction=direction, calls=calls, nnz=nnz, level_ms=level_ms)

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
        mkl_set_num_threads(self._thread_count_for(Direction.UP))
        total_t0 = perf_counter()

        t0 = perf_counter()
        self._require_plan(Direction.UP)
        X = np.asarray(primary, dtype=self._dtype, order="C")
        if X.ndim != 2:
            raise ValueError(f"primary input must be 2D, got shape {X.shape}")
        if X.shape[0] != self._n:
            raise ValueError(f"UP primary input must have {self._n} rows, got {X.shape[0]}")
        k = int(X.shape[1])
        if self._plan_up is not None and self._plan_up.k_hint is not None and k != self._plan_up.k_hint:
            warn_k_hint_mismatch(backend="MKL", direction=Direction.UP, runtime_k=k, k_hint=self._plan_up.k_hint)
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
        mkl_set_num_threads(self._thread_count_for(Direction.DOWN))
        total_t0 = perf_counter()

        t0 = perf_counter()
        self._require_plan(Direction.DOWN)
        X = np.asarray(primary, dtype=self._dtype, order="C")
        if X.ndim != 2:
            raise ValueError(f"primary input must be 2D, got shape {X.shape}")
        if X.shape[0] != self._m:
            raise ValueError(f"DOWN primary input must have {self._m} rows, got {X.shape[0]}")
        k = int(X.shape[1])
        if self._plan_down is not None and self._plan_down.k_hint is not None and k != self._plan_down.k_hint:
            warn_k_hint_mismatch(backend="MKL", direction=Direction.DOWN, runtime_k=k, k_hint=self._plan_down.k_hint)
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


__all__ = ["MklBackend", "MklPlan"]
