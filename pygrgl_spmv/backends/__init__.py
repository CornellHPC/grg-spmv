"""Backend interfaces and reference CPU implementation for GRG sparse matmul."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends.memory import MemoryUsage, RuntimeBytes, StaticBytes
from pygrgl_spmv.backends.types import InitMode, parse_init_mode

SUPPORTED_FMTS = frozenset({"csr", "csc", "coo"})
TRANSPOSE_FMT_MAP = {"csr": "csc", "csc": "csr", "coo": "coo"}


def _sparse_host_bytes(mat: sp.spmatrix) -> int:
    total = int(mat.data.nbytes)
    if hasattr(mat, "indices"):
        total += int(mat.indices.nbytes)
    if hasattr(mat, "indptr"):
        total += int(mat.indptr.nbytes)
    if hasattr(mat, "row"):
        total += int(mat.row.nbytes)
    if hasattr(mat, "col"):
        total += int(mat.col.nbytes)
    return total


def estimate_sparse_payload_bytes(
    *,
    fmt: str,
    nrows: int,
    ncols: int,
    nnz: int,
    data_itemsize: int,
    index_itemsize: int,
) -> int:
    if nnz < 0:
        raise ValueError(f"nnz must be non-negative, got {nnz}")
    if data_itemsize <= 0 or index_itemsize <= 0:
        raise ValueError(
            f"data_itemsize and index_itemsize must be positive, got {data_itemsize}, {index_itemsize}"
        )
    match fmt:
        case "csr":
            return int(nnz * (data_itemsize + index_itemsize) + (nrows + 1) * index_itemsize)
        case "csc":
            return int(nnz * (data_itemsize + index_itemsize) + (ncols + 1) * index_itemsize)
        case "coo":
            return int(nnz * (data_itemsize + 2 * index_itemsize))
        case _:
            raise ValueError(f"Unsupported sparse format for size estimate: {fmt!r}")


def estimate_common_host_static_bytes(
    *,
    level_offsets: np.ndarray,
    sample_perm: np.ndarray,
    inv_sample_perm: np.ndarray,
    coalescence_counts: np.ndarray | None,
    xtx_init: np.ndarray | None,
) -> StaticBytes:
    host = StaticBytes(
        level_offsets=int(level_offsets.nbytes),
        sample_perm=int(sample_perm.nbytes),
        inv_sample_perm=int(inv_sample_perm.nbytes),
    )
    if coalescence_counts is not None:
        host.coalescence_counts = int(coalescence_counts.nbytes)
    if xtx_init is not None:
        host.xtx_init = int(xtx_init.nbytes)
    return host


class Backend:
    """
    Base backend and CPU reference implementation.

    Subclasses may override setup/run_up/run_down for accelerated kernels.
    """

    def __init__(
        self,
        *,
        fmt_up: str | None = "csr",
        fmt_down: str | None = None,
        k_hint: int | None = None,
        log_level: str = "WARNING",
    ) -> None:
        if k_hint is not None:
            k_hint = int(k_hint)
            if k_hint <= 0:
                raise ValueError(f"k_hint must be positive or None, got {k_hint}")
        self._k_hint = k_hint

        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(getattr(logging, str(log_level).upper(), logging.WARNING))

        up = None if fmt_up is None else str(fmt_up).lower()
        down = None if fmt_down is None else str(fmt_down).lower()
        if up is None and down is None:
            raise ValueError("At least one of fmt_up/fmt_down must be provided")
        if up is not None and up not in SUPPORTED_FMTS:
            raise ValueError(f"Unsupported fmt_up {fmt_up!r}; expected one of {sorted(SUPPORTED_FMTS)}")
        if down is not None and down not in SUPPORTED_FMTS:
            raise ValueError(f"Unsupported fmt_down {fmt_down!r}; expected one of {sorted(SUPPORTED_FMTS)}")

        if up is None:
            assert down is not None
            up = TRANSPOSE_FMT_MAP[down]
        if down is None:
            down = TRANSPOSE_FMT_MAP[up]
        transpose_compatible = TRANSPOSE_FMT_MAP[up] == down

        if fmt_up is None:
            self._store_blocks_up = False
            self._store_blocks_down = True
            self._up_ops_owner = "down"
            self._down_ops_owner = "down"
        elif fmt_down is None:
            self._store_blocks_up = True
            self._store_blocks_down = False
            self._up_ops_owner = "up"
            self._down_ops_owner = "up"
        elif transpose_compatible:
            self._store_blocks_up = True
            self._store_blocks_down = False
            self._up_ops_owner = "up"
            self._down_ops_owner = "up"
        else:
            self._store_blocks_up = True
            self._store_blocks_down = True
            self._up_ops_owner = "up"
            self._down_ops_owner = "down"

        self._fmt_up = up
        self._fmt_down = down

        self._A_blocks: List[List[sp.csr_matrix]] = []
        self._level_offsets = np.empty(0, dtype=np.int64)
        self._sel_mut = sp.csr_matrix((0, 0))
        self._sel_miss = sp.csr_matrix((0, 0))
        self._sample_perm = np.empty(0, dtype=np.int64)
        self._inv_sample_perm = np.empty(0, dtype=np.int64)
        self._coalescence_counts = None
        self._xtx_init = None
        self._dtype = np.float64
        self._n = 0
        self._K = 0
        self._m = 0
        self.mem_usage = MemoryUsage()

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
        self._A_blocks = A_blocks
        self._level_offsets = np.asarray(level_offsets)
        self._n = int(n)
        self._K = int(K)
        self._m = int(sel_mut.shape[0])
        self._sel_mut = sel_mut
        self._sel_miss = sel_miss
        self._sample_perm = np.asarray(sample_perm)
        self._inv_sample_perm = np.asarray(inv_sample_perm)
        self._coalescence_counts = (
            None if coalescence_counts is None else np.asarray(coalescence_counts, dtype=np.int64)
        )
        self._dtype = np.dtype(dtype)
        self._xtx_init = None
        if self._coalescence_counts is not None:
            self._xtx_init = (2.0 * self._coalescence_counts).astype(self._dtype, copy=False).reshape(self._K)

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
        self.mem_usage.host_static.selector_mut = _sparse_host_bytes(self._sel_mut)
        self.mem_usage.host_static.selector_miss = _sparse_host_bytes(self._sel_miss)
        self.mem_usage.host_static.blocks_up = int(
            sum(_sparse_host_bytes(blk) for blocks in self._A_blocks for blk in blocks)
        )

    def estimate_static_bytes(self) -> tuple[StaticBytes, StaticBytes]:
        raise NotImplementedError(
            f"{self.__class__.__name__}.estimate_static_bytes() is required for benchmark static_est rows."
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
                if self._xtx_init is None:
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

    def _apply_init_inplace(
        self,
        node_values: np.ndarray,
        init_mode: InitMode,
        init_payload: np.ndarray | None,
    ) -> None:
        match init_mode:
            case InitMode.NONE:
                return
            case InitMode.XTX:
                if self._xtx_init is None:
                    raise ValueError("init_mode=xtx requires GRG coalescence counts")
                np.add(node_values, self._xtx_init[:, None], out=node_values)
            case InitMode.VECTOR:
                assert init_payload is not None
                np.add(node_values, init_payload[None, :], out=node_values)
            case InitMode.MATRIX:
                assert init_payload is not None
                np.add(node_values, init_payload, out=node_values)
            case _:
                raise ValueError(f"Unknown init mode: {init_mode!r}")

    def _propagate_up_inplace(self, node_values: np.ndarray) -> None:
        off = self._level_offsets
        for h in range(1, len(off) - 1):
            lo, hi = int(off[h]), int(off[h + 1])
            for j, blk in enumerate(self._A_blocks[h]):
                if blk.nnz == 0:
                    continue
                jlo, jhi = int(off[j]), int(off[j + 1])
                node_values[lo:hi] += blk @ node_values[jlo:jhi]

    def _propagate_down_inplace(self, node_values: np.ndarray) -> None:
        off = self._level_offsets
        for h in range(len(off) - 2, -1, -1):
            lo, hi = int(off[h]), int(off[h + 1])
            for src in range(h + 1, len(off) - 1):
                blk = self._A_blocks[src][h]
                if blk.nnz == 0:
                    continue
                src_lo, src_hi = int(off[src]), int(off[src + 1])
                node_values[lo:hi] += blk.T @ node_values[src_lo:src_hi]

    def run_up(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
        need_miss_output: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        X = np.asarray(primary, dtype=self._dtype, order="C")
        if X.ndim != 2:
            raise ValueError(f"primary input must be 2D, got shape {X.shape}")
        if X.shape[0] != self._n:
            raise ValueError(f"UP primary input must have {self._n} rows, got {X.shape[0]}")
        k = int(X.shape[1])

        mode = parse_init_mode(init_mode)
        payload = self._validate_init(mode, init, k)

        node_values = np.zeros((self._K, k), dtype=self._dtype)
        self._apply_init_inplace(node_values, mode, payload)
        np.add(node_values[: self._n], X[self._sample_perm], out=node_values[: self._n])
        self._propagate_up_inplace(node_values)

        if self._sel_mut.nnz == 0:
            out_mut = np.zeros((self._m, k), dtype=self._dtype)
        else:
            out_mut = np.asarray(self._sel_mut @ node_values, dtype=self._dtype)
        out_miss = None
        if need_miss_output:
            if self._sel_miss.nnz == 0:
                out_miss = np.zeros((self._m, k), dtype=self._dtype)
            else:
                out_miss = np.asarray(self._sel_miss @ node_values, dtype=self._dtype)
        self.mem_usage.record(
            stage="run_up",
            runtime_k=k,
            host_runtime=RuntimeBytes(
                level_buffers=int(node_values.nbytes),
                inputs=int(X.nbytes),
                outputs=int(out_mut.nbytes + (0 if out_miss is None else out_miss.nbytes)),
                aux=0 if payload is None else int(payload.nbytes),
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
        X = np.asarray(primary, dtype=self._dtype, order="C")
        if X.ndim != 2:
            raise ValueError(f"primary input must be 2D, got shape {X.shape}")
        if X.shape[0] != self._m:
            raise ValueError(f"DOWN primary input must have {self._m} rows, got {X.shape[0]}")
        k = int(X.shape[1])

        miss_arr = None
        if miss is not None:
            miss_arr = np.asarray(miss, dtype=self._dtype, order="C")
            if miss_arr.shape != (self._m, k):
                raise ValueError(f"miss input must have shape ({self._m}, {k}), got {miss_arr.shape}")

        mode = parse_init_mode(init_mode)
        payload = self._validate_init(mode, init, k)

        node_values = np.zeros((self._K, k), dtype=self._dtype)
        self._apply_init_inplace(node_values, mode, payload)
        if self._sel_mut.nnz > 0:
            node_values += self._sel_mut.T @ X
        if miss_arr is not None and self._sel_miss.nnz > 0:
            node_values += self._sel_miss.T @ miss_arr
        self._propagate_down_inplace(node_values)
        out = node_values[self._inv_sample_perm]
        self.mem_usage.record(
            stage="run_down",
            runtime_k=k,
            host_runtime=RuntimeBytes(
                level_buffers=int(node_values.nbytes),
                inputs=int(X.nbytes + (0 if miss_arr is None else miss_arr.nbytes)),
                outputs=int(out.nbytes),
                aux=0 if payload is None else int(payload.nbytes),
            ),
            meta={"direction": "down", "has_miss_input": bool(miss_arr is not None)},
        )
        return out


__all__ = [
    "Backend",
    "MemoryUsage",
    "estimate_common_host_static_bytes",
    "estimate_sparse_payload_bytes",
]
