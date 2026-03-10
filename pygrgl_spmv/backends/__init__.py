"""Backend interfaces and reference CPU implementation for GRG sparse matmul."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import warnings
from typing import Any, List

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends.memory import MemoryUsage, RuntimeBytes, StaticBytes

from pygrgl_spmv.backends.types import (
    Direction,
    InitMode,
    SparseFormat,
    StoredMatrix,
    parse_init_mode,
    parse_sparse_format,
    parse_store,
    transpose_compatible_format,
)


def _parse_optional_k_hint(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        key = value.strip().lower()
        if key == "none":
            return None
        value = key
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"k_hint must be positive or none, got {value!r}")
    return parsed


@dataclass(frozen=True)
class ReferencePlan:
    """Minimal sparse-storage plan for the CPU reference backend."""

    store: StoredMatrix
    fmt: SparseFormat
    k_hint: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_hint", _parse_optional_k_hint(self.k_hint))

    @classmethod
    def from_any(cls, value):
        if isinstance(value, cls):
            return value
        if hasattr(value, "store") and hasattr(value, "fmt"):
            k_hint = getattr(value, "k_hint", None)
            return cls(
                store=parse_store(getattr(value, "store")),
                fmt=parse_sparse_format(getattr(value, "fmt")),
                k_hint=_parse_optional_k_hint(k_hint),
            )
        if isinstance(value, dict):
            return cls(
                store=parse_store(value["store"]),
                fmt=parse_sparse_format(value["fmt"]),
                k_hint=_parse_optional_k_hint(value.get("k_hint")),
            )
        raise TypeError(f"Cannot construct ReferencePlan from {type(value).__name__}")

    def can_share_storage_with(self, other: "ReferencePlan") -> bool:
        if self.store == other.store:
            return self.fmt == other.fmt
        return transpose_compatible_format(self.fmt) == other.fmt


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


def selector_rows_unique_from_csr_indptr(indptr: np.ndarray) -> bool:
    """Return True when each selector row has at most one non-zero."""
    return bool(np.all(np.diff(np.asarray(indptr)) <= 1))


def build_wavefront_level_stats(ops_by_level: list[list[object]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-level wavefront stats from logical op lists.

    Returns
    -------
    calls : np.ndarray[int32]
        Number of sparse block calls per level.
    nnz : np.ndarray[int64]
        Sum of per-op nnz per level.
    """
    calls = np.fromiter((len(ops) for ops in ops_by_level), dtype=np.int32, count=len(ops_by_level))
    nnz = np.fromiter(
        (sum(int(getattr(op, "nnz", 0)) for op in ops) for ops in ops_by_level),
        dtype=np.int64,
        count=len(ops_by_level),
    )
    return calls, nnz


def log_wavefront_profile(
    logger: logging.Logger,
    *,
    direction: Direction,
    calls: np.ndarray,
    nnz: np.ndarray,
    level_ms: np.ndarray,
) -> None:
    """Log per-level wavefront timing breakdown at DEBUG level."""
    records = [
        (h, float(level_ms[h]), int(calls[h]), int(nnz[h]))
        for h in range(len(level_ms))
        if int(calls[h]) > 0
    ]
    if not records:
        return

    total_ms = sum(ms for _, ms, _, _ in records)
    logger.debug("wavefront[%s] levels=%d total=%.3fms", direction.value, len(records), total_ms)
    for h, ms, call_count, nnz_count in records:
        pct = (100.0 * ms / total_ms) if total_ms > 0.0 else 0.0
        logger.debug(
            "  level=%2d ms=%.3f (%5.1f%%) calls=%3d nnz=%d",
            h,
            ms,
            pct,
            call_count,
            nnz_count,
        )


def warn_k_hint_mismatch(*, backend: str, direction: Direction, runtime_k: int, k_hint: int) -> None:
    warnings.warn(
        (
            f"{backend} backend runtime k={runtime_k} does not match k_hint={k_hint}; "
            f"{direction.value} traversal continues on non-hinted path."
        ),
        RuntimeWarning,
        stacklevel=3,
    )


class Backend:
    """
    Base backend and CPU reference implementation.

    Subclasses may override setup/run_up/run_down for accelerated kernels.
    """

    @staticmethod
    def plan(*, fmt: str | SparseFormat = SparseFormat.CSR, store: str | StoredMatrix = StoredMatrix.N, k_hint: int | None = None) -> ReferencePlan:
        return ReferencePlan(
            store=parse_store(store),
            fmt=parse_sparse_format(fmt),
            k_hint=_parse_optional_k_hint(k_hint),
        )

    def __init__(
        self,
        *,
        plan_up,
        plan_down,
        log_level: str = "WARNING",
    ) -> None:
        self._plan_up = None if plan_up is None else (plan_up if hasattr(plan_up, "can_share_storage_with") and hasattr(plan_up, "fmt") else ReferencePlan.from_any(plan_up))
        self._plan_down = None if plan_down is None else (plan_down if hasattr(plan_down, "can_share_storage_with") and hasattr(plan_down, "fmt") else ReferencePlan.from_any(plan_down))
        if self._plan_up is None and self._plan_down is None:
            raise ValueError("At least one of plan_up/plan_down must be provided")

        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(getattr(logging, str(log_level).upper(), logging.WARNING))

        if self._plan_up is None:
            share_storage = False
            self._store_blocks_up = False
            self._store_blocks_down = True
        elif self._plan_down is None:
            share_storage = False
            self._store_blocks_up = True
            self._store_blocks_down = False
        else:
            share_storage = self._plan_up.can_share_storage_with(self._plan_down)
            self._store_blocks_up = True
            self._store_blocks_down = not share_storage
        self._up_ops_owner = "up"
        self._down_ops_owner = "up" if share_storage else "down"

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

    @property
    def _fmt_up(self) -> str:
        if self._plan_up is None:
            raise ValueError("UP plan is not configured")
        return self._plan_up.fmt.value.lower()

    @property
    def _fmt_down(self) -> str:
        if self._plan_down is None:
            raise ValueError("DOWN plan is not configured")
        return self._plan_down.fmt.value.lower()

    def _require_plan(self, direction: Direction):
        plan = self._plan_for(direction)
        if plan is None:
            raise ValueError(f"{direction.value.upper()} plan is not configured")
        return plan

    def _plan_for(self, direction: Direction):
        return self._plan_up if direction == Direction.UP else self._plan_down

    def _configured_directions(self) -> tuple[Direction, ...]:
        directions: list[Direction] = []
        if self._plan_up is not None:
            directions.append(Direction.UP)
        if self._plan_down is not None:
            directions.append(Direction.DOWN)
        return tuple(directions)


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
        self._require_plan(Direction.UP)
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
        self._require_plan(Direction.DOWN)
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
    "ReferencePlan",
    "MemoryUsage",
    "build_wavefront_level_stats",
    "estimate_common_host_static_bytes",
    "estimate_sparse_payload_bytes",
    "log_wavefront_profile",
    "selector_rows_unique_from_csr_indptr",
    "warn_k_hint_mismatch",
]
