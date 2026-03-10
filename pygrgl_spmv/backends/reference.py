"""Reference CPU backend and reference plan type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends.base import (
    BackendBase,
    BackendSetup,
    _parse_optional_k_hint,
    _sparse_host_bytes,
    estimate_common_host_static_bytes,
)
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


class ReferenceBackend(BackendBase):
    """Concrete CPU reference implementation."""

    @staticmethod
    def plan(
        *,
        fmt: str | SparseFormat = SparseFormat.CSR,
        store: str | StoredMatrix = StoredMatrix.N,
        k_hint: int | None = None,
    ) -> ReferencePlan:
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
        up = None if plan_up is None else ReferencePlan.from_any(plan_up)
        down = None if plan_down is None else ReferencePlan.from_any(plan_down)
        super().__init__(plan_up=up, plan_down=down, log_level=log_level)

    def setup(self, setup: BackendSetup) -> None:
        self._apply_setup_state(setup)

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

    def estimate_static_bytes(self):
        raise NotImplementedError(
            f"{self.__class__.__name__}.estimate_static_bytes() is required for benchmark static_est rows."
        )

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
            host_runtime={
                "level_buffers": int(node_values.nbytes),
                "inputs": int(X.nbytes),
                "outputs": int(out_mut.nbytes + (0 if out_miss is None else out_miss.nbytes)),
                "aux": 0 if payload is None else int(payload.nbytes),
            },
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
            host_runtime={
                "level_buffers": int(node_values.nbytes),
                "inputs": int(X.nbytes + (0 if miss_arr is None else miss_arr.nbytes)),
                "outputs": int(out.nbytes),
                "aux": 0 if payload is None else int(payload.nbytes),
            },
            meta={"direction": "down", "has_miss_input": bool(miss_arr is not None)},
        )
        return out


__all__ = ["ReferenceBackend", "ReferencePlan"]
