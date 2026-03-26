"""Reference CPU backend and reference plan type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends.base import (
    BackendBase,
    CallCapture,
    BackendSetup,
    _parse_optional_k_hint,
)
from pygrgl_spmv.memory import alloc_field
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
    def from_dict(cls, value: Mapping[str, object]) -> "ReferencePlan":
        return cls(
            store=parse_store(value["store"]),
            fmt=parse_sparse_format(value["fmt"]),
            k_hint=_parse_optional_k_hint(value.get("k_hint")),
        )

    def can_share_storage_with(self, other: "ReferencePlan") -> bool:
        if self.store == other.store:
            return self.fmt == other.fmt
        return transpose_compatible_format(self.fmt) == other.fmt


@dataclass
class ReferenceCall:
    node_values: np.ndarray | None = alloc_field(
        label="node_state", kind="state", owner="backend", retention="call", activity="yes", default=None
    )
    miss_output: np.ndarray | None = alloc_field(
        label="miss_output", kind="output", owner="backend", retention="call", activity="yes", default=None
    )


@dataclass
class ReferenceRetained:
    blocks: list[sp.spmatrix] = alloc_field(
        label="blocks", kind="sparse", owner="backend", retention="persistent", activity="always", default_factory=list
    )
    xtx_host: np.ndarray | None = alloc_field(
        label="xtx_host", kind="init", owner="backend", retention="persistent", activity="always", default=None
    )


class ReferenceBackend(BackendBase):
    """Concrete CPU reference implementation."""

    _SETUP_MEMORY_POLICY = {
        "_A_blocks": "retained",
        "_sel_mut": "borrowed",
        "_sel_miss": "borrowed",
        "_level_offsets": "borrowed",
        "_sample_rows": "borrowed",
        "_coalescence_counts": "borrowed",
        "_xtx_host": "retained",
    }

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

    @staticmethod
    def pair(
        *,
        plan_up: ReferencePlan | None,
        plan_down: ReferencePlan | None,
    ) -> "ReferencePlanPair":
        return ReferencePlanPair(plan_up=plan_up, plan_down=plan_down)

    def __init__(
        self,
        *,
        pair: "ReferencePlanPair",
        log_level: str = "WARNING",
        instrumentation: bool = False,
    ) -> None:
        super().__init__(
            plan_up=pair.plan_up,
            plan_down=pair.plan_down,
            log_level=log_level,
            instrumentation=instrumentation,
        )
        self._install_memory(retained=ReferenceRetained(), call_type=ReferenceCall)

    def setup(self, setup: BackendSetup) -> None:
        self._apply_setup_state(setup)
        self._sync_retained_root()
        self._bump_retained_epoch()
        self._assert_setup_memory_contract()

    def _sync_retained_root(self) -> None:
        retained = self._retained_mem
        retained.blocks = [blk for row in self._A_blocks for blk in row if blk is not None]
        retained.xtx_host = self._xtx_host

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
        if X.shape[0] != self._num_samples:
            raise ValueError(f"UP primary input must have {self._num_samples} rows, got {X.shape[0]}")
        k = int(X.shape[1])

        mode = parse_init_mode(init_mode)
        payload = self._validate_init(mode, init, k)

        node_values = np.zeros((self._num_nodes, k), dtype=self._dtype)
        self._apply_init_inplace(node_values, mode, payload)
        node_values[self._sample_rows] += X
        self._propagate_up_inplace(node_values)

        if self._sel_mut.nnz == 0:
            out_mut = np.zeros((self._num_mutations, k), dtype=self._dtype)
        else:
            out_mut = np.asarray(self._sel_mut @ node_values, dtype=self._dtype)
        out_miss = None
        if need_miss_output:
            if self._sel_miss.nnz == 0:
                out_miss = np.zeros((self._num_mutations, k), dtype=self._dtype)
            else:
                out_miss = np.asarray(self._sel_miss @ node_values, dtype=self._dtype)
        if self._capture_active:
            call = self._call_mem
            assert isinstance(call, ReferenceCall)
            call.node_values = node_values
            call.miss_output = out_miss
            self._publish_call_capture(
                CallCapture(
                    nonce=self._capture_nonce,
                    direction="up",
                    runtime_k=k,
                    active_alloc_keys=frozenset(),
                    meta={"need_miss_output": bool(need_miss_output), "mode": "host"},
                )
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
        if X.shape[0] != self._num_mutations:
            raise ValueError(f"DOWN primary input must have {self._num_mutations} rows, got {X.shape[0]}")
        k = int(X.shape[1])

        miss_arr = None
        if miss is not None:
            miss_arr = np.asarray(miss, dtype=self._dtype, order="C")
            if miss_arr.shape != (self._num_mutations, k):
                raise ValueError(f"miss input must have shape ({self._num_mutations}, {k}), got {miss_arr.shape}")

        mode = parse_init_mode(init_mode)
        payload = self._validate_init(mode, init, k)

        node_values = np.zeros((self._num_nodes, k), dtype=self._dtype)
        self._apply_init_inplace(node_values, mode, payload)
        if self._sel_mut.nnz > 0:
            node_values += self._sel_mut.T @ X
        if miss_arr is not None and self._sel_miss.nnz > 0:
            node_values += self._sel_miss.T @ miss_arr
        self._propagate_down_inplace(node_values)
        out = node_values[self._sample_rows]
        if self._capture_active:
            call = self._call_mem
            assert isinstance(call, ReferenceCall)
            call.node_values = node_values
            call.miss_output = None
            self._publish_call_capture(
                CallCapture(
                    nonce=self._capture_nonce,
                    direction="down",
                    runtime_k=k,
                    active_alloc_keys=frozenset(),
                    meta={"has_miss_input": bool(miss_arr is not None), "mode": "host"},
                )
            )
        return out

    def run_up_nodes(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
    ) -> np.ndarray:
        self._require_plan(Direction.UP)
        X = np.asarray(primary, dtype=self._dtype, order="C")
        if X.ndim != 2:
            raise ValueError(f"primary input must be 2D, got shape {X.shape}")
        if X.shape[0] != self._num_samples:
            raise ValueError(f"UP primary input must have {self._num_samples} rows, got {X.shape[0]}")
        k = int(X.shape[1])
        mode = parse_init_mode(init_mode)
        payload = self._validate_init(mode, init, k)
        node_values = np.zeros((self._num_nodes, k), dtype=self._dtype)
        self._apply_init_inplace(node_values, mode, payload)
        node_values[self._sample_rows] += X
        self._propagate_up_inplace(node_values)
        if self._capture_active:
            call = self._call_mem
            assert isinstance(call, ReferenceCall)
            call.node_values = node_values
            call.miss_output = None
            self._publish_call_capture(
                CallCapture(
                    nonce=self._capture_nonce,
                    direction="up",
                    runtime_k=k,
                    active_alloc_keys=frozenset(),
                    meta={"emit_all_nodes": True, "mode": "host"},
                )
            )
        return node_values

    def run_down_nodes(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
    ) -> np.ndarray:
        self._require_plan(Direction.DOWN)
        X = np.asarray(primary, dtype=self._dtype, order="C")
        if X.ndim != 2:
            raise ValueError(f"primary input must be 2D, got shape {X.shape}")
        if X.shape[0] != self._num_mutations:
            raise ValueError(f"DOWN primary input must have {self._num_mutations} rows, got {X.shape[0]}")
        k = int(X.shape[1])
        mode = parse_init_mode(init_mode)
        payload = self._validate_init(mode, init, k)
        node_values = np.zeros((self._num_nodes, k), dtype=self._dtype)
        self._apply_init_inplace(node_values, mode, payload)
        if self._sel_mut.nnz > 0:
            node_values += self._sel_mut.T @ X
        self._propagate_down_inplace(node_values)
        if self._capture_active:
            call = self._call_mem
            assert isinstance(call, ReferenceCall)
            call.node_values = node_values
            call.miss_output = None
            self._publish_call_capture(
                CallCapture(
                    nonce=self._capture_nonce,
                    direction="down",
                    runtime_k=k,
                    active_alloc_keys=frozenset(),
                    meta={"emit_all_nodes": True, "mode": "host"},
                )
            )
        return node_values


@dataclass(frozen=True)
class ReferencePlanPair:
    plan_up: ReferencePlan | None
    plan_down: ReferencePlan | None

    def __post_init__(self) -> None:
        if self.plan_up is None and self.plan_down is None:
            raise ValueError("At least one of plan_up/plan_down must be provided")


__all__ = ["ReferenceBackend", "ReferencePlan", "ReferencePlanPair"]
