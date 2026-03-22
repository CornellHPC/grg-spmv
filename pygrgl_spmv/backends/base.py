"""Shared backend scaffolding and helper functions."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import warnings
from typing import Any, Iterator

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends.memory import MemoryUsage, StaticBytes
from pygrgl_spmv.backends.types import Direction, InitMode


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
class BackendSetup:
    """Normalized setup payload shared by all backend implementations."""

    A_blocks: list[list[sp.csr_matrix]]
    level_offsets: np.ndarray
    num_samples: int
    num_mutations: int
    num_nodes: int
    sel_mut: sp.csr_matrix
    sel_miss: sp.csr_matrix
    sample_perm: np.ndarray
    inv_sample_perm: np.ndarray
    coalescence_counts: np.ndarray | None
    dtype: np.dtype


def iter_direction_level_pairs(direction: Direction, H: int) -> Iterator[tuple[int, int, int]]:
    """Yield ``(dst_level, src_level, row_index)`` tuples in execution order."""
    if H < 0:
        raise ValueError(f"Number of levels must be non-negative, got {H}")
    match direction:
        case Direction.UP:
            for dst_level in range(H):
                for src_level in range(dst_level):
                    yield dst_level, src_level, src_level
        case Direction.DOWN:
            for dst_level in range(H):
                for src_level in range(H - 1, dst_level, -1):
                    yield dst_level, src_level, src_level - dst_level - 1
        case _:
            raise ValueError(f"Unknown direction {direction!r}")


def _sparse_host_bytes(mat: object | None) -> int:
    if mat is None:
        return 0
    obj = getattr(mat, "_mat", mat)
    data = getattr(obj, "data", None)
    if data is None:
        return 0
    total = int(data.nbytes)
    if hasattr(obj, "indices"):
        total += int(obj.indices.nbytes)
    if hasattr(obj, "indptr"):
        total += int(obj.indptr.nbytes)
    if hasattr(obj, "row"):
        total += int(obj.row.nbytes)
    if hasattr(obj, "col"):
        total += int(obj.col.nbytes)
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


def warn_k_hint_mismatch(*, backend: str, direction: Direction, runtime_k: int, k_hint: int) -> None:
    warnings.warn(
        (
            f"{backend} backend runtime k={runtime_k} does not match k_hint={k_hint}; "
            f"{direction.value} traversal continues on non-hinted path."
        ),
        RuntimeWarning,
        stacklevel=3,
    )


def effective_k_hint(*, instrumentation: bool, k_hint: int | None) -> int | None:
    return None if instrumentation else k_hint


def warn_instrumentation_ignores_k_hint(*, backend: str, direction: Direction, k_hint: int) -> None:
    warnings.warn(
        (
            f"{backend} {direction.value} graph capture/replay disabled because instrumentation=True "
            f"ignores k_hint={k_hint} and uses the effective k_hint=none path."
        ),
        RuntimeWarning,
        stacklevel=3,
    )


class BackendBase:
    """Shared backend state, validation, and memory tracking."""

    def __init__(
        self,
        *,
        plan_up,
        plan_down,
        log_level: str = "WARNING",
        instrumentation: bool = False,
    ) -> None:
        self._plan_up = plan_up
        self._plan_down = plan_down
        if self._plan_up is None and self._plan_down is None:
            raise ValueError("At least one of plan_up/plan_down must be provided")

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._logger.setLevel(getattr(logging, str(log_level).upper(), logging.WARNING))
        self._instrumentation = bool(instrumentation)

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

        self._A_blocks: list[list[sp.csr_matrix]] = []
        self._level_offsets = np.empty(0, dtype=np.int64)
        self._sel_mut = sp.csr_matrix((0, 0))
        self._sel_miss = sp.csr_matrix((0, 0))
        self._sample_perm = np.empty(0, dtype=np.int64)
        self._inv_sample_perm = np.empty(0, dtype=np.int64)
        self._coalescence_counts = None
        self._xtx_host = None
        self._dtype = np.float64
        self._num_samples = 0
        self._num_nodes = 0
        self._num_mutations = 0
        self.mem_usage = MemoryUsage()

    def _require_plan(self, direction: Direction):
        plan = self._plan_for(direction)
        if plan is None:
            raise ValueError(f"{direction.value.upper()} plan is not configured")
        return plan

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

    def _plan_for(self, direction: Direction):
        return self._plan_up if direction == Direction.UP else self._plan_down

    def _configured_directions(self) -> tuple[Direction, ...]:
        directions: list[Direction] = []
        if self._plan_up is not None:
            directions.append(Direction.UP)
        if self._plan_down is not None:
            directions.append(Direction.DOWN)
        return tuple(directions)

    def _apply_setup_state(self, setup: BackendSetup) -> None:
        self._A_blocks = setup.A_blocks
        self._level_offsets = np.asarray(setup.level_offsets)
        self._num_samples = int(setup.num_samples)
        self._num_nodes = int(setup.num_nodes)
        self._num_mutations = int(setup.num_mutations)
        self._sel_mut = setup.sel_mut
        self._sel_miss = setup.sel_miss
        self._sample_perm = np.asarray(setup.sample_perm)
        self._inv_sample_perm = np.asarray(setup.inv_sample_perm)
        self._coalescence_counts = (
            None
            if setup.coalescence_counts is None
            else np.asarray(setup.coalescence_counts, dtype=np.int64)
        )
        self._dtype = np.dtype(setup.dtype)
        self._xtx_host = None
        if self._coalescence_counts is not None:
            self._xtx_host = (2.0 * self._coalescence_counts).astype(
                self._dtype,
                copy=False,
            ).reshape(self._num_nodes)

    def _validate_init(self, init_mode: InitMode, init: np.ndarray | None, k: int) -> np.ndarray | None:
        match init_mode:
            case InitMode.NONE:
                if init is not None:
                    raise ValueError("init payload provided with init_mode=none")
                return None
            case InitMode.XTX:
                if init is not None:
                    raise ValueError("init payload must be None when init_mode=xtx")
                if self._coalescence_counts is None:
                    raise ValueError("init_mode=xtx requires GRG coalescence counts")
                return None
            case InitMode.VECTOR:
                arr = np.asarray(init, dtype=self._dtype, order="C")
                if arr.ndim != 1 or arr.shape[0] != k:
                    raise ValueError(f"init vector must have shape ({k},), got {arr.shape}")
                return arr
            case InitMode.MATRIX:
                arr = np.asarray(init, dtype=self._dtype, order="C")
                if arr.ndim != 2 or arr.shape != (self._num_nodes, k):
                    raise ValueError(f"init matrix must have shape ({self._num_nodes}, {k}), got {arr.shape}")
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
                if self._xtx_host is None:
                    raise ValueError("init_mode=xtx requires GRG coalescence counts")
                np.add(node_values, self._xtx_host[:, None], out=node_values)
            case InitMode.VECTOR:
                assert init_payload is not None
                np.add(node_values, init_payload[None, :], out=node_values)
            case InitMode.MATRIX:
                assert init_payload is not None
                np.add(node_values, init_payload, out=node_values)
            case _:
                raise ValueError(f"Unknown init mode: {init_mode!r}")

    def _normalize_primary_input(
        self,
        *,
        direction: Direction,
        primary: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        x = np.asarray(primary, dtype=self._dtype, order="C")
        if x.ndim != 2:
            raise ValueError(f"primary input must be 2D, got shape {x.shape}")
        expected_rows = self._num_samples if direction == Direction.UP else self._num_mutations
        if x.shape[0] != expected_rows:
            raise ValueError(
                f"{direction.value.upper()} primary input must have {expected_rows} rows, got {x.shape[0]}"
            )
        return x, int(x.shape[1])

    def _normalize_down_miss_input(self, miss: np.ndarray | None, *, k: int) -> np.ndarray | None:
        if miss is None:
            return None
        miss_arr = np.asarray(miss, dtype=self._dtype, order="C")
        if miss_arr.shape != (self._num_mutations, k):
            raise ValueError(f"miss input must have shape ({self._num_mutations}, {k}), got {miss_arr.shape}")
        return miss_arr

    def setup(self, setup: BackendSetup) -> None:
        raise NotImplementedError

    def estimate_static_bytes(self) -> tuple[StaticBytes, StaticBytes]:
        raise NotImplementedError

    def run_up_nodes(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def run_down_nodes(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


__all__ = [
    "BackendBase",
    "BackendSetup",
    "_parse_optional_k_hint",
    "_sparse_host_bytes",
    "estimate_common_host_static_bytes",
    "estimate_sparse_payload_bytes",
    "iter_direction_level_pairs",
    "selector_rows_unique_from_csr_indptr",
    "warn_k_hint_mismatch",
]
