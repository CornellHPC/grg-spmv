"""Shared backend scaffolding and helper functions."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, is_dataclass
import logging
import warnings
from typing import Any, Iterator

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.memory import AllocKey, _alloc_keys_for_value
from pygrgl_spmv.backends.types import Direction, InitMode, StoredMatrix

_RESERVED_CAPTURE_META_KEYS = frozenset({"direction", "runtime_k", "active_alloc_keys"})


def _require_struct_dtype(dtype: np.dtype, *, label: str) -> np.dtype:
    dt = np.dtype(dtype)
    if dt not in {np.dtype(np.int32), np.dtype(np.int64)}:
        raise TypeError(f"{label} must use int32 or int64, got {dt}")
    return dt


def _struct_dtype_for_bound(max_value: int) -> np.dtype:
    if int(max_value) < 0:
        raise ValueError(f"structural bound must be non-negative, got {max_value}")
    if int(max_value) > int(np.iinfo(np.int32).max):
        return np.dtype(np.int64)
    return np.dtype(np.int32)


def _layout_struct_dtypes(fmt: str, *, nrows: int, ncols: int, nnz: int) -> tuple[np.dtype, np.dtype]:
    token = str(fmt).strip().lower()
    if token == "csr":
        return _struct_dtype_for_bound(max(int(nnz), 0)), _struct_dtype_for_bound(max(int(ncols) - 1, 0))
    if token == "csc":
        return _struct_dtype_for_bound(max(int(nnz), 0)), _struct_dtype_for_bound(max(int(nrows) - 1, 0))
    if token == "coo":
        return _struct_dtype_for_bound(max(int(nrows) - 1, 0)), _struct_dtype_for_bound(max(int(ncols) - 1, 0))
    raise ValueError(f"unknown sparse format for structural dtype bounds: {fmt!r}")


def _copy_struct_checked(dst: Any, values: Any, *, label: str) -> None:
    dst_arr = np.asarray(dst)
    src_arr = np.asarray(values)
    source = _require_struct_dtype(src_arr.dtype, label=label)
    target = _require_struct_dtype(dst_arr.dtype, label=label)
    if dst_arr.shape != src_arr.shape:
        raise ValueError(f"{label} shape mismatch: expected {dst_arr.shape}, got {src_arr.shape}")
    if source == target:
        np.copyto(dst_arr, src_arr, casting="no")
        return
    if source.itemsize < target.itemsize:
        np.copyto(dst_arr, src_arr, casting="safe")
        return
    if src_arr.size:
        lo = int(np.asarray(src_arr).min())
        hi = int(np.asarray(src_arr).max())
        info = np.iinfo(target)
        if lo < int(info.min) or hi > int(info.max):
            raise ValueError(f"{label} exceeds {target} range")
    np.copyto(dst_arr, src_arr, casting="unsafe")


def _validate_struct_int_array(values: Any, *, label: str) -> None:
    arr = np.asarray(values)
    _require_struct_dtype(arr.dtype, label=label)
    if arr.size and int(arr.min()) < 0:
        raise ValueError(f"{label} must be non-negative")


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
    coalescence_counts: np.ndarray | None
    dtype: np.dtype


@dataclass(frozen=True)
class CallCapture:
    nonce: int
    direction: str
    runtime_k: int
    active_alloc_keys: frozenset[AllocKey] = field(default_factory=frozenset)
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "nonce", int(self.nonce))
        token = str(self.direction)
        if token not in {"up", "down"}:
            raise ValueError(f"Unknown CallCapture direction {self.direction!r}; expected 'up' or 'down'")
        object.__setattr__(self, "direction", token)
        object.__setattr__(self, "runtime_k", int(self.runtime_k))
        normalized: set[AllocKey] = set()
        for key in self.active_alloc_keys:
            if isinstance(key, tuple):
                normalized.add(tuple(key))
                continue
            if isinstance(key, list):
                normalized.add(tuple(key))
                continue
            raise TypeError(f"CallCapture active_alloc_keys entries must be tuple/list, got {type(key).__name__}")
        object.__setattr__(self, "active_alloc_keys", frozenset(normalized))
        meta = dict(self.meta)
        reserved = sorted(_RESERVED_CAPTURE_META_KEYS.intersection(meta))
        if reserved:
            raise ValueError(f"CallCapture.meta contains reserved semantic key(s): {reserved}")
        object.__setattr__(self, "meta", meta)


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
    """Shared backend state, validation, and fail-fast call-capture hooks."""

    _SETUP_MEMORY_FIELDS = (
        "_A_blocks",
        "_sel_mut",
        "_sel_miss",
        "_level_offsets",
        "_coalescence_counts",
        "_xtx_host",
    )
    _SETUP_MEMORY_POLICY: dict[str, str] = {}

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
        self._coalescence_counts = None
        self._xtx_host = None
        self._dtype = np.float64
        self._num_samples = 0
        self._num_nodes = 0
        self._num_mutations = 0
        self._capture_nonce = 0
        self._capture_active = False
        self._last_call_capture: CallCapture | None = None
        self._retained_mem = None
        self._call_mem_type = None
        self._call_mem = None
        self._retained_epoch = 0

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

    def _stored_matrix(self, direction: Direction, *, dst_level: int, src_level: int) -> sp.spmatrix:
        """Return the block in the exact sparse orientation retained by the direction plan."""
        plan = self._require_plan(direction)
        base = self._A_blocks[dst_level][src_level] if direction == Direction.UP else self._A_blocks[src_level][dst_level]
        return base if plan.store == StoredMatrix.N else base.T

    def _apply_setup_state(self, setup: BackendSetup) -> None:
        self._require_memory_installed()
        if self._capture_active:
            raise RuntimeError(f"{self.__class__.__name__} cannot apply setup state while a call capture is active")
        if self._call_mem is not None:
            raise RuntimeError(f"{self.__class__.__name__} cannot apply setup state while call memory is live")
        _validate_struct_int_array(setup.level_offsets, label="level_offsets")
        for dst_level, row in enumerate(setup.A_blocks):
            for src_level, block in enumerate(row):
                if not sp.isspmatrix_csr(block):
                    raise TypeError(f"A_blocks[{dst_level}][{src_level}] must be CSR, got {type(block).__name__}")
                _validate_struct_int_array(block.indices, label=f"A_blocks[{dst_level}][{src_level}].indices")
                _validate_struct_int_array(block.indptr, label=f"A_blocks[{dst_level}][{src_level}].indptr")
        _validate_struct_int_array(setup.sel_mut.indices, label="sel_mut.indices")
        _validate_struct_int_array(setup.sel_mut.indptr, label="sel_mut.indptr")
        _validate_struct_int_array(setup.sel_miss.indices, label="sel_miss.indices")
        _validate_struct_int_array(setup.sel_miss.indptr, label="sel_miss.indptr")
        self._A_blocks = setup.A_blocks
        self._level_offsets = np.asarray(setup.level_offsets)
        self._num_samples = int(setup.num_samples)
        self._num_nodes = int(setup.num_nodes)
        self._num_mutations = int(setup.num_mutations)
        self._sel_mut = setup.sel_mut
        self._sel_miss = setup.sel_miss
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

    @contextmanager
    def _call_capture_scope(self):
        self._require_memory_installed()
        if self._capture_active:
            raise RuntimeError(f"{self.__class__.__name__} call capture is already active")
        if self._call_mem is not None:
            raise RuntimeError(f"{self.__class__.__name__} call memory is unexpectedly live")
        if self._call_mem_type is None:
            raise RuntimeError(f"{self.__class__.__name__} call memory type is not installed")
        self._capture_nonce += 1
        nonce = int(self._capture_nonce)
        self._capture_active = True
        self._last_call_capture = None
        self._call_mem = self._call_mem_type()
        try:
            yield nonce
        finally:
            self._capture_active = False
            self._last_call_capture = None
            self._call_mem = None

    def _publish_call_capture(self, capture: CallCapture) -> None:
        if not self._capture_active:
            raise RuntimeError(f"{self.__class__.__name__} published call capture outside an active capture scope")
        if self._call_mem is None:
            raise RuntimeError(f"{self.__class__.__name__} has no call memory while publishing call capture")
        if int(capture.nonce) != int(self._capture_nonce):
            raise RuntimeError(
                f"Stale call-capture nonce {capture.nonce} for {self.__class__.__name__}; current nonce is {self._capture_nonce}"
            )
        if self._last_call_capture is not None:
            raise RuntimeError(f"{self.__class__.__name__} published duplicate call capture")
        self._last_call_capture = capture

    def _consume_call_capture(
        self,
        *,
        expected_nonce: int,
        expected_direction: Direction,
        expected_k: int,
    ) -> CallCapture:
        capture = self._last_call_capture
        if capture is None:
            raise RuntimeError(f"{self.__class__.__name__} failed to publish call capture")
        if int(capture.nonce) != int(expected_nonce):
            raise RuntimeError(
                f"{self.__class__.__name__} published nonce {capture.nonce}, expected {expected_nonce}"
            )
        if str(capture.direction) != str(expected_direction.value):
            raise RuntimeError(
                f"{self.__class__.__name__} published direction {capture.direction!r}, expected {expected_direction.value!r}"
            )
        if int(capture.runtime_k) != int(expected_k):
            raise RuntimeError(
                f"{self.__class__.__name__} published runtime_k {capture.runtime_k}, expected {expected_k}"
            )
        return capture

    def _require_memory_installed(self) -> None:
        if self._retained_mem is None or not (is_dataclass(self._retained_mem) and not isinstance(self._retained_mem, type)):
            raise RuntimeError(f"{self.__class__.__name__} retained memory root is not installed")
        if self._call_mem_type is None or not (isinstance(self._call_mem_type, type) and is_dataclass(self._call_mem_type)):
            raise RuntimeError(f"{self.__class__.__name__} call memory type is not installed")

    def _install_memory(self, *, retained, call_type) -> None:
        if not (is_dataclass(retained) and not isinstance(retained, type)):
            raise RuntimeError(f"{self.__class__.__name__} retained memory root must be a dataclass instance")
        if not (isinstance(call_type, type) and is_dataclass(call_type)):
            raise RuntimeError(f"{self.__class__.__name__} call memory type must be a dataclass type")
        self._retained_mem = retained
        self._call_mem_type = call_type
        self._call_mem = None
        self._retained_epoch = 0

    @staticmethod
    def _alloc_keys(*values: Any) -> frozenset[AllocKey]:
        keys: set[AllocKey] = set()
        for value in values:
            keys.update(_alloc_keys_for_value(value))
        return frozenset(keys)

    def _bump_retained_epoch(self) -> None:
        self._retained_epoch += 1

    def _assert_setup_memory_contract(self) -> None:
        self._require_memory_installed()
        policy = dict(getattr(self, "_SETUP_MEMORY_POLICY", {}))
        expected = set(self._SETUP_MEMORY_FIELDS)
        actual = set(policy)
        if actual != expected:
            raise RuntimeError(
                f"{self.__class__.__name__} setup memory policy keys mismatch: actual={sorted(actual)!r}, expected={sorted(expected)!r}"
            )
        if self._call_mem is not None:
            raise RuntimeError(f"{self.__class__.__name__} call memory must be absent outside an active capture scope")
        retained_ids = _alloc_keys_for_value(self._retained_mem)
        for name in self._SETUP_MEMORY_FIELDS:
            mode = str(policy[name])
            if mode not in {"retained", "borrowed", "dropped"}:
                raise RuntimeError(f"{self.__class__.__name__} setup memory policy for {name} is invalid: {mode!r}")
            value = getattr(self, name)
            if mode == "dropped":
                if not self._is_dropped_setup_value(value):
                    raise RuntimeError(f"{self.__class__.__name__} expected dropped setup field {name} to be empty")
                continue
            if mode == "retained":
                missing = _alloc_keys_for_value(value) - retained_ids
                if missing:
                    raise RuntimeError(f"{self.__class__.__name__} retained setup field {name} is not represented in backend retained memory")

    @staticmethod
    def _is_dropped_setup_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, dict):
            return len(value) == 0
        if isinstance(value, list):
            return len(value) == 0
        if isinstance(value, tuple):
            return len(value) == 0
        if sp.issparse(value):
            return tuple(value.shape) == (0, 0)
        if isinstance(value, np.ndarray):
            return int(value.size) == 0
        return False

    def setup(self, setup: BackendSetup) -> None:
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
    "CallCapture",
    "BackendSetup",
    "_parse_optional_k_hint",
    "_copy_struct_checked",
    "_layout_struct_dtypes",
    "_require_struct_dtype",
    "_struct_dtype_for_bound",
    "iter_direction_level_pairs",
    "selector_rows_unique_from_csr_indptr",
    "warn_k_hint_mismatch",
]
