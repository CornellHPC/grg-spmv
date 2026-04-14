"""MKL backend — Intel MKL-accelerated fused GRG matmul traversal."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends import (
    BackendBase,
    CallCapture,
    BackendSetup,
    iter_direction_level_pairs,
    selector_rows_unique_from_csr_indptr,
    warn_k_hint_mismatch,
)
from pygrgl_spmv.memory import alloc_field, child_field
from pygrgl_spmv.backends.mkl.ffi import (
    MklSparseHandle,
    mkl_get_max_threads,
    mkl_set_num_threads,
    mkl_set_num_threads_local,
)
from pygrgl_spmv._rss import rss_bytes
from pygrgl_spmv.backends.types import (
    Direction,
    InitMode,
    StoredMatrix,
    parse_init_mode,
)
from pygrgl_spmv.backends.mkl.plan import MklPlan, MklPlanPair


@dataclass(frozen=True)
class _MklBlockOp:
    """One sparse block application in a level wavefront."""

    src_level: int
    handle: MklSparseHandle
    transpose: bool
    nnz: int


@dataclass(frozen=True)
class _MklDirectionSpec:
    direction: Direction
    plan: MklPlan
    thread_count: int
    store_blocks: bool
    ops_owner: Direction
    stage_name: str


@dataclass
class MklCall:
    node_values: np.ndarray | None = alloc_field(
        label="node_state", kind="state", owner="backend", retention="call", activity="yes", default=None
    )
    miss_output: np.ndarray | None = alloc_field(
        label="miss_output", kind="output", owner="backend", retention="call", activity="yes", default=None
    )
    level_ms: np.ndarray | None = alloc_field(
        label="level_ms", kind="temporary", owner="backend", retention="call", activity="yes", default=None
    )


@dataclass
class MklRetained:
    blocks_up: list[sp.spmatrix] = alloc_field(
        label="blocks_up", kind="sparse", owner="backend", retention="persistent", activity="always", default_factory=list
    )
    blocks_down: list[sp.spmatrix] = alloc_field(
        label="blocks_down", kind="sparse", owner="backend", retention="persistent", activity="always", default_factory=list
    )
    selector_mut_rows: np.ndarray | None = alloc_field(
        label="selector_mut", kind="selector", owner="backend", retention="persistent", activity="always", default=None
    )
    selector_mut_cols: np.ndarray | None = alloc_field(
        label="selector_mut", kind="selector", owner="backend", retention="persistent", activity="always", default=None
    )
    selector_miss_rows: np.ndarray | None = alloc_field(
        label="selector_miss", kind="selector", owner="backend", retention="persistent", activity="always", default=None
    )
    selector_miss_cols: np.ndarray | None = alloc_field(
        label="selector_miss", kind="selector", owner="backend", retention="persistent", activity="always", default=None
    )
    xtx_host: np.ndarray | None = alloc_field(
        label="xtx_host", kind="init", owner="backend", retention="persistent", activity="always", default=None
    )
    shared_ones: np.ndarray | None = alloc_field(
        label="shared_ones", kind="sparse", owner="backend", retention="persistent", activity="always", default=None
    )


def _needs_transpose(direction: Direction, store: StoredMatrix) -> bool:
    return (direction == Direction.DOWN) != (store == StoredMatrix.T)


def _level_call_stats(ops_by_level: list[list[object]]) -> tuple[np.ndarray, np.ndarray]:
    calls = np.fromiter((len(ops) for ops in ops_by_level), dtype=np.int32, count=len(ops_by_level))
    nnz = np.fromiter(
        (sum(int(getattr(op, "nnz", 0)) for op in ops) for ops in ops_by_level),
        dtype=np.int64,
        count=len(ops_by_level),
    )
    return calls, nnz


def _log_level_timing(
    logger: logging.Logger,
    *,
    direction: Direction,
    calls: np.ndarray,
    nnz: np.ndarray,
    level_ms: np.ndarray,
) -> None:
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


def _a_blocks_nbytes(a_blocks: list[list[sp.spmatrix]]) -> tuple[int, int, int]:
    """Return (data_bytes, indices_bytes, indptr_bytes) across all A_blocks scipy matrices."""
    data = indices = indptr = 0
    for row in a_blocks:
        for mat in row:
            if mat is None or mat.nnz == 0:
                continue
            data += mat.data.nbytes
            if hasattr(mat, "indices"):
                indices += mat.indices.nbytes
                indptr += mat.indptr.nbytes
            else:
                indices += mat.row.nbytes + mat.col.nbytes
    return data, indices, indptr


def _blocks_nbytes(blocks: list[list[MklSparseHandle | None]]) -> tuple[int, int, int]:
    """Return (values_bytes, indices_bytes, indptr_bytes) across all handles in the grid."""
    values = indices = indptr = 0
    for row in blocks:
        for handle in row:
            if handle is None:
                continue
            mat = handle._mat
            values += mat.data.nbytes
            if hasattr(mat, "indices"):  # CSR or CSC
                indices += mat.indices.nbytes
                indptr += mat.indptr.nbytes
            else:  # COO
                indices += mat.row.nbytes + mat.col.nbytes
    return values, indices, indptr


class MklBackend(BackendBase):
    """MKL-accelerated backend using the Inspector-Executor Sparse BLAS API.

    Persistent MKL handles are created once in setup().
    Each run_up()/run_down() call performs one fused traversal.
    """

    _SETUP_MEMORY_POLICY = {
        "_A_blocks": "dropped",
        "_sel_mut": "dropped",
        "_sel_miss": "dropped",
        "_level_offsets": "borrowed",
        "_coalescence_counts": "borrowed",
        "_xtx_host": "retained",
    }

    def __init__(
        self,
        *,
        pair: MklPlanPair,
        log_level: str = "WARNING",
        instrumentation: bool = False,
    ):
        super().__init__(
            plan_up=pair.plan_up,
            plan_down=pair.plan_down,
            log_level=log_level,
            instrumentation=instrumentation,
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
        self._selector_rows: dict[str, np.ndarray] = {}
        self._selector_cols: dict[str, np.ndarray] = {}
        self._selector_row_unique: dict[str, bool] = {}
        self._install_memory(retained=MklRetained(), call_type=MklCall)

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

    def _direction_spec(self, direction: Direction) -> _MklDirectionSpec | None:
        plan = self._plan_for(direction)
        if plan is None:
            return None
        return _MklDirectionSpec(
            direction=direction,
            plan=plan,
            thread_count=self._thread_count_for(direction),
            store_blocks=self._store_blocks_up if direction == Direction.UP else self._store_blocks_down,
            ops_owner=Direction.UP if (self._up_ops_owner if direction == Direction.UP else self._down_ops_owner) == "up" else Direction.DOWN,
            stage_name="run_up" if direction == Direction.UP else "run_down",
        )

    def _require_direction_spec(self, direction: Direction) -> _MklDirectionSpec:
        spec = self._direction_spec(direction)
        if spec is None:
            raise ValueError(f"{direction.value.upper()} plan is not configured")
        return spec

    def _grid_for(self, direction: Direction) -> list[list[MklSparseHandle | None]]:
        return self._blocks_up if direction == Direction.UP else self._blocks_down

    def _build_direction_handles(
        self,
        spec: _MklDirectionSpec,
    ) -> list[list[MklSparseHandle | None]]:
        num_levels = len(self._level_offsets) - 1
        if not spec.store_blocks:
            return [[] for _ in range(num_levels)]

        rows = [
            [None] * (dst_level if spec.direction == Direction.UP else max(num_levels - dst_level - 1, 0))
            for dst_level in range(num_levels)
        ]
        for dst_level, src_level, row_index in iter_direction_level_pairs(spec.direction, num_levels):
            stored = self._stored_matrix(spec.direction, dst_level=dst_level, src_level=src_level)
            rows[dst_level][row_index] = (
                None if stored.nnz == 0
                else MklSparseHandle(stored, spec.plan.fmt.value.lower(), shared_values=self._shared_ones)
            )
        return rows

    @staticmethod
    def _handle_payload_arrays(handles: list[list[MklSparseHandle | None]]) -> list[np.ndarray]:
        arrays: list[np.ndarray] = []
        for row in handles:
            for handle in row:
                if handle is None:
                    continue
                mat = getattr(handle, "_mat", None)
                if mat is None:
                    continue
                for attr in ("data", "indices", "indptr", "row", "col"):
                    value = getattr(mat, attr, None)
                    if value is not None:
                        arrays.append(np.asarray(value))
        return arrays

    @staticmethod
    def _handle_payload_matrices(handles: list[list[MklSparseHandle | None]]) -> list[sp.spmatrix]:
        mats: list[sp.spmatrix] = []
        for row in handles:
            for handle in row:
                if handle is None:
                    continue
                mat = getattr(handle, "_mat", None)
                if mat is not None:
                    mats.append(mat)
        return mats

    def _sync_retained_root(self) -> None:
        retained = self._retained_mem
        retained.blocks_up = self._handle_payload_matrices(self._blocks_up)
        retained.blocks_down = self._handle_payload_matrices(self._blocks_down)
        retained.selector_mut_rows = self._selector_rows.get("mut")
        retained.selector_mut_cols = self._selector_cols.get("mut")
        retained.selector_miss_rows = self._selector_rows.get("miss")
        retained.selector_miss_cols = self._selector_cols.get("miss")
        retained.xtx_host = self._xtx_host
        retained.shared_ones = self._shared_ones

    def _build_direction_ops(self, spec: _MklDirectionSpec) -> list[list[_MklBlockOp]]:
        num_levels = len(self._level_offsets) - 1
        ops: list[list[_MklBlockOp]] = [[] for _ in range(num_levels)]
        owner_direction = spec.ops_owner
        owner_plan = self._plan_for(owner_direction)
        if owner_plan is None:
            raise RuntimeError(
                f"{spec.direction.value.upper()} ops requested shared {owner_direction.value.upper()} storage without a configured plan"
            )
        owner_grid = self._grid_for(owner_direction)

        for dst_level, src_level, row_index in iter_direction_level_pairs(spec.direction, num_levels):
            owner_dst = dst_level if owner_direction == spec.direction else src_level
            owner_src = src_level if owner_direction == spec.direction else dst_level
            owner_row_index = owner_src if owner_direction == Direction.UP else owner_src - owner_dst - 1
            handle = owner_grid[owner_dst][owner_row_index]
            if handle is None:
                continue
            ops[dst_level].append(
                _MklBlockOp(
                    src_level=src_level,
                    handle=handle,
                    transpose=_needs_transpose(spec.direction, owner_plan.store),
                    nnz=int(handle.nnz),
                )
            )
        return ops

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
                            "optimize": plan.optimize,
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
                    entry["optimize"] = entry["optimize"] or plan.optimize
        expected = 1000
        for entry in usage.values():
            handle = entry["handle"]
            assert isinstance(handle, MklSparseHandle)
            transpose = bool(entry["transpose"])
            if entry["optimize"]:
                handle.set_mv_hint(transpose=transpose, expected_calls=expected)
                hint_k = entry["k_hint"]
                if hint_k is not None and int(hint_k) > 1:
                    handle.set_mm_hint(int(hint_k), transpose=transpose, expected_calls=expected)
                handle.optimize()

    def setup(
        self,
        setup: BackendSetup,
    ) -> None:
        rss_before = rss_bytes()
        mkl_set_num_threads(self._n_threads_setup)
        self._apply_setup_state(setup)
        self._dtype = np.float64
        self._xtx_host = None
        if self._coalescence_counts is not None:
            self._xtx_host = (2.0 * self._coalescence_counts.astype(self._dtype, copy=False)).reshape(self._num_nodes)

        requested_dtype = np.dtype(setup.dtype)
        if requested_dtype != np.float64:
            self._logger.info(
                "MKL backend uses float64 kernels; requested dtype %s is cast to float64 internally",
                requested_dtype,
            )

        num_levels = len(self._level_offsets) - 1
        self._blocks_up = [[] for _ in range(num_levels)]
        self._blocks_down = [[] for _ in range(num_levels)]
        self._ops_up = [[] for _ in range(num_levels)]
        self._ops_down = [[] for _ in range(num_levels)]

        a_blocks_data, a_blocks_idx, a_blocks_iptr = _a_blocks_nbytes(self._A_blocks)

        max_nnz = max(
            (mat.nnz for row in self._A_blocks for mat in row if mat is not None and mat.nnz > 0),
            default=0,
        )
        self._shared_ones = np.ones(max(max_nnz, 1), dtype=np.float64)

        up_spec = self._direction_spec(Direction.UP)
        down_spec = self._direction_spec(Direction.DOWN)
        if up_spec is not None:
            self._blocks_up = self._build_direction_handles(up_spec)
            self._ops_up = self._build_direction_ops(up_spec)
        if down_spec is not None:
            self._blocks_down = self._build_direction_handles(down_spec)
            self._ops_down = self._build_direction_ops(down_spec)
        rss_after_handles = rss_bytes()

        self._selector_rows = {}
        self._selector_cols = {}
        self._selector_row_unique = {}
        self._selector_rows["mut"], self._selector_cols["mut"], self._selector_row_unique["mut"] = (
            self._build_selector_indices(self._sel_mut)
        )
        self._selector_rows["miss"], self._selector_cols["miss"], self._selector_row_unique["miss"] = (
            self._build_selector_indices(self._sel_miss)
        )
        self._configure_handle_hints()
        rss_after_hints = rss_bytes()

        self._log_memory_stats(
            rss_before, rss_after_handles, rss_after_hints,
            a_blocks_data, a_blocks_idx, a_blocks_iptr,
        )

        self._logger.info(
            (
                "MklBackend setup: fmt_up=%s fmt_down=%s k_hint=%s n_threads=%s (actual=%d) "
                "store_up=%s store_down=%s up_owner=%s down_owner=%s optimize_up=%s optimize_down=%s"
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
            None if self._plan_up is None else self._plan_up.optimize,
            None if self._plan_down is None else self._plan_down.optimize,
        )
        self._refresh_level_stats()
        self._sync_retained_root()
        self._bump_retained_epoch()
        self._A_blocks = []
        self._sel_mut = sp.csr_matrix((0, 0))
        self._sel_miss = sp.csr_matrix((0, 0))
        self._assert_setup_memory_contract()

    def _refresh_level_stats(self) -> None:
        self._ops_up_calls, self._ops_up_nnz = _level_call_stats(self._ops_up)
        self._ops_down_calls, self._ops_down_nnz = _level_call_stats(self._ops_down)

    def _log_memory_stats(
        self,
        rss_before: int | None,
        rss_after_handles: int | None,
        rss_after_hints: int | None,
        a_blocks_data: int,
        a_blocks_idx: int,
        a_blocks_iptr: int,
    ) -> None:
        if not self._logger.isEnabledFor(logging.INFO):
            return

        MiB = 1024.0 * 1024.0

        up_val, up_idx, up_iptr = _blocks_nbytes(self._blocks_up)
        dn_val, dn_idx, dn_iptr = _blocks_nbytes(self._blocks_down)

        sel_mut_bytes = sum(
            a.nbytes for a in (self._selector_rows.get("mut"), self._selector_cols.get("mut"))
            if a is not None
        )
        sel_miss_bytes = sum(
            a.nbytes for a in (self._selector_rows.get("miss"), self._selector_cols.get("miss"))
            if a is not None
        )
        xtx_bytes = self._xtx_host.nbytes if self._xtx_host is not None else 0

        shared_actual_bytes = self._shared_ones.nbytes if self._shared_ones is not None else 0
        theoretical = shared_actual_bytes + up_idx + up_iptr + dn_idx + dn_iptr + sel_mut_bytes + sel_miss_bytes + xtx_bytes

        self._logger.info(
            "MklBackend memory components: "
            "blocks_up_values=%.1fMiB blocks_up_indices=%.1fMiB blocks_up_indptr=%.1fMiB "
            "blocks_down_values=%.1fMiB blocks_down_indices=%.1fMiB blocks_down_indptr=%.1fMiB "
            "selector_mut=%.1fMiB selector_miss=%.1fMiB xtx_host=%.1fMiB",
            up_val / MiB, up_idx / MiB, up_iptr / MiB,
            dn_val / MiB, dn_idx / MiB, dn_iptr / MiB,
            sel_mut_bytes / MiB, sel_miss_bytes / MiB, xtx_bytes / MiB,
        )
        self._logger.info(
            "MklBackend memory shared_values: logical_values=%.1fMiB actual_buffer=%.1fMiB savings=%.1fMiB",
            (up_val + dn_val) / MiB,
            shared_actual_bytes / MiB,
            (up_val + dn_val - shared_actual_bytes) / MiB,
        )

        a_blocks_total = a_blocks_data + a_blocks_idx + a_blocks_iptr
        self._logger.info(
            "MklBackend memory a_blocks_input: data=%.1fMiB indices=%.1fMiB indptr=%.1fMiB total=%.1fMiB (freed after setup)",
            a_blocks_data / MiB, a_blocks_idx / MiB, a_blocks_iptr / MiB, a_blocks_total / MiB,
        )

        if all(v is not None for v in (rss_before, rss_after_handles, rss_after_hints)):
            delta_handles = rss_after_handles - rss_before
            delta_optimize = rss_after_hints - rss_after_handles
            self._logger.info(
                "MklBackend memory stages: rss_delta_handles=%+.1fMiB rss_delta_mkl_optimize=%+.1fMiB",
                delta_handles / MiB,
                delta_optimize / MiB,
            )

        rss_after = rss_after_hints
        if rss_after is not None and rss_before is not None:
            delta = rss_after - rss_before
            self._logger.info(
                "MklBackend memory summary: theoretical=%.1fMiB rss=%.1fMiB rss_delta=%+.1fMiB",
                theoretical / MiB,
                rss_after / MiB,
                delta / MiB,
            )
        else:
            self._logger.info(
                "MklBackend memory summary: theoretical=%.1fMiB rss=unavailable",
                theoretical / MiB,
            )

    def _log_wavefront_levels(self, direction: Direction, level_ms: np.ndarray) -> None:
        match direction:
            case Direction.UP:
                calls = self._ops_up_calls
                nnz = self._ops_up_nnz
            case Direction.DOWN:
                calls = self._ops_down_calls
                nnz = self._ops_down_nnz
            case _:
                raise ValueError(f"Unknown direction for wavefront profile: {direction!r}")
        _log_level_timing(self._logger, direction=direction, calls=calls, nnz=nnz, level_ms=level_ms)

    def _propagate_direction_inplace(
        self,
        spec: _MklDirectionSpec,
        node_values: np.ndarray,
        level_ms: np.ndarray | None = None,
    ) -> None:
        off = self._level_offsets
        k = node_values.shape[1]
        ops_by_level = self._ops_up if spec.direction == Direction.UP else self._ops_down
        level_iter = (
            range(1, len(off) - 1)
            if spec.direction == Direction.UP
            else range(len(off) - 2, -1, -1)
        )
        for h in level_iter:
            lo, hi = int(off[h]), int(off[h + 1])
            t_level = perf_counter() if level_ms is not None else None
            for op in ops_by_level[h]:
                src_lo, src_hi = int(off[op.src_level]), int(off[op.src_level + 1])
                if k == 1:
                    op.handle.mv(
                        node_values[src_lo:src_hi, 0],
                        node_values[lo:hi, 0],
                        alpha=1.0,
                        beta=1.0,
                        transpose=op.transpose,
                    )
                else:
                    op.handle.mm(
                        node_values[src_lo:src_hi],
                        node_values[lo:hi],
                        alpha=1.0,
                        beta=1.0,
                        transpose=op.transpose,
                    )
            if level_ms is not None and t_level is not None:
                level_ms[h] = (perf_counter() - t_level) * 1000.0

    def _selector_forward(self, node_values: np.ndarray, selector: str) -> np.ndarray:
        rows = self._selector_rows[selector]
        cols = self._selector_cols[selector]
        unique_rows = self._selector_row_unique[selector]
        k = node_values.shape[1]
        out = np.zeros((self._num_mutations, k), dtype=self._dtype)
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

    def _run_direction(
        self,
        spec: _MklDirectionSpec,
        primary: np.ndarray,
        *,
        miss: np.ndarray | None,
        init_mode: InitMode,
        init: np.ndarray | None,
        need_miss_output: bool,
        emit_all_nodes: bool,
    ) -> tuple[np.ndarray, np.ndarray | None] | np.ndarray:
        mkl_set_num_threads_local(spec.thread_count)
        x, k = self._normalize_primary_input(direction=spec.direction, primary=primary)
        if spec.plan.k_hint is not None and int(k) != int(spec.plan.k_hint):
            warn_k_hint_mismatch(
                backend="MKL",
                direction=spec.direction,
                runtime_k=k,
                k_hint=int(spec.plan.k_hint),
            )

        miss_arr = self._normalize_down_miss_input(miss, k=k) if spec.direction == Direction.DOWN else None

        mode = parse_init_mode(init_mode)
        init_payload = self._validate_init(mode, init, k)

        node_values = np.zeros((self._num_nodes, k), dtype=self._dtype)
        self._apply_init_inplace(node_values, mode, init_payload)

        if spec.direction == Direction.UP:
            np.add(node_values[: self._num_samples], x, out=node_values[: self._num_samples])
        else:
            self._selector_backward_add(x, "mut", node_values)
            if miss_arr is not None:
                self._selector_backward_add(miss_arr, "miss", node_values)

        track_wave = self._instrumentation and self._logger.isEnabledFor(logging.DEBUG)
        level_ms = np.zeros(len(self._level_offsets) - 1, dtype=np.float64) if track_wave else None
        self._logger.info("grg=%s direction=%s Begin with %d threads", self._grg_name or "<unknown>", spec.direction.value, spec.thread_count)
        t0 = perf_counter()
        self._propagate_direction_inplace(spec, node_values, level_ms=level_ms)
        self._logger.info("grg=%s direction=%s time=%.3fms", self._grg_name or "<unknown>", spec.direction.value, (perf_counter() - t0) * 1000.0)

        if emit_all_nodes:
            if track_wave and level_ms is not None:
                self._log_wavefront_levels(spec.direction, level_ms)
            if self._capture_active:
                call = self._call_mem
                assert isinstance(call, MklCall)
                call.node_values = node_values
                call.miss_output = None
                call.level_ms = level_ms
                self._publish_call_capture(
                    CallCapture(
                        nonce=self._capture_nonce,
                        direction=spec.direction.value,
                        runtime_k=k,
                        active_alloc_keys=frozenset(),
                        meta={
                            "emit_all_nodes": True,
                            "mode": "instrumented" if self._instrumentation else "n/a",
                        },
                    )
                )
            return node_values

        if spec.direction == Direction.UP:
            out_mut = self._selector_forward(node_values, "mut")

            out_miss = None
            if need_miss_output:
                out_miss = self._selector_forward(node_values, "miss")

            if track_wave and level_ms is not None:
                self._log_wavefront_levels(spec.direction, level_ms)
            if self._capture_active:
                call = self._call_mem
                assert isinstance(call, MklCall)
                call.node_values = node_values
                call.miss_output = out_miss
                call.level_ms = level_ms
                self._publish_call_capture(
                    CallCapture(
                        nonce=self._capture_nonce,
                        direction=spec.direction.value,
                        runtime_k=k,
                        active_alloc_keys=frozenset(),
                        meta={
                            "need_miss_output": bool(need_miss_output),
                            "mode": "instrumented" if self._instrumentation else "n/a",
                        },
                    )
                )
            return out_mut, out_miss

        out = node_values[: self._num_samples]

        if track_wave and level_ms is not None:
            self._log_wavefront_levels(spec.direction, level_ms)
        if self._capture_active:
            call = self._call_mem
            assert isinstance(call, MklCall)
            call.node_values = node_values
            call.miss_output = None
            call.level_ms = level_ms
            self._publish_call_capture(
                CallCapture(
                    nonce=self._capture_nonce,
                    direction=spec.direction.value,
                    runtime_k=k,
                    active_alloc_keys=frozenset(),
                    meta={
                        "has_miss_input": bool(miss_arr is not None),
                        "mode": "instrumented" if self._instrumentation else "n/a",
                    },
                )
            )
        return out

    def run_up(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
        need_miss_output: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        spec = self._require_direction_spec(Direction.UP)
        out_mut, out_miss = self._run_direction(
            spec,
            primary,
            miss=None,
            init_mode=init_mode,
            init=init,
            need_miss_output=need_miss_output,
            emit_all_nodes=False,
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
        spec = self._require_direction_spec(Direction.DOWN)
        out = self._run_direction(
            spec,
            primary,
            miss=miss,
            init_mode=init_mode,
            init=init,
            need_miss_output=False,
            emit_all_nodes=False,
        )
        return out

    def run_up_nodes(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
    ) -> np.ndarray:
        spec = self._require_direction_spec(Direction.UP)
        out = self._run_direction(
            spec,
            primary,
            miss=None,
            init_mode=init_mode,
            init=init,
            need_miss_output=False,
            emit_all_nodes=True,
        )
        return out

    def run_down_nodes(
        self,
        primary: np.ndarray,
        *,
        init_mode: InitMode,
        init: np.ndarray | None = None,
    ) -> np.ndarray:
        spec = self._require_direction_spec(Direction.DOWN)
        out = self._run_direction(
            spec,
            primary,
            miss=None,
            init_mode=init_mode,
            init=init,
            need_miss_output=False,
            emit_all_nodes=True,
        )
        return out

__all__ = ["MklBackend", "MklPlan"]
