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
    BackendBase,
    BackendSetup,
    _sparse_host_bytes,
    estimate_common_host_static_bytes,
    estimate_sparse_payload_bytes,
    iter_direction_level_pairs,
    selector_rows_unique_from_csr_indptr,
    warn_k_hint_mismatch,
)
from pygrgl_spmv.backends.memory import RuntimeBytes, StaticBytes
from pygrgl_spmv.backends.mkl.ffi import (
    MklSparseHandle,
    mkl_get_max_threads,
    mkl_set_num_threads,
)
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


def _stored_block(base_block: sp.spmatrix, store: StoredMatrix) -> sp.spmatrix:
    return base_block if store == StoredMatrix.N else base_block.T


def _needs_transpose(direction: Direction, store: StoredMatrix) -> bool:
    return (direction == Direction.DOWN) != (store == StoredMatrix.T)


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


class MklBackend(BackendBase):
    """
    MKL-accelerated backend using the Inspector-Executor Sparse BLAS API.

    Persistent MKL handles are created once in setup().
    Each run_up()/run_down() call performs one fused traversal.
    """

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
        A_blocks: List[List[sp.csr_matrix]],
    ) -> list[list[MklSparseHandle | None]]:
        num_levels = len(self._level_offsets) - 1
        if not spec.store_blocks:
            return [[] for _ in range(num_levels)]

        rows = [
            [None] * (dst_level if spec.direction == Direction.UP else max(num_levels - dst_level - 1, 0))
            for dst_level in range(num_levels)
        ]
        for dst_level, src_level, row_index in iter_direction_level_pairs(spec.direction, num_levels):
            base_block = (
                A_blocks[dst_level][src_level]
                if spec.direction == Direction.UP
                else A_blocks[src_level][dst_level]
            )
            stored = _stored_block(base_block, spec.plan.store)
            rows[dst_level][row_index] = (
                None if stored.nnz == 0 else MklSparseHandle(stored, spec.plan.fmt.value.lower())
            )
        return rows

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
        setup: BackendSetup,
    ) -> None:
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

        up_spec = self._direction_spec(Direction.UP)
        down_spec = self._direction_spec(Direction.DOWN)
        if up_spec is not None:
            self._blocks_up = self._build_direction_handles(up_spec, self._A_blocks)
            self._ops_up = self._build_direction_ops(up_spec)
        if down_spec is not None:
            self._blocks_down = self._build_direction_handles(down_spec, self._A_blocks)
            self._ops_down = self._build_direction_ops(down_spec)

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

        self.mem_usage.reset()
        common_host = estimate_common_host_static_bytes(
            level_offsets=self._level_offsets,
            sample_perm=self._sample_perm,
            inv_sample_perm=self._inv_sample_perm,
            coalescence_counts=self._coalescence_counts,
            xtx_init=self._xtx_host,
        )
        self.mem_usage.host_static.level_offsets = common_host.level_offsets
        self.mem_usage.host_static.sample_perm = common_host.sample_perm
        self.mem_usage.host_static.inv_sample_perm = common_host.inv_sample_perm
        self.mem_usage.host_static.coalescence_counts = common_host.coalescence_counts
        self.mem_usage.host_static.xtx_init = common_host.xtx_init
        self.mem_usage.host_static.blocks_up = int(
            sum(_sparse_host_bytes(getattr(h, "_mat", None)) for row in self._blocks_up for h in row)
        )
        self.mem_usage.host_static.blocks_down = int(
            sum(_sparse_host_bytes(getattr(h, "_mat", None)) for row in self._blocks_down for h in row)
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
        self._refresh_level_stats()

    def _refresh_level_stats(self) -> None:
        self._ops_up_calls, self._ops_up_nnz = _level_call_stats(self._ops_up)
        self._ops_down_calls, self._ops_down_nnz = _level_call_stats(self._ops_down)

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
        mkl_set_num_threads(spec.thread_count)
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
            np.add(node_values[: self._num_samples], x[self._sample_perm], out=node_values[: self._num_samples])
        else:
            self._selector_backward_add(x, "mut", node_values)
            if miss_arr is not None:
                self._selector_backward_add(miss_arr, "miss", node_values)

        track_wave = self._instrumentation and self._logger.isEnabledFor(logging.DEBUG)
        level_ms = np.zeros(len(self._level_offsets) - 1, dtype=np.float64) if track_wave else None
        self._propagate_direction_inplace(spec, node_values, level_ms=level_ms)

        if emit_all_nodes:
            if track_wave and level_ms is not None:
                self._log_wavefront_levels(spec.direction, level_ms)
            self.mem_usage.record(
                stage=spec.stage_name,
                runtime_k=k,
                host_runtime=RuntimeBytes(
                    level_buffers=int(node_values.nbytes),
                    inputs=int(x.nbytes + (0 if miss_arr is None else miss_arr.nbytes)),
                    outputs=int(node_values.nbytes),
                    aux=int(
                        (0 if init_payload is None else init_payload.nbytes)
                        + (0 if level_ms is None else level_ms.nbytes)
                    ),
                ),
                meta={
                    "direction": spec.direction.value,
                    "emit_all_nodes": True,
                    "mode": "instrumented" if self._instrumentation else "n/a",
                },
            )
            return node_values

        if spec.direction == Direction.UP:
            out_mut = self._selector_forward(node_values, "mut")

            out_miss = None
            if need_miss_output:
                out_miss = self._selector_forward(node_values, "miss")

            if track_wave and level_ms is not None:
                self._log_wavefront_levels(spec.direction, level_ms)
            self.mem_usage.record(
                stage=spec.stage_name,
                runtime_k=k,
                host_runtime=RuntimeBytes(
                    level_buffers=int(node_values.nbytes),
                    inputs=int(x.nbytes),
                    outputs=int(out_mut.nbytes + (0 if out_miss is None else out_miss.nbytes)),
                    aux=int(
                        (0 if init_payload is None else init_payload.nbytes)
                        + (0 if level_ms is None else level_ms.nbytes)
                    ),
                ),
                meta={
                    "direction": spec.direction.value,
                    "need_miss_output": bool(need_miss_output),
                    "mode": "instrumented" if self._instrumentation else "n/a",
                },
            )
            return out_mut, out_miss

        out = node_values[self._inv_sample_perm]

        if track_wave and level_ms is not None:
            self._log_wavefront_levels(spec.direction, level_ms)
        self.mem_usage.record(
            stage=spec.stage_name,
            runtime_k=k,
            host_runtime=RuntimeBytes(
                level_buffers=int(node_values.nbytes),
                inputs=int(x.nbytes + (0 if miss_arr is None else miss_arr.nbytes)),
                outputs=int(out.nbytes),
                aux=int(
                    (0 if init_payload is None else init_payload.nbytes)
                    + (0 if level_ms is None else level_ms.nbytes)
                ),
            ),
            meta={
                "direction": spec.direction.value,
                "has_miss_input": bool(miss_arr is not None),
                "mode": "instrumented" if self._instrumentation else "n/a",
            },
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

    def estimate_static_bytes(self) -> tuple[StaticBytes, StaticBytes]:
        host = estimate_common_host_static_bytes(
            level_offsets=self._level_offsets,
            sample_perm=self._sample_perm,
            inv_sample_perm=self._inv_sample_perm,
            coalescence_counts=self._coalescence_counts,
            xtx_init=self._xtx_host,
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
