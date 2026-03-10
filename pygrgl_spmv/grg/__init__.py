"""Public SpmvGRG API and matmul orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pygrgl

from pygrgl_spmv.backends import ReferenceBackend
from pygrgl_spmv.backends.registry import create_backend
from pygrgl_spmv.backends.types import Direction, InitMode, parse_direction
from pygrgl_spmv.grg.cache import cache_path_for_grg, load_operator_npz, save_operator_npz
from pygrgl_spmv.grg.compile import CompiledOperatorState, compile_grg


class SpmvGRG:
    """Matmul-focused GRG operator for genotype matrix G (n x m)."""

    def __init__(self, path, backend_config: dict[str, Any], dtype, index_dtype, cache_dir: str | Path = "pygrgl_spmv_cache"):
        self._dtype = np.dtype(dtype)
        self._index_dtype = np.dtype(index_dtype)
        log_level = str(backend_config.get("log_level", "WARNING"))
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(getattr(logging, log_level.upper(), logging.WARNING))
        self._backend = create_backend(backend_config)

        path = Path(path)
        assert path.suffix == ".grg", f"Not a GRG file: {path}"
        cache_root = Path(cache_dir).expanduser()
        npz_path = cache_path_for_grg(path, cache_root)
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        self._state = self._load_state(path=path, npz_path=npz_path)
        self._backend.setup(self._state.to_backend_setup(self._dtype))
        self._state.A_blocks = None

    def _load_state(self, *, path: Path, npz_path: Path) -> CompiledOperatorState:
        if npz_path.exists():
            self._logger.info("Loading cached SpmvGRG from %s", npz_path)
            try:
                return load_operator_npz(npz_path, self._dtype, self._index_dtype)
            except (KeyError, ValueError) as exc:
                self._logger.warning(
                    "Cached SpmvGRG at %s is invalid (%s); rebuilding from %s",
                    npz_path,
                    exc,
                    path,
                )
        else:
            self._logger.info("Building SpmvGRG with %s", path)
        return self._build_and_cache_state(path=path, npz_path=npz_path)

    def _build_and_cache_state(self, *, path: Path, npz_path: Path) -> CompiledOperatorState:
        grg = pygrgl.load_immutable_grg(str(path), load_up_edges=True)
        self._state = compile_grg(grg, dtype=self._dtype, index_dtype=self._index_dtype)
        self._build_init_bias_cache()
        save_operator_npz(self._state, npz_path)
        return self._state

    def _build_init_bias_cache(self) -> None:
        """Precompute per-output init bias vectors for fast init paths."""
        helper = ReferenceBackend(
            plan_up=ReferenceBackend.plan(fmt="CSR", store="N", k_hint=None),
            plan_down=ReferenceBackend.plan(fmt="CSC", store="T", k_hint=None),
            log_level="WARNING",
        )
        helper.setup(self._state.to_backend_setup(self._dtype))
        zeros_up = np.zeros((self.n, 1), dtype=self._dtype)
        zeros_down = np.zeros((self.m, 1), dtype=self._dtype)
        init_vec = np.ones(1, dtype=self._dtype)

        up_bias, _ = helper.run_up(zeros_up, init_mode=InitMode.VECTOR, init=init_vec, need_miss_output=False)
        down_bias = helper.run_down(zeros_down, init_mode=InitMode.VECTOR, init=init_vec, miss=None)
        self._state.init_vector_up_bias = np.asarray(up_bias[:, 0], dtype=self._dtype).reshape(self.m)
        self._state.init_vector_down_bias = np.asarray(down_bias[:, 0], dtype=self._dtype).reshape(self.n)
        self._state.init_xtx_up_bias = None
        self._state.init_xtx_down_bias = None
        if self.coalescence_counts is not None:
            up_xtx, _ = helper.run_up(zeros_up, init_mode=InitMode.XTX, init=None, need_miss_output=False)
            down_xtx = helper.run_down(zeros_down, init_mode=InitMode.XTX, init=None, miss=None)
            self._state.init_xtx_up_bias = np.asarray(up_xtx[:, 0], dtype=self._dtype).reshape(self.m)
            self._state.init_xtx_down_bias = np.asarray(down_xtx[:, 0], dtype=self._dtype).reshape(self.n)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.m)

    def _parse_direction(self, direction: Any) -> Direction:
        match direction:
            case str():
                return parse_direction(direction)
            case pygrgl.TraversalDirection.UP:
                return Direction.UP
            case pygrgl.TraversalDirection.DOWN:
                return Direction.DOWN
            case _:
                raise ValueError(
                    f"Unknown direction: {direction!r}. Expected 'up', 'down', "
                    "pygrgl.TraversalDirection.UP, or pygrgl.TraversalDirection.DOWN"
                )

    def _parse_init(
        self,
        init: Any,
        rows: int,
        input_dtype: np.dtype,
    ) -> tuple[InitMode, np.ndarray | None]:
        if init is None:
            return InitMode.NONE, None
        if isinstance(init, str):
            if init != "xtx":
                raise ValueError(f"Unexpected init value: {init}")
            if self.coalescence_counts is None:
                raise ValueError(
                    "init='xtx' requires per-node coalescence counts in the GRG. "
                    "This GRG was loaded without coalescence counts."
                )
            return InitMode.XTX, None

        init_arr = np.asarray(init)
        if init_arr.dtype != input_dtype:
            raise TypeError(
                f"The init matrix must match the dtype of the input matrix. Got: {init_arr.dtype}"
            )

        if init_arr.ndim == 1:
            if init_arr.shape[0] != rows:
                raise ValueError(
                    "If init has a single dimension, it must match the number of rows in the input matrix"
                )
            return InitMode.VECTOR, np.asarray(init_arr, dtype=self._dtype, order="C")

        if init_arr.ndim == 2:
            if init_arr.shape != (rows, self.K):
                raise ValueError("If init is a matrix, it must match the dimensions ROW x NODES")
            init_nodes = init_arr[:, self.node_perm].T
            return InitMode.MATRIX, np.asarray(init_nodes, dtype=self._dtype, order="C")

        raise ValueError("init must be None, 'xtx', a vector, or a matrix")

    def _validate_miss(
        self,
        miss: Any,
        rows: int,
        direction: Direction,
        input_dtype: np.dtype,
    ) -> np.ndarray:
        miss_arr = np.asarray(miss)
        if miss_arr.dtype != input_dtype:
            raise TypeError(
                f'The "miss" input must match the dtype of the input matrix. Got: {miss_arr.dtype}'
            )
        if miss_arr.ndim != 2:
            raise ValueError(f'"miss" must be a two-dimension numpy array (matrix). ndim={miss_arr.ndim}')
        if miss_arr.shape[0] != rows:
            raise ValueError(
                f'"miss" has {miss_arr.shape[0]} rows, but must match the input/output matrices ({rows})'
            )
        if miss_arr.shape[1] != self.m:
            match direction:
                case Direction.DOWN:
                    raise ValueError(
                        'The "miss" matrix must match the number of columns in the input matrix. '
                        f"Got: {miss_arr.shape[1]}"
                    )
                case Direction.UP:
                    raise ValueError(
                        'The "miss" matrix must match the number of columns in the output matrix. '
                        f"Got: {miss_arr.shape[1]}"
                    )
                case _:
                    raise ValueError(f"Unsupported direction for miss validation: {direction!r}")
        return miss_arr

    def _apply_init_bias(
        self,
        result_col: np.ndarray,
        *,
        direction: Direction,
        init_mode: InitMode,
        init_payload: np.ndarray | None,
    ) -> None:
        if init_mode == InitMode.XTX:
            bias = self._init_xtx_up_bias if direction == Direction.UP else self._init_xtx_down_bias
            assert bias is not None
            result_col += bias[:, None]
            return
        if init_mode == InitMode.VECTOR:
            bias = self._init_vector_up_bias if direction == Direction.UP else self._init_vector_down_bias
            assert init_payload is not None
            assert bias is not None
            result_col += bias[:, None] * init_payload[None, :]

    def _finish_matmul(
        self,
        result_col: np.ndarray,
        *,
        direction: Direction,
        rows: int,
        cols: int,
        by_individual: bool,
        init_mode: InitMode,
        has_miss: bool,
        timing_pairs: list[tuple[str, float]],
        total_t0: float,
        record_timing,
    ) -> np.ndarray:
        t0 = perf_counter()
        out = result_col.T.astype(self._dtype, copy=False)
        record_timing("output_cast", t0)
        if self._logger.isEnabledFor(logging.INFO):
            self._print_matmul_timings(
                direction=direction,
                rows=rows,
                cols=cols,
                by_individual=by_individual,
                init_mode=init_mode.value,
                miss=has_miss,
                timings=timing_pairs,
                total_ms=(perf_counter() - total_t0) * 1000.0,
            )
        return out

    def matmul(
        self,
        input_matrix: np.ndarray,
        direction: Any,
        by_individual: bool = False,
        init: Any = None,
        miss: Any = None,
    ) -> np.ndarray:
        """pygrgl-compatible matrix multiplication."""
        timing_pairs: list[tuple[str, float]] = []
        total_t0 = perf_counter()

        def _record(name: str, t0: float) -> None:
            if self._logger.isEnabledFor(logging.INFO):
                timing_pairs.append((name, (perf_counter() - t0) * 1000.0))

        t0 = perf_counter()
        X_in = np.asarray(input_matrix)
        if X_in.ndim != 2:
            raise ValueError("matmul() only supports two-dimensional numpy arrays as input.")
        rows, cols = X_in.shape
        if rows == 0 or cols == 0:
            raise ValueError("matmul() requires non-zero dimensions.")

        direction_name = self._parse_direction(direction)
        expect_sample_cols = self.num_individuals if by_individual else self.n
        if direction_name == Direction.UP:
            if cols != expect_sample_cols:
                raise ValueError(
                    "Input matrix has wrong number of columns for UP direction "
                    "(numSamples or numIndividuals depending on by_individual)"
                )
        else:
            if cols != self.m:
                raise ValueError("Input matrix has wrong number of columns for DOWN direction (numMutations)")
        _record("validate_input", t0)

        if init is not None and miss is not None:
            raise ValueError('The "miss" parameter cannot be mixed with the "init" parameter')

        t0 = perf_counter()
        init_mode, init_payload = self._parse_init(init, rows, X_in.dtype)
        _record("parse_init", t0)
        backend_init_mode, backend_init_payload = (
            (InitMode.NONE, None) if init_mode in (InitMode.VECTOR, InitMode.XTX) else (init_mode, init_payload)
        )

        t0 = perf_counter()
        X = np.asarray(X_in, dtype=self._dtype, order="C")
        X_col = X.T
        _record("cast_and_transpose", t0)

        if direction_name == Direction.UP:
            miss_arr = None
            if miss is not None:
                t0 = perf_counter()
                miss_arr = self._validate_miss(miss, rows, direction_name, X_in.dtype)
                _record("validate_miss", t0)

            if by_individual:
                t0 = perf_counter()
                X_col = X_col[self.sample_to_individual]
                _record("by_individual_expand", t0)

            t0 = perf_counter()
            result_col, miss_col = self._backend.run_up(
                X_col,
                init_mode=backend_init_mode,
                init=backend_init_payload,
                need_miss_output=(miss_arr is not None),
            )
            _record("backend_run", t0)

            if init_mode != InitMode.NONE:
                t0 = perf_counter()
                self._apply_init_bias(
                    result_col,
                    direction=direction_name,
                    init_mode=init_mode,
                    init_payload=init_payload,
                )
                _record("init_bias_add", t0)

            if miss_arr is not None and miss_col is not None:
                t0 = perf_counter()
                miss_arr += miss_col.T.astype(miss_arr.dtype, copy=False)
                _record("write_miss", t0)

            return self._finish_matmul(
                result_col,
                direction=direction_name,
                rows=rows,
                cols=cols,
                by_individual=by_individual,
                init_mode=init_mode,
                has_miss=(miss is not None),
                timing_pairs=timing_pairs,
                total_t0=total_t0,
                record_timing=_record,
            )

        miss_col = None
        miss_arr = None
        if miss is not None:
            t0 = perf_counter()
            miss_arr = self._validate_miss(miss, rows, direction_name, X_in.dtype)
            miss_col = np.asarray(miss_arr.T, dtype=self._dtype, order="C")
            _record("validate_miss", t0)

        t0 = perf_counter()
        result_col = self._backend.run_down(
            X_col,
            miss=miss_col,
            init_mode=backend_init_mode,
            init=backend_init_payload,
        )
        _record("backend_run", t0)

        if init_mode != InitMode.NONE:
            t0 = perf_counter()
            self._apply_init_bias(
                result_col,
                direction=direction_name,
                init_mode=init_mode,
                init_payload=init_payload,
            )
            _record("init_bias_add", t0)

        if by_individual:
            t0 = perf_counter()
            result_by_individual = np.zeros((self.num_individuals, rows), dtype=self._dtype)
            np.add.at(result_by_individual, self.sample_to_individual, result_col)
            result_col = result_by_individual
            _record("by_individual_reduce", t0)

        return self._finish_matmul(
            result_col,
            direction=direction_name,
            rows=rows,
            cols=cols,
            by_individual=by_individual,
            init_mode=init_mode,
            has_miss=(miss is not None),
            timing_pairs=timing_pairs,
            total_t0=total_t0,
            record_timing=_record,
        )

    def _print_matmul_timings(
        self,
        direction: Direction,
        rows: int,
        cols: int,
        by_individual: bool,
        init_mode: str,
        miss: bool,
        timings: list[tuple[str, float]],
        total_ms: float,
    ) -> None:
        miss_mode = "on" if miss else "off"
        timing_body = " ".join(f"{stage}={ms:.3f}ms" for stage, ms in timings)
        self._logger.info(
            "SpmvGRG.matmul[%s] rows=%d cols=%d by_individual=%s init=%s miss=%s "
            "path=fused_single_traversal=on %s total=%.3fms",
            direction.value,
            rows,
            cols,
            by_individual,
            init_mode,
            miss_mode,
            timing_body,
            total_ms,
        )


def _state_property(name: str) -> property:
    def getter(self: SpmvGRG):
        return getattr(self._state, name)
    return property(getter)


for _public_name, _state_name in (
    ("level_offsets", "level_offsets"),
    ("sample_perm", "sample_perm"),
    ("_inv_sample_perm", "inv_sample_perm"),
    ("node_perm", "node_perm"),
    ("_inv_node_perm", "inv_node_perm"),
    ("sel_mut", "sel_mut"),
    ("sel_miss", "sel_miss"),
    ("n", "n"),
    ("m", "m"),
    ("K", "K"),
    ("ploidy", "ploidy"),
    ("num_individuals", "num_individuals"),
    ("sample_to_individual", "sample_to_individual"),
    ("coalescence_counts", "coalescence_counts"),
    ("_init_vector_up_bias", "init_vector_up_bias"),
    ("_init_vector_down_bias", "init_vector_down_bias"),
    ("_init_xtx_up_bias", "init_xtx_up_bias"),
    ("_init_xtx_down_bias", "init_xtx_down_bias"),
):
    setattr(SpmvGRG, _public_name, _state_property(_state_name))
del _public_name
del _state_name

__all__ = ["SpmvGRG"]
