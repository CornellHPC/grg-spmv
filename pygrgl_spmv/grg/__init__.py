"""Public SpmvGRG API and matmul orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pygrgl

from pygrgl_spmv.backends import BackendBase, ReferenceBackend, ReferencePlanPair
from pygrgl_spmv.backends.types import Direction, InitMode, parse_direction
from pygrgl_spmv.grg.artifact import artifact_path_for_grg, load_grg_spmv, save_grg_spmv
from pygrgl_spmv.grg.compile import CompiledOperatorState, compile_grg

_NUCLEOTIDE_DECODE = {0b00: "A", 0b01: "T", 0b10: "C", 0b11: "G"}


def _decode_allele(buf: np.ndarray, idx: int) -> str:
    byte = int(buf[idx])
    length = (byte >> 6) & 0b11
    return "".join(_NUCLEOTIDE_DECODE[(byte >> (j * 2)) & 0b11] for j in range(length))


class SpmvGRG:
    """Matmul-focused GRG operator for genotype matrix G (num_samples x num_mutations)."""

    def __init__(
        self,
        path,
        backend: BackendBase,
        dtype,
        index_dtype,
        artifact_dir: str | Path = "pygrgl_spmv_artifacts",
    ):
        self._dtype = np.dtype(dtype)
        self._index_dtype = np.dtype(index_dtype)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(logging.WARNING)
        if not isinstance(backend, BackendBase):
            raise TypeError(f"SpmvGRG backend must be a BackendBase instance, got {type(backend).__name__}")
        self._backend = backend

        source_path = Path(path)
        self._artifact_path: Path
        if source_path.suffix == ".grg":
            artifact_root = Path(artifact_dir).expanduser()
            self._artifact_path = artifact_path_for_grg(source_path, artifact_root)
            self._artifact_path.parent.mkdir(parents=True, exist_ok=True)
            self._state = self._load_or_build_from_grg(source_path=source_path, artifact_path=self._artifact_path)
        elif source_path.suffix == ".grg_spmv":
            self._artifact_path = source_path
            self._state = load_grg_spmv(self._artifact_path, self._dtype, self._index_dtype)
        else:
            raise ValueError(f"Unsupported SpmvGRG input path {source_path}; expected .grg or .grg_spmv")

        self._backend.setup(self._state.to_backend_setup(self._dtype))
        self._state.A_blocks = None

    def _load_or_build_from_grg(self, *, source_path: Path, artifact_path: Path) -> CompiledOperatorState:
        if artifact_path.exists():
            self._logger.info("Loading SpmvGRG artifact from %s", artifact_path)
            try:
                return load_grg_spmv(artifact_path, self._dtype, self._index_dtype)
            except (KeyError, ValueError) as exc:
                self._logger.warning(
                    "SpmvGRG artifact at %s is invalid (%s); rebuilding from %s",
                    artifact_path,
                    exc,
                    source_path,
                )
        else:
            self._logger.info("Building SpmvGRG from %s", source_path)
        return self._build_and_save_artifact(source_path=source_path, artifact_path=artifact_path)

    def _build_and_save_artifact(self, *, source_path: Path, artifact_path: Path) -> CompiledOperatorState:
        grg = pygrgl.load_immutable_grg(str(source_path), load_up_edges=True)
        state = compile_grg(grg, dtype=self._dtype, index_dtype=self._index_dtype)
        self._build_init_biases(state)
        save_grg_spmv(state, artifact_path)
        return state

    def _build_init_biases(self, state: CompiledOperatorState) -> None:
        helper = ReferenceBackend(
            pair=ReferencePlanPair(
                plan_up=ReferenceBackend.plan(fmt="CSR", store="N", k_hint=None),
                plan_down=ReferenceBackend.plan(fmt="CSC", store="T", k_hint=None),
            ),
            log_level="WARNING",
        )
        helper.setup(state.to_backend_setup(self._dtype))
        zeros_up = np.zeros((state.num_samples, 1), dtype=self._dtype)
        zeros_down = np.zeros((state.num_mutations, 1), dtype=self._dtype)
        init_vec = np.ones(1, dtype=self._dtype)

        up_bias, _ = helper.run_up(zeros_up, init_mode=InitMode.VECTOR, init=init_vec, need_miss_output=False)
        down_bias = helper.run_down(zeros_down, init_mode=InitMode.VECTOR, init=init_vec, miss=None)
        state.init_vector_up_bias = np.asarray(up_bias[:, 0], dtype=self._dtype).reshape(state.num_mutations)
        state.init_vector_down_bias = np.asarray(down_bias[:, 0], dtype=self._dtype).reshape(state.num_samples)
        state.init_xtx_up_bias = None
        state.init_xtx_down_bias = None
        if state.coalescence_counts is not None:
            up_xtx, _ = helper.run_up(zeros_up, init_mode=InitMode.XTX, init=None, need_miss_output=False)
            down_xtx = helper.run_down(zeros_down, init_mode=InitMode.XTX, init=None, miss=None)
            state.init_xtx_up_bias = np.asarray(up_xtx[:, 0], dtype=self._dtype).reshape(state.num_mutations)
            state.init_xtx_down_bias = np.asarray(down_xtx[:, 0], dtype=self._dtype).reshape(state.num_samples)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_samples, self.num_mutations)

    @property
    def artifact_path(self) -> Path:
        return self._artifact_path

    @property
    def num_samples(self):
        return self._state.num_samples

    @property
    def num_individuals(self):
        return self._state.num_individuals

    @property
    def num_mutations(self):
        return self._state.num_mutations

    @property
    def ploidy(self):
        return self._state.ploidy

    @property
    def num_nodes(self):
        return self._state.num_nodes

    @property
    def num_edges(self):
        return self._state.num_edges

    @property
    def has_missing_data(self):
        return self._state.has_missing_data

    @property
    def level_offsets(self):
        return self._state.level_offsets

    @property
    def sample_perm(self):
        return self._state.sample_perm

    @property
    def inv_sample_perm(self):
        return self._state.inv_sample_perm

    @property
    def node_perm(self):
        return self._state.node_perm

    @property
    def inv_node_perm(self):
        return self._state.inv_node_perm

    @property
    def sel_mut(self):
        return self._state.sel_mut

    @property
    def sel_miss(self):
        return self._state.sel_miss

    @property
    def sample_to_individual(self):
        return self._state.sample_to_individual

    @property
    def coalescence_counts(self):
        return self._state.coalescence_counts

    @property
    def init_vector_up_bias(self):
        return self._state.init_vector_up_bias

    @property
    def init_vector_down_bias(self):
        return self._state.init_vector_down_bias

    @property
    def init_xtx_up_bias(self):
        return self._state.init_xtx_up_bias

    @property
    def init_xtx_down_bias(self):
        return self._state.init_xtx_down_bias

    def get_mutation_by_id(self, mutation_id: int):
        idx = int(mutation_id)
        if idx < 0 or idx >= self.num_mutations:
            raise IndexError(f"Mutation id out of range: {mutation_id}")
        return pygrgl.Mutation(
            float(self._state.mutation_positions[idx]),
            _decode_allele(self._state.mutation_alleles, idx),
            _decode_allele(self._state.mutation_ref_alleles, idx),
            float(self._state.mutation_times[idx]),
        )

    def _parse_direction(self, direction: str | Direction | pygrgl.TraversalDirection) -> Direction:
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

    def _parse_init(self, init: str | np.ndarray | None, rows: int, input_dtype: np.dtype) -> tuple[InitMode, np.ndarray | None]:
        if init is None:
            return InitMode.NONE, None
        if isinstance(init, str):
            if init != "xtx":
                raise ValueError(f"Unexpected init value: {init}")
            if self.coalescence_counts is None:
                raise ValueError(
                    "init='xtx' requires per-node coalescence counts in the GRG. "
                    "This SpmvGRG instance was loaded without coalescence counts."
                )
            return InitMode.XTX, None

        init_arr = np.asarray(init)
        if init_arr.dtype != input_dtype:
            raise TypeError(f"The init matrix must match the dtype of the input matrix. Got: {init_arr.dtype}")

        if init_arr.ndim == 1:
            if init_arr.shape[0] != rows:
                raise ValueError("If init has a single dimension, it must match the number of rows in the input matrix")
            return InitMode.VECTOR, np.asarray(init_arr, dtype=self._dtype, order="C")

        if init_arr.ndim == 2:
            if init_arr.shape != (rows, self.num_nodes):
                raise ValueError(
                    f"If init is a matrix, it must match the dimensions ({rows}, {self.num_nodes})"
                )
            init_nodes = init_arr[:, self.node_perm].T
            return InitMode.MATRIX, np.asarray(init_nodes, dtype=self._dtype, order="C")

        raise ValueError("init must be None, 'xtx', a vector, or a matrix")

    def _validate_miss(self, miss: np.ndarray, rows: int, direction: Direction, input_dtype: np.dtype) -> np.ndarray:
        miss_arr = np.asarray(miss)
        if miss_arr.dtype != input_dtype:
            raise TypeError(f'The "miss" input must match the dtype of the input matrix. Got: {miss_arr.dtype}')
        if miss_arr.ndim != 2:
            raise ValueError(f'"miss" must be a two-dimension numpy array (matrix). ndim={miss_arr.ndim}')
        if miss_arr.shape[0] != rows:
            raise ValueError(f'"miss" has {miss_arr.shape[0]} rows, but must match the input/output matrices ({rows})')
        if miss_arr.shape[1] != self.num_mutations:
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

    def _apply_endpoint_init_bias(
        self,
        result_internal: np.ndarray,
        *,
        direction: Direction,
        init_mode: InitMode,
        init_payload: np.ndarray | None,
    ) -> None:
        if init_mode == InitMode.XTX:
            bias = self.init_xtx_up_bias if direction == Direction.UP else self.init_xtx_down_bias
            assert bias is not None
            result_internal += bias[:, None]
            return
        if init_mode == InitMode.VECTOR:
            bias = self.init_vector_up_bias if direction == Direction.UP else self.init_vector_down_bias
            assert init_payload is not None
            assert bias is not None
            result_internal += bias[:, None] * init_payload[None, :]

    def _finish_endpoint_output(self, result_internal: np.ndarray) -> np.ndarray:
        return result_internal.T.astype(self._dtype, copy=False)

    def _finish_node_output(self, node_values_internal: np.ndarray) -> np.ndarray:
        node_values = np.asarray(node_values_internal, dtype=self._dtype, order="C")
        reordered = node_values[self.inv_node_perm]
        return reordered.T.astype(self._dtype, copy=False)

    def matmul(
        self,
        input,
        direction,
        emit_all_nodes: bool = False,
        by_individual: bool = False,
        init=None,
        miss=None,
    ):
        X_in = np.asarray(input)
        if X_in.ndim != 2:
            raise ValueError("matmul() only supports two-dimensional numpy arrays as input.")
        rows, cols = X_in.shape
        if rows == 0 or cols == 0:
            raise ValueError("matmul() requires non-zero dimensions.")

        direction_name = self._parse_direction(direction)
        expected_input_cols = self.num_individuals if by_individual and direction_name == Direction.UP else (
            self.num_samples if direction_name == Direction.UP else self.num_mutations
        )
        if cols != expected_input_cols:
            if direction_name == Direction.UP:
                raise ValueError(
                    "Input matrix has wrong number of columns for UP direction "
                    "(numSamples or numIndividuals depending on by_individual)"
                )
            raise ValueError("Input matrix has wrong number of columns for DOWN direction (numMutations)")

        if emit_all_nodes and miss is not None:
            raise RuntimeError('The "miss" parameter cannot be mixed with the "emit_all_nodes" parameter')
        if init is not None and miss is not None:
            raise ValueError('The "miss" parameter cannot be mixed with the "init" parameter')

        init_mode, init_payload = self._parse_init(init, rows, X_in.dtype)
        if emit_all_nodes:
            backend_init_mode = init_mode
            backend_init_payload = init_payload
        else:
            backend_init_mode, backend_init_payload = (
                (InitMode.NONE, None) if init_mode in (InitMode.VECTOR, InitMode.XTX) else (init_mode, init_payload)
            )

        input_matrix = np.asarray(X_in, dtype=self._dtype, order="C")
        input_internal = input_matrix.T

        if direction_name == Direction.UP:
            if by_individual:
                input_internal = input_internal[self.sample_to_individual]
            if emit_all_nodes:
                node_values = self._backend.run_up_nodes(
                    input_internal,
                    init_mode=backend_init_mode,
                    init=backend_init_payload,
                )
                return self._finish_node_output(node_values)

            miss_output = None
            if miss is not None:
                miss_output = self._validate_miss(miss, rows, direction_name, X_in.dtype)
            result_internal, miss_internal = self._backend.run_up(
                input_internal,
                init_mode=backend_init_mode,
                init=backend_init_payload,
                need_miss_output=(miss_output is not None),
            )
            if init_mode != InitMode.NONE:
                self._apply_endpoint_init_bias(
                    result_internal,
                    direction=direction_name,
                    init_mode=init_mode,
                    init_payload=init_payload,
                )
            if miss_output is not None and miss_internal is not None:
                miss_output += miss_internal.T.astype(miss_output.dtype, copy=False)
            return self._finish_endpoint_output(result_internal)

        if emit_all_nodes:
            node_values = self._backend.run_down_nodes(
                input_internal,
                init_mode=backend_init_mode,
                init=backend_init_payload,
            )
            return self._finish_node_output(node_values)

        miss_internal = None
        miss_output = None
        if miss is not None:
            miss_output = self._validate_miss(miss, rows, direction_name, X_in.dtype)
            miss_internal = np.asarray(miss_output.T, dtype=self._dtype, order="C")

        result_internal = self._backend.run_down(
            input_internal,
            miss=miss_internal,
            init_mode=backend_init_mode,
            init=backend_init_payload,
        )
        if init_mode != InitMode.NONE:
            self._apply_endpoint_init_bias(
                result_internal,
                direction=direction_name,
                init_mode=init_mode,
                init_payload=init_payload,
            )
        if by_individual:
            result_by_individual = np.zeros((self.num_individuals, rows), dtype=self._dtype)
            np.add.at(result_by_individual, self.sample_to_individual, result_internal)
            result_internal = result_by_individual
        return self._finish_endpoint_output(result_internal)


__all__ = ["SpmvGRG"]
