"""Public SpmvGRG API and matmul orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pygrgl
import scipy.sparse as sp

from pygrgl_spmv.backends import BackendBase, ReferenceBackend, ReferencePlanPair
from pygrgl_spmv.backends.types import Direction, InitMode, parse_direction
from pygrgl_spmv.grg.artifact import artifact_path_for_grg, load_grg_spmv, save_grg_spmv
from pygrgl_spmv.grg.compile import (
    CompiledOperatorState,
    VALID_INTRA_BLOCK_ORDERINGS,
    VALID_ORDERINGS,
    compile_grg,
)
from pygrgl_spmv.memory import (
    MemoryLedger,
    alloc_field,
    capture_snapshot,
)


@dataclass
class OperatorRetainedMem:
    level_offsets: np.ndarray = alloc_field(label="level_offsets", kind="mapping", owner="operator", retention="persistent", activity="always")
    sample_rows: np.ndarray = alloc_field(label="sample_rows", kind="mapping", owner="operator", retention="persistent", activity="always")
    node_perm: np.ndarray = alloc_field(label="node_perm", kind="mapping", owner="operator", retention="persistent", activity="always")
    inv_node_perm: np.ndarray = alloc_field(label="inv_node_perm", kind="mapping", owner="operator", retention="persistent", activity="always")
    sample_to_individual: np.ndarray = alloc_field(label="sample_to_individual", kind="mapping", owner="operator", retention="persistent", activity="always")
    mutation_positions: np.ndarray = alloc_field(label="mutation_positions", kind="tables", owner="operator", retention="persistent", activity="always")
    mutation_times: np.ndarray = alloc_field(label="mutation_times", kind="tables", owner="operator", retention="persistent", activity="always")
    mutation_alleles: np.ndarray = alloc_field(label="mutation_alleles", kind="tables", owner="operator", retention="persistent", activity="always")
    mutation_allele_offsets: np.ndarray = alloc_field(label="mutation_allele_offsets", kind="tables", owner="operator", retention="persistent", activity="always")
    mutation_ref_alleles: np.ndarray = alloc_field(label="mutation_ref_alleles", kind="tables", owner="operator", retention="persistent", activity="always")
    mutation_ref_allele_offsets: np.ndarray = alloc_field(label="mutation_ref_allele_offsets", kind="tables", owner="operator", retention="persistent", activity="always")
    sel_mut: sp.spmatrix = alloc_field(label="sel_mut", kind="selector", owner="operator", retention="persistent", activity="always")
    sel_miss: sp.spmatrix = alloc_field(label="sel_miss", kind="selector", owner="operator", retention="persistent", activity="always")
    coalescence_counts: np.ndarray | None = alloc_field(label="coalescence_counts", kind="coalescence", owner="operator", retention="persistent", activity="always", default=None)
    init_vector_up_bias: np.ndarray | None = alloc_field(label="init_vector_up_bias", kind="init", owner="operator", retention="persistent", activity="always", default=None)
    init_vector_down_bias: np.ndarray | None = alloc_field(label="init_vector_down_bias", kind="init", owner="operator", retention="persistent", activity="always", default=None)
    init_xtx_up_bias: np.ndarray | None = alloc_field(label="init_xtx_up_bias", kind="init", owner="operator", retention="persistent", activity="always", default=None)
    init_xtx_down_bias: np.ndarray | None = alloc_field(label="init_xtx_down_bias", kind="init", owner="operator", retention="persistent", activity="always", default=None)


@dataclass
class OperatorCallMem:
    caller_input: np.ndarray | None = alloc_field(label="input", kind="input", owner="caller", retention="call", activity="yes", default=None)
    caller_miss_input: np.ndarray | None = alloc_field(label="miss_input", kind="input", owner="caller", retention="call", activity="yes", default=None)
    caller_miss_output: np.ndarray | None = alloc_field(label="miss_output", kind="output", owner="caller", retention="call", activity="yes", default=None)
    caller_init: np.ndarray | None = alloc_field(label="init", kind="input", owner="caller", retention="call", activity="yes", default=None)
    caller_output: np.ndarray | None = alloc_field(label="output", kind="output", owner="caller", retention="call", activity="yes", default=None)
    caller_aux_output: np.ndarray | None = alloc_field(label="aux_output", kind="output", owner="caller", retention="call", activity="yes", default=None)
    input_internal: np.ndarray | None = alloc_field(label="input_internal", kind="input", owner="operator", retention="call", activity="yes", default=None)
    miss_internal: np.ndarray | None = alloc_field(label="miss_internal", kind="input", owner="operator", retention="call", activity="yes", default=None)
    init_payload: np.ndarray | None = alloc_field(label="init_payload", kind="init", owner="operator", retention="call", activity="yes", default=None)
    backend_init_payload: np.ndarray | None = alloc_field(label="backend_init_payload", kind="init", owner="operator", retention="call", activity="yes", default=None)
    input_by_individual: np.ndarray | None = alloc_field(label="input_by_individual", kind="temporary", owner="operator", retention="call", activity="yes", default=None)

_NUCLEOTIDE_DECODE = ["A", "T", "C", "G"]


def _decode_allele(data: np.ndarray, offsets: np.ndarray, idx: int) -> str:
    start = int(offsets[idx])
    end = int(offsets[idx + 1])
    return "".join(
        _NUCLEOTIDE_DECODE[(int(data[j // 4]) >> ((j % 4) * 2)) & 0b11]
        for j in range(start, end)
    )


class SpmvGRG:
    """Matmul-focused GRG operator for genotype matrix G (num_samples x num_mutations)."""

    def __init__(
        self,
        path,
        backend: BackendBase,
        dtype,
        index_dtype,
        artifact_dir: str | Path = "pygrgl_spmv_artifacts",
        *,
        ordering: str = "height",
        intra_block_ordering: str = "rcm_mincol",
    ):
        self._dtype = np.dtype(dtype)
        self._index_dtype = np.dtype(index_dtype)
        self._requested_ordering = self._parse_ordering(ordering)
        self._requested_intra_block_ordering = self._parse_intra_block_ordering(intra_block_ordering)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(logging.WARNING)
        if not isinstance(backend, BackendBase):
            raise TypeError(f"SpmvGRG backend must be a BackendBase instance, got {type(backend).__name__}")
        self._backend = backend

        source_path = Path(path)
        self._artifact_path: Path
        if source_path.suffix == ".grg":
            artifact_root = Path(artifact_dir).expanduser()
            self._artifact_path = artifact_path_for_grg(
                source_path,
                artifact_root,
                ordering=self._requested_ordering,
                intra_block_ordering=self._requested_intra_block_ordering,
            )
            self._artifact_path.parent.mkdir(parents=True, exist_ok=True)
            self._compiled = self._load_or_build_from_grg(source_path=source_path, artifact_path=self._artifact_path)
        elif source_path.suffix == ".grg_spmv":
            self._artifact_path = source_path
            self._compiled = load_grg_spmv(self._artifact_path, self._dtype, self._index_dtype)
        else:
            raise ValueError(f"Unsupported SpmvGRG input path {source_path}; expected .grg or .grg_spmv")

        self.memory = MemoryLedger()
        self._backend.setup(self._compiled.to_backend_setup(self._dtype))
        self._compiled.A_blocks = None
        self._retained_mem = self._build_retained_mem()
        self._seen_retained_epoch = -1
        self._refresh_retained_snapshot(force=True)

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
        compiled = compile_grg(
            grg,
            dtype=self._dtype,
            index_dtype=self._index_dtype,
            ordering=self._requested_ordering,
            intra_block_ordering=self._requested_intra_block_ordering,
        )
        self._build_init_biases(compiled)
        save_grg_spmv(compiled, artifact_path)
        return compiled

    def _build_init_biases(self, compiled: CompiledOperatorState) -> None:
        helper = ReferenceBackend(
            pair=ReferencePlanPair(
                plan_up=ReferenceBackend.plan(fmt="CSR", store="N", k_hint=None),
                plan_down=ReferenceBackend.plan(fmt="CSC", store="T", k_hint=None),
            ),
            log_level="WARNING",
        )
        helper.setup(compiled.to_backend_setup(self._dtype))
        zeros_up = np.zeros((compiled.num_samples, 1), dtype=self._dtype)
        zeros_down = np.zeros((compiled.num_mutations, 1), dtype=self._dtype)
        init_vec = np.ones(1, dtype=self._dtype)

        up_bias, _ = helper.run_up(zeros_up, init_mode=InitMode.VECTOR, init=init_vec, need_miss_output=False)
        down_bias = helper.run_down(zeros_down, init_mode=InitMode.VECTOR, init=init_vec, miss=None)
        compiled.init_vector_up_bias = np.asarray(up_bias[:, 0], dtype=self._dtype).reshape(compiled.num_mutations)
        compiled.init_vector_down_bias = np.asarray(down_bias[:, 0], dtype=self._dtype).reshape(compiled.num_samples)
        compiled.init_xtx_up_bias = None
        compiled.init_xtx_down_bias = None
        if compiled.coalescence_counts is not None:
            up_xtx, _ = helper.run_up(zeros_up, init_mode=InitMode.XTX, init=None, need_miss_output=False)
            down_xtx = helper.run_down(zeros_down, init_mode=InitMode.XTX, init=None, miss=None)
            compiled.init_xtx_up_bias = np.asarray(up_xtx[:, 0], dtype=self._dtype).reshape(compiled.num_mutations)
            compiled.init_xtx_down_bias = np.asarray(down_xtx[:, 0], dtype=self._dtype).reshape(compiled.num_samples)

    def _build_retained_mem(self) -> OperatorRetainedMem:
        return OperatorRetainedMem(
            level_offsets=self._compiled.level_offsets,
            sample_rows=self._compiled.sample_rows,
            node_perm=self._compiled.node_perm,
            inv_node_perm=self._compiled.inv_node_perm,
            sample_to_individual=self._compiled.sample_to_individual,
            mutation_positions=self._compiled.mutation_positions,
            mutation_times=self._compiled.mutation_times,
            mutation_alleles=self._compiled.mutation_alleles,
            mutation_allele_offsets=self._compiled.mutation_allele_offsets,
            mutation_ref_alleles=self._compiled.mutation_ref_alleles,
            mutation_ref_allele_offsets=self._compiled.mutation_ref_allele_offsets,
            coalescence_counts=self._compiled.coalescence_counts,
            init_vector_up_bias=self._compiled.init_vector_up_bias,
            init_vector_down_bias=self._compiled.init_vector_down_bias,
            init_xtx_up_bias=self._compiled.init_xtx_up_bias,
            init_xtx_down_bias=self._compiled.init_xtx_down_bias,
            sel_mut=self._compiled.sel_mut,
            sel_miss=self._compiled.sel_miss,
        )

    def _refresh_retained_snapshot(self, *, force: bool = False) -> None:
        current_epoch = int(self._backend._retained_epoch)
        if not force and current_epoch == self._seen_retained_epoch:
            return
        self.memory.retained = capture_snapshot(self._retained_mem, self._backend._retained_mem, stage="retained", runtime_k=None)
        self._seen_retained_epoch = current_epoch

    def _record_last_call_snapshot(
        self,
        *,
        stage: str,
        operator_call: OperatorCallMem,
        backend_call,
        capture,
    ) -> None:
        self.memory.last_call = capture_snapshot(
            operator_call,
            backend_call,
            stage=stage,
            runtime_k=capture.runtime_k,
            direction=capture.direction,
            active_alloc_keys=capture.active_alloc_keys,
            meta=dict(capture.meta),
        )
        self._refresh_retained_snapshot()

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_samples, self.num_mutations)

    @property
    def artifact_path(self) -> Path:
        return self._artifact_path

    @property
    def num_samples(self):
        return self._compiled.num_samples

    @property
    def num_individuals(self):
        return self._compiled.num_individuals

    @property
    def num_mutations(self):
        return self._compiled.num_mutations

    @property
    def ploidy(self):
        return self._compiled.ploidy

    @property
    def num_nodes(self):
        return self._compiled.num_nodes

    @property
    def num_edges(self):
        return self._compiled.num_edges

    @property
    def has_missing_data(self):
        return self._compiled.has_missing_data

    @property
    def level_offsets(self):
        return self._compiled.level_offsets

    @property
    def sample_rows(self):
        return self._compiled.sample_rows

    @property
    def ordering(self):
        return self._compiled.ordering

    @property
    def intra_block_ordering(self):
        return self._compiled.intra_block_ordering

    @property
    def node_perm(self):
        return self._compiled.node_perm

    @property
    def inv_node_perm(self):
        return self._compiled.inv_node_perm

    @property
    def sel_mut(self):
        return self._compiled.sel_mut

    @property
    def sel_miss(self):
        return self._compiled.sel_miss

    @property
    def sample_to_individual(self):
        return self._compiled.sample_to_individual

    @property
    def coalescence_counts(self):
        return self._compiled.coalescence_counts

    @property
    def init_vector_up_bias(self):
        return self._compiled.init_vector_up_bias

    @property
    def init_vector_down_bias(self):
        return self._compiled.init_vector_down_bias

    @property
    def init_xtx_up_bias(self):
        return self._compiled.init_xtx_up_bias

    @property
    def init_xtx_down_bias(self):
        return self._compiled.init_xtx_down_bias

    def get_mutation_by_id(self, mutation_id: int):
        idx = int(mutation_id)
        if idx < 0 or idx >= self.num_mutations:
            raise IndexError(f"Mutation id out of range: {mutation_id}")
        return pygrgl.Mutation(
            float(self._compiled.mutation_positions[idx]),
            _decode_allele(self._compiled.mutation_alleles, self._compiled.mutation_allele_offsets, idx),
            _decode_allele(self._compiled.mutation_ref_alleles, self._compiled.mutation_ref_allele_offsets, idx),
            float(self._compiled.mutation_times[idx]),
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

    def _parse_ordering(self, ordering: str) -> str:
        token = str(ordering).strip().lower()
        if token not in VALID_ORDERINGS:
            raise ValueError(f"Unknown ordering {ordering!r}; expected one of {sorted(VALID_ORDERINGS)}")
        return token

    def _parse_intra_block_ordering(self, intra_block_ordering: str) -> str:
        token = str(intra_block_ordering).strip().lower()
        if token not in VALID_INTRA_BLOCK_ORDERINGS:
            raise ValueError(
                "Unknown intra_block_ordering "
                f"{intra_block_ordering!r}; expected one of {sorted(VALID_INTRA_BLOCK_ORDERINGS)}"
            )
        return token

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
        if not isinstance(init, np.ndarray):
            raise TypeError(f"init must be None, 'xtx', or a numpy.ndarray, got {type(init).__name__}")

        init_arr = init
        if init_arr.dtype != input_dtype:
            raise TypeError(f"The init matrix must match the dtype of the input matrix. Got: {init_arr.dtype}")

        if init_arr.ndim == 1:
            if init_arr.shape[0] != rows:
                raise ValueError("If init has a single dimension, it must match the number of rows in the input matrix")
            return InitMode.VECTOR, init_arr.astype(self._dtype, order="C", copy=False)

        if init_arr.ndim == 2:
            if init_arr.shape != (rows, self.num_nodes):
                raise ValueError(
                    f"If init is a matrix, it must match the dimensions ({rows}, {self.num_nodes})"
                )
            init_nodes = init_arr[:, self.node_perm].T
            return InitMode.MATRIX, init_nodes.astype(self._dtype, order="C", copy=False)

        raise ValueError("init must be None, 'xtx', a vector, or a matrix")

    def _validate_miss(self, miss: np.ndarray, rows: int, direction: Direction, input_dtype: np.dtype) -> np.ndarray:
        if not isinstance(miss, np.ndarray):
            raise TypeError(f'The "miss" input must be a numpy.ndarray. Got: {type(miss).__name__}')
        miss_arr = miss
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
        if not isinstance(input, np.ndarray):
            raise TypeError(f"matmul() requires input to be a numpy.ndarray, got {type(input).__name__}")
        X_in = input
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

        input_matrix = X_in.astype(self._dtype, order="C", copy=False)
        input_internal = input_matrix.T
        caller_init = init if isinstance(init, np.ndarray) else None
        operator_call = OperatorCallMem(
            caller_input=X_in,
            caller_init=caller_init,
            input_internal=input_matrix,
            init_payload=init_payload,
            backend_init_payload=backend_init_payload,
        )

        with self._backend._call_capture_scope() as capture_nonce:
            if direction_name == Direction.UP:
                if by_individual:
                    input_internal = input_internal[self.sample_to_individual]
                    operator_call.input_by_individual = input_internal
                if emit_all_nodes:
                    node_values = self._backend.run_up_nodes(
                        input_internal,
                        init_mode=backend_init_mode,
                        init=backend_init_payload,
                    )
                    capture = self._backend._consume_call_capture(
                        expected_nonce=capture_nonce,
                        expected_direction=direction_name,
                        expected_k=rows,
                    )
                    output = self._finish_node_output(node_values)
                    operator_call.caller_output = output
                    assert self._backend._call_mem is not None
                    self._record_last_call_snapshot(
                        stage="run_up",
                        operator_call=operator_call,
                        backend_call=self._backend._call_mem,
                        capture=capture,
                    )
                    return output

                miss_output = None
                if miss is not None:
                    miss_output = self._validate_miss(miss, rows, direction_name, X_in.dtype)
                result_internal, miss_internal = self._backend.run_up(
                    input_internal,
                    init_mode=backend_init_mode,
                    init=backend_init_payload,
                    need_miss_output=(miss_output is not None),
                )
                capture = self._backend._consume_call_capture(
                    expected_nonce=capture_nonce,
                    expected_direction=direction_name,
                    expected_k=rows,
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
                output = self._finish_endpoint_output(result_internal)
                operator_call.caller_output = output
                operator_call.caller_miss_output = miss_output
                assert self._backend._call_mem is not None
                self._record_last_call_snapshot(
                    stage="run_up",
                    operator_call=operator_call,
                    backend_call=self._backend._call_mem,
                    capture=capture,
                )
                return output

            if emit_all_nodes:
                node_values = self._backend.run_down_nodes(
                    input_internal,
                    init_mode=backend_init_mode,
                    init=backend_init_payload,
                )
                capture = self._backend._consume_call_capture(
                    expected_nonce=capture_nonce,
                    expected_direction=direction_name,
                    expected_k=rows,
                )
                output = self._finish_node_output(node_values)
                operator_call.caller_output = output
                assert self._backend._call_mem is not None
                self._record_last_call_snapshot(
                    stage="run_down",
                    operator_call=operator_call,
                    backend_call=self._backend._call_mem,
                    capture=capture,
                )
                return output

            miss_internal = None
            miss_output = None
            if miss is not None:
                miss_output = self._validate_miss(miss, rows, direction_name, X_in.dtype)
                miss_internal = miss_output.T.astype(self._dtype, order="C", copy=False)
            result_internal = self._backend.run_down(
                input_internal,
                miss=miss_internal,
                init_mode=backend_init_mode,
                init=backend_init_payload,
            )
            capture = self._backend._consume_call_capture(
                expected_nonce=capture_nonce,
                expected_direction=direction_name,
                expected_k=rows,
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
            output = self._finish_endpoint_output(result_internal)
            operator_call.caller_output = output
            operator_call.caller_miss_input = miss_output
            operator_call.miss_internal = miss_internal
            assert self._backend._call_mem is not None
            self._record_last_call_snapshot(
                stage="run_down",
                operator_call=operator_call,
                backend_call=self._backend._call_mem,
                capture=capture,
            )
            return output


__all__ = ["SpmvGRG"]
