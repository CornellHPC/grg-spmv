"""Artifact conversion and runtime-bound GRG operator semantics."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pygrgl

from pygrgl_spmv._rss import rss_checkpoint
from pygrgl_spmv.backends.types import Direction, InitMode, parse_direction, parse_init_mode
from pygrgl_spmv.grg.artifact import _load_grg_spmv_host, artifact_path_for_grg, save_grg_spmv
from pygrgl_spmv.grg.compile import CompiledOperatorState, compile_grg

if TYPE_CHECKING:
    from pygrgl_spmv.backends.reference import ReferenceLayout, ReferenceRuntime

_NUCLEOTIDE_DECODE = ["A", "T", "C", "G"]
_COMPILE_LOGGER = logging.getLogger("pygrgl_spmv.grg.compile")


@dataclass(frozen=True)
class RuntimeRequirements:
    max_k_up: int
    max_k_down: int
    need_down_miss_input: bool
    need_up_miss_output: bool
    need_init_vector: bool
    need_init_matrix: bool
    need_init_xtx: bool

    def __post_init__(self) -> None:
        if int(self.max_k_up) < 1:
            raise ValueError(f"max_k_up must be >= 1, got {self.max_k_up}")
        if int(self.max_k_down) < 1:
            raise ValueError(f"max_k_down must be >= 1, got {self.max_k_down}")


@dataclass(frozen=True)
class _CudaMatmulSpec:
    direction: Direction
    k: int
    emit_all_nodes: bool
    by_individual: bool
    init_mode: InitMode
    backend_init_mode: InitMode
    use_miss: bool
    input_cols: int
    output_cols: int
    apply_endpoint_bias: bool


def _decode_allele(data: np.ndarray, offsets: np.ndarray, idx: int) -> str:
    start = int(offsets[idx])
    end = int(offsets[idx + 1])
    return "".join(
        _NUCLEOTIDE_DECODE[(int(data[j // 4]) >> ((j % 4) * 2)) & 0b11]
        for j in range(start, end)
    )


def _validate_runtime_requirements(
    requirements: RuntimeRequirements,
    *,
    direction: Direction,
    k: int,
    init_mode: InitMode,
    uses_miss: bool,
) -> None:
    if direction == Direction.UP and int(k) > int(requirements.max_k_up):
        raise ValueError(f"UP runtime k={k} exceeds declared max_k_up={requirements.max_k_up}")
    if direction == Direction.DOWN and int(k) > int(requirements.max_k_down):
        raise ValueError(f"DOWN runtime k={k} exceeds declared max_k_down={requirements.max_k_down}")
    if direction == Direction.UP and uses_miss and not requirements.need_up_miss_output:
        raise ValueError("UP miss output was not declared in RuntimeRequirements")
    if direction == Direction.DOWN and uses_miss and not requirements.need_down_miss_input:
        raise ValueError("DOWN miss input was not declared in RuntimeRequirements")
    if init_mode == InitMode.VECTOR and not requirements.need_init_vector:
        raise ValueError("init vector use was not declared in RuntimeRequirements")
    if init_mode == InitMode.MATRIX and not requirements.need_init_matrix:
        raise ValueError("init matrix use was not declared in RuntimeRequirements")
    if init_mode == InitMode.XTX and not requirements.need_init_xtx:
        raise ValueError("init='xtx' use was not declared in RuntimeRequirements")


def _validate_internal_init(
    state: CompiledOperatorState,
    dtype: np.dtype,
    init_mode: InitMode,
    init: np.ndarray | None,
    k: int,
) -> np.ndarray | None:
    match init_mode:
        case InitMode.NONE:
            if init is not None:
                raise ValueError("init payload provided with init_mode=none")
            return None
        case InitMode.XTX:
            if init is not None:
                raise ValueError("init payload must be None with init_mode=xtx")
            if state.coalescence_counts is None:
                raise ValueError("init_mode=xtx requires GRG coalescence counts")
            return None
        case InitMode.VECTOR:
            arr = np.asarray(init, dtype=dtype, order="C")
            if arr.ndim != 1 or arr.shape != (k,):
                raise ValueError(f"init vector must have shape ({k},), got {arr.shape}")
            return arr
        case InitMode.MATRIX:
            arr = np.asarray(init, dtype=dtype, order="C")
            if arr.ndim != 2 or arr.shape != (state.num_nodes, k):
                raise ValueError(f"init matrix must have shape ({state.num_nodes}, {k}), got {arr.shape}")
            return arr
        case _:
            raise ValueError(f"unknown init mode {init_mode!r}")


def _apply_internal_init(
    state: CompiledOperatorState,
    dtype: np.dtype,
    node_values: np.ndarray,
    *,
    init_mode: InitMode,
    init_payload: np.ndarray | None,
) -> None:
    match init_mode:
        case InitMode.NONE:
            return
        case InitMode.XTX:
            if state.coalescence_counts is None:
                raise ValueError("init_mode=xtx requires GRG coalescence counts")
            node_values += (2.0 * state.coalescence_counts.astype(dtype, copy=False))[:, None]
        case InitMode.VECTOR:
            assert init_payload is not None
            node_values += init_payload[None, :]
        case InitMode.MATRIX:
            assert init_payload is not None
            node_values += init_payload
        case _:
            raise ValueError(f"unknown init mode {init_mode!r}")


def _reference_run_up(
    state: CompiledOperatorState,
    dtype: np.dtype,
    primary: np.ndarray,
    *,
    init_mode: InitMode,
    init_payload: np.ndarray | None,
    need_miss_output: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    x = np.asarray(primary, dtype=dtype, order="C")
    node_values = np.zeros((state.num_nodes, x.shape[1]), dtype=dtype)
    _apply_internal_init(state, dtype, node_values, init_mode=init_mode, init_payload=init_payload)
    node_values[: state.num_samples] += x
    assert state.A_blocks is not None
    for dst_level in range(1, len(state.level_offsets) - 1):
        lo = int(state.level_offsets[dst_level])
        hi = int(state.level_offsets[dst_level + 1])
        for src_level, block in enumerate(state.A_blocks[dst_level]):
            if block.nnz == 0:
                continue
            src_lo = int(state.level_offsets[src_level])
            src_hi = int(state.level_offsets[src_level + 1])
            node_values[lo:hi] += block @ node_values[src_lo:src_hi]
    out_mut = np.asarray(state.sel_mut @ node_values, dtype=dtype) if state.sel_mut.nnz else np.zeros((state.num_mutations, x.shape[1]), dtype=dtype)
    out_miss = None
    if need_miss_output:
        out_miss = (
            np.asarray(state.sel_miss @ node_values, dtype=dtype)
            if state.sel_miss.nnz
            else np.zeros((state.num_mutations, x.shape[1]), dtype=dtype)
        )
    return out_mut, out_miss


def _reference_run_down(
    state: CompiledOperatorState,
    dtype: np.dtype,
    primary: np.ndarray,
    *,
    miss: np.ndarray | None,
    init_mode: InitMode,
    init_payload: np.ndarray | None,
) -> np.ndarray:
    x = np.asarray(primary, dtype=dtype, order="C")
    node_values = np.zeros((state.num_nodes, x.shape[1]), dtype=dtype)
    _apply_internal_init(state, dtype, node_values, init_mode=init_mode, init_payload=init_payload)
    if state.sel_mut.nnz:
        node_values += state.sel_mut.T @ x
    if miss is not None and state.sel_miss.nnz:
        node_values += state.sel_miss.T @ np.asarray(miss, dtype=dtype, order="C")
    assert state.A_blocks is not None
    for dst_level in range(len(state.level_offsets) - 2, -1, -1):
        lo = int(state.level_offsets[dst_level])
        hi = int(state.level_offsets[dst_level + 1])
        for src_level in range(len(state.level_offsets) - 2, dst_level, -1):
            block = state.A_blocks[src_level][dst_level]
            if block.nnz == 0:
                continue
            src_lo = int(state.level_offsets[src_level])
            src_hi = int(state.level_offsets[src_level + 1])
            node_values[lo:hi] += block.T @ node_values[src_lo:src_hi]
    return node_values[: state.num_samples]


def _build_init_biases(compiled: CompiledOperatorState, dtype: np.dtype) -> None:
    if compiled.A_blocks is None:
        raise RuntimeError("compiled operator blocks are required to build init biases")
    zeros_up = np.zeros((compiled.num_samples, 1), dtype=dtype)
    zeros_down = np.zeros((compiled.num_mutations, 1), dtype=dtype)
    init_vec = np.ones(1, dtype=dtype)

    up_bias, _ = _reference_run_up(
        compiled,
        dtype,
        zeros_up,
        init_mode=InitMode.VECTOR,
        init_payload=init_vec,
        need_miss_output=False,
    )
    down_bias = _reference_run_down(
        compiled,
        dtype,
        zeros_down,
        miss=None,
        init_mode=InitMode.VECTOR,
        init_payload=init_vec,
    )
    compiled.init_vector_up_bias = np.asarray(up_bias[:, 0], dtype=dtype).reshape(compiled.num_mutations)
    compiled.init_vector_down_bias = np.asarray(down_bias[:, 0], dtype=dtype).reshape(compiled.num_samples)
    compiled.init_xtx_up_bias = None
    compiled.init_xtx_down_bias = None
    if compiled.coalescence_counts is not None:
        up_xtx, _ = _reference_run_up(
            compiled,
            dtype,
            zeros_up,
            init_mode=InitMode.XTX,
            init_payload=None,
            need_miss_output=False,
        )
        down_xtx = _reference_run_down(
            compiled,
            dtype,
            zeros_down,
            miss=None,
            init_mode=InitMode.XTX,
            init_payload=None,
        )
        compiled.init_xtx_up_bias = np.asarray(up_xtx[:, 0], dtype=dtype).reshape(compiled.num_mutations)
        compiled.init_xtx_down_bias = np.asarray(down_xtx[:, 0], dtype=dtype).reshape(compiled.num_samples)


class BoundGRG:
    """Lightweight runtime-bound GRG operator."""

    def __init__(self, runtime, artifact_index: int, state: CompiledOperatorState, artifact_path: Path):
        self._runtime = runtime
        self._artifact_index = int(artifact_index)
        self._state = state
        self._artifact_path = Path(artifact_path)
        self._dtype = np.dtype(runtime.layout.dtype)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_samples, self.num_mutations)

    @property
    def artifact_path(self) -> Path:
        return self._artifact_path

    @property
    def num_samples(self) -> int:
        return int(self._state.num_samples)

    @property
    def num_individuals(self) -> int:
        return int(self._state.num_individuals)

    @property
    def num_mutations(self) -> int:
        return int(self._state.num_mutations)

    @property
    def ploidy(self) -> int:
        return int(self._state.ploidy)

    @property
    def num_nodes(self) -> int:
        return int(self._state.num_nodes)

    @property
    def num_edges(self) -> int:
        return int(self._state.num_edges)

    @property
    def has_missing_data(self) -> bool:
        return bool(self._state.has_missing_data)

    @property
    def level_offsets(self) -> np.ndarray:
        return self._state.level_offsets

    @property
    def node_perm(self) -> np.ndarray:
        return self._state.node_perm

    @property
    def inv_node_perm(self) -> np.ndarray:
        return self._state.inv_node_perm

    @property
    def sel_mut(self):
        return self._state.sel_mut

    @property
    def sel_miss(self):
        return self._state.sel_miss

    @property
    def sample_to_individual(self) -> np.ndarray:
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
            raise IndexError(f"mutation id out of range: {mutation_id}")
        return pygrgl.Mutation(
            float(self._state.mutation_positions[idx]),
            _decode_allele(self._state.mutation_alleles, self._state.mutation_allele_offsets, idx),
            _decode_allele(self._state.mutation_ref_alleles, self._state.mutation_ref_allele_offsets, idx),
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
                    f"unknown direction: {direction!r}. Expected 'up', 'down', "
                    "pygrgl.TraversalDirection.UP, or pygrgl.TraversalDirection.DOWN"
                )

    def _parse_init(
        self,
        init: str | np.ndarray | None,
        rows: int,
        input_dtype: np.dtype,
        *,
        reorder_matrix: bool,
    ) -> tuple[InitMode, np.ndarray | None]:
        if init is None:
            return InitMode.NONE, None
        if isinstance(init, str):
            if init != "xtx":
                raise ValueError(f"unexpected init value: {init}")
            if self.coalescence_counts is None:
                raise ValueError(
                    "init='xtx' requires per-node coalescence counts in the GRG. "
                    "This artifact was loaded without coalescence counts."
                )
            return InitMode.XTX, None
        if not isinstance(init, np.ndarray):
            raise TypeError(f"init must be None, 'xtx', or a numpy.ndarray, got {type(init).__name__}")
        if init.dtype != input_dtype:
            raise TypeError(f"the init matrix must match the dtype of the input matrix. Got: {init.dtype}")
        if init.ndim == 1:
            if init.shape[0] != rows:
                raise ValueError("if init has a single dimension, it must match the number of rows in the input matrix")
            return InitMode.VECTOR, init.astype(self._dtype, order="C", copy=False)
        if init.ndim == 2:
            if init.shape != (rows, self.num_nodes):
                raise ValueError(f"if init is a matrix, it must match the dimensions ({rows}, {self.num_nodes})")
            if reorder_matrix:
                init_nodes = init[:, self.node_perm].T
                return InitMode.MATRIX, init_nodes.astype(self._dtype, order="C", copy=False)
            return InitMode.MATRIX, init.astype(self._dtype, order="C", copy=False)
        raise ValueError("init must be None, 'xtx', a vector, or a matrix")

    def _validate_miss(self, miss: np.ndarray, rows: int, direction: Direction, input_dtype: np.dtype) -> np.ndarray:
        if not isinstance(miss, np.ndarray):
            raise TypeError(f'the "miss" input must be a numpy.ndarray. Got: {type(miss).__name__}')
        if miss.dtype != input_dtype:
            raise TypeError(f'the "miss" input must match the dtype of the input matrix. Got: {miss.dtype}')
        if miss.ndim != 2:
            raise ValueError(f'"miss" must be a two-dimension numpy array (matrix). ndim={miss.ndim}')
        if miss.shape[0] != rows:
            raise ValueError(f'"miss" has {miss.shape[0]} rows, but must match the input/output matrices ({rows})')
        if miss.shape[1] != self.num_mutations:
            if direction == Direction.DOWN:
                raise ValueError(f'the "miss" matrix must match the number of columns in the input matrix. Got: {miss.shape[1]}')
            raise ValueError(f'the "miss" matrix must match the number of columns in the output matrix. Got: {miss.shape[1]}')
        return miss

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
            assert bias is not None
            assert init_payload is not None
            result_internal += bias[:, None] * init_payload[None, :]

    def _finish_endpoint_output(self, result_internal: np.ndarray) -> np.ndarray:
        return result_internal.T.astype(self._dtype, copy=False)

    def _finish_node_output(self, node_values_internal: np.ndarray) -> np.ndarray:
        node_values = np.asarray(node_values_internal, dtype=self._dtype, order="C")
        reordered = node_values[self.inv_node_perm]
        return reordered.T.astype(self._dtype, copy=False)

    def _prepare_cuda_spec(
        self,
        *,
        direction: str | Direction | pygrgl.TraversalDirection,
        k: int,
        emit_all_nodes: bool,
        by_individual: bool,
        init_mode: str | InitMode,
        use_miss: bool,
    ) -> _CudaMatmulSpec:
        direction_name = self._parse_direction(direction)
        init_mode_name = parse_init_mode(init_mode)
        rows = int(k)
        if rows < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if emit_all_nodes and use_miss:
            raise RuntimeError('the "use_miss" parameter cannot be mixed with "emit_all_nodes=True"')
        if init_mode_name != InitMode.NONE and use_miss:
            raise ValueError('the "use_miss" parameter cannot be mixed with a non-none init_mode')
        if init_mode_name == InitMode.XTX and self.coalescence_counts is None:
            raise ValueError(
                "init_mode='xtx' requires per-node coalescence counts in the GRG. "
                "This artifact was loaded without coalescence counts."
            )
        _validate_runtime_requirements(
            self._runtime.layout.requirements,
            direction=direction_name,
            k=rows,
            init_mode=init_mode_name,
            uses_miss=bool(use_miss),
        )
        input_cols = (
            self.num_individuals
            if by_individual and direction_name == Direction.UP
            else (self.num_samples if direction_name == Direction.UP else self.num_mutations)
        )
        if emit_all_nodes:
            output_cols = self.num_nodes
        elif direction_name == Direction.UP:
            output_cols = self.num_mutations
        elif by_individual:
            output_cols = self.num_individuals
        else:
            output_cols = self.num_samples
        backend_init_mode = (
            init_mode_name
            if emit_all_nodes or init_mode_name == InitMode.MATRIX
            else InitMode.NONE
        )
        return _CudaMatmulSpec(
            direction=direction_name,
            k=rows,
            emit_all_nodes=bool(emit_all_nodes),
            by_individual=bool(by_individual),
            init_mode=init_mode_name,
            backend_init_mode=backend_init_mode,
            use_miss=bool(use_miss),
            input_cols=int(input_cols),
            output_cols=int(output_cols),
            apply_endpoint_bias=bool(
                (not emit_all_nodes) and init_mode_name in {InitMode.VECTOR, InitMode.XTX}
            ),
        )

    def _copy_host_to_cuda(self, dst, src: np.ndarray) -> None:
        import torch

        arr = np.asarray(src, dtype=self._dtype, order="C")
        stream = getattr(self._runtime, "_torch_caller_stream", None)
        if stream is None:
            dst.copy_(torch.from_numpy(arr), non_blocking=False)
            return
        with torch.cuda.device(dst.device):
            with torch.cuda.stream(stream()):
                dst.copy_(torch.from_numpy(arr), non_blocking=False)

    def _copy_cuda_to_host(self, src) -> np.ndarray:
        import torch

        stream = getattr(self._runtime, "_torch_caller_stream", None)
        if stream is None:
            return src.detach().cpu().numpy().astype(self._dtype, copy=False).copy()
        with torch.cuda.device(src.device):
            with torch.cuda.stream(stream()):
                host = src.detach().cpu()
        return host.numpy().astype(self._dtype, copy=False).copy()

    def prepare_matmul_cuda(
        self,
        *,
        direction,
        k: int,
        emit_all_nodes: bool = False,
        by_individual: bool = False,
        init_mode: str | InitMode = "none",
        use_miss: bool = False,
    ):
        prepare = getattr(self._runtime, "_prepare_matmul_cuda", None)
        if prepare is None:
            raise NotImplementedError("prepare_matmul_cuda() is implemented only for CUDA backends")
        spec = self._prepare_cuda_spec(
            direction=direction,
            k=int(k),
            emit_all_nodes=bool(emit_all_nodes),
            by_individual=bool(by_individual),
            init_mode=init_mode,
            use_miss=bool(use_miss),
        )
        return prepare(self, spec)

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
        x_in = input
        if x_in.ndim != 2:
            raise ValueError("matmul() only supports two-dimensional numpy arrays as input.")
        rows, cols = x_in.shape
        if rows == 0 or cols == 0:
            raise ValueError("matmul() requires non-zero dimensions.")

        direction_name = self._parse_direction(direction)
        expected_cols = (
            self.num_individuals
            if by_individual and direction_name == Direction.UP
            else (self.num_samples if direction_name == Direction.UP else self.num_mutations)
        )
        if cols != expected_cols:
            if direction_name == Direction.UP:
                raise ValueError("input matrix has wrong number of columns for UP direction (numSamples or numIndividuals depending on by_individual)")
            raise ValueError("input matrix has wrong number of columns for DOWN direction (numMutations)")

        if emit_all_nodes and miss is not None:
            raise RuntimeError('the "miss" parameter cannot be mixed with the "emit_all_nodes" parameter')
        if init is not None and miss is not None:
            raise ValueError('the "miss" parameter cannot be mixed with the "init" parameter')

        if getattr(self._runtime, "_prepare_matmul_cuda", None) is not None:
            init_mode, init_payload = self._parse_init(init, rows, x_in.dtype, reorder_matrix=False)
            spec = self._prepare_cuda_spec(
                direction=direction_name,
                k=rows,
                emit_all_nodes=bool(emit_all_nodes),
                by_individual=bool(by_individual),
                init_mode=init_mode,
                use_miss=bool(miss is not None),
            )
            miss_matrix = None if miss is None else self._validate_miss(miss, rows, direction_name, x_in.dtype)
            with self._runtime._call_scope():
                with self.prepare_matmul_cuda(
                    direction=spec.direction,
                    k=spec.k,
                    emit_all_nodes=spec.emit_all_nodes,
                    by_individual=spec.by_individual,
                    init_mode=spec.init_mode,
                    use_miss=spec.use_miss,
                ) as op:
                    self._copy_host_to_cuda(op.input, x_in)
                    if spec.direction == Direction.DOWN and miss_matrix is not None:
                        self._copy_host_to_cuda(op.miss_input, miss_matrix)
                    if init_mode == InitMode.VECTOR:
                        assert init_payload is not None
                        self._copy_host_to_cuda(op.init_vector, init_payload)
                    elif init_mode == InitMode.MATRIX:
                        assert init_payload is not None
                        self._copy_host_to_cuda(op.init_matrix, init_payload)
                    run_prelocked = getattr(op, "_call_prelocked", None)
                    if run_prelocked is None:
                        raise RuntimeError("prepared CUDA op is missing _call_prelocked()")
                    run_prelocked()
                    result = self._copy_cuda_to_host(op.output)
                    if spec.direction == Direction.UP and miss_matrix is not None:
                        miss_matrix += self._copy_cuda_to_host(op.miss_output).astype(miss_matrix.dtype, copy=False)
                    return result

        init_mode, init_payload = self._parse_init(init, rows, x_in.dtype, reorder_matrix=True)
        _validate_runtime_requirements(
            self._runtime.layout.requirements,
            direction=direction_name,
            k=rows,
            init_mode=init_mode,
            uses_miss=bool(miss is not None),
        )

        if emit_all_nodes:
            backend_init_mode = init_mode
            backend_init_payload = init_payload
        else:
            backend_init_mode, backend_init_payload = (
                (InitMode.NONE, None)
                if init_mode in {InitMode.VECTOR, InitMode.XTX}
                else (init_mode, init_payload)
            )

        input_matrix = x_in.astype(self._dtype, order="C", copy=False)
        input_internal = input_matrix.T
        with self._runtime._call_scope():
            if direction_name == Direction.UP:
                if by_individual:
                    input_internal = input_internal[self.sample_to_individual]
                if emit_all_nodes:
                    node_values = self._runtime._run(
                        self._artifact_index,
                        direction_name,
                        input_internal,
                        miss=None,
                        init_mode=backend_init_mode,
                        init_payload=backend_init_payload,
                        need_miss_output=False,
                        emit_all_nodes=True,
                    )
                    return self._finish_node_output(node_values)

                miss_output = None if miss is None else self._validate_miss(miss, rows, direction_name, x_in.dtype)
                result_internal, miss_internal = self._runtime._run(
                    self._artifact_index,
                    direction_name,
                    input_internal,
                    miss=None,
                    init_mode=backend_init_mode,
                    init_payload=backend_init_payload,
                    need_miss_output=miss_output is not None,
                    emit_all_nodes=False,
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
                node_values = self._runtime._run(
                    self._artifact_index,
                    direction_name,
                    input_internal,
                    miss=None,
                    init_mode=backend_init_mode,
                    init_payload=backend_init_payload,
                    need_miss_output=False,
                    emit_all_nodes=True,
                )
                return self._finish_node_output(node_values)

            miss_internal = None
            if miss is not None:
                miss_output = self._validate_miss(miss, rows, direction_name, x_in.dtype)
                miss_internal = miss_output.T.astype(self._dtype, order="C", copy=False)
            result_internal = self._runtime._run(
                self._artifact_index,
                direction_name,
                input_internal,
                miss=miss_internal,
                init_mode=backend_init_mode,
                init_payload=backend_init_payload,
                need_miss_output=False,
                emit_all_nodes=False,
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


def convert(
    source: str | os.PathLike[str] | pygrgl.ImmutableGRG,
    output_dir: str | os.PathLike[str] = "pygrgl_spmv_artifacts",
    *,
    dtype=np.float64,
    name: str | None = None,
) -> Path:
    """Compile a GRG and write a `.grg_spmv` artifact.

    Path-based sources default to an artifact path derived from the fully
    resolved source path under ``output_dir``. Passing ``name=`` overrides that
    default and writes ``output_dir / f"{name}.grg_spmv"`` instead.
    """

    dtype = np.dtype(dtype)
    if output_dir is None:
        raise ValueError("convert() requires an output_dir in the runtime-owned API")

    if isinstance(source, (str, os.PathLike)):
        source_path = Path(os.fspath(source))
        if source_path.suffix != ".grg":
            raise ValueError(f"expected a .grg file, got {source_path}")
        stem = source_path.stem if name is None else str(name)
        grg = pygrgl.load_immutable_grg(str(source_path), load_up_edges=False)
        prev_rss = rss_checkpoint(_COMPILE_LOGGER, "artifact:grg_loaded", None)
        own_grg = True
    else:
        if name is None:
            raise ValueError("name is required when source is an ImmutableGRG object")
        stem = str(name)
        grg = source
        prev_rss = None
        own_grg = False

    compiled = compile_grg(grg)
    prev_rss = rss_checkpoint(_COMPILE_LOGGER, "artifact:compiled", prev_rss)
    if own_grg:
        del grg
        prev_rss = rss_checkpoint(_COMPILE_LOGGER, "artifact:grg_dropped", prev_rss)
    _build_init_biases(compiled, dtype)
    prev_rss = rss_checkpoint(_COMPILE_LOGGER, "artifact:init_biases", prev_rss)

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(source, (str, os.PathLike)) and name is None:
        artifact_path = artifact_path_for_grg(source_path, out_dir)
    else:
        artifact_path = out_dir / f"{stem}.grg_spmv"
    save_grg_spmv(compiled, artifact_path)
    rss_checkpoint(_COMPILE_LOGGER, "artifact:saved", prev_rss)
    return artifact_path


__all__ = ["BoundGRG", "RuntimeRequirements", "convert"]
