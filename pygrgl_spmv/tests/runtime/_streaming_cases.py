"""Synthetic streamed-artifact helpers shared by GPU runtime tests."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import gc
from pathlib import Path
import resource

import numpy as np
import pytest
import scipy.sparse as sp

from pygrgl_spmv.grg import _build_init_biases
from pygrgl_spmv.grg.artifact import save_grg_spmv
from pygrgl_spmv.grg.compile import CompiledOperatorState
from pygrgl_spmv.grg.sparse import binary_csr_from_parts
from pygrgl_spmv.tests.runtime._runtime_builders import build_cusparse_layout, build_triton_layout

_DENSITY_DEN = 100
# This is a streamed-stress fixture restriction, not a runtime rule. Keeping the
# synthetic owner blocks below the int32/int64 structural cliff avoids coupling
# artifact size, stored dtype, and planner byte thresholds in a way that is easy
# to get subtly wrong, and it lets Triton and cuSPARSE share the same artifact.
_GPU_STRESS_NNZ_CAP = 1 << 30


@dataclass(frozen=True)
class StreamStressCase:
    backend_name: str
    n: int
    bandwidth: int
    nnz: int
    block_struct_bytes: int
    total_vram_bytes: int
    available_host_bytes: int
    shifts: tuple[int, int, int]


@dataclass(frozen=True)
class ThreeBlockMode:
    name: str
    requested_ring_buffer_size: int
    budget_blocks: int
    resident_blocks: tuple[tuple[int, int], ...]
    streamed_blocks: tuple[tuple[int, int], ...]
    slot_count: int
    expect_warning: bool


def clear_cupy_state() -> None:
    gc.collect()
    import cupy as cp

    cp.cuda.runtime.deviceSynchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()


def clear_torch_state() -> None:
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def _mem_available_bytes() -> int:
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def _current_rss_bytes() -> int:
    with open("/proc/self/status", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def _effective_available_host_bytes() -> int:
    available = _mem_available_bytes()
    try:
        soft_limit, _hard_limit = resource.getrlimit(resource.RLIMIT_RSS)
    except (AttributeError, OSError, ValueError):
        return available
    if int(soft_limit) <= 0 or int(soft_limit) == resource.RLIM_INFINITY:
        return available
    return min(int(available), max(int(soft_limit) - _current_rss_bytes(), 0))


def _bandwidth_for_n(n: int) -> int:
    return (int(n) + _DENSITY_DEN - 1) // _DENSITY_DEN


def _banded_nnz(n: int, bandwidth: int) -> int:
    return int(n) * int(bandwidth)


def _block_struct_bytes(*, n: int, bandwidth: int, itemsize: int = np.dtype(np.int32).itemsize) -> int:
    return int(itemsize) * (_banded_nnz(n, bandwidth) + int(n) + 1)


def _fits_half_total_gpu_case(total_vram_bytes: int, n: int) -> bool:
    bandwidth = _bandwidth_for_n(n)
    return _banded_nnz(n, bandwidth) < _GPU_STRESS_NNZ_CAP and _block_struct_bytes(n=n, bandwidth=bandwidth) <= int(total_vram_bytes) // 2


def _largest_n_for_half_total(total_vram_bytes: int, *, fits_case) -> int:
    if int(total_vram_bytes) <= 0:
        raise ValueError(f"total_vram_bytes must be positive, got {total_vram_bytes}")
    lo = 0
    hi = 1
    while fits_case(int(total_vram_bytes), hi):
        lo = hi
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits_case(int(total_vram_bytes), mid):
            lo = mid
        else:
            hi = mid
    if lo < 1:
        raise RuntimeError(f"no valid stress case fits half total VRAM={total_vram_bytes}")
    return int(lo)


def _prepare_stream_stress_case(*, backend_name: str, total_vram_bytes: int) -> StreamStressCase:
    available_host_bytes = _effective_available_host_bytes()
    n = _largest_n_for_half_total(int(total_vram_bytes), fits_case=_fits_half_total_gpu_case)
    bandwidth = _bandwidth_for_n(n)
    block_struct_bytes = _block_struct_bytes(n=n, bandwidth=bandwidth)
    required_host_bytes = 3 * int(block_struct_bytes)
    if available_host_bytes < required_host_bytes:
        pytest.skip(
            f"{backend_name} large streamed tests need {required_host_bytes} host bytes for three blocks, "
            f"but only {available_host_bytes} are available"
        )
    return StreamStressCase(
        backend_name=backend_name,
        n=int(n),
        bandwidth=int(bandwidth),
        nnz=int(_banded_nnz(n, bandwidth)),
        block_struct_bytes=int(block_struct_bytes),
        total_vram_bytes=int(total_vram_bytes),
        available_host_bytes=int(available_host_bytes),
        shifts=(0, int(bandwidth), 2 * int(bandwidth)),
    )


@functools.cache
def prepare_triton_stream_stress_case() -> StreamStressCase:
    import torch

    clear_torch_state()
    with torch.cuda.device(0):
        _, total_vram_bytes = (int(v) for v in torch.cuda.mem_get_info())
    return _prepare_stream_stress_case(
        backend_name="triton",
        total_vram_bytes=total_vram_bytes,
    )


@functools.cache
def prepare_cusparse_stream_stress_case() -> StreamStressCase:
    import cupy as cp

    clear_cupy_state()
    with cp.cuda.Device(0):
        _, total_vram_bytes = (int(v) for v in cp.cuda.runtime.memGetInfo())
    return _prepare_stream_stress_case(
        backend_name="cusparse",
        total_vram_bytes=total_vram_bytes,
    )


def _attach_binary_csr(indices: np.ndarray, indptr: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
    shape_tuple = tuple(int(v) for v in shape)
    idx = np.asarray(indices)
    ptr = np.asarray(indptr)
    if idx.ndim != 1 or ptr.ndim != 1:
        raise ValueError("CSR indices and indptr must be one-dimensional")
    data = np.empty(0, dtype=np.bool_) if idx.size == 0 else np.broadcast_to(np.ones(1, dtype=np.bool_), (int(idx.size),))
    matrix = sp.csr_matrix(shape_tuple, dtype=np.bool_)
    matrix.data = data
    matrix.indices = idx
    matrix.indptr = ptr
    matrix._shape = shape_tuple
    matrix.has_sorted_indices = False
    matrix.has_canonical_format = False
    return matrix


def _build_band_arrays(n: int, bandwidth: int, shift: int, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    n = int(n)
    bandwidth = int(bandwidth)
    struct_dtype = np.dtype(dtype)
    indices_2d = np.empty((n, bandwidth), dtype=struct_dtype)
    rows = np.arange(n, dtype=struct_dtype)[:, None]
    base = np.arange(bandwidth, dtype=struct_dtype)[None, :] + struct_dtype.type(int(shift))
    np.add(rows, base, out=indices_2d)
    np.mod(indices_2d, n, out=indices_2d)
    indices = indices_2d.reshape(-1)
    indptr = np.arange(0, (n + 1) * bandwidth, bandwidth, dtype=struct_dtype)
    return indices, indptr


def _build_band_csr(n: int, bandwidth: int, shift: int, dtype: np.dtype) -> sp.csr_matrix:
    indices, indptr = _build_band_arrays(n, bandwidth, shift, dtype)
    return _attach_binary_csr(indices, indptr, (int(n), int(n)))


def _allele_tables(num_mutations: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(num_mutations + 1, dtype=np.uint32)
    data = np.frombuffer(b"A" * int(num_mutations), dtype=np.uint8).copy()
    return data, offsets


def _identity_permutation(num_nodes: int, struct_dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    perm = np.arange(num_nodes, dtype=np.dtype(struct_dtype))
    return perm, perm.copy()


def _synthetic_state(
    *,
    blocks: list[list[sp.csr_matrix]],
    level_offsets: np.ndarray,
    num_samples: int,
    num_mutations: int,
    num_nodes: int,
    sel_mut: sp.csr_matrix,
    sel_miss: sp.csr_matrix,
    dtype=np.float64,
) -> CompiledOperatorState:
    struct_dtype = np.dtype(level_offsets.dtype)
    node_perm, inv_node_perm = _identity_permutation(num_nodes, struct_dtype)
    sample_to_individual = np.arange(num_samples, dtype=struct_dtype)
    alleles, allele_offsets = _allele_tables(num_mutations)
    ref_alleles, ref_offsets = _allele_tables(num_mutations)
    state = CompiledOperatorState(
        A_blocks=blocks,
        level_offsets=np.asarray(level_offsets),
        node_perm=node_perm,
        inv_node_perm=inv_node_perm,
        sel_mut=sel_mut,
        sel_miss=sel_miss,
        num_samples=int(num_samples),
        num_mutations=int(num_mutations),
        num_nodes=int(num_nodes),
        ploidy=1,
        num_individuals=int(num_samples),
        num_edges=int(sum(block.nnz for row in blocks for block in row)),
        has_missing_data=False,
        sample_to_individual=sample_to_individual,
        mutation_positions=np.arange(num_mutations, dtype=np.float64),
        mutation_times=np.zeros((num_mutations,), dtype=np.float64),
        mutation_alleles=alleles,
        mutation_allele_offsets=allele_offsets,
        mutation_ref_alleles=ref_alleles,
        mutation_ref_allele_offsets=ref_offsets,
        coalescence_counts=None,
    )
    _build_init_biases(state, np.dtype(dtype))
    return state


def write_three_level_band_artifact(
    artifact_dir: Path,
    name: str,
    *,
    n: int,
    bandwidth: int,
    struct_dtype=None,
    shifts: tuple[int, int, int] | None = None,
) -> Path:
    n = int(n)
    bandwidth = int(bandwidth)
    if struct_dtype is None:
        struct_dtype = np.int32
    struct_dtype = np.dtype(struct_dtype)
    shifts = (0, bandwidth, 2 * bandwidth) if shifts is None else tuple(int(value) for value in shifts)
    path = Path(artifact_dir) / f"{name}.grg_spmv"
    if path.exists():
        return path
    block10 = _build_band_csr(n, bandwidth, shifts[0], struct_dtype)
    block20 = _build_band_csr(n, bandwidth, shifts[1], struct_dtype)
    block21 = _build_band_csr(n, bandwidth, shifts[2], struct_dtype)
    sel_mut = binary_csr_from_parts(
        indices=np.arange(2 * n, 3 * n, dtype=struct_dtype),
        indptr=np.arange(n + 1, dtype=struct_dtype),
        shape=(n, 3 * n),
        shared_data=True,
    )
    sel_miss = binary_csr_from_parts(
        indices=np.empty(0, dtype=struct_dtype),
        indptr=np.zeros(n + 1, dtype=struct_dtype),
        shape=(n, 3 * n),
        shared_data=True,
    )
    state = _synthetic_state(
        blocks=[[], [block10], [block20, block21]],
        level_offsets=np.asarray([0, n, 2 * n, 3 * n], dtype=struct_dtype),
        num_samples=n,
        num_mutations=n,
        num_nodes=3 * n,
        sel_mut=sel_mut,
        sel_miss=sel_miss,
    )
    save_grg_spmv(state, path)
    return path


def write_overlap_band_artifact(
    artifact_dir: Path,
    name: str,
    *,
    n: int,
    bandwidth: int,
) -> Path:
    n = int(n)
    bandwidth = int(bandwidth)
    dtype = np.dtype(np.int32)
    shifts = tuple(idx * bandwidth for idx in range(6))
    block10 = _build_band_csr(n, bandwidth, shifts[0], dtype)
    block20 = _build_band_csr(n, bandwidth, shifts[1], dtype)
    block21 = _build_band_csr(n, bandwidth, shifts[2], dtype)
    block30 = _build_band_csr(n, bandwidth, shifts[3], dtype)
    block31 = _build_band_csr(n, bandwidth, shifts[4], dtype)
    block32 = _build_band_csr(n, bandwidth, shifts[5], dtype)
    sel_mut = binary_csr_from_parts(
        indices=np.arange(3 * n, 4 * n, dtype=dtype),
        indptr=np.arange(n + 1, dtype=dtype),
        shape=(n, 4 * n),
        shared_data=True,
    )
    sel_miss = binary_csr_from_parts(
        indices=np.empty(0, dtype=dtype),
        indptr=np.zeros(n + 1, dtype=dtype),
        shape=(n, 4 * n),
        shared_data=True,
    )
    state = _synthetic_state(
        blocks=[[], [block10], [block20, block21], [block30, block31, block32]],
        level_offsets=np.asarray([0, n, 2 * n, 3 * n, 4 * n], dtype=dtype),
        num_samples=n,
        num_mutations=n,
        num_nodes=4 * n,
        sel_mut=sel_mut,
        sel_miss=sel_miss,
    )
    path = Path(artifact_dir) / f"{name}.grg_spmv"
    save_grg_spmv(state, path)
    return path


def _equal_block_budget_components(layout) -> tuple[int, int, int]:
    block_nbytes = {
        int(block.nbytes)
        for artifact in layout.artifacts
        for block in (*artifact.blocks_up, *artifact.blocks_down)
    }
    if not block_nbytes:
        raise RuntimeError("expected at least one sparse block in streamed stress layout")
    if len(block_nbytes) != 1:
        raise RuntimeError(f"streamed stress thresholds require equal-sized blocks, got {sorted(block_nbytes)}")
    fixed_bytes = int(layout.bytes_total) - int(layout.bytes_by_category["resident_sparse"])
    block_bytes = next(iter(block_nbytes))
    owner_block_count = sum(1 for artifact in layout.artifacts for block in (*artifact.blocks_up, *artifact.blocks_down))
    return int(fixed_bytes), int(block_bytes), int(owner_block_count)


def _stream_thresholds_from_layout(layout, *, max_ring_buffer_size: int) -> tuple[int, ...]:
    fixed_bytes, block_bytes, _owner_block_count = _equal_block_budget_components(layout)
    return tuple(int(fixed_bytes + ring * block_bytes) for ring in range(1, int(max_ring_buffer_size) + 1))


THREE_BLOCK_TRANSITION_MODES = (
    ThreeBlockMode(
        name="ring0-resident",
        requested_ring_buffer_size=0,
        budget_blocks=3,
        resident_blocks=((1, 0), (2, 0), (2, 1)),
        streamed_blocks=(),
        slot_count=0,
        expect_warning=False,
    ),
    ThreeBlockMode(
        name="ring1-stream",
        requested_ring_buffer_size=1,
        budget_blocks=1,
        resident_blocks=(),
        streamed_blocks=((1, 0), (2, 0), (2, 1)),
        slot_count=1,
        expect_warning=False,
    ),
    ThreeBlockMode(
        name="ring1-hybrid",
        requested_ring_buffer_size=1,
        budget_blocks=2,
        resident_blocks=((2, 1),),
        streamed_blocks=((1, 0), (2, 0)),
        slot_count=1,
        expect_warning=False,
    ),
    ThreeBlockMode(
        name="ring1-resident",
        requested_ring_buffer_size=1,
        budget_blocks=3,
        resident_blocks=((1, 0), (2, 0), (2, 1)),
        streamed_blocks=(),
        slot_count=0,
        expect_warning=True,
    ),
    ThreeBlockMode(
        name="ring2-stream",
        requested_ring_buffer_size=2,
        budget_blocks=2,
        resident_blocks=(),
        streamed_blocks=((1, 0), (2, 0), (2, 1)),
        slot_count=2,
        expect_warning=False,
    ),
    ThreeBlockMode(
        name="ring2-resident",
        requested_ring_buffer_size=2,
        budget_blocks=3,
        resident_blocks=((1, 0), (2, 0), (2, 1)),
        streamed_blocks=(),
        slot_count=0,
        expect_warning=True,
    ),
    ThreeBlockMode(
        name="ring3-resident",
        requested_ring_buffer_size=3,
        budget_blocks=3,
        resident_blocks=((1, 0), (2, 0), (2, 1)),
        streamed_blocks=(),
        slot_count=0,
        expect_warning=True,
    ),
)

THREE_BLOCK_EXACTNESS_MODES = tuple(
    mode
    for mode in THREE_BLOCK_TRANSITION_MODES
    if mode.name in {"ring0-resident", "ring1-stream", "ring1-hybrid", "ring2-stream", "ring2-resident"}
)


def three_block_mode_budget_bytes(mode: ThreeBlockMode, *, fixed_bytes: int, block_bytes: int) -> int:
    return int(fixed_bytes + int(mode.budget_blocks) * int(block_bytes))


def triton_ring_thresholds(
    artifact_path: Path,
    *,
    requirements,
    total_vram_bytes: int,
    max_ring_buffer_size: int,
) -> tuple[int, ...]:
    layout = build_triton_layout(
        [artifact_path],
        requirements=requirements,
        ring_buffer_size=0,
        vram_budget_bytes=max(int(total_vram_bytes) * 4, 1),
    )
    return _stream_thresholds_from_layout(layout, max_ring_buffer_size=int(max_ring_buffer_size))


def cusparse_ring_thresholds(
    artifact_path: Path,
    *,
    requirements,
    total_vram_bytes: int,
    max_ring_buffer_size: int,
) -> tuple[int, ...]:
    layout = build_cusparse_layout(
        [artifact_path],
        requirements=requirements,
        ring_buffer_size=0,
        vram_budget_bytes=max(int(total_vram_bytes) * 4, 1),
    )
    return _stream_thresholds_from_layout(layout, max_ring_buffer_size=int(max_ring_buffer_size))


def _window_sum(x: np.ndarray, *, shift: int, bandwidth: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {arr.shape}")
    n = int(arr.shape[0])
    start = (np.arange(n, dtype=np.int64) + int(shift)) % n
    dup = np.concatenate((arr, arr), axis=0)
    prefix = np.empty((dup.shape[0] + 1, arr.shape[1]), dtype=np.float64)
    prefix[0].fill(0.0)
    np.cumsum(dup, axis=0, out=prefix[1:])
    stop = start + int(bandwidth)
    return prefix[stop] - prefix[start]


def _band_sum(x: np.ndarray, *, shift: int, bandwidth: int) -> np.ndarray:
    return _window_sum(x, shift=shift, bandwidth=bandwidth)


def _band_sum_transpose(x: np.ndarray, *, shift: int, bandwidth: int) -> np.ndarray:
    return _window_sum(x, shift=-int(shift) - int(bandwidth) + 1, bandwidth=bandwidth)


def expected_up(x: np.ndarray, *, shifts: tuple[int, int, int], bandwidth: int) -> np.ndarray:
    level1 = _band_sum(x, shift=shifts[0], bandwidth=bandwidth)
    return _band_sum(x, shift=shifts[1], bandwidth=bandwidth) + _band_sum(level1, shift=shifts[2], bandwidth=bandwidth)


def expected_down(x: np.ndarray, *, shifts: tuple[int, int, int], bandwidth: int) -> np.ndarray:
    level1 = _band_sum_transpose(x, shift=shifts[2], bandwidth=bandwidth)
    return _band_sum_transpose(level1, shift=shifts[0], bandwidth=bandwidth) + _band_sum_transpose(
        x,
        shift=shifts[1],
        bandwidth=bandwidth,
    )


__all__ = [
    "StreamStressCase",
    "THREE_BLOCK_EXACTNESS_MODES",
    "THREE_BLOCK_TRANSITION_MODES",
    "ThreeBlockMode",
    "_equal_block_budget_components",
    "clear_cupy_state",
    "clear_torch_state",
    "cusparse_ring_thresholds",
    "expected_down",
    "expected_up",
    "prepare_cusparse_stream_stress_case",
    "prepare_triton_stream_stress_case",
    "three_block_mode_budget_bytes",
    "triton_ring_thresholds",
    "write_three_level_band_artifact",
    "write_overlap_band_artifact",
]
