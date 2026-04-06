"""Synthetic cases for streamed GPU tests."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import gc
import resource

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends import BackendSetup

_SHARED_BINARY_TRUE = np.ones(1, dtype=np.bool_)
_INT32_MAX = int(np.iinfo(np.int32).max)
_DENSITY_DEN = 100


@dataclass(frozen=True)
class LargeBandCase:
    backend_name: str
    n: int
    bandwidth: int
    nnz: int
    struct_dtype: np.dtype
    block_bytes: int
    total_vram_bytes: int
    available_ram_bytes: int
    shifts: tuple[int, int, int]


def clear_gpu_state() -> None:
    gc.collect()
    try:
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
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


def _effective_available_ram_bytes() -> int:
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


def _block_bytes(*, n: int, bandwidth: int, itemsize: int) -> int:
    return int(itemsize) * (_banded_nnz(n, bandwidth) + int(n) + 1)


def _int32_structure_valid(*, n: int, nnz: int) -> bool:
    return int(nnz) <= _INT32_MAX and 3 * int(n) <= _INT32_MAX


def _smallest_n_for_total_vram(*, total_vram_bytes: int, itemsize: int) -> int:
    lo = 1
    hi = 1
    while 3 * _block_bytes(n=hi, bandwidth=_bandwidth_for_n(hi), itemsize=itemsize) <= int(total_vram_bytes):
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        block_bytes = _block_bytes(n=mid, bandwidth=_bandwidth_for_n(mid), itemsize=itemsize)
        if 3 * int(block_bytes) > int(total_vram_bytes):
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def _make_large_band_case(
    *,
    backend_name: str,
    total_vram_bytes: int,
    available_ram_bytes: int,
    struct_dtype: np.dtype,
) -> LargeBandCase:
    dtype = np.dtype(struct_dtype)
    n = _smallest_n_for_total_vram(total_vram_bytes=total_vram_bytes, itemsize=dtype.itemsize)
    bandwidth = _bandwidth_for_n(n)
    nnz = _banded_nnz(n, bandwidth)
    return LargeBandCase(
        backend_name=backend_name,
        n=int(n),
        bandwidth=int(bandwidth),
        nnz=int(nnz),
        struct_dtype=dtype,
        block_bytes=int(_block_bytes(n=n, bandwidth=bandwidth, itemsize=dtype.itemsize)),
        total_vram_bytes=int(total_vram_bytes),
        available_ram_bytes=int(available_ram_bytes),
        shifts=(0, int(bandwidth), 2 * int(bandwidth)),
    )


def _prepare_large_band_case(*, backend_name: str, total_vram_bytes: int) -> LargeBandCase:
    import pytest

    available_ram_bytes = _effective_available_ram_bytes()
    case32 = _make_large_band_case(
        backend_name=backend_name,
        total_vram_bytes=total_vram_bytes,
        available_ram_bytes=available_ram_bytes,
        struct_dtype=np.int32,
    )
    if _int32_structure_valid(n=case32.n, nnz=case32.nnz):
        if int(case32.available_ram_bytes) < 3 * int(case32.block_bytes):
            pytest.skip(
                f"{backend_name} large streamed tests need {3 * case32.block_bytes} host bytes for three blocks, "
                f"but only {case32.available_ram_bytes} are available"
            )
        return case32

    case64 = _make_large_band_case(
        backend_name=backend_name,
        total_vram_bytes=total_vram_bytes,
        available_ram_bytes=available_ram_bytes,
        struct_dtype=np.int64,
    )
    if backend_name == "cusparse" and int(case64.nnz) > _INT32_MAX:
        pytest.skip("no valid cuSPARSE large streamed case exists below the CUDA 12.9 near-2^31 SpMM boundary")
    if int(case64.available_ram_bytes) < 3 * int(case64.block_bytes):
        pytest.skip(
            f"{backend_name} large streamed tests need {3 * case64.block_bytes} host bytes for three blocks, "
            f"but only {case64.available_ram_bytes} are available"
        )
    return case64


@functools.cache
def prepare_triton_large_band_case() -> LargeBandCase:
    import torch

    clear_gpu_state()
    with torch.cuda.device(0):
        _free_vram_bytes, total_vram_bytes = (int(v) for v in torch.cuda.mem_get_info())
    return _prepare_large_band_case(backend_name="triton", total_vram_bytes=total_vram_bytes)


@functools.cache
def prepare_cusparse_large_band_case() -> LargeBandCase:
    import cupy as cp

    clear_gpu_state()
    with cp.cuda.Device(0):
        _free_vram_bytes, total_vram_bytes = (int(v) for v in cp.cuda.runtime.memGetInfo())
    return _prepare_large_band_case(backend_name="cusparse", total_vram_bytes=total_vram_bytes)


def _attach_binary_csr(indices: np.ndarray, indptr: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
    shape_tuple = tuple(int(v) for v in shape)
    if len(shape_tuple) != 2:
        raise ValueError(f"CSR shape must have 2 dimensions, got {shape_tuple}")
    idx = np.asarray(indices)
    ptr = np.asarray(indptr)
    if idx.ndim != 1 or ptr.ndim != 1:
        raise ValueError("CSR indices and indptr must be one-dimensional")
    if idx.dtype not in {np.dtype(np.int32), np.dtype(np.int64)}:
        raise TypeError(f"indices must use int32 or int64, got {idx.dtype}")
    if ptr.dtype not in {np.dtype(np.int32), np.dtype(np.int64)}:
        raise TypeError(f"indptr must use int32 or int64, got {ptr.dtype}")
    if ptr.size != shape_tuple[0] + 1:
        raise ValueError(f"indptr must have length {shape_tuple[0] + 1}, got {ptr.size}")
    if int(ptr[0]) != 0:
        raise ValueError("indptr[0] must be 0")
    if int(ptr[-1]) != int(idx.size):
        raise ValueError(f"indptr[-1] must equal nnz={idx.size}, got {int(ptr[-1])}")
    if np.any(ptr[1:] < ptr[:-1]):
        raise ValueError("indptr must be nondecreasing")
    nnz = int(idx.size)
    if nnz == 0:
        data = np.empty(0, dtype=np.bool_)
    else:
        data = np.broadcast_to(_SHARED_BINARY_TRUE, (nnz,))

    # Preserve explicit int64 stress structure: SciPy's normal CSR constructor
    # re-infers structural dtypes and may downcast back to int32 because the
    # values still fit, which would invalidate the ring-3 OOM contract.
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
    if bandwidth <= 0 or bandwidth > n:
        raise ValueError(f"bandwidth must be in [1, n], got bandwidth={bandwidth}, n={n}")
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


def build_large_band_setup(case: LargeBandCase) -> BackendSetup:
    n = int(case.n)
    dtype = np.dtype(case.struct_dtype)
    block10 = _build_band_csr(n, case.bandwidth, case.shifts[0], dtype)
    block20 = _build_band_csr(n, case.bandwidth, case.shifts[1], dtype)
    block21 = _build_band_csr(n, case.bandwidth, case.shifts[2], dtype)
    sel_mut = _attach_binary_csr(
        np.arange(2 * n, 3 * n, dtype=dtype),
        np.arange(n + 1, dtype=dtype),
        (n, 3 * n),
    )
    sel_miss = _attach_binary_csr(
        np.empty(0, dtype=dtype),
        np.zeros(n + 1, dtype=dtype),
        (n, 3 * n),
    )
    return BackendSetup(
        A_blocks=[[], [block10], [block20, block21]],
        level_offsets=np.asarray([0, n, 2 * n, 3 * n], dtype=dtype),
        num_samples=n,
        num_mutations=n,
        num_nodes=3 * n,
        sel_mut=sel_mut,
        sel_miss=sel_miss,
        coalescence_counts=None,
        dtype=np.float64,
    )


def build_overlap_band_setup(*, n: int = 4096, bandwidth: int = 64) -> BackendSetup:
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
    sel_mut = _attach_binary_csr(
        np.arange(3 * n, 4 * n, dtype=dtype),
        np.arange(n + 1, dtype=dtype),
        (n, 4 * n),
    )
    sel_miss = _attach_binary_csr(
        np.empty(0, dtype=dtype),
        np.zeros(n + 1, dtype=dtype),
        (n, 4 * n),
    )
    return BackendSetup(
        A_blocks=[[], [block10], [block20, block21], [block30, block31, block32]],
        level_offsets=np.asarray([0, n, 2 * n, 3 * n, 4 * n], dtype=dtype),
        num_samples=n,
        num_mutations=n,
        num_nodes=4 * n,
        sel_mut=sel_mut,
        sel_miss=sel_miss,
        coalescence_counts=None,
        dtype=np.float64,
    )


def _window_sum(x: np.ndarray, *, shift: int, bandwidth: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {arr.shape}")
    n = int(arr.shape[0])
    if int(bandwidth) <= 0 or int(bandwidth) > n:
        raise ValueError(f"bandwidth must be in [1, n], got bandwidth={bandwidth}, n={n}")
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
    "LargeBandCase",
    "build_large_band_setup",
    "build_overlap_band_setup",
    "clear_gpu_state",
    "expected_down",
    "expected_up",
    "prepare_cusparse_large_band_case",
    "prepare_triton_large_band_case",
]
