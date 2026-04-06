"""Helpers for constructing canonical binary CSR matrices."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

_SHARED_BINARY_TRUE = np.ones(1, dtype=np.bool_)


def pick_small_signed_int_dtype(max_value: int) -> np.dtype:
    value = int(max_value)
    if value < 0:
        raise ValueError(f"Structural integer values must be non-negative, got {value}")
    if value <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def finalize_int_array(values, *, label: str, non_negative: bool = True) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{label} must be an integer array, got {arr.dtype}")
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.size == 0:
        if arr.dtype == np.dtype(np.int32):
            return arr
        return arr.astype(np.int32, copy=False)

    arr64 = np.asarray(arr, dtype=np.int64)
    min_value = int(arr64.min())
    max_value = int(arr64.max())
    if non_negative and min_value < 0:
        raise ValueError(f"{label} must be non-negative")
    target = pick_small_signed_int_dtype(max_value if non_negative else max(abs(min_value), abs(max_value)))
    if arr.dtype == target:
        return arr
    return arr.astype(target, copy=False)


def binary_csr_from_parts(
    *,
    indices,
    indptr,
    shape: Sequence[int],
    shared_data: bool = False,
) -> sp.csr_matrix:
    """Build a bool-backed binary CSR matrix from saved CSR arrays."""
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
    nnz = int(idx.size)
    if nnz == 0:
        data = np.empty(0, dtype=np.bool_)
    elif shared_data:
        data = np.broadcast_to(_SHARED_BINARY_TRUE, (nnz,))
    else:
        data = np.ones(nnz, dtype=np.bool_)
    return sp.csr_matrix(
        (
            data,
            idx,
            ptr,
        ),
        shape=shape_tuple,
    )


__all__ = ["binary_csr_from_parts", "finalize_int_array", "pick_small_signed_int_dtype"]
