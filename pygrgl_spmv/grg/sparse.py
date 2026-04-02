"""Helpers for constructing canonical binary CSR matrices."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

_SHARED_BINARY_TRUE = np.ones(1, dtype=np.bool_)


def binary_csr_from_csr_parts(
    *,
    indices,
    indptr,
    shape: Sequence[int],
    index_dtype: np.dtype,
    shared_data: bool = False,
) -> sp.csr_matrix:
    """Build a bool-backed binary CSR matrix from saved CSR arrays."""
    shape_tuple = tuple(int(v) for v in shape)
    if len(shape_tuple) != 2:
        raise ValueError(f"CSR shape must have 2 dimensions, got {shape_tuple}")
    idx = np.asarray(indices, dtype=index_dtype)
    ptr = np.asarray(indptr, dtype=index_dtype)
    if idx.ndim != 1 or ptr.ndim != 1:
        raise ValueError("CSR indices and indptr must be one-dimensional")
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


__all__ = ["binary_csr_from_csr_parts"]
