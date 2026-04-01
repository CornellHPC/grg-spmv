"""Helpers for constructing canonical binary CSR matrices."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp


def binary_csr_from_csr_parts(
    *,
    indices,
    indptr,
    shape: Sequence[int],
    index_dtype: np.dtype,
) -> sp.csr_matrix:
    """Build a binary CSR matrix from saved CSR arrays.

    Data values are stored as bool (True = 1), saving 4–8× versus float32/64.
    All backends either ignore the data array (cuSPARSE, Triton) or convert
    it themselves (MKL casts to float64; Reference relies on scipy promotion).
    """
    shape_tuple = tuple(int(v) for v in shape)
    if len(shape_tuple) != 2:
        raise ValueError(f"CSR shape must have 2 dimensions, got {shape_tuple}")
    idx = np.asarray(indices, dtype=index_dtype)
    ptr = np.asarray(indptr, dtype=index_dtype)
    if idx.ndim != 1 or ptr.ndim != 1:
        raise ValueError("CSR indices and indptr must be one-dimensional")
    return sp.csr_matrix(
        (
            np.ones(int(idx.size), dtype=np.bool_),
            idx,
            ptr,
        ),
        shape=shape_tuple,
    )


__all__ = ["binary_csr_from_csr_parts"]
