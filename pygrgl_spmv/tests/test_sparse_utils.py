"""Tests for canonical binary CSR helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pygrgl_spmv.grg.sparse import binary_csr_from_csr_parts


def test_binary_csr_from_csr_parts_rehydrates_binary_matrix():
    matrix = binary_csr_from_csr_parts(
        indices=np.array([1, 0, 2], dtype=np.int32),
        indptr=np.array([0, 1, 3], dtype=np.int32),
        shape=(2, 3),
        index_dtype=np.int64,
    )
    assert matrix.dtype == np.bool_
    np.testing.assert_array_equal(
        matrix.toarray(),
        np.array([[False, True, False], [True, False, True]]),
    )


def test_binary_csr_from_csr_parts_validates_shapes():
    with pytest.raises(ValueError, match="2 dimensions"):
        binary_csr_from_csr_parts(
            indices=np.array([0], dtype=np.int32),
            indptr=np.array([0, 1], dtype=np.int32),
            shape=(1,),
            index_dtype=np.int32,
        )
