"""Tests for canonical binary CSR helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pygrgl_spmv.grg.sparse import binary_csr_from_parts, finalize_int_array, pick_small_signed_int_dtype


def test_pick_small_signed_int_dtype():
    assert pick_small_signed_int_dtype(7) == np.dtype(np.int32)
    assert pick_small_signed_int_dtype(np.iinfo(np.int32).max) == np.dtype(np.int32)
    assert pick_small_signed_int_dtype(np.iinfo(np.int32).max + 1) == np.dtype(np.int64)


def test_finalize_int_array_preserves_existing_int32():
    arr = np.array([0, 1, 2], dtype=np.int32)
    out = finalize_int_array(arr, label="arr")
    assert out is arr


def test_finalize_int_array_downcasts_safe_int64():
    arr = np.array([0, 1, 2], dtype=np.int64)
    out = finalize_int_array(arr, label="arr")
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, arr)


def test_finalize_int_array_downcasts_empty_int64_to_int32():
    arr = np.empty(0, dtype=np.int64)
    out = finalize_int_array(arr, label="arr")
    assert out.dtype == np.int32
    assert out.shape == (0,)


def test_finalize_int_array_rejects_floats():
    with pytest.raises(TypeError, match="integer array"):
        finalize_int_array(np.array([0.0, 1.0]), label="arr")


def test_binary_csr_from_parts_rehydrates_binary_matrix():
    matrix = binary_csr_from_parts(
        indices=np.array([1, 0, 2], dtype=np.int32),
        indptr=np.array([0, 1, 3], dtype=np.int64),
        shape=(2, 3),
    )
    assert matrix.dtype == np.bool_
    assert matrix.indices.dtype == np.int32
    np.testing.assert_array_equal(
        matrix.toarray(),
        np.array([[False, True, False], [True, False, True]]),
    )


def test_binary_csr_from_parts_preserves_int64_indices_when_values_require_it():
    matrix = binary_csr_from_parts(
        indices=np.array([2_500_000_000], dtype=np.int64),
        indptr=np.array([0, 1], dtype=np.int32),
        shape=(1, 3_000_000_000),
    )
    assert matrix.indices.dtype == np.int64
    assert matrix.indptr.dtype == np.int64
    assert matrix.nnz == 1


def test_binary_csr_from_parts_can_share_immutable_data():
    matrix = binary_csr_from_parts(
        indices=np.array([1, 0, 2], dtype=np.int32),
        indptr=np.array([0, 1, 3], dtype=np.int64),
        shape=(2, 3),
        shared_data=True,
    )
    assert matrix.dtype == np.bool_
    assert matrix.data.strides == (0,)
    assert not matrix.data.flags.writeable
    np.testing.assert_array_equal(
        matrix.toarray(),
        np.array([[False, True, False], [True, False, True]]),
    )
    with pytest.raises(ValueError, match="read-only"):
        matrix.data[0] = False


def test_binary_csr_from_parts_validates_shapes():
    with pytest.raises(ValueError, match="2 dimensions"):
        binary_csr_from_parts(
            indices=np.array([0], dtype=np.int32),
            indptr=np.array([0, 1], dtype=np.int32),
            shape=(1,),
        )
