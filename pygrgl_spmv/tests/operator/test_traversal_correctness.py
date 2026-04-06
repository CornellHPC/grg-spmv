"""Operator traversal correctness tests across backends."""

from __future__ import annotations

import numpy as np
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.grg.compile import compile_grg
from pygrgl_spmv.tests.conftest import DATA_DTYPE, K_MATRIX, binary_pm1, matmul_expect_k_hint_warning, tol


def _run_up(op: SpmvGRG, X_col_major: np.ndarray) -> np.ndarray:
    return matmul_expect_k_hint_warning(op, X_col_major.T, "up").T


def _run_down(op: SpmvGRG, X_col_major: np.ndarray) -> np.ndarray:
    return matmul_expect_k_hint_warning(op, X_col_major.T, "down").T


@pytest.mark.smoke
def test_shape_smoke(backend_config, primary_grg_path, spmv_cache_dir):
    import pygrgl

    grg = pygrgl.load_immutable_grg(primary_grg_path)
    op = SpmvGRG(primary_grg_path, backend_config, DATA_DTYPE, artifact_dir=spmv_cache_dir)
    assert op.shape == (grg.num_samples, grg.num_mutations)


@pytest.mark.smoke
def test_forward_smoke(op, gt_small):
    X, Y_expected = gt_small.get("forward", 2, seed=42, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_backward_smoke(op, gt_small):
    X, Y_expected = gt_small.get("backward", 2, seed=42, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("k", K_MATRIX)
def test_forward_full(op, gt_small, k):
    X, Y_expected = gt_small.get("forward", k, seed=123, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("k", K_MATRIX)
def test_backward_full(op, gt_small, k):
    X, Y_expected = gt_small.get("backward", k, seed=123, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


def test_stable_height_order_forward_path(backend_config, primary_grg_path, gt_small, spmv_cache_dir):
    op = SpmvGRG(primary_grg_path, backend_config, DATA_DTYPE, artifact_dir=spmv_cache_dir)
    X, Y_expected = gt_small.get("forward", 4, seed=11, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


def test_stable_height_order_backward_path(backend_config, primary_grg_path, gt_small, spmv_cache_dir):
    op = SpmvGRG(primary_grg_path, backend_config, DATA_DTYPE, artifact_dir=spmv_cache_dir)
    X, Y_expected = gt_small.get("backward", 4, seed=11, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


def test_stable_height_order_keeps_sample_prefix(op):
    np.testing.assert_array_equal(op.node_perm[: op.num_samples], np.arange(op.num_samples, dtype=op.node_perm.dtype))
    np.testing.assert_array_equal(op.inv_node_perm[: op.num_samples], np.arange(op.num_samples, dtype=op.inv_node_perm.dtype))
    assert int(op.level_offsets[1]) >= op.num_samples


def test_compiled_blocks_have_sorted_unique_columns(primary_grg_path):
    import pygrgl

    grg = pygrgl.load_immutable_grg(primary_grg_path, load_up_edges=False)
    state = compile_grg(grg)
    assert state.A_blocks is not None
    assert state.level_offsets.dtype == np.int32
    assert state.node_perm.dtype == np.int32
    assert state.inv_node_perm.dtype == np.int32
    assert state.sample_to_individual.dtype == np.int32
    assert state.sel_mut.indices.dtype == np.int32
    assert state.sel_mut.indptr.dtype == np.int32
    assert state.sel_miss.indices.dtype == np.int32
    assert state.sel_miss.indptr.dtype == np.int32
    for level_blocks in state.A_blocks:
        for block in level_blocks:
            assert block.indices.dtype == np.int32
            assert block.indptr.dtype == np.int32
            indptr = np.asarray(block.indptr)
            indices = np.asarray(block.indices)
            for row in range(block.shape[0]):
                lo = int(indptr[row])
                hi = int(indptr[row + 1])
                row_indices = indices[lo:hi]
                if row_indices.size <= 1:
                    continue
                assert np.all(row_indices[1:] > row_indices[:-1])


def test_compiled_nonempty_blocks_share_read_only_bool_data(primary_grg_path):
    import pygrgl

    grg = pygrgl.load_immutable_grg(primary_grg_path, load_up_edges=False)
    state = compile_grg(grg)
    assert state.A_blocks is not None
    data_arrays = [block.data for level_blocks in state.A_blocks for block in level_blocks if block.nnz > 0]
    assert data_arrays
    first = data_arrays[0]
    assert first.dtype == np.bool_
    assert first.strides == (0,)
    assert not first.flags.writeable
    for data in data_arrays[1:]:
        assert data.dtype == np.bool_
        assert data.strides == (0,)
        assert not data.flags.writeable
        assert np.shares_memory(data, first)


def _ones_input(_rng, shape, dtype):
    return np.ones(shape, dtype=dtype)


_ones_input.__name__ = "ones"


def test_exact_binary_forward(op, gt_small):
    X, Y_expected = gt_small.get("forward", 4, seed=100, dtype=DATA_DTYPE, input_fn=binary_pm1)
    np.testing.assert_array_equal(_run_up(op, X), Y_expected)


def test_exact_binary_backward(op, gt_small):
    X, Y_expected = gt_small.get("backward", 4, seed=100, dtype=DATA_DTYPE, input_fn=binary_pm1)
    np.testing.assert_array_equal(_run_down(op, X), Y_expected)


def test_allele_counts(op, gt_small):
    X, Y_expected = gt_small.get("forward", 1, seed=0, dtype=DATA_DTYPE, input_fn=_ones_input)
    np.testing.assert_array_equal(_run_up(op, X), Y_expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_dtype_forward(backend_config, primary_grg_path, gt_small, spmv_cache_dir, dtype):
    op = SpmvGRG(primary_grg_path, backend_config, dtype, artifact_dir=spmv_cache_dir)
    X, Y_expected = gt_small.get("forward", 4, seed=200, dtype=dtype)
    atol, rtol = tol(dtype)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_dtype_backward(backend_config, primary_grg_path, gt_small, spmv_cache_dir, dtype):
    op = SpmvGRG(primary_grg_path, backend_config, dtype, artifact_dir=spmv_cache_dir)
    X, Y_expected = gt_small.get("backward", 4, seed=200, dtype=dtype)
    atol, rtol = tol(dtype)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


def test_repeated_forward_is_stable(op, gt_small):
    X, _ = gt_small.get("forward", 4, seed=300, dtype=DATA_DTYPE)
    results = [_run_up(op, X) for _ in range(5)]
    for r in results[1:]:
        np.testing.assert_allclose(r, results[0], atol=1e-10)


def test_repeated_backward_is_stable(op, gt_small):
    X, _ = gt_small.get("backward", 4, seed=300, dtype=DATA_DTYPE)
    results = [_run_down(op, X) for _ in range(5)]
    for r in results[1:]:
        np.testing.assert_allclose(r, results[0], atol=1e-10)


def test_zero_forward(op):
    X = np.zeros((op.num_samples, 4), dtype=DATA_DTYPE)
    np.testing.assert_array_equal(_run_up(op, X), 0.0)


def test_zero_backward(op):
    X = np.zeros((op.num_mutations, 4), dtype=DATA_DTYPE)
    np.testing.assert_array_equal(_run_down(op, X), 0.0)
