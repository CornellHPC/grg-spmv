from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv import ReferenceRuntime
from pygrgl_spmv.grg.compile import compile_grg
from pygrgl_spmv.tests.conftest import DATA_DTYPE, binary_pm1, tol
from pygrgl_spmv.tests.runtime._runtime_builders import build_reference_layout, full_requirements

_K_MATRIX = [1, 2, 3, 7, 8, 9, 16, 20]


@contextmanager
def _open_reference_grg(artifact, *, dtype=np.float64, max_k: int = 20):
    with ReferenceRuntime(
        build_reference_layout(
            [artifact],
            dtype=dtype,
            requirements=full_requirements(max_k_up=max_k, max_k_down=max_k),
        )
    ) as runtime:
        yield runtime.grgs[0]


def _expected(grg, matrix: np.ndarray, direction: pygrgl.TraversalDirection) -> np.ndarray:
    return np.asarray(pygrgl.matmul(grg, matrix, direction))


def _ones_input(_rng, shape, dtype):
    return np.ones(shape, dtype=dtype)


def test_shape(primary_artifact, primary_grg):
    with _open_reference_grg(primary_artifact) as grg:
        assert grg.shape == (primary_grg.num_samples, primary_grg.num_mutations)


def test_forward_smoke(primary_artifact, primary_grg):
    with _open_reference_grg(primary_artifact) as grg:
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, primary_grg.num_samples), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, "up"), _expected(primary_grg, x, pygrgl.TraversalDirection.UP), atol=atol, rtol=rtol)


def test_backward_smoke(primary_artifact, primary_grg):
    with _open_reference_grg(primary_artifact) as grg:
        rng = np.random.default_rng(43)
        x = rng.standard_normal((2, primary_grg.num_mutations), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, "down"), _expected(primary_grg, x, pygrgl.TraversalDirection.DOWN), atol=atol, rtol=rtol)


@pytest.mark.parametrize("k", _K_MATRIX)
def test_forward_k_sweep(primary_artifact, primary_grg, k):
    with _open_reference_grg(primary_artifact, max_k=max(k, 20)) as grg:
        rng = np.random.default_rng(100 + int(k))
        x = rng.standard_normal((int(k), primary_grg.num_samples), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, "up"), _expected(primary_grg, x, pygrgl.TraversalDirection.UP), atol=atol, rtol=rtol)


@pytest.mark.parametrize("k", _K_MATRIX)
def test_backward_k_sweep(primary_artifact, primary_grg, k):
    with _open_reference_grg(primary_artifact, max_k=max(k, 20)) as grg:
        rng = np.random.default_rng(200 + int(k))
        x = rng.standard_normal((int(k), primary_grg.num_mutations), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, "down"), _expected(primary_grg, x, pygrgl.TraversalDirection.DOWN), atol=atol, rtol=rtol)


def test_stable_height_order_forward_path(primary_artifact, primary_grg):
    with _open_reference_grg(primary_artifact) as grg:
        rng = np.random.default_rng(11)
        x = rng.standard_normal((4, primary_grg.num_samples), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, "up"), _expected(primary_grg, x, pygrgl.TraversalDirection.UP), atol=atol, rtol=rtol)


def test_stable_height_order_backward_path(primary_artifact, primary_grg):
    with _open_reference_grg(primary_artifact) as grg:
        rng = np.random.default_rng(12)
        x = rng.standard_normal((4, primary_grg.num_mutations), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, "down"), _expected(primary_grg, x, pygrgl.TraversalDirection.DOWN), atol=atol, rtol=rtol)


def test_stable_height_order_keeps_sample_prefix(primary_artifact):
    with _open_reference_grg(primary_artifact) as grg:
        np.testing.assert_array_equal(grg.node_perm[: grg.num_samples], np.arange(grg.num_samples, dtype=grg.node_perm.dtype))
        np.testing.assert_array_equal(grg.inv_node_perm[: grg.num_samples], np.arange(grg.num_samples, dtype=grg.inv_node_perm.dtype))
        assert int(grg.level_offsets[1]) >= grg.num_samples


def test_compiled_blocks_have_sorted_unique_columns(primary_grg_path):
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
                if row_indices.size > 1:
                    assert np.all(row_indices[1:] > row_indices[:-1])


def test_compiled_nonempty_blocks_share_read_only_bool_data(primary_grg_path):
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


def test_exact_binary_forward(primary_artifact, primary_grg):
    with _open_reference_grg(primary_artifact) as grg:
        rng = np.random.default_rng(300)
        x = binary_pm1(rng, (4, primary_grg.num_samples), DATA_DTYPE)
        np.testing.assert_array_equal(grg.matmul(x, "up"), _expected(primary_grg, x, pygrgl.TraversalDirection.UP))


def test_exact_binary_backward(primary_artifact, primary_grg):
    with _open_reference_grg(primary_artifact) as grg:
        rng = np.random.default_rng(301)
        x = binary_pm1(rng, (4, primary_grg.num_mutations), DATA_DTYPE)
        np.testing.assert_array_equal(grg.matmul(x, "down"), _expected(primary_grg, x, pygrgl.TraversalDirection.DOWN))


def test_allele_counts(primary_artifact, primary_grg):
    with _open_reference_grg(primary_artifact) as grg:
        rng = np.random.default_rng(0)
        x = _ones_input(rng, (1, primary_grg.num_samples), DATA_DTYPE)
        np.testing.assert_array_equal(grg.matmul(x, "up"), _expected(primary_grg, x, pygrgl.TraversalDirection.UP))


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_dtype_forward(primary_artifact, primary_grg, dtype):
    with _open_reference_grg(primary_artifact, dtype=dtype, max_k=8) as grg:
        rng = np.random.default_rng(400)
        x = rng.standard_normal((4, primary_grg.num_samples), dtype=dtype)
        atol, rtol = tol(dtype)
        np.testing.assert_allclose(grg.matmul(x, "up"), _expected(primary_grg, x, pygrgl.TraversalDirection.UP), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_dtype_backward(primary_artifact, primary_grg, dtype):
    with _open_reference_grg(primary_artifact, dtype=dtype, max_k=8) as grg:
        rng = np.random.default_rng(401)
        x = rng.standard_normal((4, primary_grg.num_mutations), dtype=dtype)
        atol, rtol = tol(dtype)
        np.testing.assert_allclose(grg.matmul(x, "down"), _expected(primary_grg, x, pygrgl.TraversalDirection.DOWN), atol=atol, rtol=rtol)


def test_repeated_forward_is_stable(primary_artifact, primary_grg):
    with _open_reference_grg(primary_artifact) as grg:
        rng = np.random.default_rng(500)
        x = rng.standard_normal((4, primary_grg.num_samples), dtype=DATA_DTYPE)
        results = [grg.matmul(x, "up") for _ in range(5)]
        for result in results[1:]:
            np.testing.assert_allclose(result, results[0], atol=1e-10)


def test_repeated_backward_is_stable(primary_artifact, primary_grg):
    with _open_reference_grg(primary_artifact) as grg:
        rng = np.random.default_rng(501)
        x = rng.standard_normal((4, primary_grg.num_mutations), dtype=DATA_DTYPE)
        results = [grg.matmul(x, "down") for _ in range(5)]
        for result in results[1:]:
            np.testing.assert_allclose(result, results[0], atol=1e-10)


def test_zero_forward(primary_artifact):
    with _open_reference_grg(primary_artifact) as grg:
        x = np.zeros((4, grg.num_samples), dtype=DATA_DTYPE)
        np.testing.assert_array_equal(grg.matmul(x, "up"), 0.0)


def test_zero_backward(primary_artifact):
    with _open_reference_grg(primary_artifact) as grg:
        x = np.zeros((4, grg.num_mutations), dtype=DATA_DTYPE)
        np.testing.assert_array_equal(grg.matmul(x, "down"), 0.0)

