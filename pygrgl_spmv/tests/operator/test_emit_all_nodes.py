"""Node-output matmul semantics for emit_all_nodes=True across backends."""

from __future__ import annotations

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.tests.conftest import DATA_DTYPE, matmul_expect_k_hint_warning, tol


def _expected(grg, matrix, direction, **kwargs):
    return pygrgl.matmul(grg, matrix, direction, emit_all_nodes=True, **kwargs)


def _actual(op, matrix, direction, **kwargs):
    return matmul_expect_k_hint_warning(op, matrix, direction, emit_all_nodes=True, **kwargs)


@pytest.mark.smoke
@pytest.mark.parametrize("direction", [pygrgl.TraversalDirection.UP, pygrgl.TraversalDirection.DOWN], ids=["up", "down"])
def test_emit_all_nodes_baseline(op, grg_ref, direction):
    rng = np.random.default_rng(6101 if direction == pygrgl.TraversalDirection.UP else 6102)
    cols = grg_ref.num_samples if direction == pygrgl.TraversalDirection.UP else grg_ref.num_mutations
    matrix = rng.standard_normal((3, cols), dtype=DATA_DTYPE)
    expected = _expected(grg_ref, matrix, direction)
    actual = _actual(op, matrix, direction)
    atol, rtol = tol(DATA_DTYPE)
    assert actual.shape == (matrix.shape[0], op.num_nodes)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("direction", [pygrgl.TraversalDirection.UP, pygrgl.TraversalDirection.DOWN], ids=["up", "down"])
def test_emit_all_nodes_by_individual(op, grg_ref, direction):
    rng = np.random.default_rng(6201 if direction == pygrgl.TraversalDirection.UP else 6202)
    cols = grg_ref.num_individuals if direction == pygrgl.TraversalDirection.UP else grg_ref.num_mutations
    matrix = rng.standard_normal((4, cols), dtype=DATA_DTYPE)
    expected = _expected(grg_ref, matrix, direction, by_individual=True)
    actual = _actual(op, matrix, direction, by_individual=True)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def test_emit_all_nodes_init_modes(op, grg_ref):
    rng = np.random.default_rng(6303)
    rows = 3
    x_up = rng.standard_normal((rows, grg_ref.num_samples), dtype=DATA_DTYPE)
    x_down = rng.standard_normal((rows, grg_ref.num_mutations), dtype=DATA_DTYPE)
    init_vec = rng.standard_normal(rows, dtype=DATA_DTYPE)
    init_mat = rng.standard_normal((rows, grg_ref.num_nodes), dtype=DATA_DTYPE)

    init_modes = [init_vec, init_mat]
    if op.coalescence_counts is not None:
        init_modes.append("xtx")

    atol, rtol = tol(DATA_DTYPE)
    for init in init_modes:
        np.testing.assert_allclose(
            _actual(op, x_up, pygrgl.TraversalDirection.UP, init=init),
            _expected(grg_ref, x_up, pygrgl.TraversalDirection.UP, init=init),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            _actual(op, x_down, pygrgl.TraversalDirection.DOWN, init=init),
            _expected(grg_ref, x_down, pygrgl.TraversalDirection.DOWN, init=init),
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("direction", [pygrgl.TraversalDirection.UP, pygrgl.TraversalDirection.DOWN], ids=["up", "down"])
def test_emit_all_nodes_dtype(backend_config, primary_grg_path, grg_ref, spmv_cache_dir, dtype, direction):
    op = SpmvGRG(primary_grg_path, backend_config, dtype, artifact_dir=spmv_cache_dir)
    rng = np.random.default_rng(6401 if direction == pygrgl.TraversalDirection.UP else 6402)
    cols = grg_ref.num_samples if direction == pygrgl.TraversalDirection.UP else grg_ref.num_mutations
    matrix = rng.standard_normal((2, cols), dtype=dtype)
    expected = _expected(grg_ref, matrix, direction)
    actual = _actual(op, matrix, direction)
    atol, rtol = tol(dtype)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def test_emit_all_nodes_repeated_is_stable(op, grg_ref):
    rng = np.random.default_rng(6505)
    matrix = rng.standard_normal((2, grg_ref.num_samples), dtype=DATA_DTYPE)
    results = [_actual(op, matrix, pygrgl.TraversalDirection.UP) for _ in range(3)]
    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], atol=1e-10)


def test_emit_all_nodes_zero_input(op):
    up = np.zeros((2, op.num_samples), dtype=DATA_DTYPE)
    down = np.zeros((2, op.num_mutations), dtype=DATA_DTYPE)
    np.testing.assert_array_equal(_actual(op, up, pygrgl.TraversalDirection.UP), 0.0)
    np.testing.assert_array_equal(_actual(op, down, pygrgl.TraversalDirection.DOWN), 0.0)


def test_emit_all_nodes_rejects_miss(op_missing):
    up = np.ones((2, op_missing.num_individuals), dtype=DATA_DTYPE)
    down = np.ones((2, op_missing.num_mutations), dtype=DATA_DTYPE)
    miss = np.zeros((2, op_missing.num_mutations), dtype=DATA_DTYPE)

    with pytest.raises(RuntimeError, match='The "miss" parameter cannot be mixed with the "emit_all_nodes" parameter'):
        op_missing.matmul(
            up,
            pygrgl.TraversalDirection.UP,
            emit_all_nodes=True,
            by_individual=True,
            miss=miss,
        )
    with pytest.raises(RuntimeError, match='The "miss" parameter cannot be mixed with the "emit_all_nodes" parameter'):
        op_missing.matmul(
            down,
            pygrgl.TraversalDirection.DOWN,
            emit_all_nodes=True,
            by_individual=True,
            miss=miss,
        )
