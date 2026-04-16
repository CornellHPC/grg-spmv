from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv import ReferenceRuntime
from pygrgl_spmv.tests.conftest import DATA_DTYPE, tol
from pygrgl_spmv.tests.runtime._runtime_builders import build_reference_layout, full_requirements


@contextmanager
def _open_reference_grg(artifact, *, dtype=np.float64):
    with ReferenceRuntime(build_reference_layout([artifact], dtype=dtype, requirements=full_requirements(max_k_up=8, max_k_down=8))) as runtime:
        yield runtime.grgs[0]


@pytest.mark.parametrize("direction", [pygrgl.TraversalDirection.UP, pygrgl.TraversalDirection.DOWN], ids=["up", "down"])
def test_emit_all_nodes_baseline(primary_artifact, primary_grg, direction):
    with _open_reference_grg(primary_artifact) as grg:
        rng = np.random.default_rng(6101 if direction == pygrgl.TraversalDirection.UP else 6102)
        cols = primary_grg.num_samples if direction == pygrgl.TraversalDirection.UP else primary_grg.num_mutations
        matrix = rng.standard_normal((3, cols), dtype=DATA_DTYPE)
        expected = np.asarray(pygrgl.matmul(primary_grg, matrix, direction, emit_all_nodes=True))
        actual = grg.matmul(matrix, direction, emit_all_nodes=True)
        atol, rtol = tol(DATA_DTYPE)
        assert actual.shape == (matrix.shape[0], grg.num_nodes)
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("direction", [pygrgl.TraversalDirection.UP, pygrgl.TraversalDirection.DOWN], ids=["up", "down"])
def test_emit_all_nodes_by_individual(primary_artifact, primary_grg, direction):
    if primary_grg.num_individuals == primary_grg.num_samples:
        pytest.skip("fixture is not grouped by individuals")
    with _open_reference_grg(primary_artifact) as grg:
        rng = np.random.default_rng(6201 if direction == pygrgl.TraversalDirection.UP else 6202)
        cols = primary_grg.num_individuals if direction == pygrgl.TraversalDirection.UP else primary_grg.num_mutations
        matrix = rng.standard_normal((4, cols), dtype=DATA_DTYPE)
        expected = np.asarray(pygrgl.matmul(primary_grg, matrix, direction, emit_all_nodes=True, by_individual=True))
        actual = grg.matmul(matrix, direction, emit_all_nodes=True, by_individual=True)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def test_emit_all_nodes_init_modes(primary_artifact, primary_grg):
    with _open_reference_grg(primary_artifact) as grg:
        rng = np.random.default_rng(6303)
        rows = 3
        x_up = rng.standard_normal((rows, primary_grg.num_samples), dtype=DATA_DTYPE)
        x_down = rng.standard_normal((rows, primary_grg.num_mutations), dtype=DATA_DTYPE)
        init_vec = rng.standard_normal((rows,), dtype=DATA_DTYPE)
        init_mat = rng.standard_normal((rows, primary_grg.num_nodes), dtype=DATA_DTYPE)
        init_modes: list[object] = [init_vec, init_mat]
        if grg.coalescence_counts is not None:
            init_modes.append("xtx")
        atol, rtol = tol(DATA_DTYPE)
        for init in init_modes:
            np.testing.assert_allclose(
                grg.matmul(x_up, pygrgl.TraversalDirection.UP, emit_all_nodes=True, init=init),
                np.asarray(pygrgl.matmul(primary_grg, x_up, pygrgl.TraversalDirection.UP, emit_all_nodes=True, init=init)),
                atol=atol,
                rtol=rtol,
            )
            np.testing.assert_allclose(
                grg.matmul(x_down, pygrgl.TraversalDirection.DOWN, emit_all_nodes=True, init=init),
                np.asarray(pygrgl.matmul(primary_grg, x_down, pygrgl.TraversalDirection.DOWN, emit_all_nodes=True, init=init)),
                atol=atol,
                rtol=rtol,
            )


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("direction", [pygrgl.TraversalDirection.UP, pygrgl.TraversalDirection.DOWN], ids=["up", "down"])
def test_emit_all_nodes_dtype(primary_artifact, primary_grg, dtype, direction):
    with _open_reference_grg(primary_artifact, dtype=dtype) as grg:
        rng = np.random.default_rng(6401 if direction == pygrgl.TraversalDirection.UP else 6402)
        cols = primary_grg.num_samples if direction == pygrgl.TraversalDirection.UP else primary_grg.num_mutations
        matrix = rng.standard_normal((2, cols), dtype=dtype)
        expected = np.asarray(pygrgl.matmul(primary_grg, matrix, direction, emit_all_nodes=True))
        actual = grg.matmul(matrix, direction, emit_all_nodes=True)
        atol, rtol = tol(dtype)
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def test_emit_all_nodes_repeated_is_stable(primary_artifact):
    with _open_reference_grg(primary_artifact) as grg:
        rng = np.random.default_rng(6505)
        matrix = rng.standard_normal((2, grg.num_samples), dtype=DATA_DTYPE)
        results = [grg.matmul(matrix, pygrgl.TraversalDirection.UP, emit_all_nodes=True) for _ in range(3)]
        for result in results[1:]:
            np.testing.assert_allclose(result, results[0], atol=1e-10)


def test_emit_all_nodes_zero_input(primary_artifact):
    with _open_reference_grg(primary_artifact) as grg:
        up = np.zeros((2, grg.num_samples), dtype=DATA_DTYPE)
        down = np.zeros((2, grg.num_mutations), dtype=DATA_DTYPE)
        np.testing.assert_array_equal(grg.matmul(up, pygrgl.TraversalDirection.UP, emit_all_nodes=True), 0.0)
        np.testing.assert_array_equal(grg.matmul(down, pygrgl.TraversalDirection.DOWN, emit_all_nodes=True), 0.0)


def test_emit_all_nodes_rejects_miss(missing_artifact):
    with _open_reference_grg(missing_artifact) as grg:
        up = np.ones((2, grg.num_individuals), dtype=DATA_DTYPE)
        down = np.ones((2, grg.num_mutations), dtype=DATA_DTYPE)
        miss = np.zeros((2, grg.num_mutations), dtype=DATA_DTYPE)
        with pytest.raises(RuntimeError, match="cannot be mixed"):
            grg.matmul(up, pygrgl.TraversalDirection.UP, emit_all_nodes=True, by_individual=True, miss=miss)
        with pytest.raises(RuntimeError, match="cannot be mixed"):
            grg.matmul(down, pygrgl.TraversalDirection.DOWN, emit_all_nodes=True, by_individual=True, miss=miss)

