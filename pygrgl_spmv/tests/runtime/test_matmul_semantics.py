from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv import ReferenceRuntime
from pygrgl_spmv.tests.conftest import DATA_DTYPE, tol
from pygrgl_spmv.tests.runtime._runtime_builders import build_reference_layout, full_requirements


@contextmanager
def _open_reference_grg(artifact, *, requirements=None, dtype=np.float64):
    layout = build_reference_layout(
        [artifact],
        dtype=dtype,
        requirements=full_requirements() if requirements is None else requirements,
    )
    with ReferenceRuntime(layout) as runtime:
        yield runtime.grgs[0]


def test_by_individual(primary_artifact, primary_grg):
    if primary_grg.num_individuals == primary_grg.num_samples:
        pytest.skip("fixture is not grouped by individuals")
    with _open_reference_grg(primary_artifact) as grg:
        rng = np.random.default_rng(1101)
        rows = 5
        x_up = rng.standard_normal((rows, primary_grg.num_individuals), dtype=DATA_DTYPE)
        x_down = rng.standard_normal((rows, primary_grg.num_mutations), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(
            grg.matmul(x_up, pygrgl.TraversalDirection.UP, by_individual=True),
            np.asarray(pygrgl.matmul(primary_grg, x_up, pygrgl.TraversalDirection.UP, by_individual=True)),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            grg.matmul(x_down, pygrgl.TraversalDirection.DOWN, by_individual=True),
            np.asarray(pygrgl.matmul(primary_grg, x_down, pygrgl.TraversalDirection.DOWN, by_individual=True)),
            atol=atol,
            rtol=rtol,
        )


def test_init_modes_both_directions(primary_artifact, primary_grg):
    with _open_reference_grg(primary_artifact) as grg:
        rng = np.random.default_rng(2203)
        rows = 4
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
                grg.matmul(x_up, pygrgl.TraversalDirection.UP, init=init),
                np.asarray(pygrgl.matmul(primary_grg, x_up, pygrgl.TraversalDirection.UP, init=init)),
                atol=atol,
                rtol=rtol,
            )
            np.testing.assert_allclose(
                grg.matmul(x_down, pygrgl.TraversalDirection.DOWN, init=init),
                np.asarray(pygrgl.matmul(primary_grg, x_down, pygrgl.TraversalDirection.DOWN, init=init)),
                atol=atol,
                rtol=rtol,
            )


def test_missing_input_output(missing_artifact, missing_grg):
    if not missing_grg.has_missing_data:
        pytest.skip("dataset has no missingness nodes")
    if missing_grg.num_individuals == missing_grg.num_samples:
        pytest.skip("fixture is not grouped by individuals")

    with _open_reference_grg(missing_artifact) as grg:
        rng = np.random.default_rng(3303)
        rows = 3
        x_up = rng.standard_normal((rows, missing_grg.num_individuals), dtype=DATA_DTYPE)
        miss_expected = np.zeros((rows, missing_grg.num_mutations), dtype=DATA_DTYPE)
        miss_actual = np.zeros((rows, missing_grg.num_mutations), dtype=DATA_DTYPE)
        expected_up = np.asarray(
            pygrgl.matmul(
                missing_grg,
                x_up,
                pygrgl.TraversalDirection.UP,
                by_individual=True,
                miss=miss_expected,
            )
        )
        actual_up = grg.matmul(
            x_up,
            pygrgl.TraversalDirection.UP,
            by_individual=True,
            miss=miss_actual,
        )

        x_down = rng.standard_normal((rows, missing_grg.num_mutations), dtype=DATA_DTYPE)
        miss_down = rng.standard_normal((rows, missing_grg.num_mutations), dtype=DATA_DTYPE)
        expected_down = np.asarray(
            pygrgl.matmul(
                missing_grg,
                x_down,
                pygrgl.TraversalDirection.DOWN,
                by_individual=True,
                miss=miss_down.copy(),
            )
        )
        actual_down = grg.matmul(
            x_down,
            pygrgl.TraversalDirection.DOWN,
            by_individual=True,
            miss=miss_down.copy(),
        )

        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(actual_up, expected_up, atol=atol, rtol=rtol)
        np.testing.assert_allclose(miss_actual, miss_expected, atol=atol, rtol=rtol)
        np.testing.assert_allclose(actual_down, expected_down, atol=atol, rtol=rtol)


def test_bool_init_dtype_strict_behavior(primary_artifact):
    with _open_reference_grg(primary_artifact) as grg:
        rng = np.random.default_rng(5505)
        rows = 4
        up_bool = rng.integers(0, 2, size=(rows, grg.num_samples), dtype=np.int8).astype(bool)
        down_bool = rng.integers(0, 2, size=(rows, grg.num_mutations), dtype=np.int8).astype(bool)
        init_vec_bool = rng.integers(0, 2, size=(rows,), dtype=np.int8).astype(bool)
        init_mat_bool = rng.integers(0, 2, size=(rows, grg.num_nodes), dtype=np.int8).astype(bool)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(
            grg.matmul(up_bool, pygrgl.TraversalDirection.UP, init=init_vec_bool),
            grg.matmul(up_bool.astype(DATA_DTYPE), pygrgl.TraversalDirection.UP, init=init_vec_bool.astype(DATA_DTYPE)),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            grg.matmul(up_bool, pygrgl.TraversalDirection.UP, init=init_mat_bool),
            grg.matmul(up_bool.astype(DATA_DTYPE), pygrgl.TraversalDirection.UP, init=init_mat_bool.astype(DATA_DTYPE)),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            grg.matmul(down_bool, pygrgl.TraversalDirection.DOWN, init=init_vec_bool),
            grg.matmul(
                down_bool.astype(DATA_DTYPE),
                pygrgl.TraversalDirection.DOWN,
                init=init_vec_bool.astype(DATA_DTYPE),
            ),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            grg.matmul(down_bool, pygrgl.TraversalDirection.DOWN, init=init_mat_bool),
            grg.matmul(
                down_bool.astype(DATA_DTYPE),
                pygrgl.TraversalDirection.DOWN,
                init=init_mat_bool.astype(DATA_DTYPE),
            ),
            atol=atol,
            rtol=rtol,
        )
        with pytest.raises(TypeError, match="dtype"):
            grg.matmul(up_bool.astype(DATA_DTYPE), pygrgl.TraversalDirection.UP, init=init_vec_bool)


def test_matmul_requires_numpy_array_inputs(primary_artifact):
    with _open_reference_grg(primary_artifact) as grg:
        rows = 2
        x_up = np.ones((rows, grg.num_samples), dtype=DATA_DTYPE)
        miss_up = np.zeros((rows, grg.num_mutations), dtype=DATA_DTYPE)
        init_vec = np.ones((rows,), dtype=DATA_DTYPE)

        with pytest.raises(TypeError, match="numpy.ndarray"):
            grg.matmul(x_up.tolist(), pygrgl.TraversalDirection.UP)
        with pytest.raises(TypeError, match="numpy.ndarray"):
            grg.matmul(x_up, pygrgl.TraversalDirection.UP, miss=miss_up.tolist())
        with pytest.raises(TypeError, match="numpy.ndarray"):
            grg.matmul(x_up, pygrgl.TraversalDirection.UP, init=init_vec.tolist())


def test_runtime_requirements_reject_undeclared_modes(primary_artifact):
    requirements = full_requirements(
        max_k_up=1,
        max_k_down=1,
        need_down_miss_input=False,
        need_up_miss_output=False,
        need_init_vector=False,
        need_init_matrix=False,
        need_init_xtx=False,
    )
    with _open_reference_grg(primary_artifact, requirements=requirements) as grg:
        up = np.ones((1, grg.num_samples), dtype=DATA_DTYPE)
        down = np.ones((1, grg.num_mutations), dtype=DATA_DTYPE)
        miss = np.zeros((1, grg.num_mutations), dtype=DATA_DTYPE)
        with pytest.raises(ValueError, match="UP miss output"):
            grg.matmul(up, "up", miss=miss)
        with pytest.raises(ValueError, match="DOWN miss input"):
            grg.matmul(down, "down", miss=miss)
        with pytest.raises(ValueError, match="init vector"):
            grg.matmul(up, "up", init=np.ones((1,), dtype=DATA_DTYPE))
        with pytest.raises(ValueError, match="init matrix"):
            grg.matmul(up, "up", init=np.ones((1, grg.num_nodes), dtype=DATA_DTYPE))
        if grg.coalescence_counts is not None:
            with pytest.raises(ValueError, match="xtx"):
                grg.matmul(up, "up", init="xtx")

