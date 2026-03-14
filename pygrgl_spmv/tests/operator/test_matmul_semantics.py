"""Public matmul option semantics: by_individual, init, miss, dtype rules."""

from __future__ import annotations

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv.tests.conftest import DATA_DTYPE, matmul_expect_k_hint_warning, tol


@pytest.mark.smoke
def test_init_modes_up_down_smoke(op, grg_ref):
    rng = np.random.default_rng(2202)
    rows = 3
    x_up = rng.standard_normal((rows, grg_ref.num_samples), dtype=DATA_DTYPE)
    x_down = rng.standard_normal((rows, grg_ref.num_mutations), dtype=DATA_DTYPE)
    init_vec = rng.standard_normal(rows, dtype=DATA_DTYPE)

    atol, rtol = tol(DATA_DTYPE)
    expected_up = pygrgl.matmul(grg_ref, x_up, pygrgl.TraversalDirection.UP, init=init_vec)
    expected_down = pygrgl.matmul(grg_ref, x_down, pygrgl.TraversalDirection.DOWN, init=init_vec)
    np.testing.assert_allclose(
        matmul_expect_k_hint_warning(op, x_up, pygrgl.TraversalDirection.UP, init=init_vec),
        expected_up,
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        matmul_expect_k_hint_warning(op, x_down, pygrgl.TraversalDirection.DOWN, init=init_vec),
        expected_down,
        atol=atol,
        rtol=rtol,
    )


def test_by_individual(op, grg_ref):
    if grg_ref.num_individuals == grg_ref.num_samples:
        pytest.skip("Dataset is not grouped by individuals")

    rng = np.random.default_rng(1101)
    rows = 5
    x_up = rng.standard_normal((rows, grg_ref.num_individuals), dtype=DATA_DTYPE)
    x_down = rng.standard_normal((rows, grg_ref.num_mutations), dtype=DATA_DTYPE)

    expected_up = pygrgl.matmul(grg_ref, x_up, pygrgl.TraversalDirection.UP, by_individual=True)
    expected_down = pygrgl.matmul(grg_ref, x_down, pygrgl.TraversalDirection.DOWN, by_individual=True)
    actual_up = matmul_expect_k_hint_warning(op, x_up, pygrgl.TraversalDirection.UP, by_individual=True)
    actual_down = matmul_expect_k_hint_warning(op, x_down, pygrgl.TraversalDirection.DOWN, by_individual=True)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(actual_up, expected_up, atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual_down, expected_down, atol=atol, rtol=rtol)


def test_init_modes_both_directions(op, grg_ref):
    rng = np.random.default_rng(2203)
    rows = 4
    x_up = rng.standard_normal((rows, grg_ref.num_samples), dtype=DATA_DTYPE)
    x_down = rng.standard_normal((rows, grg_ref.num_mutations), dtype=DATA_DTYPE)
    init_vec = rng.standard_normal(rows, dtype=DATA_DTYPE)
    init_mat = rng.standard_normal((rows, grg_ref.num_nodes), dtype=DATA_DTYPE)

    init_modes = [init_vec, init_mat]
    if op.coalescence_counts is not None:
        init_modes.append("xtx")

    atol, rtol = tol(DATA_DTYPE)
    for init in init_modes:
        expected_up = pygrgl.matmul(grg_ref, x_up, pygrgl.TraversalDirection.UP, init=init)
        expected_down = pygrgl.matmul(grg_ref, x_down, pygrgl.TraversalDirection.DOWN, init=init)
        np.testing.assert_allclose(
            matmul_expect_k_hint_warning(op, x_up, pygrgl.TraversalDirection.UP, init=init),
            expected_up,
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            matmul_expect_k_hint_warning(op, x_down, pygrgl.TraversalDirection.DOWN, init=init),
            expected_down,
            atol=atol,
            rtol=rtol,
        )


def test_missing_input_output(op_missing, missing_grg_ref):
    if not missing_grg_ref.has_missing_data:
        pytest.skip("Dataset has no missingness nodes")
    if missing_grg_ref.num_individuals == missing_grg_ref.num_samples:
        pytest.skip("Dataset is not grouped by individuals")

    rng = np.random.default_rng(3303)
    rows = 3

    x_up = rng.standard_normal((rows, missing_grg_ref.num_individuals), dtype=DATA_DTYPE)
    miss_expected = np.zeros((rows, missing_grg_ref.num_mutations), dtype=DATA_DTYPE)
    miss_actual = np.zeros((rows, missing_grg_ref.num_mutations), dtype=DATA_DTYPE)
    expected_up = pygrgl.matmul(
        missing_grg_ref,
        x_up,
        pygrgl.TraversalDirection.UP,
        by_individual=True,
        miss=miss_expected,
    )
    actual_up = matmul_expect_k_hint_warning(
        op_missing,
        x_up,
        pygrgl.TraversalDirection.UP,
        by_individual=True,
        miss=miss_actual,
    )

    x_down = rng.standard_normal((rows, missing_grg_ref.num_mutations), dtype=DATA_DTYPE)
    miss_down = rng.standard_normal((rows, missing_grg_ref.num_mutations), dtype=DATA_DTYPE)
    expected_down = pygrgl.matmul(
        missing_grg_ref,
        x_down,
        pygrgl.TraversalDirection.DOWN,
        by_individual=True,
        miss=miss_down.copy(),
    )
    actual_down = matmul_expect_k_hint_warning(
        op_missing,
        x_down,
        pygrgl.TraversalDirection.DOWN,
        by_individual=True,
        miss=miss_down.copy(),
    )

    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(actual_up, expected_up, atol=atol, rtol=rtol)
    np.testing.assert_allclose(miss_actual, miss_expected, atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual_down, expected_down, atol=atol, rtol=rtol)


def test_bool_init_dtype_strict_behavior(op):
    rng = np.random.default_rng(5505)
    rows = 4

    up_bool = rng.integers(0, 2, size=(rows, op.num_samples), dtype=np.int8).astype(bool)
    down_bool = rng.integers(0, 2, size=(rows, op.num_mutations), dtype=np.int8).astype(bool)
    init_vec_bool = rng.integers(0, 2, size=(rows,), dtype=np.int8).astype(bool)
    init_mat_bool = rng.integers(0, 2, size=(rows, op.num_nodes), dtype=np.int8).astype(bool)

    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(
        matmul_expect_k_hint_warning(op, up_bool, pygrgl.TraversalDirection.UP, init=init_vec_bool),
        matmul_expect_k_hint_warning(
            op,
            up_bool.astype(DATA_DTYPE),
            pygrgl.TraversalDirection.UP,
            init=init_vec_bool.astype(DATA_DTYPE),
        ),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        matmul_expect_k_hint_warning(op, up_bool, pygrgl.TraversalDirection.UP, init=init_mat_bool),
        matmul_expect_k_hint_warning(
            op,
            up_bool.astype(DATA_DTYPE),
            pygrgl.TraversalDirection.UP,
            init=init_mat_bool.astype(DATA_DTYPE),
        ),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        matmul_expect_k_hint_warning(op, down_bool, pygrgl.TraversalDirection.DOWN, init=init_vec_bool),
        matmul_expect_k_hint_warning(
            op,
            down_bool.astype(DATA_DTYPE),
            pygrgl.TraversalDirection.DOWN,
            init=init_vec_bool.astype(DATA_DTYPE),
        ),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        matmul_expect_k_hint_warning(op, down_bool, pygrgl.TraversalDirection.DOWN, init=init_mat_bool),
        matmul_expect_k_hint_warning(
            op,
            down_bool.astype(DATA_DTYPE),
            pygrgl.TraversalDirection.DOWN,
            init=init_mat_bool.astype(DATA_DTYPE),
        ),
        atol=atol,
        rtol=rtol,
    )

    with pytest.raises(TypeError):
        matmul_expect_k_hint_warning(op, up_bool.astype(DATA_DTYPE), pygrgl.TraversalDirection.UP, init=init_vec_bool)
