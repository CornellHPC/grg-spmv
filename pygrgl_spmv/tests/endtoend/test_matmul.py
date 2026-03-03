"""End-to-end matrix-multiplication tests for immutable GRGs."""

from __future__ import annotations

import subprocess

import numpy as np
import pygrgl
import pytest

from .conftest import grg_to_matrix


def _direction_helper(grg: pygrgl.GRG, op, direction: pygrgl.TraversalDirection):
    rows = 40
    size = grg.num_samples if direction == pygrgl.TraversalDirection.UP else grg.num_mutations
    rng = np.random.default_rng(42 if direction == pygrgl.TraversalDirection.UP else 43)
    mat = rng.random((rows, size), dtype=np.float64)

    # Result 1: independent dot_product calls to GRG.
    dot_result = np.array([pygrgl.dot_product(grg, mat[i], direction) for i in range(rows)])

    # Result 2: explicit numpy matrix multiplication.
    genotype_matrix = grg_to_matrix(grg, diploid=False)
    if direction == pygrgl.TraversalDirection.UP:
        np_result = mat @ genotype_matrix
    else:
        np_result = mat @ genotype_matrix.T

    # Result 3: backend operator path.
    op_result = op.matmul(mat, direction)

    np.testing.assert_allclose(dot_result, np_result)
    np.testing.assert_allclose(op_result, np_result)


@pytest.mark.smoke
def test_different_methods(backend_config, basic_grg, basic_grg_path, make_operator):
    op = make_operator(basic_grg_path)
    _direction_helper(basic_grg, op, pygrgl.TraversalDirection.UP)
    _direction_helper(basic_grg, op, pygrgl.TraversalDirection.DOWN)


@pytest.mark.smoke
def test_diploid(backend_config, basic_grg, basic_grg_path, make_operator):
    op = make_operator(basic_grg_path)
    rows = 12
    rng = np.random.default_rng(100)

    # (rows x N_indiv) @ (N_indiv x M)
    genotype_matrix = grg_to_matrix(basic_grg, diploid=True)
    rand_matrix = rng.random((rows, basic_grg.num_individuals), dtype=np.float64)
    np_result = rand_matrix @ genotype_matrix
    ref_result = pygrgl.matmul(
        basic_grg, rand_matrix, pygrgl.TraversalDirection.UP, by_individual=True
    )
    op_result = op.matmul(rand_matrix, pygrgl.TraversalDirection.UP, by_individual=True)
    np.testing.assert_allclose(ref_result, np_result)
    np.testing.assert_allclose(op_result, np_result)

    # (rows x M) @ (M x N_indiv)
    genotype_t = genotype_matrix.T
    rand_matrix = rng.random((rows, basic_grg.num_mutations), dtype=np.float64)
    np_result = rand_matrix @ genotype_t
    ref_result = pygrgl.matmul(
        basic_grg, rand_matrix, pygrgl.TraversalDirection.DOWN, by_individual=True
    )
    op_result = op.matmul(rand_matrix, pygrgl.TraversalDirection.DOWN, by_individual=True)
    np.testing.assert_allclose(ref_result, np_result)
    np.testing.assert_allclose(op_result, np_result)


def test_xtx_init(backend_config, basic_grg, basic_grg_path, make_operator):
    op = make_operator(basic_grg_path)
    node_xx_count = [0 for _ in range(basic_grg.num_nodes)]
    assert basic_grg.ploidy == 2
    for node_id in range(basic_grg.num_nodes):
        curr_coals = basic_grg.get_num_individual_coals(node_id)
        assert curr_coals != pygrgl.COAL_COUNT_NOT_SET
        coal_modifier = 2 * curr_coals
        if basic_grg.is_sample(node_id):
            node_xx_count[node_id] = 1
        else:
            child_sum = sum(node_xx_count[child] for child in basic_grg.get_down_edges(node_id))
            node_xx_count[node_id] = child_sum + coal_modifier

    X = np.ones((2, basic_grg.num_samples), dtype=np.int32)
    ref = pygrgl.matmul(
        basic_grg,
        X,
        pygrgl.TraversalDirection.UP,
        init="xtx",
    )
    got = op.matmul(X, pygrgl.TraversalDirection.UP, init="xtx")
    np.testing.assert_allclose(got, ref)

    for mut_id, node_id in basic_grg.get_mutation_node_pairs():
        if node_id == pygrgl.INVALID_NODE:
            continue
        assert ref[0, mut_id] == node_xx_count[node_id]
        assert ref[1, mut_id] == node_xx_count[node_id]


def test_vector_init(backend_config, basic_grg, basic_grg_path, make_operator):
    op = make_operator(basic_grg_path)
    rows = 10
    init = np.ones(rows, dtype=np.float64) * 2.0
    X = np.ones((rows, basic_grg.num_samples), dtype=np.float64)

    ref_without = pygrgl.matmul(basic_grg, X, pygrgl.TraversalDirection.UP)
    ref_with = pygrgl.matmul(basic_grg, X, pygrgl.TraversalDirection.UP, init=init)
    got_without = op.matmul(X, pygrgl.TraversalDirection.UP)
    got_with = op.matmul(X, pygrgl.TraversalDirection.UP, init=init)

    assert got_without.shape == got_with.shape
    np.testing.assert_allclose(got_without, ref_without)
    np.testing.assert_allclose(got_with, ref_with)
    assert np.all(got_with >= 2 * got_without)


def test_matrix_init(backend_config, basic_grg, basic_grg_path, make_operator):
    op = make_operator(basic_grg_path)
    rows = 10
    X = np.ones((rows, basic_grg.num_samples), dtype=np.int64)
    init = np.zeros((rows, basic_grg.num_nodes), dtype=np.int64)

    ref_without = pygrgl.matmul(basic_grg, X, pygrgl.TraversalDirection.UP)
    ref_with = pygrgl.matmul(basic_grg, X, pygrgl.TraversalDirection.UP, init=init)
    got_without = op.matmul(X, pygrgl.TraversalDirection.UP)
    got_with = op.matmul(X, pygrgl.TraversalDirection.UP, init=init)

    np.testing.assert_allclose(got_without, ref_without)
    np.testing.assert_allclose(got_with, ref_with)
    np.testing.assert_array_equal(ref_without, ref_with)

    tweak_id = np.random.default_rng(123).integers(basic_grg.num_samples, basic_grg.num_nodes)
    collected_mut_ids = []

    def collect(start_id: int):
        muts = basic_grg.get_mutations_for_node(start_id)
        collected_mut_ids.extend(muts)
        for parent_id in basic_grg.get_up_edges(start_id):
            collect(parent_id)

    collect(int(tweak_id))
    init[4, int(tweak_id)] = 100

    ref_tweaked = pygrgl.matmul(
        basic_grg, X, pygrgl.TraversalDirection.UP, init=init
    )
    got_tweaked = op.matmul(X, pygrgl.TraversalDirection.UP, init=init)
    np.testing.assert_allclose(got_tweaked, ref_tweaked)

    for i in range(got_tweaked.shape[0]):
        for j in range(got_tweaked.shape[1]):
            if i != 4:
                assert got_tweaked[i, j] == pytest.approx(got_without[i, j])
            elif j not in collected_mut_ids:
                assert got_tweaked[i, j] == pytest.approx(got_without[i, j])
            else:
                assert got_tweaked[i, j] > got_without[i, j]


def test_init_failures(backend_config, basic_grg_path, make_operator):
    op = make_operator(basic_grg_path)
    grg = pygrgl.load_immutable_grg(str(basic_grg_path), load_up_edges=True)

    with pytest.raises((ValueError, TypeError)):
        op.matmul(
            np.ones((1, grg.num_samples), dtype=np.float64),
            pygrgl.TraversalDirection.UP,
            init=np.ones((1, 200000), dtype=np.float64),
        )
    with pytest.raises((ValueError, TypeError)):
        op.matmul(
            np.ones((1, grg.num_samples), dtype=np.float64),
            pygrgl.TraversalDirection.UP,
            init=np.ones((2, grg.num_samples), dtype=np.float64),
        )
    with pytest.raises((ValueError, TypeError)):
        op.matmul(
            np.ones((1, grg.num_samples), dtype=np.int32),
            pygrgl.TraversalDirection.UP,
            init=np.ones((1, grg.num_samples), dtype=np.float64),
        )
    with pytest.raises((ValueError, TypeError)):
        op.matmul(
            np.ones((1, grg.num_samples), dtype=np.int32),
            pygrgl.TraversalDirection.UP,
            init="test",
        )
    with pytest.raises((ValueError, TypeError)):
        op.matmul(
            np.ones((9, grg.num_samples), dtype=np.float64),
            pygrgl.TraversalDirection.UP,
            init=np.ones(10, dtype=np.float64),
        )


def test_split_consistency(backend_config, basic_grg, basic_grg_path, make_operator, tmp_path):
    """
    Splitting a GRG and summing per-piece results matches full multiplication.
    """
    op_full = make_operator(basic_grg_path)
    rows = 4
    rng = np.random.default_rng(777)
    in_matrix = rng.standard_normal((rows, basic_grg.num_mutations), dtype=np.float64)
    full_result = op_full.matmul(in_matrix, pygrgl.TraversalDirection.DOWN)

    split_dir = tmp_path / "split"
    subprocess.check_call(
        [
            "grg",
            "split",
            "-j",
            "4",
            str(basic_grg_path),
            "-s",
            "1000000",
            "-o",
            str(split_dir),
        ]
    )

    part_infos = []
    for fn in split_dir.glob("*.grg"):
        grg_part = pygrgl.load_immutable_grg(str(fn))
        part_infos.append((grg_part.bp_range[0], fn, grg_part.num_mutations))
    part_infos.sort(key=lambda x: x[0])

    total_muts = sum(info[2] for info in part_infos)
    assert total_muts == basic_grg.num_mutations

    split_result = None
    start = 0
    for _, fn, num_muts in part_infos:
        end = start + num_muts
        sub_matrix = in_matrix[:, start:end]
        op_part = make_operator(fn)
        part_out = op_part.matmul(sub_matrix, pygrgl.TraversalDirection.DOWN)
        split_result = part_out.copy() if split_result is None else (split_result + part_out)
        start = end
    assert start == in_matrix.shape[1]

    np.testing.assert_allclose(full_result, split_result)
