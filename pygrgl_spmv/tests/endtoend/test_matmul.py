from __future__ import annotations

from contextlib import contextmanager
import subprocess

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv import ReferenceRuntime
from pygrgl_spmv.tests.runtime._runtime_builders import (
    build_layout_for_backend,
    build_reference_layout,
    full_requirements,
    nonreference_backend_cases,
    runtime_cls_for_backend,
)

from .conftest import grg_to_matrix


@contextmanager
def _open_reference_grg(artifact, *, max_k: int):
    with ReferenceRuntime(build_reference_layout([artifact], requirements=full_requirements(max_k_up=max_k, max_k_down=max_k))) as runtime:
        yield runtime.grgs[0]


@contextmanager
def _open_backend_grg(backend_name: str, artifact, *, k: int):
    requirements = full_requirements(max_k_up=k, max_k_down=k)
    layout = build_layout_for_backend(backend_name, [artifact], requirements=requirements)
    runtime_cls = runtime_cls_for_backend(backend_name)
    with runtime_cls(layout) as runtime:
        yield runtime.grgs[0]


def _direction_helper(grg: pygrgl.GRG, op, direction: pygrgl.TraversalDirection, *, rows: int):
    size = grg.num_samples if direction == pygrgl.TraversalDirection.UP else grg.num_mutations
    rng = np.random.default_rng(42 if direction == pygrgl.TraversalDirection.UP else 43)
    mat = rng.random((rows, size), dtype=np.float64)
    dot_result = np.array([pygrgl.dot_product(grg, mat[i], direction) for i in range(rows)])
    genotype_matrix = grg_to_matrix(grg, diploid=False)
    np_result = mat @ genotype_matrix if direction == pygrgl.TraversalDirection.UP else mat @ genotype_matrix.T
    op_result = op.matmul(mat, direction)
    np.testing.assert_allclose(dot_result, np_result)
    np.testing.assert_allclose(op_result, np_result)


def test_different_methods_reference(primary_grg, primary_artifact):
    with _open_reference_grg(primary_artifact, max_k=40) as grg:
        _direction_helper(primary_grg, grg, pygrgl.TraversalDirection.UP, rows=40)
        _direction_helper(primary_grg, grg, pygrgl.TraversalDirection.DOWN, rows=40)


@pytest.mark.parametrize("backend_name", nonreference_backend_cases())
def test_different_methods_nonreference(primary_grg, primary_artifact, backend_name):
    rows = 8
    with _open_backend_grg(backend_name, primary_artifact, k=rows) as grg:
        _direction_helper(primary_grg, grg, pygrgl.TraversalDirection.UP, rows=rows)
        _direction_helper(primary_grg, grg, pygrgl.TraversalDirection.DOWN, rows=rows)


def test_diploid_reference(primary_grg, primary_artifact):
    with _open_reference_grg(primary_artifact, max_k=12) as grg:
        rows = 12
        rng = np.random.default_rng(100)
        genotype_matrix = grg_to_matrix(primary_grg, diploid=True)
        rand_matrix = rng.random((rows, primary_grg.num_individuals), dtype=np.float64)
        np_result = rand_matrix @ genotype_matrix
        ref_result = np.asarray(pygrgl.matmul(primary_grg, rand_matrix, pygrgl.TraversalDirection.UP, by_individual=True))
        op_result = grg.matmul(rand_matrix, pygrgl.TraversalDirection.UP, by_individual=True)
        np.testing.assert_allclose(ref_result, np_result)
        np.testing.assert_allclose(op_result, np_result)

        genotype_t = genotype_matrix.T
        rand_matrix = rng.random((rows, primary_grg.num_mutations), dtype=np.float64)
        np_result = rand_matrix @ genotype_t
        ref_result = np.asarray(pygrgl.matmul(primary_grg, rand_matrix, pygrgl.TraversalDirection.DOWN, by_individual=True))
        op_result = grg.matmul(rand_matrix, pygrgl.TraversalDirection.DOWN, by_individual=True)
        np.testing.assert_allclose(ref_result, np_result)
        np.testing.assert_allclose(op_result, np_result)


@pytest.mark.parametrize("backend_name", nonreference_backend_cases())
def test_diploid_nonreference(primary_grg, primary_artifact, backend_name):
    rows = 6
    with _open_backend_grg(backend_name, primary_artifact, k=rows) as grg:
        rng = np.random.default_rng(101)
        genotype_matrix = grg_to_matrix(primary_grg, diploid=True)
        rand_matrix = rng.random((rows, primary_grg.num_individuals), dtype=np.float64)
        np_result = rand_matrix @ genotype_matrix
        op_result = grg.matmul(rand_matrix, pygrgl.TraversalDirection.UP, by_individual=True)
        np.testing.assert_allclose(op_result, np_result)


def test_xtx_init(primary_grg, primary_artifact):
    with _open_reference_grg(primary_artifact, max_k=2) as grg:
        node_xx_count = [0 for _ in range(primary_grg.num_nodes)]
        assert primary_grg.ploidy == 2
        for node_id in range(primary_grg.num_nodes):
            curr_coals = primary_grg.get_num_individual_coals(node_id)
            assert curr_coals != pygrgl.COAL_COUNT_NOT_SET
            coal_modifier = 2 * curr_coals
            if primary_grg.is_sample(node_id):
                node_xx_count[node_id] = 1
            else:
                child_sum = sum(node_xx_count[child] for child in primary_grg.get_down_edges(node_id))
                node_xx_count[node_id] = child_sum + coal_modifier
        X = np.ones((2, primary_grg.num_samples), dtype=np.int32)
        ref = np.asarray(pygrgl.matmul(primary_grg, X, pygrgl.TraversalDirection.UP, init="xtx"))
        got = grg.matmul(X, pygrgl.TraversalDirection.UP, init="xtx")
        np.testing.assert_allclose(got, ref)
        for mut_id, node_id in primary_grg.get_mutation_node_pairs():
            if node_id == pygrgl.INVALID_NODE:
                continue
            assert ref[0, mut_id] == node_xx_count[node_id]
            assert ref[1, mut_id] == node_xx_count[node_id]


def test_vector_init(primary_grg, primary_artifact):
    with _open_reference_grg(primary_artifact, max_k=10) as grg:
        rows = 10
        init = np.ones(rows, dtype=np.float64) * 2.0
        X = np.ones((rows, primary_grg.num_samples), dtype=np.float64)
        ref_without = np.asarray(pygrgl.matmul(primary_grg, X, pygrgl.TraversalDirection.UP))
        ref_with = np.asarray(pygrgl.matmul(primary_grg, X, pygrgl.TraversalDirection.UP, init=init))
        got_without = grg.matmul(X, pygrgl.TraversalDirection.UP)
        got_with = grg.matmul(X, pygrgl.TraversalDirection.UP, init=init)
        np.testing.assert_allclose(got_without, ref_without)
        np.testing.assert_allclose(got_with, ref_with)
        assert np.all(got_with >= 2 * got_without)


def test_matrix_init(primary_grg, primary_artifact):
    with _open_reference_grg(primary_artifact, max_k=10) as grg:
        rows = 10
        X = np.ones((rows, primary_grg.num_samples), dtype=np.int64)
        init = np.zeros((rows, primary_grg.num_nodes), dtype=np.int64)
        ref_without = np.asarray(pygrgl.matmul(primary_grg, X, pygrgl.TraversalDirection.UP))
        ref_with = np.asarray(pygrgl.matmul(primary_grg, X, pygrgl.TraversalDirection.UP, init=init))
        got_without = grg.matmul(X, pygrgl.TraversalDirection.UP)
        got_with = grg.matmul(X, pygrgl.TraversalDirection.UP, init=init)
        np.testing.assert_allclose(got_without, ref_without)
        np.testing.assert_allclose(got_with, ref_with)
        np.testing.assert_array_equal(ref_without, ref_with)

        tweak_id = np.random.default_rng(123).integers(primary_grg.num_samples, primary_grg.num_nodes)
        collected_mut_ids = []

        def collect(start_id: int):
            muts = primary_grg.get_mutations_for_node(start_id)
            collected_mut_ids.extend(muts)
            for parent_id in primary_grg.get_up_edges(start_id):
                collect(parent_id)

        collect(int(tweak_id))
        init[4, int(tweak_id)] = 100
        ref_tweaked = np.asarray(pygrgl.matmul(primary_grg, X, pygrgl.TraversalDirection.UP, init=init))
        got_tweaked = grg.matmul(X, pygrgl.TraversalDirection.UP, init=init)
        np.testing.assert_allclose(got_tweaked, ref_tweaked)
        for i in range(got_tweaked.shape[0]):
            for j in range(got_tweaked.shape[1]):
                if i != 4 or j not in collected_mut_ids:
                    assert got_tweaked[i, j] == pytest.approx(got_without[i, j])
                else:
                    assert got_tweaked[i, j] > got_without[i, j]


def test_init_failures(primary_grg_path, primary_artifact):
    grg_ref = pygrgl.load_immutable_grg(primary_grg_path, load_up_edges=True)
    with _open_reference_grg(primary_artifact, max_k=10) as grg:
        with pytest.raises((ValueError, TypeError)):
            grg.matmul(np.ones((1, grg_ref.num_samples), dtype=np.float64), pygrgl.TraversalDirection.UP, init=np.ones((1, 200000), dtype=np.float64))
        with pytest.raises((ValueError, TypeError)):
            grg.matmul(np.ones((1, grg_ref.num_samples), dtype=np.float64), pygrgl.TraversalDirection.UP, init=np.ones((2, grg_ref.num_samples), dtype=np.float64))
        with pytest.raises((ValueError, TypeError)):
            grg.matmul(np.ones((1, grg_ref.num_samples), dtype=np.int32), pygrgl.TraversalDirection.UP, init=np.ones((1, grg_ref.num_samples), dtype=np.float64))
        with pytest.raises((ValueError, TypeError)):
            grg.matmul(np.ones((1, grg_ref.num_samples), dtype=np.int32), pygrgl.TraversalDirection.UP, init="test")
        with pytest.raises((ValueError, TypeError)):
            grg.matmul(np.ones((9, grg_ref.num_samples), dtype=np.float64), pygrgl.TraversalDirection.UP, init=np.ones(10, dtype=np.float64))


def test_split_consistency(primary_grg, primary_grg_path, primary_artifact, tmp_path):
    with _open_reference_grg(primary_artifact, max_k=4) as grg:
        rows = 4
        rng = np.random.default_rng(777)
        in_matrix = rng.standard_normal((rows, primary_grg.num_mutations), dtype=np.float64)
        full_result = grg.matmul(in_matrix, pygrgl.TraversalDirection.DOWN)

    split_dir = tmp_path / "split"
    subprocess.check_call(
        [
            "grg",
            "split",
            "-j",
            "4",
            str(primary_grg_path),
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
    assert sum(info[2] for info in part_infos) == primary_grg.num_mutations

    split_result = None
    start = 0
    for _bp_start, fn, num_muts in part_infos:
        end = start + num_muts
        sub_matrix = in_matrix[:, start:end]
        part_artifact = fn.with_suffix(".grg_spmv")
        from pygrgl_spmv import convert

        convert(str(fn), part_artifact.parent, name=part_artifact.stem)
        with _open_reference_grg(part_artifact.parent / f"{part_artifact.stem}.grg_spmv", max_k=4) as part_grg:
            part_out = part_grg.matmul(sub_matrix, pygrgl.TraversalDirection.DOWN)
        split_result = part_out.copy() if split_result is None else (split_result + part_out)
        start = end

    assert start == in_matrix.shape[1]
    np.testing.assert_allclose(full_result, split_result)
