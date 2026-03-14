"""End-to-end tests for matmul missingness semantics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pygrgl
import pytest

from .conftest import allele_frequencies, grg_to_matrix, samples_below_node
from pygrgl_spmv.tests.conftest import matmul_expect_k_hint_warning


MISSING_INDIVS = 21
MISSING_SAMPLES = 25


@pytest.mark.smoke
def test_missing_nodes_and_counts(backend_config, missing_grg):
    total_miss_idv = 0
    total_miss = 0
    for mut_id, node, miss in missing_grg.get_mutation_node_miss():
        if node != pygrgl.INVALID_NODE and miss != pygrgl.INVALID_NODE:
            rs = samples_below_node(missing_grg, node)
            ms = samples_below_node(missing_grg, miss)
            assert len(set(rs)) == len(rs)
            assert len(set(ms)) == len(ms)
            total_miss += len(ms)
            total_miss_idv += len(set(sample // 2 for sample in ms))
    assert total_miss == MISSING_SAMPLES
    assert total_miss_idv == MISSING_INDIVS


@pytest.mark.smoke
def test_missing_matmul_semantics(backend_config, missing_grg, missing_grg_path, make_operator):
    op = make_operator(missing_grg_path)
    X = grg_to_matrix(missing_grg, diploid=True)
    # Only non-0/1/2 entries should correspond to missingness mean-imputation.
    nonstandard = np.where((X > 0) & (X != 1) & (X != 2))[0]
    assert len(nonstandard) == MISSING_INDIVS

    rows = 7
    rng = np.random.default_rng(303)
    freqs = allele_frequencies(missing_grg)

    # UP direction: miss is output.
    rv_up = rng.standard_normal((rows, missing_grg.num_individuals), dtype=np.float64)
    numpy_up = rv_up @ X
    miss_ref = np.zeros((rows, missing_grg.num_mutations), dtype=np.float64)
    ref_up = pygrgl.matmul(
        missing_grg,
        rv_up,
        pygrgl.TraversalDirection.UP,
        by_individual=True,
        miss=miss_ref,
    )
    miss_op = np.zeros((rows, missing_grg.num_mutations), dtype=np.float64)
    got_up = matmul_expect_k_hint_warning(
        op,
        rv_up,
        pygrgl.TraversalDirection.UP,
        by_individual=True,
        miss=miss_op,
    )

    # mean-impute equivalence: explicit = known + miss * freq
    np.testing.assert_allclose(numpy_up, ref_up + (miss_ref * freqs))
    np.testing.assert_allclose(got_up, ref_up)
    np.testing.assert_allclose(miss_op, miss_ref)

    # DOWN direction: miss is input.
    rv_down = rng.standard_normal((rows, missing_grg.num_mutations), dtype=np.float64)
    numpy_down = rv_down @ X.T
    miss_in = np.array([freqs]) * rv_down
    ref_down = pygrgl.matmul(
        missing_grg,
        rv_down,
        pygrgl.TraversalDirection.DOWN,
        by_individual=True,
        miss=miss_in.copy(),
    )
    got_down = matmul_expect_k_hint_warning(
        op,
        rv_down,
        pygrgl.TraversalDirection.DOWN,
        by_individual=True,
        miss=miss_in.copy(),
    )

    np.testing.assert_allclose(numpy_down, ref_down)
    np.testing.assert_allclose(got_down, ref_down)


def test_shared_site_missingness_affects_all_variants(
    backend_config, missing_grg, missing_grg_path, make_operator
):
    """
    For sites with multiple variants and a shared missingness node,
    missing-output columns should match for those variants.
    """
    op = make_operator(missing_grg_path)
    mut_info = {
        mut_id: missing_grg.get_mutation_by_id(mut_id)
        for mut_id in range(missing_grg.num_mutations)
    }
    groups = defaultdict(list)
    for mut_id, node, miss_node in missing_grg.get_mutation_node_miss():
        if miss_node == pygrgl.INVALID_NODE:
            continue
        key = (mut_info[mut_id].position, miss_node)
        groups[key].append(mut_id)

    shared_groups = [mut_ids for mut_ids in groups.values() if len(mut_ids) > 1]
    if not shared_groups:
        pytest.skip("No shared (position, missingness-node) groups in this fixture")

    rng = np.random.default_rng(404)
    rows = 3
    rv = rng.standard_normal((rows, missing_grg.num_individuals), dtype=np.float64)
    miss_ref = np.zeros((rows, missing_grg.num_mutations), dtype=np.float64)
    _ = pygrgl.matmul(
        missing_grg,
        rv,
        pygrgl.TraversalDirection.UP,
        by_individual=True,
        miss=miss_ref,
    )
    miss_op = np.zeros((rows, missing_grg.num_mutations), dtype=np.float64)
    _ = matmul_expect_k_hint_warning(
        op,
        rv,
        pygrgl.TraversalDirection.UP,
        by_individual=True,
        miss=miss_op,
    )
    np.testing.assert_allclose(miss_op, miss_ref)

    for mut_ids in shared_groups:
        baseline = miss_op[:, mut_ids[0]]
        for mut_id in mut_ids[1:]:
            np.testing.assert_allclose(miss_op[:, mut_id], baseline)
