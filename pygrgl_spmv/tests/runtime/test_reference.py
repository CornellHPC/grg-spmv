from __future__ import annotations

import numpy as np
import pygrgl

from pygrgl_spmv import ReferenceRuntime
from pygrgl_spmv.backends.base import sparse_structure_nbytes, stored_block_shape
from pygrgl_spmv.tests.conftest import DATA_DTYPE, tol
from pygrgl_spmv.tests.runtime._runtime_builders import build_reference_layout, full_requirements


def test_reference_runtime_matches_pygrgl_up_down(primary_artifact, primary_grg):
    layout = build_reference_layout([primary_artifact], requirements=full_requirements(max_k_up=4, max_k_down=4))
    with ReferenceRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(42)
        x_up = rng.standard_normal((3, primary_grg.num_samples), dtype=np.float64)
        x_down = rng.standard_normal((3, primary_grg.num_mutations), dtype=np.float64)
        atol, rtol = tol(np.float64)
        np.testing.assert_allclose(
            grg.matmul(x_up, pygrgl.TraversalDirection.UP),
            np.asarray(pygrgl.matmul(primary_grg, x_up, pygrgl.TraversalDirection.UP)),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            grg.matmul(x_down, pygrgl.TraversalDirection.DOWN),
            np.asarray(pygrgl.matmul(primary_grg, x_down, pygrgl.TraversalDirection.DOWN)),
            atol=atol,
            rtol=rtol,
        )


def test_reference_layout_bytes_are_exact(primary_artifact):
    requirements = full_requirements(max_k_up=8, max_k_down=6)
    layout = build_reference_layout([primary_artifact], requirements=requirements)
    artifact_layout = layout.artifacts[0]
    expected_sparse = 0
    for block in artifact_layout.blocks_up:
        nrows, ncols = stored_block_shape(block.stored_shape[0], block.stored_shape[1], store=layout.pair.plan_up.store)
        expected_sparse += int(sparse_structure_nbytes(layout.pair.plan_up.fmt, nrows=nrows, ncols=ncols, nnz=block.nnz) + block.nnz)
    state_layout = build_reference_layout([primary_artifact], requirements=requirements)
    with ReferenceRuntime(state_layout) as runtime:
        state = runtime._artifacts[0].state
        selector_bytes = int(state.sel_mut.indices.nbytes + state.sel_mut.indptr.nbytes + state.sel_mut.data.nbytes)
        selector_bytes += int(state.sel_miss.indices.nbytes + state.sel_miss.indptr.nbytes + state.sel_miss.data.nbytes)
        expected = {
            "resident_sparse": expected_sparse,
            "selectors": selector_bytes,
            "workspace_up": int(state.num_nodes * int(requirements.max_k_up) * np.dtype(DATA_DTYPE).itemsize),
            "workspace_down": int(state.num_nodes * int(requirements.max_k_down) * np.dtype(DATA_DTYPE).itemsize),
        }
        assert state_layout.bytes_by_category == expected
        assert state_layout.bytes_total == int(sum(expected.values()))
        assert sum(item.nbytes for item in state_layout.budget_items) == state_layout.bytes_total
        assert state_layout.required_budget_for_full_residency == state_layout.bytes_total
