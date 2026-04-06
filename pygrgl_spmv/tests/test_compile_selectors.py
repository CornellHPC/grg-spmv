"""Focused tests for GRG selector compilation edge cases."""

from __future__ import annotations

import pygrgl_spmv.grg.compile as compile_module
import numpy as np
import pygrgl
import pytest

from pygrgl_spmv.grg.compile import _build_level_blocks, _build_selectors, compile_grg


class _BlockStubGrg:
    def __init__(self, down_edges):
        self._down_edges = [tuple(children) for children in down_edges]
        self.num_edges = sum(len(children) for children in self._down_edges)

    def get_down_edges(self, node_id):
        return self._down_edges[int(node_id)]


class _MutationStub:
    def __init__(self, position: float, allele: str, ref_allele: str, time: float):
        self.position = position
        self.allele = allele
        self.ref_allele = ref_allele
        self.time = time


class _CompileOrderStubGrg:
    num_samples = 1
    num_mutations = 1
    num_nodes = 2
    num_edges = 1
    ploidy = 1
    num_individuals = 1
    has_missing_data = False
    has_individual_coals = False

    def __init__(self, events):
        self._events = events

    def get_down_edges(self, node_id):
        return () if int(node_id) == 0 else (0,)

    def get_mutation_node_miss(self):
        self._events.append("mutation_rows")
        return [(0, 1, int(pygrgl.INVALID_NODE))]

    def get_num_individual_coals(self, _node_id):
        return int(pygrgl.COAL_COUNT_NOT_SET)

    def get_mutation_by_id(self, mutation_id):
        if int(mutation_id) != 0:
            raise IndexError(mutation_id)
        return _MutationStub(position=1.0, allele="A", ref_allele="C", time=0.0)


def _selectors(rows, *, num_mutations: int, num_nodes: int = 8):
    return _build_selectors(
        list(rows),
        inv_node_perm=np.arange(num_nodes, dtype=np.int32),
        num_mutations=num_mutations,
        num_nodes=num_nodes,
        node_id_scratch_dtype=np.int32,
    )


def test_build_selectors_accepts_repeated_mutation_rows_with_shared_missing_node():
    invalid = int(pygrgl.INVALID_NODE)
    sel_mut, sel_miss = _selectors(
        [
            (0, 3, 6),
            (0, 4, 6),
            (1, 5, invalid),
        ],
        num_mutations=2,
    )

    np.testing.assert_array_equal(
        sel_mut.toarray(),
        np.array(
            [
                [False, False, False, True, True, False, False, False],
                [False, False, False, False, False, True, False, False],
            ],
            dtype=np.bool_,
        ),
    )
    np.testing.assert_array_equal(
        sel_miss.toarray(),
        np.array(
            [
                [False, False, False, False, False, False, True, False],
                [False, False, False, False, False, False, False, False],
            ],
            dtype=np.bool_,
        ),
    )
    assert sel_mut.dtype == np.bool_
    assert sel_miss.dtype == np.bool_
    np.testing.assert_array_equal(sel_mut.data, np.ones(sel_mut.nnz, dtype=np.bool_))
    np.testing.assert_array_equal(sel_miss.data, np.ones(sel_miss.nnz, dtype=np.bool_))


def test_build_selectors_accepts_repeated_mutation_rows_with_distinct_missing_nodes():
    invalid = int(pygrgl.INVALID_NODE)
    _sel_mut, sel_miss = _selectors(
        [
            (0, 2, 6),
            (0, 3, 7),
            (1, 4, invalid),
        ],
        num_mutations=2,
    )

    np.testing.assert_array_equal(
        sel_miss.toarray(),
        np.array(
            [
                [False, False, False, False, False, False, True, True],
                [False, False, False, False, False, False, False, False],
            ],
            dtype=np.bool_,
        ),
    )
    assert sel_miss.dtype == np.bool_
    np.testing.assert_array_equal(sel_miss.data, np.ones(sel_miss.nnz, dtype=np.bool_))


def test_build_selectors_rejects_incomplete_mutation_rows():
    with pytest.raises(ValueError, match="incomplete"):
        _selectors([(0, 2, int(pygrgl.INVALID_NODE))], num_mutations=2)


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([(0, 2, -1), (2, 3, -1), (2, 4, -1)], "sorted by ascending mutation id"),
        ([(1, 2, -1), (0, 3, -1), (2, 4, -1)], "sorted by ascending mutation id"),
        ([(0, 2, -1), (1, 3, -1), (3, 4, -1)], "out of range"),
    ],
    ids=["skipped-id", "descending-id", "out-of-range-id"],
)
def test_build_selectors_rejects_invalid_mutation_row_order(rows, match):
    with pytest.raises(ValueError, match=match):
        _selectors(rows, num_mutations=3)


def test_build_level_blocks_finalizes_each_block_and_uses_int32_for_empty_blocks():
    grg = _BlockStubGrg(
        [
            (),
            (),
            (0,),
            (2,),
        ]
    )
    A_blocks = _build_level_blocks(
        grg,
        node_levels=np.array([0, 0, 1, 2], dtype=np.int32),
        level_offsets=np.array([0, 2, 3, 4], dtype=np.int32),
        inv_node_perm=np.array([0, 1, 2, 3], dtype=np.int32),
        node_id_scratch_dtype=np.dtype(np.int64),
    )

    block10 = A_blocks[1][0]
    block20 = A_blocks[2][0]
    block21 = A_blocks[2][1]

    assert block10.indices.dtype == np.int32
    assert block10.indptr.dtype == np.int32
    assert block20.indices.dtype == np.int32
    assert block20.indptr.dtype == np.int32
    assert block21.indices.dtype == np.int32
    assert block21.indptr.dtype == np.int32

    np.testing.assert_array_equal(block10.toarray(), np.array([[True, False]], dtype=np.bool_))
    np.testing.assert_array_equal(block20.toarray(), np.array([[False, False]], dtype=np.bool_))
    np.testing.assert_array_equal(block21.toarray(), np.array([[True]], dtype=np.bool_))


def test_compile_grg_loads_mutation_rows_only_after_blocks(monkeypatch):
    events: list[str] = []
    real_build_level_blocks = compile_module._build_level_blocks

    def _wrapped_build_level_blocks(*args, **kwargs):
        events.append("blocks:start")
        result = real_build_level_blocks(*args, **kwargs)
        events.append("blocks:end")
        return result

    monkeypatch.setattr(compile_module, "_build_level_blocks", _wrapped_build_level_blocks)

    state = compile_grg(_CompileOrderStubGrg(events))

    assert state.A_blocks is not None
    assert state.sel_mut.nnz == 1
    assert events.count("mutation_rows") == 1
    assert events.index("blocks:end") < events.index("mutation_rows")


def test_compile_grg_rejects_empty_grgs():
    with pytest.raises(ValueError, match="non-empty GRGs"):
        compile_grg(pygrgl.MutableGRG(0, 1))


def test_compile_grg_rejects_mutable_grgs():
    with pytest.raises(ValueError, match="immutable GRGs"):
        compile_grg(pygrgl.MutableGRG(2, 1))
