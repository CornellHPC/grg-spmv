"""Focused tests for GRG selector compilation edge cases."""

from __future__ import annotations

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv.grg.compile import _build_selectors, compile_grg


class _SelectorRowsStub:
    def __init__(self, rows):
        self._rows = list(rows)

    def get_mutation_node_miss(self):
        return list(self._rows)


def _selectors(rows, *, num_mutations: int, num_nodes: int = 8):
    return _build_selectors(
        _SelectorRowsStub(rows),
        inv_node_perm=np.arange(num_nodes, dtype=np.int32),
        num_mutations=num_mutations,
        num_nodes=num_nodes,
        index_dtype=np.int32,
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


def test_compile_grg_rejects_empty_grgs():
    with pytest.raises(ValueError, match="non-empty GRGs"):
        compile_grg(pygrgl.MutableGRG(0, 1), index_dtype=np.int32)


def test_compile_grg_rejects_mutable_grgs():
    with pytest.raises(ValueError, match="immutable GRGs"):
        compile_grg(pygrgl.MutableGRG(2, 1), index_dtype=np.int32)
