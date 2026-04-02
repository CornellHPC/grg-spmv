"""Focused tests for the flat physical-allocation collector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pygrgl_spmv.grg.sparse import binary_csr_from_csr_parts
from pygrgl_spmv.memory import alloc_field, capture_snapshot, child_field, ignore_field, tree_rows


@dataclass
class _AliasCall:
    left: np.ndarray | None = alloc_field(label="left", kind="input", owner="caller", default=None)
    right: np.ndarray | None = alloc_field(label="right", kind="temporary", owner="operator", default=None)


@dataclass
class _AliasRoot:
    call: _AliasCall = child_field(retention="call", activity="yes")


@dataclass
class _AliasCallReversed:
    right: np.ndarray | None = alloc_field(label="right", kind="temporary", owner="operator", default=None)
    left: np.ndarray | None = alloc_field(label="left", kind="input", owner="caller", default=None)


@dataclass
class _AliasRootReversed:
    call: _AliasCallReversed = child_field(retention="call", activity="yes")


@dataclass
class _Buf:
    value: np.ndarray | None = alloc_field(label="buf", kind="temporary", default=None)


@dataclass
class _StageMap:
    staging: dict[int, _Buf] = child_field(owner="backend", retention="staging", activity="no", direction="up", slot_k_from_dict_key=True, default_factory=dict)


@dataclass
class _IgnoreRoot:
    payload: np.ndarray | None = ignore_field(default=None)


@dataclass
class _BrokenChild:
    payload: np.ndarray | None = None


@dataclass
class _BrokenRoot:
    child: _BrokenChild = child_field(owner="operator", retention="call", activity="yes")


@dataclass
class _ConflictCall:
    call_buf: np.ndarray | None = alloc_field(label="call_buf", kind="input", owner="caller", default=None)


@dataclass
class _ConflictRetained:
    retained_buf: np.ndarray | None = alloc_field(label="retained_buf", kind="mapping", owner="operator", default=None)


@dataclass
class _ConflictRoot:
    call: _ConflictCall = child_field(retention="call", activity="yes")
    retained: _ConflictRetained = child_field(retention="persistent", activity="always")


@dataclass
class _SparseRoot:
    mat: object = alloc_field(label="mat", kind="sparse", owner="operator", retention="persistent", activity="always")


def test_capture_snapshot_merges_alias_roles_and_keeps_one_physical_allocation():
    arr = np.arange(8, dtype=np.float64)
    snapshot = capture_snapshot(_AliasRootReversed(call=_AliasCallReversed(right=arr, left=arr)), stage="run_up", runtime_k=1)
    assert len(snapshot.allocations) == 1
    row = snapshot.allocations[0]
    assert row.owner is None
    assert row.owners == frozenset({"caller", "operator"})
    assert row.retention == "call"
    assert row.activity == "yes"
    assert row.labels == frozenset({"left", "right"})
    assert row.kinds == frozenset({"input", "temporary"})


def test_capture_snapshot_merges_shared_base_views():
    arr = np.arange(16, dtype=np.float64)
    left = arr[2:]
    right = arr.view()
    snapshot = capture_snapshot(_AliasRoot(call=_AliasCall(left=left, right=right)), stage="run_up", runtime_k=1)
    assert len(snapshot.allocations) == 1
    assert snapshot.allocations[0].nbytes == arr.nbytes


def test_capture_snapshot_merges_distinct_bindings_for_one_physical_allocation():
    arr = np.arange(8, dtype=np.float64)
    root = _ConflictRoot(
        call=_ConflictCall(call_buf=arr),
        retained=_ConflictRetained(retained_buf=arr),
    )
    snapshot = capture_snapshot(root, stage="run_up", runtime_k=1)
    assert len(snapshot.allocations) == 1
    row = snapshot.allocations[0]
    assert row.owners == frozenset({"caller", "operator"})
    assert row.retentions == frozenset({"call", "persistent"})
    assert row.activities == frozenset({"yes", "always"})


def test_capture_snapshot_rejects_unclassified_fields():
    with pytest.raises(RuntimeError, match="missing memory metadata"):
        capture_snapshot(_BrokenRoot(child=_BrokenChild(payload=np.arange(4, dtype=np.float64))), stage="run_up", runtime_k=1)


def test_capture_snapshot_rejects_ignored_allocations():
    with pytest.raises(RuntimeError, match="ignore field"):
        capture_snapshot(_IgnoreRoot(payload=np.arange(4, dtype=np.float64)), stage="run_up", runtime_k=1)


def test_capture_snapshot_rejects_cpu_torch_tensors():
    torch = pytest.importorskip("torch")

    @dataclass
    class _TorchRoot:
        payload: object | None = alloc_field(
            label="payload",
            kind="state",
            owner="backend",
            retention="call",
            activity="yes",
            default=None,
        )

    with pytest.raises(AssertionError, match="torch tensors to live on CUDA"):
        capture_snapshot(_TorchRoot(payload=torch.ones((4,))), stage="run_up", runtime_k=1)


def test_capture_snapshot_derives_slot_k_from_dict_key():
    root = _StageMap(staging={4: _Buf(value=np.arange(4, dtype=np.float64))})
    snapshot = capture_snapshot(root, stage="run_up", runtime_k=1)
    assert len(snapshot.allocations) == 1
    assert snapshot.allocations[0].slot_k == 4
    assert snapshot.allocations[0].direction == "up"
    assert snapshot.allocations[0].retention == "staging"


def test_tree_rows_emit_multi_slot_retained_binding():
    arr = np.arange(4, dtype=np.float64)
    snapshot = capture_snapshot(
        _StageMap(staging={1: _Buf(value=arr), 2: _Buf(value=arr)}),
        stage="retained",
        runtime_k=None,
    )
    assert len(snapshot.allocations) == 1
    assert snapshot.allocations[0].slot_ks == frozenset({1, 2})
    rows = tree_rows(snapshot)
    assert any(row.path == ("cpu_live", "retained", "up", "k=1|2") for row in rows)


def test_tree_rows_preserve_total_leaf_bytes():
    arr1 = np.arange(4, dtype=np.float64)
    arr2 = np.arange(8, dtype=np.float64)
    snapshot = capture_snapshot(
        _AliasRoot(call=_AliasCall(left=arr1, right=arr2)),
        stage="run_up",
        runtime_k=2,
        direction="up",
        meta={"mode": "dynamic"},
    )
    rows = tree_rows(snapshot)
    leaf_total = sum(row.nbytes for row in rows if row.is_leaf())
    flat_total = sum(row.nbytes for row in snapshot.allocations)
    assert leaf_total == flat_total
    assert rows[0].path == ("cpu_live",)


def test_capture_snapshot_handles_shared_sparse_data_without_breaking():
    matrix = binary_csr_from_csr_parts(
        indices=np.arange(6, dtype=np.int32),
        indptr=np.array([0, 6], dtype=np.int32),
        shape=(1, 6),
        index_dtype=np.int32,
        shared_data=True,
    )
    snapshot = capture_snapshot(_SparseRoot(mat=matrix), stage="retained", runtime_k=None)
    leaf_sizes = sorted(row.nbytes for row in snapshot.allocations)
    assert leaf_sizes == [1, 8, 24]
