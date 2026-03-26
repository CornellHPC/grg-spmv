"""Tests for the concrete CPU reference behavior in Backend base class."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import pytest

from pygrgl_spmv.backends import BackendBase, BackendSetup, CallCapture, ReferenceBackend, ReferencePlanPair
from pygrgl_spmv.backends.types import InitMode
from pygrgl_spmv.memory import alloc_field, capture_snapshot, child_field


def _make_toy_backend(coalescence_counts: np.ndarray | None = None) -> ReferenceBackend:
    # Levels: [samples(0,1)] -> [node2] -> [node3]
    level_offsets = np.array([0, 2, 3, 4], dtype=np.int64)
    A_blocks = [
        [],
        [sp.csr_matrix(np.array([[1.0, 1.0]], dtype=np.float64))],
        [
            sp.csr_matrix((1, 2), dtype=np.float64),
            sp.csr_matrix(np.array([[1.0]], dtype=np.float64)),
        ],
    ]
    sel_mut = sp.csr_matrix(
        (
            np.ones(2, dtype=np.float64),
            (np.array([0, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64)),
        ),
        shape=(2, 4),
    )
    sel_miss = sp.csr_matrix(
        (
            np.ones(2, dtype=np.float64),
            (np.array([0, 1], dtype=np.int64), np.array([0, 1], dtype=np.int64)),
        ),
        shape=(2, 4),
    )
    sample_perm = np.array([0, 1], dtype=np.int64)
    inv_sample_perm = np.array([0, 1], dtype=np.int64)

    backend = ReferenceBackend(
        pair=ReferencePlanPair(
            plan_up=ReferenceBackend.plan(fmt="CSR", store="N", k_hint=None),
            plan_down=ReferenceBackend.plan(fmt="CSC", store="T", k_hint=None),
        ),
    )
    backend.setup(
        BackendSetup(
            A_blocks=A_blocks,
            level_offsets=level_offsets,
            num_samples=2,
            num_mutations=2,
            num_nodes=4,
            sel_mut=sel_mut,
            sel_miss=sel_miss,
            sample_perm=sample_perm,
            inv_sample_perm=inv_sample_perm,
            coalescence_counts=coalescence_counts,
            dtype=np.float64,
        )
    )
    return backend


@dataclass
class _NestedCall:
    payload: np.ndarray | None = alloc_field(label="payload", kind="temporary", default=None)


@dataclass
class _TestCall:
    node_values: np.ndarray | None = alloc_field(
        label="node_state", kind="state", owner="backend", retention="call", activity="yes", default=None
    )
    nested: _NestedCall = child_field(owner="backend", retention="call", activity="yes", default_factory=_NestedCall)


@dataclass
class _TestRetained:
    kept: np.ndarray | None = alloc_field(
        label="kept", kind="mapping", owner="backend", retention="persistent", activity="always", default=None
    )


class _DummyBackend(BackendBase):
    _SETUP_MEMORY_POLICY = {
        "_A_blocks": "retained",
        "_sel_mut": "borrowed",
        "_sel_miss": "borrowed",
        "_level_offsets": "borrowed",
        "_sample_perm": "borrowed",
        "_inv_sample_perm": "borrowed",
        "_coalescence_counts": "borrowed",
        "_xtx_host": "dropped",
    }

    def __init__(self):
        super().__init__(
            plan_up=ReferenceBackend.plan(fmt="CSR", store="N", k_hint=None),
            plan_down=None,
        )
        self._install_memory(retained=_TestRetained(), call_type=_TestCall)

    def setup(self, setup: BackendSetup) -> None:
        self._apply_setup_state(setup)

    def run_up(self, primary: np.ndarray, *, init_mode: InitMode, init: np.ndarray | None = None, need_miss_output: bool = False):
        raise NotImplementedError

    def run_down(self, primary: np.ndarray, *, miss: np.ndarray | None = None, init_mode: InitMode, init: np.ndarray | None = None):
        raise NotImplementedError

    def run_up_nodes(self, primary: np.ndarray, *, init_mode: InitMode, init: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    def run_down_nodes(self, primary: np.ndarray, *, init_mode: InitMode, init: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError
@pytest.mark.smoke
def test_backend_base_run_up_and_run_down_cpu_reference():
    backend = _make_toy_backend()

    up_in = np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float64)
    up_out, miss_out = backend.run_up(
        up_in,
        init_mode=InitMode.NONE,
        need_miss_output=True,
    )
    expected_mut = np.vstack([up_in[0] + up_in[1], up_in[0] + up_in[1]])
    expected_miss = np.vstack([up_in[0], up_in[1]])
    np.testing.assert_allclose(up_out, expected_mut)
    assert miss_out is not None
    np.testing.assert_allclose(miss_out, expected_miss)

    down_in = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    down_miss = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float64)
    down_out = backend.run_down(
        down_in,
        miss=down_miss,
        init_mode=InitMode.NONE,
    )
    expected_down = np.vstack(
        [
            down_miss[0] + down_in[0] + down_in[1],
            down_miss[1] + down_in[0] + down_in[1],
        ]
    )
    np.testing.assert_allclose(down_out, expected_down)


def test_backend_base_xtx_requires_coalescence_counts():
    backend = _make_toy_backend(coalescence_counts=None)
    x = np.ones((2, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="coalescence counts"):
        backend.run_up(x, init_mode=InitMode.XTX)


def test_backend_base_xtx_runs_when_counts_present():
    backend = _make_toy_backend(coalescence_counts=np.array([0, 0, 2, 3], dtype=np.int64))
    x = np.ones((2, 1), dtype=np.float64)
    out, _ = backend.run_up(x, init_mode=InitMode.XTX, need_miss_output=False)
    assert out.shape == (2, 1)


def test_call_capture_scope_creates_and_drops_call_memory():
    backend = _DummyBackend()
    assert backend._call_mem is None
    with backend._call_capture_scope() as nonce:
        assert nonce == 1
        assert isinstance(backend._call_mem, _TestCall)
    assert backend._call_mem is None
    assert backend._capture_active is False


def test_call_capture_scope_clears_state_on_exception():
    backend = _DummyBackend()
    with pytest.raises(RuntimeError, match="boom"):
        with backend._call_capture_scope():
            assert isinstance(backend._call_mem, _TestCall)
            backend._call_mem.node_values = np.ones((4,), dtype=np.float64)
            backend._call_mem.nested.payload = np.ones((2,), dtype=np.float64)
            raise RuntimeError("boom")
    assert backend._call_mem is None
    assert backend._capture_active is False


def test_call_capture_rejects_reserved_semantic_meta_keys():
    with pytest.raises(ValueError, match="reserved semantic key"):
        CallCapture(
            nonce=1,
            direction="up",
            runtime_k=1,
            meta={"active_alloc_keys": frozenset()},
        )


def test_setup_memory_contract_rejects_live_dropped_field():
    backend = _DummyBackend()
    backend._xtx_host = np.ones((4,), dtype=np.float64)
    with pytest.raises(RuntimeError, match="expected dropped setup field _xtx_host"):
        backend._assert_setup_memory_contract()


def test_setup_memory_contract_rejects_missing_retained_representation():
    backend = _DummyBackend()
    backend._A_blocks = [[sp.csr_matrix(np.array([[1.0]], dtype=np.float64))]]
    with pytest.raises(RuntimeError, match="retained setup field _A_blocks is not represented"):
        backend._assert_setup_memory_contract()


def test_reference_backend_setup_snapshot_counts_retained_blocks_and_xtx_host():
    backend = _make_toy_backend(coalescence_counts=np.array([0, 0, 2, 3], dtype=np.int64))
    snapshot = capture_snapshot(backend._retained_mem, stage="setup", runtime_k=None)
    labels = [row.labels for row in snapshot.allocations]
    assert any("blocks" in entry for entry in labels)
    assert any("xtx_host" in entry for entry in labels)


def test_reference_backend_direct_run_does_not_leave_call_memory_live():
    backend = _make_toy_backend()
    x = np.ones((2, 1), dtype=np.float64)
    _ = backend.run_up(x, init_mode=InitMode.NONE, need_miss_output=False)
    assert backend._call_mem is None
    backend.setup(
        BackendSetup(
            A_blocks=backend._A_blocks,
            level_offsets=backend._level_offsets,
            num_samples=backend._num_samples,
            num_mutations=backend._num_mutations,
            num_nodes=backend._num_nodes,
            sel_mut=backend._sel_mut,
            sel_miss=backend._sel_miss,
            sample_perm=backend._sample_perm,
            inv_sample_perm=backend._inv_sample_perm,
            coalescence_counts=backend._coalescence_counts,
            dtype=np.float64,
        )
    )
