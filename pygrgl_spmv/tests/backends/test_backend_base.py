"""Tests for the concrete CPU reference behavior in Backend base class."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import pytest

from pygrgl_spmv.backends import BackendSetup, ReferenceBackend
from pygrgl_spmv.backends.types import InitMode


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
        plan_up=ReferenceBackend.plan(fmt="CSR", store="N", k_hint=None),
        plan_down=ReferenceBackend.plan(fmt="CSC", store="T", k_hint=None),
    )
    backend.setup(
        BackendSetup(
            A_blocks=A_blocks,
            level_offsets=level_offsets,
            n=2,
            K=4,
            sel_mut=sel_mut,
            sel_miss=sel_miss,
            sample_perm=sample_perm,
            inv_sample_perm=inv_sample_perm,
            coalescence_counts=coalescence_counts,
            dtype=np.float64,
        )
    )
    return backend
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


def test_backend_base_estimate_static_bytes_not_implemented():
    backend = _make_toy_backend()
    with pytest.raises(NotImplementedError, match="estimate_static_bytes"):
        _ = backend.estimate_static_bytes()
