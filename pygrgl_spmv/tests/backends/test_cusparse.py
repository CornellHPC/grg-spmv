"""cuSPARSE backend tests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import warnings
from pathlib import Path

import numpy as np
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.backends.cusparse import CusparseBackend, CusparsePlan, CusparsePlanPair, is_valid_combo
from pygrgl_spmv.backends.types import Direction
from pygrgl_spmv.memory import alloc_field, capture_snapshot, live_snapshot, tree_rows
from pygrgl_spmv.tests.conftest import (
    DATA_DTYPE,
    INDEX_DTYPE,
    K_CORE,
    K_MATRIX,
    binary_pm1,
    invalid_fmt_algo_params,
    make_cusparse_backend,
    make_cusparse_plan,
    tol,
    valid_fmt_algo_params,
)

cp = pytest.importorskip("cupy")
pytestmark = [pytest.mark.gpu, pytest.mark.cusparse]

_CACHE_DIR = Path(".pytest_cache") / "pygrgl_spmv_npz"


def _make_op(
    grg_path,
    *,
    fmt_up="csr",
    fmt_down=None,
    k_hint=None,
    algo_up="default",
    algo_down="default",
    scratch_up="none",
    scratch_down="none",
    log_level="WARNING",
    instrumentation=False,
    dtype=DATA_DTYPE,
    cache_dir=_CACHE_DIR,
):
    return SpmvGRG(
        grg_path,
        make_cusparse_backend(
            fmt_up=fmt_up,
            fmt_down=fmt_down,
            k_hint=k_hint,
            algo_up=algo_up,
            algo_down=algo_down,
            scratch_up=scratch_up,
            scratch_down=scratch_down,
            log_level=log_level,
            instrumentation=instrumentation,
        ),
        dtype,
        INDEX_DTYPE,
        artifact_dir=cache_dir,
    )


def _run_up(op, X_col_major):
    return op.matmul(X_col_major.T, "up").T


def _run_down(op, X_col_major):
    return op.matmul(X_col_major.T, "down").T


def test_cupy_memory_accounting_uses_logical_bytes():
    @dataclass
    class _CupyRoot:
        payload: object | None = alloc_field(
            label="payload",
            kind="temporary",
            owner="backend",
            retention="persistent",
            activity="always",
            default=None,
        )

    arr = cp.zeros((3,), dtype=cp.float64)
    snapshot = capture_snapshot(_CupyRoot(payload=arr), stage="retained", runtime_k=None)
    assert len(snapshot.allocations) == 1
    assert snapshot.allocations[0].nbytes == int(arr.nbytes)


@pytest.mark.parametrize(
    "plan_up",
    [
        pytest.param(make_cusparse_plan(k_hint=None, store="N", fmt="CSR", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"), id="direct-row"),
        pytest.param(make_cusparse_plan(k_hint=None, store="N", fmt="CSR", op_a="N", op_b="T", order_b="COL", order_c="ROW", algo="DEFAULT"), id="reinterpret"),
        pytest.param(make_cusparse_plan(k_hint=None, store="N", fmt="CSR", op_a="N", op_b="N", order_b="COL", order_c="ROW", algo="DEFAULT"), id="repack"),
    ],
)
def test_forward_explicit_dense_view_plans(primary_grg_path, gt_small, plan_up):
    op = SpmvGRG(
        primary_grg_path,
        CusparseBackend(pair=CusparsePlanPair.from_dicts(plan_up, None), log_level="WARNING"),
        DATA_DTYPE,
        INDEX_DTYPE,
        artifact_dir=_CACHE_DIR,
    )
    X, Y_expected = gt_small.get("forward", 4, seed=841, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(op.matmul(X.T, "up").T, Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "plan_down",
    [
        pytest.param(make_cusparse_plan(k_hint=None, store="T", fmt="CSC", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"), id="direct-row"),
        pytest.param(make_cusparse_plan(k_hint=None, store="T", fmt="CSC", op_a="N", op_b="T", order_b="COL", order_c="ROW", algo="DEFAULT"), id="reinterpret"),
        pytest.param(make_cusparse_plan(k_hint=None, store="T", fmt="CSC", op_a="N", op_b="N", order_b="COL", order_c="ROW", algo="DEFAULT"), id="repack"),
    ],
)
def test_backward_explicit_dense_view_plans(primary_grg_path, gt_small, plan_down):
    op = SpmvGRG(
        primary_grg_path,
        CusparseBackend(pair=CusparsePlanPair.from_dicts(None, plan_down), log_level="WARNING"),
        DATA_DTYPE,
        INDEX_DTYPE,
        artifact_dir=_CACHE_DIR,
    )
    X, Y_expected = gt_small.get("backward", 4, seed=842, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(op.matmul(X.T, "down").T, Y_expected, atol=atol, rtol=rtol)


class TestSmokeCoreConfigMatrix:
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "cfg,k",
        [
            pytest.param(
                {"fmt_up": "csr", "fmt_down": None, "algo_up": "default", "algo_down": "default", "k_hint": 1},
                1,
                id="up-csr-down-auto-default-k1",
            ),
            pytest.param(
                {"fmt_up": None, "fmt_down": "csc", "algo_up": "default", "algo_down": "default", "k_hint": 4},
                4,
                id="up-auto-down-csc-default-k4",
            ),
            pytest.param(
                {"fmt_up": "coo", "fmt_down": "coo", "algo_up": "coo_alg1", "algo_down": "coo_alg2", "k_hint": None},
                2,
                id="up-coo-down-coo-mixed-dyn",
            ),
        ],
    )
    def test_forward(self, primary_grg_path, gt_small, cfg, k):
        op = _make_op(primary_grg_path, **cfg)
        X, Y_expected = gt_small.get("forward", k, seed=7201, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "cfg,k",
        [
            pytest.param(
                {"fmt_up": "csr", "fmt_down": None, "algo_up": "default", "algo_down": "default", "k_hint": 1},
                1,
                id="up-csr-down-auto-default-k1",
            ),
            pytest.param(
                {"fmt_up": None, "fmt_down": "csc", "algo_up": "default", "algo_down": "default", "k_hint": 4},
                4,
                id="up-auto-down-csc-default-k4",
            ),
            pytest.param(
                {"fmt_up": "coo", "fmt_down": "coo", "algo_up": "coo_alg1", "algo_down": "coo_alg2", "k_hint": None},
                2,
                id="up-coo-down-coo-mixed-dyn",
            ),
        ],
    )
    def test_backward(self, primary_grg_path, gt_small, cfg, k):
        op = _make_op(primary_grg_path, **cfg)
        X, Y_expected = gt_small.get("backward", k, seed=7202, dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt,algo", valid_fmt_algo_params())
def test_construct_for_valid_fmt_algo(primary_grg_path, fmt, algo):
    _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=4, algo_up=algo, algo_down=algo)


@pytest.mark.parametrize("fmt,algo", invalid_fmt_algo_params())
def test_construct_rejects_invalid_fmt_algo(primary_grg_path, fmt, algo):
    with pytest.raises(ValueError, match="Unsupported cuSPARSE"):
        _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=4, algo_up=algo, algo_down=algo)


def test_csr_alg3_transpose_validity_matrix():
    assert is_valid_combo("csr", False, "csr_alg3")
    assert not is_valid_combo("csc", False, "csr_alg3")
    assert not is_valid_combo("csr", True, "csr_alg3")
    assert not is_valid_combo("csc", True, "csr_alg3")
    assert not is_valid_combo("coo", False, "csr_alg3")
    assert not is_valid_combo("coo", True, "csr_alg3")


def test_csr_alg3_rejects_csc_non_transpose_path(primary_grg_path):
    with pytest.raises(ValueError, match="Unsupported cuSPARSE UP plan"):
        _make_op(primary_grg_path, fmt_up="csc", fmt_down="csc", k_hint=None, algo_up="csr_alg3", algo_down="default")


def test_csr_alg3_rejects_csc_transpose_path(primary_grg_path):
    with pytest.raises(ValueError, match="Unsupported cuSPARSE UP plan"):
        _make_op(primary_grg_path, fmt_up=None, fmt_down="csc", k_hint=None, algo_up="csr_alg3", algo_down="default")


def test_csr_alg3_rejects_csr_transpose_path(primary_grg_path):
    with pytest.raises(ValueError, match="Unsupported cuSPARSE DOWN plan"):
        SpmvGRG(
            primary_grg_path,
            CusparseBackend(
                pair=CusparsePlanPair.from_dicts(
                    make_cusparse_plan(
                        k_hint=None,
                        store="N",
                        fmt="CSR",
                        op_a="N",
                        op_b="N",
                        order_b="ROW",
                        order_c="ROW",
                        algo="DEFAULT",
                    ),
                    make_cusparse_plan(
                        k_hint=None,
                        store="N",
                        fmt="CSR",
                        op_a="T",
                        op_b="N",
                        order_b="ROW",
                        order_c="ROW",
                        algo="CSR_ALG3",
                    ),
                ),
                log_level="WARNING",
            ),
            DATA_DTYPE,
            INDEX_DTYPE,
            artifact_dir=_CACHE_DIR,
        )


def test_fmt_inference_from_down(primary_grg_path):
    _make_op(primary_grg_path, fmt_up=None, fmt_down="csc", k_hint=4)


def test_both_fmts_none_rejected(primary_grg_path):
    with pytest.raises(ValueError, match="fmt_up/fmt_down"):
        _make_op(primary_grg_path, fmt_up=None, fmt_down=None)


def test_non_string_algo_rejected(primary_grg_path):
    with pytest.raises(TypeError, match="algo_up"):
        _make_op(primary_grg_path, algo_up=123, algo_down="default")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="algo_down"):
        _make_op(primary_grg_path, algo_up="default", algo_down=123)  # type: ignore[arg-type]


@pytest.mark.parametrize("fmt,algo", valid_fmt_algo_params())
def test_forward_dynamic(primary_grg_path, gt_small, fmt, algo):
    op = _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=None, algo_up=algo, algo_down=algo)
    X, Y_expected = gt_small.get("forward", 4, seed=42, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt,algo", valid_fmt_algo_params())
def test_backward_dynamic(primary_grg_path, gt_small, fmt, algo):
    op = _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=None, algo_up=algo, algo_down=algo)
    X, Y_expected = gt_small.get("backward", 4, seed=42, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt,algo", valid_fmt_algo_params())
def test_forward_graph(primary_grg_path, gt_small, fmt, algo):
    op = _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=4, algo_up=algo, algo_down=algo)
    X, Y_expected = gt_small.get("forward", 4, seed=52, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt,algo", valid_fmt_algo_params())
def test_backward_graph(primary_grg_path, gt_small, fmt, algo):
    op = _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=4, algo_up=algo, algo_down=algo)
    X, Y_expected = gt_small.get("backward", 4, seed=52, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt,algo", valid_fmt_algo_params())
@pytest.mark.parametrize("k", K_CORE)
def test_k_sweep_graph(primary_grg_path, gt_small, fmt, algo, k):
    op = _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=k, algo_up=algo, algo_down=algo)
    Xf, Yf = gt_small.get("forward", k, seed=62, dtype=DATA_DTYPE)
    Xb, Yb = gt_small.get("backward", k, seed=63, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, Xf), Yf, atol=atol, rtol=rtol)
    np.testing.assert_allclose(_run_down(op, Xb), Yb, atol=atol, rtol=rtol)


def test_hint_none_keeps_graphs_disabled(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=None)
    X, _ = gt_small.get("forward", 4, seed=5000, dtype=DATA_DTYPE)
    _ = _run_up(op, X)
    ws = op._backend._workspaces.dynamic_up
    assert ws is not None
    assert op._backend._workspaces.graph_up is None
    assert op._backend._workspaces.graph_down is None
    assert ws.k == 4
    assert ws.graph is None
    assert op._backend._workspaces.dynamic_down is None


def test_graph_workspaces_built_during_setup(primary_grg_path):
    op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=4)
    backend = op._backend
    assert backend._workspaces.graph_up is not None
    assert backend._workspaces.graph_down is not None
    assert backend._workspaces.graph_up.graph is not None
    assert backend._workspaces.graph_down.graph is not None
    assert backend._workspaces.dynamic_up is None
    assert backend._workspaces.dynamic_down is None
    assert op.memory.retained is not None
    assert any(
        row.path[:4] == ("cuda_live", "retained", "up", "k=4") and row.retention == "captured"
        for row in tree_rows(op.memory.retained)
    )


def test_graph_workspace_keeps_optional_buffers_lazy(primary_grg_path):
    op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=1)
    backend = op._backend
    ws_up = backend._workspaces.graph_up
    ws_down = backend._workspaces.graph_down
    assert ws_up is not None and ws_down is not None
    before_up = id(ws_up)
    before_down = id(ws_down)
    assert backend._staging_up_by_k == {}

    X = np.ones((op.num_samples, 1), dtype=DATA_DTYPE)
    _ = op.matmul(X.T, "up", emit_all_nodes=True, init="xtx").T

    assert 1 in backend._staging_up_by_k
    assert backend._staging_up_by_k[1].xtx_bias is not None
    assert id(backend._workspaces.graph_up) == before_up
    assert id(backend._workspaces.graph_down) == before_down


def test_dynamic_workspace_reused_for_same_k(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4)
    X1, _ = gt_small.get("forward", 1, seed=5006, dtype=DATA_DTYPE)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _ = _run_up(op, X1)
    ws1 = op._backend._workspaces.dynamic_up
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _ = _run_up(op, X1)
    ws2 = op._backend._workspaces.dynamic_up
    assert ws1 is ws2
    assert ws2 is not None
    assert ws2.k == 1


def test_dynamic_workspace_recreated_when_k_changes(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4)
    X1, _ = gt_small.get("forward", 1, seed=5007, dtype=DATA_DTYPE)
    X2, _ = gt_small.get("forward", 2, seed=5008, dtype=DATA_DTYPE)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _ = _run_up(op, X1)
    ws1 = op._backend._workspaces.dynamic_up
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _ = _run_up(op, X2)
    ws2 = op._backend._workspaces.dynamic_up
    assert ws1 is not ws2
    assert ws2 is not None
    assert ws2.k == 2


def test_graph_fallback_preserves_static_graph(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4)
    X4, Y4_exp = gt_small.get("backward", 4, seed=5001, dtype=DATA_DTYPE)
    X1, Y1_exp = gt_small.get("backward", 1, seed=5002, dtype=DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X4), Y4_exp, atol=1e-5, rtol=1e-5)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        np.testing.assert_allclose(_run_down(op, X1), Y1_exp, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_run_down(op, X4), Y4_exp, atol=1e-5, rtol=1e-5)
    assert op._backend._workspaces.graph_down is not None
    assert op._backend._workspaces.graph_down.graph is not None


def test_debug_log_level_keeps_graph_mode(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4, log_level="DEBUG")
    x_up, _ = gt_small.get("forward", 4, seed=5003, dtype=DATA_DTYPE)
    x_down, _ = gt_small.get("backward", 4, seed=5004, dtype=DATA_DTYPE)
    _ = _run_up(op, x_up)
    assert op.memory.last_call is not None
    assert op.memory.last_call.meta["mode"] == "graph"
    assert op.memory.last_call.active_alloc_keys
    merged = live_snapshot(op.memory.retained, op.memory.last_call)
    assert merged is not None
    assert any(
        row.owner == "backend"
        and row.direction == "up"
        and row.slot_k == 4
        and row.retention in {"captured", "staging"}
        and row.activity == "yes"
        for row in merged.allocations
    )
    _ = _run_down(op, x_down)
    assert op.memory.last_call is not None
    assert op.memory.last_call.meta["mode"] == "graph"
    assert op._backend._workspaces.graph_up is not None
    assert op._backend._workspaces.graph_down is not None
    assert op._backend._workspaces.graph_up.graph is not None
    assert op._backend._workspaces.graph_down.graph is not None


def test_debug_log_level_reports_block_memory(primary_grg_path, caplog):
    with caplog.at_level(logging.DEBUG):
        op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=1, log_level="DEBUG")
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("cuSPARSE blocks_up" in msg and "rows=" in msg and "nnz=" in msg and "indices_bytes=" in msg for msg in messages)
    assert any("cuSPARSE block dir=up" in msg and "cols=" in msg and "nnz=" in msg for msg in messages)
    del op


def test_instrumentation_disables_graph_mode(primary_grg_path, gt_small):
    with pytest.warns(RuntimeWarning, match="instrumentation=True"):
        op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4, instrumentation=True)
    x_up, y_up = gt_small.get("forward", 4, seed=5005, dtype=DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, x_up), y_up, atol=1e-5, rtol=1e-5)
    assert op.memory.last_call is not None
    assert op.memory.last_call.meta["mode"] == "instrumented"
    assert op._backend._workspaces.dynamic_up is not None
    assert op._backend._workspaces.graph_up is None
    assert op._backend._workspaces.graph_down is None
    assert op.memory.retained is not None
    assert not any(row.retention == "captured" for row in tree_rows(op.memory.retained))


def test_transpose_compatible_storage_aliases_payload_but_not_descriptors(primary_grg_path):
    up = make_cusparse_plan(
        k_hint=None,
        store="N",
        fmt="CSR",
        op_a="N",
        op_b="N",
        order_b="ROW",
        order_c="ROW",
        algo="DEFAULT",
    )
    down = make_cusparse_plan(
        k_hint=None,
        store="T",
        fmt="CSC",
        op_a="N",
        op_b="N",
        order_b="ROW",
        order_c="ROW",
        algo="DEFAULT",
    )
    op = SpmvGRG(
        primary_grg_path,
        CusparseBackend(pair=CusparsePlanPair.from_dicts(up, down), log_level="WARNING"),
        DATA_DTYPE,
        INDEX_DTYPE,
        artifact_dir=_CACHE_DIR,
    )
    backend = op._backend
    H = len(backend._level_offsets) - 1
    for dst_level in range(H):
        for row_index, down_block in enumerate(backend._blocks_down[dst_level]):
            if down_block is None:
                continue
            src_level = row_index + dst_level + 1
            up_block = backend._blocks_up[src_level][dst_level]
            assert up_block is not None
            assert tuple(buf.data.ptr for buf in down_block.index_buffers) == tuple(buf.data.ptr for buf in up_block.index_buffers)
            assert down_block.data_ptr == up_block.data_ptr
            assert down_block.graph_desc.value != up_block.graph_desc.value
            assert down_block.dynamic_desc.value != up_block.dynamic_desc.value
            return
    raise AssertionError("expected at least one shared cuSPARSE block alias")


def test_cusparse_csr_csc_blocks_share_one_shared_ones_pointer(primary_grg_path):
    op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=None)
    backend = op._backend
    shared = backend._shared_ones
    assert shared is not None
    data_ptrs = {
        int(block.data_ptr)
        for grid in (backend._blocks_up, backend._blocks_down)
        for row in grid
        for block in row
        if block is not None
    }
    assert data_ptrs == {int(shared.ptr)}


def test_cusparse_coo_blocks_share_one_shared_ones_pointer(primary_grg_path):
    op = _make_op(primary_grg_path, fmt_up="coo", fmt_down="coo", k_hint=None, algo_up="coo_alg1", algo_down="coo_alg2")
    backend = op._backend
    shared = backend._shared_ones
    assert shared is not None
    data_ptrs = {
        int(block.data_ptr)
        for grid in (backend._blocks_up, backend._blocks_down)
        for row in grid
        for block in row
        if block is not None
    }
    assert data_ptrs == {int(shared.ptr)}


def test_cusparse_shared_ones_use_materialized_array_when_vmm_unsupported(primary_grg_path, monkeypatch, caplog):
    import pygrgl_spmv.backends.cusparse.backend as cusparse_backend_mod

    monkeypatch.setattr(cusparse_backend_mod.CudaVmmDriver, "current_context", lambda self: 1)
    monkeypatch.setattr(cusparse_backend_mod.CudaVmmDriver, "vmm_supported", lambda self, device_id: False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.INFO):
            op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=None, log_level="INFO")
    backend = op._backend
    shared = backend._shared_ones
    assert shared is not None
    assert shared.vmm is False
    assert shared.physical_nbytes == shared.logical_nbytes
    assert caught == []
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("cuSPARSE shared ones: mode=materialized reason=vmm-unsupported" in msg for msg in messages)
    data_ptrs = {
        int(block.data_ptr)
        for grid in (backend._blocks_up, backend._blocks_down)
        for row in grid
        for block in row
        if block is not None
    }
    assert data_ptrs == {int(shared.ptr)}


def test_cusparse_shared_ones_log_materialized_when_vmm_saves_no_memory(primary_grg_path, monkeypatch, caplog):
    import pygrgl_spmv.backends.cusparse.backend as cusparse_backend_mod

    monkeypatch.setattr(cusparse_backend_mod.CudaVmmDriver, "current_context", lambda self: 1)
    monkeypatch.setattr(cusparse_backend_mod.CudaVmmDriver, "vmm_supported", lambda self, device_id: True)
    monkeypatch.setattr(
        cusparse_backend_mod.CudaVmmDriver,
        "allocation_granularity",
        lambda self, device_id, *, recommended: 1 << 30 if not recommended else 1 << 31,
    )
    monkeypatch.setattr(
        cusparse_backend_mod.CusparseBackend,
        "_build_vmm_shared_ones",
        lambda self, **kwargs: (_ for _ in ()).throw(AssertionError("VMM path should not be used")),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.INFO):
            op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=None, log_level="INFO")
    shared = op._backend._shared_ones
    assert shared is not None
    assert shared.vmm is False
    assert caught == []
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("cuSPARSE shared ones: mode=materialized reason=no-memory-savings" in msg for msg in messages)


def test_cusparse_shared_ones_log_vmm_when_physical_bytes_drop(primary_grg_path, monkeypatch, caplog):
    import pygrgl_spmv.backends.cusparse.backend as cusparse_backend_mod

    original_max_block_nnz = cusparse_backend_mod.CusparseBackend._max_block_nnz

    def fake_build_vmm_shared_ones(self, *, driver, device_id, logical_nbytes, tile_nbytes):
        arr = self._cp.ones((logical_nbytes // int(self._dtype.itemsize),), dtype=self._dtype)
        return cusparse_backend_mod._SharedOnes(
            ptr=int(arr.data.ptr),
            logical_nbytes=logical_nbytes,
            physical_nbytes=tile_nbytes,
            vmm=True,
            _materialized=arr,
            _reserved_nbytes=cusparse_backend_mod._round_up(logical_nbytes, tile_nbytes),
        )

    monkeypatch.setattr(cusparse_backend_mod.CusparseBackend, "_max_block_nnz", lambda self: max(original_max_block_nnz(self), 16))
    monkeypatch.setattr(cusparse_backend_mod.CudaVmmDriver, "current_context", lambda self: 1)
    monkeypatch.setattr(cusparse_backend_mod.CudaVmmDriver, "vmm_supported", lambda self, device_id: True)
    monkeypatch.setattr(
        cusparse_backend_mod.CudaVmmDriver,
        "allocation_granularity",
        lambda self, device_id, *, recommended: 64 if not recommended else 128,
    )
    monkeypatch.setattr(cusparse_backend_mod.CusparseBackend, "_build_vmm_shared_ones", fake_build_vmm_shared_ones)
    with caplog.at_level(logging.INFO):
        op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=None, log_level="INFO")
    shared = op._backend._shared_ones
    assert shared is not None
    assert shared.vmm is True
    messages = [rec.getMessage() for rec in caplog.records]
    assert any(
        "cuSPARSE shared ones: mode=vmm reason=physical-bytes-reduced" in msg and "reserve_alignment=64" in msg
        for msg in messages
    )


def test_cusparse_shared_ones_warn_when_vmm_build_fails(primary_grg_path, monkeypatch, caplog):
    import pygrgl_spmv.backends.cusparse.backend as cusparse_backend_mod

    original_max_block_nnz = cusparse_backend_mod.CusparseBackend._max_block_nnz

    monkeypatch.setattr(cusparse_backend_mod.CusparseBackend, "_max_block_nnz", lambda self: max(original_max_block_nnz(self), 16))
    monkeypatch.setattr(cusparse_backend_mod.CudaVmmDriver, "current_context", lambda self: 1)
    monkeypatch.setattr(cusparse_backend_mod.CudaVmmDriver, "vmm_supported", lambda self, device_id: True)
    monkeypatch.setattr(
        cusparse_backend_mod.CudaVmmDriver,
        "allocation_granularity",
        lambda self, device_id, *, recommended: 64 if not recommended else 128,
    )
    monkeypatch.setattr(
        cusparse_backend_mod.CusparseBackend,
        "_build_vmm_shared_ones",
        lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with caplog.at_level(logging.INFO):
        with pytest.warns(RuntimeWarning, match="cuSPARSE shared ones falling back to one materialized all-ones array: boom"):
            op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=None, log_level="INFO")
    shared = op._backend._shared_ones
    assert shared is not None
    assert shared.vmm is False
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("cuSPARSE shared ones: mode=materialized reason=vmm-build-failed" in msg for msg in messages)


def test_cusparse_vmm_driver_forwards_reserve_alignment(monkeypatch):
    import ctypes

    import pygrgl_spmv.backends.cusparse.ffi as cusparse_ffi

    class _FakeCudaDriverLib:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def cuInit(self, flags):
            return cusparse_ffi.CUDA_SUCCESS

        def cuMemAddressReserve(self, addr_ptr, size, alignment, addr, flags):
            self.calls.append((int(size.value), int(alignment.value)))
            ctypes.cast(addr_ptr, ctypes.POINTER(ctypes.c_ulonglong))[0] = 0x1234000
            return cusparse_ffi.CUDA_SUCCESS

    fake = _FakeCudaDriverLib()
    monkeypatch.setattr(cusparse_ffi, "_load_cuda_driver_library", lambda: fake)

    driver = cusparse_ffi.CudaVmmDriver()
    addr = driver.address_reserve(4096, alignment_bytes=2097152)

    assert addr == 0x1234000
    assert fake.calls == [(4096, 2097152)]


def test_setup_releases_host_sparse_payloads(primary_grg_path):
    op = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=None)
    backend = op._backend
    assert backend._A_blocks == []
    assert backend._sel_mut.shape == (0, 0)
    assert backend._sel_miss.shape == (0, 0)


class _NvtxRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, object]]] = []

    @contextmanager
    def range(self, name: str, /, **fields: object):
        self.events.append(("range", name, dict(fields)))
        yield

    def mark(self, name: str, /, **fields: object) -> None:
        self.events.append(("mark", name, dict(fields)))


def test_instrumented_cusparse_emits_nvtx_markers(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=None, instrumentation=True)
    recorder = _NvtxRecorder()
    op._backend._nvtx = recorder
    x_up, _ = gt_small.get("forward", 2, seed=5009, dtype=DATA_DTYPE)
    _ = _run_up(op, x_up)
    names = [name for _, name, _ in recorder.events]
    assert "run_direction" in names
    assert "seed_direction" in names
    assert "wavefront" in names
    assert "level" in names
    assert "wait_ready" in names
    assert "launch" in names
    assert "publish_level" in names
    assert "event.record_ready" in names
    assert "join_ready" in names
    assert "collect_outputs" in names


def test_cusparse_scratch_buffers_allocated_for_enabled_levels(primary_grg_path):
    op = _make_op(
        primary_grg_path,
        fmt_up="csr",
        fmt_down="csc",
        k_hint=4,
        scratch_up="1",
        scratch_down="0",
    )
    backend = op._backend
    ws_up = backend._workspaces.graph_up
    ws_down = backend._workspaces.graph_down
    assert ws_up is not None and ws_down is not None
    assert len(ws_up.scratch_views_by_level[1]) == len(backend._ops_up[1]) > 0
    assert len(ws_up.scratch_dst_descs_by_level[1]) == len(backend._ops_up[1])
    assert len(ws_up.scratch_done_events_by_level[1]) == len(backend._ops_up[1])
    assert ws_up.scratch_views_by_level[0] == []
    assert len(ws_down.scratch_views_by_level[0]) == len(backend._ops_down[0]) > 0
    assert ws_down.scratch_views_by_level[1] == []


def test_instrumented_cusparse_scratch_emits_reduction_markers(primary_grg_path, gt_small):
    op = _make_op(
        primary_grg_path,
        fmt_up="csr",
        fmt_down=None,
        k_hint=None,
        scratch_up="1",
        instrumentation=True,
    )
    recorder = _NvtxRecorder()
    op._backend._nvtx = recorder
    x_up, _ = gt_small.get("forward", 2, seed=5010, dtype=DATA_DTYPE)
    _ = _run_up(op, x_up)
    names = [name for _, name, _ in recorder.events]
    assert "helper_launch" in names
    assert "event.record_scratch_done" in names
    assert "wait_scratch_done" in names
    assert "reduce_add" in names


def test_cusparse_scratch_enabled_up_matches_reference(primary_grg_path, gt_small):
    op = _make_op(
        primary_grg_path,
        fmt_up="csr",
        fmt_down=None,
        k_hint=None,
        scratch_up="1",
    )
    X, Y_expected = gt_small.get("forward", 4, seed=5011, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


def test_cusparse_scratch_enabled_down_matches_reference(primary_grg_path, gt_small):
    op = _make_op(
        primary_grg_path,
        fmt_up=None,
        fmt_down="csc",
        k_hint=None,
        scratch_down="0",
    )
    X, Y_expected = gt_small.get("backward", 4, seed=5012, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


def test_cusparse_graph_vs_dynamic_with_scratch(primary_grg_path):
    op_graph = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=4, scratch_up="1", scratch_down="0")
    op_dynamic = _make_op(primary_grg_path, fmt_up="csr", fmt_down="csc", k_hint=None, scratch_up="1", scratch_down="0")
    rng = np.random.default_rng(7010)
    V = rng.standard_normal((op_graph.num_samples, 4), dtype=DATA_DTYPE)
    W = rng.standard_normal((op_graph.num_mutations, 4), dtype=DATA_DTYPE)
    np.testing.assert_allclose(op_graph.matmul(V.T, "up").T, op_dynamic.matmul(V.T, "up").T, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(op_graph.matmul(W.T, "down").T, op_dynamic.matmul(W.T, "down").T, atol=1e-5, rtol=1e-5)


def test_graph_vs_dynamic(primary_grg_path):
    op_graph = _make_op(primary_grg_path, fmt_up="csr", k_hint=4)
    op_dynamic = _make_op(primary_grg_path, fmt_up="csr", k_hint=None)
    rng = np.random.default_rng(7001)
    V = rng.standard_normal((op_graph.num_samples, 4), dtype=DATA_DTYPE)
    W = rng.standard_normal((op_graph.num_mutations, 4), dtype=DATA_DTYPE)
    np.testing.assert_allclose(op_graph.matmul(V.T, "up").T, op_dynamic.matmul(V.T, "up").T, atol=1e-5)
    np.testing.assert_allclose(op_graph.matmul(W.T, "down").T, op_dynamic.matmul(W.T, "down").T, atol=1e-5)


def test_failure_modes(primary_grg_path, gt_small):
    for k, seed in [(1, 9001), (2, 9002), (3, 9003), (7, 9004)]:
        op = _make_op(primary_grg_path, fmt_up="csr", k_hint=k)
        X, Y_expected = gt_small.get("forward", k, seed=seed, dtype=DATA_DTYPE)
        np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=1e-5, rtol=1e-5)


def test_f_order_input(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=None)
    X, Y_expected = gt_small.get("forward", 4, seed=9005, dtype=DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, np.asfortranarray(X)), Y_expected, atol=1e-5, rtol=1e-5)


def test_none_hint_emits_no_warning(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=None)
    X, _ = gt_small.get("forward", 3, seed=9101, dtype=DATA_DTYPE)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _run_up(op, X)
    assert len(rec) == 0


def test_warn_every_mismatch_call(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4)
    X1, _ = gt_small.get("forward", 1, seed=9102, dtype=DATA_DTYPE)
    X2, _ = gt_small.get("forward", 2, seed=9103, dtype=DATA_DTYPE)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _run_up(op, X1)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _run_up(op, X1)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        _run_up(op, X2)


def test_first_mismatch_call_matches_reference(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4, log_level="INFO")
    X, Y_expected = gt_small.get("forward", 1, seed=123, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        Y = _run_up(op, X)
    np.testing.assert_allclose(Y, Y_expected, atol=atol, rtol=rtol)


def test_repeated_mismatch_calls_match_reference(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4, log_level="INFO")
    X, Y_expected = gt_small.get("forward", 1, seed=123, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        Y_first = _run_up(op, X)
    with pytest.warns(RuntimeWarning, match="k_hint"):
        Y_second = _run_up(op, X)
    np.testing.assert_allclose(Y_first, Y_expected, atol=atol, rtol=rtol)
    np.testing.assert_allclose(Y_second, Y_expected, atol=atol, rtol=rtol)


def test_runtime_logging_reports_versions_and_warns_for_non_doc_cuda(caplog, monkeypatch):
    import pygrgl_spmv.backends.cusparse as cusparse_backend

    cusparse_backend.cusparse_plan._runtime_cuda_version.cache_clear()
    monkeypatch.setattr(cusparse_backend.cusparse_plan, "_probe_cuda_version", lambda: (12, 8, 0))
    try:
        with caplog.at_level(logging.INFO):
            backend = cusparse_backend.CusparseBackend(
                pair=cusparse_backend.CusparsePlanPair.from_dicts(
                    make_cusparse_plan(
                        k_hint=None,
                        store="N",
                        fmt="CSR",
                        op_a="N",
                        op_b="N",
                        order_b="ROW",
                        order_c="ROW",
                        algo="DEFAULT",
                    ),
                    None,
                ),
                log_level="INFO",
            )
    finally:
        cusparse_backend.cusparse_plan._runtime_cuda_version.cache_clear()
    try:
        messages = [rec.getMessage() for rec in caplog.records]
        assert any("CUDA runtime=12.8.0" in msg and "cuSPARSE=" in msg for msg in messages)
        assert any("CUDA 12.9.0" in msg and "12.8.0" in msg for msg in messages)
    finally:
        del backend


def test_runtime_logging_skips_warning_for_doc_cuda(caplog, monkeypatch):
    import pygrgl_spmv.backends.cusparse as cusparse_backend

    cusparse_backend.cusparse_plan._runtime_cuda_version.cache_clear()
    monkeypatch.setattr(cusparse_backend.cusparse_plan, "_probe_cuda_version", lambda: (12, 9, 0))
    try:
        with caplog.at_level(logging.INFO):
            backend = cusparse_backend.CusparseBackend(
                pair=cusparse_backend.CusparsePlanPair.from_dicts(
                    make_cusparse_plan(
                        k_hint=None,
                        store="N",
                        fmt="CSR",
                        op_a="N",
                        op_b="N",
                        order_b="ROW",
                        order_c="ROW",
                        algo="DEFAULT",
                    ),
                    None,
                ),
                log_level="INFO",
            )
    finally:
        cusparse_backend.cusparse_plan._runtime_cuda_version.cache_clear()
    try:
        messages = [rec.getMessage() for rec in caplog.records]
        assert any("CUDA runtime=12.9.0" in msg and "cuSPARSE=" in msg for msg in messages)
        assert not any("CusparsePlan semantics are grounded in the CUDA 12.9.0" in msg for msg in messages)
    finally:
        del backend


def test_cusparse_buffer_size_warning_on_plan_disagreement(primary_grg_path, monkeypatch, caplog):
    op = _make_op(primary_grg_path, fmt_up="csr", fmt_down=None, k_hint=None)
    backend = op._backend
    monkeypatch.setattr(type(backend._plan_up), "need_buffer", property(lambda self: False))
    monkeypatch.setattr(backend._cslib, "spmm_buffer_size", lambda *args: 8)
    monkeypatch.setattr(backend._cslib, "spmm_preprocess", lambda *args: None)
    with caplog.at_level(logging.WARNING):
        ws = backend._build_direction_workspace(Direction.UP, 1, use_graph_descs=False)
    try:
        messages = [rec.getMessage() for rec in caplog.records]
        assert any("bufferSize disagrees with plan.need_buffer" in msg for msg in messages)
    finally:
        ws.destroy(cslib=backend._cslib)


def test_cusparse_skips_preprocess_when_plan_disables_it(primary_grg_path, monkeypatch):
    op = _make_op(primary_grg_path, fmt_up="csr", fmt_down=None, k_hint=None)
    backend = op._backend
    monkeypatch.setattr(type(backend._plan_up), "need_preprocess", property(lambda self: False))
    monkeypatch.setattr(backend._cslib, "spmm_buffer_size", lambda *args: 8)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(backend._cslib, "spmm_preprocess", lambda *args: calls.append(args))
    ws = backend._build_direction_workspace(Direction.UP, 1, use_graph_descs=False)
    try:
        assert calls == []
    finally:
        ws.destroy(cslib=backend._cslib)


def test_cusparse_runtime_loads_after_torch_import():
    pytest.importorskip("torch")
    from pygrgl_spmv.backends.cusparse.ffi import CuSparseLib

    lib = CuSparseLib()
    try:
        assert lib.version
    finally:
        lib.destroy()


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
def test_primary_grg_exact_binary_forward(primary_grg_path, gt_primary, fmt):
    op = _make_op(primary_grg_path, fmt_up=fmt, k_hint=None)
    X, Y_expected = gt_primary.get("forward", 4, seed=8001, dtype=DATA_DTYPE, input_fn=binary_pm1)
    np.testing.assert_array_equal(_run_up(op, X), Y_expected)


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
def test_primary_grg_exact_binary_backward(primary_grg_path, gt_primary, fmt):
    op = _make_op(primary_grg_path, fmt_up=fmt, k_hint=None)
    X, Y_expected = gt_primary.get("backward", 4, seed=8001, dtype=DATA_DTYPE, input_fn=binary_pm1)
    np.testing.assert_array_equal(_run_down(op, X), Y_expected)
