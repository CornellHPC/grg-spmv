"""cuSPARSE backend tests."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.backends.cusparse import is_valid_combo
from pygrgl_spmv.tests.conftest import (
    DATA_DTYPE,
    INDEX_DTYPE,
    K_CORE,
    K_MATRIX,
    binary_pm1,
    make_backend_config,
    make_cusparse_config,
    make_cusparse_plan,
    make_fmt_algo_params,
    tol,
    valid_fmt_algo_params,
)

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu

_CACHE_DIR = Path(".pytest_cache") / "pygrgl_spmv_npz"


def _make_op(
    grg_path,
    *,
    fmt_up="csr",
    fmt_down=None,
    k_hint=None,
    algo_up="default",
    algo_down="default",
    log_level="WARNING",
    dtype=DATA_DTYPE,
    cache_dir=_CACHE_DIR,
):
    return SpmvGRG(
        grg_path,
        make_cusparse_config(
            fmt_up=fmt_up,
            fmt_down=fmt_down,
            k_hint=k_hint,
            algo_up=algo_up,
            algo_down=algo_down,
            log_level=log_level,
        ),
        dtype,
        INDEX_DTYPE,
        cache_dir=cache_dir,
    )


def _run_up(op, X_col_major):
    return op.matmul(X_col_major.T, "up").T


def _run_down(op, X_col_major):
    return op.matmul(X_col_major.T, "down").T


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
        make_backend_config("cusparse", plan_up=plan_up, plan_down=None, log_level="WARNING"),
        DATA_DTYPE,
        INDEX_DTYPE,
        cache_dir=_CACHE_DIR,
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
        make_backend_config("cusparse", plan_up=None, plan_down=plan_down, log_level="WARNING"),
        DATA_DTYPE,
        INDEX_DTYPE,
        cache_dir=_CACHE_DIR,
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


@pytest.mark.parametrize("fmt,algo", make_fmt_algo_params())
def test_construct_for_fmt_algo(primary_grg_path, fmt, algo):
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
            make_backend_config(
                "cusparse",
                plan_up=make_cusparse_plan(
                    k_hint=None,
                    store="N",
                    fmt="CSR",
                    op_a="N",
                    op_b="N",
                    order_b="ROW",
                    order_c="ROW",
                    algo="DEFAULT",
                ),
                plan_down=make_cusparse_plan(
                    k_hint=None,
                    store="N",
                    fmt="CSR",
                    op_a="T",
                    op_b="N",
                    order_b="ROW",
                    order_c="ROW",
                    algo="CSR_ALG3",
                ),
                log_level="WARNING",
            ),
            DATA_DTYPE,
            INDEX_DTYPE,
            cache_dir=_CACHE_DIR,
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


@pytest.mark.parametrize("fmt,algo", make_fmt_algo_params())
def test_forward_dynamic(primary_grg_path, gt_small, fmt, algo):
    op = _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=None, algo_up=algo, algo_down=algo)
    X, Y_expected = gt_small.get("forward", 4, seed=42, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt,algo", make_fmt_algo_params())
def test_backward_dynamic(primary_grg_path, gt_small, fmt, algo):
    op = _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=None, algo_up=algo, algo_down=algo)
    X, Y_expected = gt_small.get("backward", 4, seed=42, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt,algo", make_fmt_algo_params())
def test_forward_graph(primary_grg_path, gt_small, fmt, algo):
    op = _make_op(primary_grg_path, fmt_up=fmt, fmt_down=fmt, k_hint=4, algo_up=algo, algo_down=algo)
    X, Y_expected = gt_small.get("forward", 4, seed=52, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(_run_up(op, X), Y_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt,algo", make_fmt_algo_params())
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
    ws = op._backend._workspaces.dynamic
    assert ws is not None
    assert op._backend._workspaces.static_up is None
    assert op._backend._workspaces.static_down is None
    assert ws.k == 4
    assert ws.up.graph is None
    assert ws.down.graph is None


def test_dynamic_workspace_reused_for_same_k(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4)
    X1, _ = gt_small.get("forward", 1, seed=5006, dtype=DATA_DTYPE)
    _ = _run_up(op, X1)
    ws1 = op._backend._workspaces.dynamic
    _ = _run_up(op, X1)
    ws2 = op._backend._workspaces.dynamic
    assert ws1 is ws2
    assert ws2 is not None
    assert ws2.k == 1


def test_dynamic_workspace_recreated_when_k_changes(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4)
    X1, _ = gt_small.get("forward", 1, seed=5007, dtype=DATA_DTYPE)
    X2, _ = gt_small.get("forward", 2, seed=5008, dtype=DATA_DTYPE)
    _ = _run_up(op, X1)
    ws1 = op._backend._workspaces.dynamic
    _ = _run_up(op, X2)
    ws2 = op._backend._workspaces.dynamic
    assert ws1 is not ws2
    assert ws2 is not None
    assert ws2.k == 2


def test_graph_fallback_preserves_static_graph(primary_grg_path, gt_small):
    op = _make_op(primary_grg_path, fmt_up="csr", k_hint=4)
    X4, Y4_exp = gt_small.get("backward", 4, seed=5001, dtype=DATA_DTYPE)
    X1, Y1_exp = gt_small.get("backward", 1, seed=5002, dtype=DATA_DTYPE)
    np.testing.assert_allclose(_run_down(op, X4), Y4_exp, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_run_down(op, X1), Y1_exp, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_run_down(op, X4), Y4_exp, atol=1e-5, rtol=1e-5)
    assert op._backend._workspaces.static_down is not None
    assert op._backend._workspaces.static_down.down.graph is not None


def test_graph_vs_dynamic(primary_grg_path):
    op_graph = _make_op(primary_grg_path, fmt_up="csr", k_hint=4)
    op_dynamic = _make_op(primary_grg_path, fmt_up="csr", k_hint=None)
    rng = np.random.default_rng(7001)
    V = rng.standard_normal((op_graph.n, 4), dtype=DATA_DTYPE)
    W = rng.standard_normal((op_graph.m, 4), dtype=DATA_DTYPE)
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


def test_runtime_logging_reports_versions_and_warns_for_non_doc_cuda(caplog, monkeypatch):
    import pygrgl_spmv.backends.cusparse as cusparse_backend

    cusparse_backend.cusparse_plan._runtime_cuda_version.cache_clear()
    monkeypatch.setattr(cusparse_backend.cusparse_plan, "_probe_cuda_version", lambda: (12, 8, 0))
    try:
        with caplog.at_level(logging.INFO):
            backend = cusparse_backend.CusparseBackend(
                plan_up=make_cusparse_plan(
                    k_hint=None,
                    store="N",
                    fmt="CSR",
                    op_a="N",
                    op_b="N",
                    order_b="ROW",
                    order_c="ROW",
                    algo="DEFAULT",
                ),
                plan_down=None,
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
                plan_up=make_cusparse_plan(
                    k_hint=None,
                    store="N",
                    fmt="CSR",
                    op_a="N",
                    op_b="N",
                    order_b="ROW",
                    order_c="ROW",
                    algo="DEFAULT",
                ),
                plan_down=None,
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
