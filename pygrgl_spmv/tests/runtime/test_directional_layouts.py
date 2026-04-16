from __future__ import annotations

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv.backends.mkl import MklPlan, MklPlanPair
from pygrgl_spmv.backends.reference import ReferencePlan, ReferencePlanPair
from pygrgl_spmv.tests.conftest import DATA_DTYPE, tol
from pygrgl_spmv.tests.runtime._runtime_builders import build_layout_for_backend, full_requirements, runtime_cls_for_backend

_BACKENDS = [
    pytest.param("reference", id="reference"),
    pytest.param("mkl", id="mkl", marks=pytest.mark.mkl),
    pytest.param("cusparse", id="cusparse", marks=[pytest.mark.gpu, pytest.mark.cusparse]),
    pytest.param("triton", id="triton", marks=[pytest.mark.gpu, pytest.mark.triton]),
]


def _up_only_pair(backend_name: str):
    match backend_name:
        case "reference":
            return ReferencePlanPair(plan_up=ReferencePlan(store="N", fmt="CSR"), plan_down=None)
        case "mkl":
            return MklPlanPair(plan_up=MklPlan(store="N", fmt="CSR", n_threads=1), plan_down=None)
        case "triton":
            from pygrgl_spmv.backends.triton import TritonPlan, TritonPlanPair

            return TritonPlanPair(plan_up=TritonPlan(store="N", fmt="CSR", scratch="none"), plan_down=None)
        case "cusparse":
            from pygrgl_spmv.backends.cusparse import CusparsePlan, CusparsePlanPair

            return CusparsePlanPair(
                plan_up=CusparsePlan(store="N", fmt="CSR", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"),
                plan_down=None,
            )
        case _:
            raise ValueError(backend_name)


def _down_only_pair(backend_name: str):
    match backend_name:
        case "reference":
            return ReferencePlanPair(plan_up=None, plan_down=ReferencePlan(store="T", fmt="CSC"))
        case "mkl":
            return MklPlanPair(plan_up=None, plan_down=MklPlan(store="T", fmt="CSC", n_threads=1))
        case "triton":
            from pygrgl_spmv.backends.triton import TritonPlan, TritonPlanPair

            return TritonPlanPair(plan_up=None, plan_down=TritonPlan(store="T", fmt="CSC", scratch="none"))
        case "cusparse":
            from pygrgl_spmv.backends.cusparse import CusparsePlan, CusparsePlanPair

            return CusparsePlanPair(
                plan_up=None,
                plan_down=CusparsePlan(store="T", fmt="CSC", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"),
            )
        case _:
            raise ValueError(backend_name)


def _nonsharing_full_pair(backend_name: str):
    match backend_name:
        case "reference":
            return ReferencePlanPair(
                plan_up=ReferencePlan(store="N", fmt="CSR"),
                plan_down=ReferencePlan(store="T", fmt="COO"),
            )
        case "mkl":
            return MklPlanPair(
                plan_up=MklPlan(store="N", fmt="CSR", n_threads=1),
                plan_down=MklPlan(store="T", fmt="COO", n_threads=1),
            )
        case "triton":
            from pygrgl_spmv.backends.triton import TritonPlan, TritonPlanPair

            return TritonPlanPair(
                plan_up=TritonPlan(store="N", fmt="CSR", scratch="none"),
                plan_down=TritonPlan(store="T", fmt="CSR", scratch="none"),
            )
        case "cusparse":
            from pygrgl_spmv.backends.cusparse import CusparsePlan, CusparsePlanPair

            return CusparsePlanPair(
                plan_up=CusparsePlan(store="N", fmt="COO", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"),
                plan_down=CusparsePlan(store="T", fmt="COO", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"),
            )
        case _:
            raise ValueError(backend_name)


def _requirements_for(_backend_name: str, *, k: int):
    return full_requirements(max_k_up=int(k), max_k_down=int(k))


@pytest.mark.parametrize("backend_name", _BACKENDS)
@pytest.mark.parametrize("direction", ["up", "down"], ids=["up", "down"])
def test_one_sided_layout_runs_enabled_direction(primary_artifact, primary_grg, backend_name, direction):
    pair = _up_only_pair(backend_name) if direction == "up" else _down_only_pair(backend_name)
    k = 4
    layout = build_layout_for_backend(
        backend_name,
        [primary_artifact],
        pair=pair,
        requirements=_requirements_for(backend_name, k=k),
    )
    runtime_cls = runtime_cls_for_backend(backend_name)
    cols = primary_grg.num_samples if direction == "up" else primary_grg.num_mutations
    with runtime_cls(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(8400 if direction == "up" else 8401)
        x = rng.standard_normal((k, cols), dtype=DATA_DTYPE)
        expected = np.asarray(pygrgl.matmul(primary_grg, x, pygrgl.TraversalDirection.UP if direction == "up" else pygrgl.TraversalDirection.DOWN))
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, direction), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("backend_name", _BACKENDS)
def test_disabled_direction_raises(primary_artifact, backend_name):
    k = 2
    layout = build_layout_for_backend(
        backend_name,
        [primary_artifact],
        pair=_up_only_pair(backend_name),
        requirements=_requirements_for(backend_name, k=k),
    )
    runtime_cls = runtime_cls_for_backend(backend_name)
    with runtime_cls(layout) as runtime:
        (grg,) = runtime.grgs
        with pytest.raises(ValueError, match="not configured"):
            grg.matmul(np.ones((k, grg.num_mutations), dtype=DATA_DTYPE), "down")


@pytest.mark.parametrize("backend_name", _BACKENDS)
def test_one_sided_layout_drops_unused_storage(primary_artifact, backend_name):
    k = 4
    full_layout = build_layout_for_backend(
        backend_name,
        [primary_artifact],
        pair=_nonsharing_full_pair(backend_name),
        requirements=_requirements_for(backend_name, k=k),
    )
    up_only_layout = build_layout_for_backend(
        backend_name,
        [primary_artifact],
        pair=_up_only_pair(backend_name),
        requirements=_requirements_for(backend_name, k=k),
    )
    assert up_only_layout.bytes_total < full_layout.bytes_total
    assert up_only_layout.bytes_by_category["workspace_down"] == 0
    assert up_only_layout.bytes_by_category["resident_sparse"] < full_layout.bytes_by_category["resident_sparse"]


@pytest.mark.parametrize("backend_name", _BACKENDS)
@pytest.mark.parametrize("direction", ["up", "down"], ids=["up", "down"])
def test_runtime_k_below_declared_max_supports_matrix_init(primary_artifact, primary_grg, backend_name, direction):
    max_k = 4
    runtime_k = 2
    layout = build_layout_for_backend(
        backend_name,
        [primary_artifact],
        requirements=full_requirements(max_k_up=max_k, max_k_down=max_k),
    )
    runtime_cls = runtime_cls_for_backend(backend_name)
    cols = primary_grg.num_samples if direction == "up" else primary_grg.num_mutations
    with runtime_cls(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(9100 if direction == "up" else 9101)
        x = rng.standard_normal((runtime_k, cols), dtype=DATA_DTYPE)
        init = rng.standard_normal((runtime_k, primary_grg.num_nodes), dtype=DATA_DTYPE)
        expected = np.asarray(pygrgl.matmul(primary_grg, x, pygrgl.TraversalDirection.UP if direction == "up" else pygrgl.TraversalDirection.DOWN, init=init))
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, direction, init=init), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("backend_name", _BACKENDS)
@pytest.mark.parametrize("direction", ["up", "down"], ids=["up", "down"])
def test_runtime_k_below_declared_max_supports_emit_all_nodes_vector_init(primary_artifact, primary_grg, backend_name, direction):
    max_k = 4
    runtime_k = 1
    layout = build_layout_for_backend(
        backend_name,
        [primary_artifact],
        requirements=full_requirements(max_k_up=max_k, max_k_down=max_k),
    )
    runtime_cls = runtime_cls_for_backend(backend_name)
    cols = primary_grg.num_samples if direction == "up" else primary_grg.num_mutations
    with runtime_cls(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(9200 if direction == "up" else 9201)
        x = rng.standard_normal((runtime_k, cols), dtype=DATA_DTYPE)
        init = rng.standard_normal((runtime_k,), dtype=DATA_DTYPE)
        expected = np.asarray(pygrgl.matmul(primary_grg, x, pygrgl.TraversalDirection.UP if direction == "up" else pygrgl.TraversalDirection.DOWN, init=init, emit_all_nodes=True))
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, direction, init=init, emit_all_nodes=True), expected, atol=atol, rtol=rtol)
