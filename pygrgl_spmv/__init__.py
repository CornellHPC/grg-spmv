"""Runtime-owned GRG sparse matmul package."""

from pygrgl_spmv.backends.mkl import MklPlan, MklPlanPair, MklRuntime, plan_mkl_layout
from pygrgl_spmv.backends.reference import ReferencePlan, ReferencePlanPair, ReferenceRuntime, plan_reference_layout
from pygrgl_spmv.grg import RuntimeRequirements, convert
from pygrgl_spmv.adaptor import (
    CapturedBoundGRG,
    CaptureSpec,
    RunConfigs,
    MklBackendConfig,
    CusparseBackendConfig,
    make_backend_mkl,
    make_backend_cusparse,
    make_runconfig_matmul,
    make_runconfig_pca,
    load_grg_spmv_single,
    load_grg_spmv_multi,
)

__all__ = [
    "MklPlan",
    "MklPlanPair",
    "MklRuntime",
    "ReferencePlan",
    "ReferencePlanPair",
    "ReferenceRuntime",
    "RuntimeRequirements",
    "convert",
    "plan_mkl_layout",
    "plan_reference_layout",
    "CapturedBoundGRG",
    "CaptureSpec",
    "RunConfigs",
    "MklBackendConfig",
    "CusparseBackendConfig",
    "make_backend_mkl",
    "make_backend_cusparse",
    "make_runconfig_matmul",
    "make_runconfig_pca",
    "load_grg_spmv_single",
    "load_grg_spmv_multi",
]
