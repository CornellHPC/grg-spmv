"""Public backend exports."""

from pygrgl_spmv.backends.base import (
    BackendBase,
    CallCapture,
    BackendSetup,
    _parse_optional_k_hint,
    effective_k_hint,
    iter_direction_level_pairs,
    selector_rows_unique_from_csr_indptr,
    warn_instrumentation_ignores_k_hint,
    warn_k_hint_mismatch,
)
from pygrgl_spmv.backends.reference import ReferenceBackend, ReferencePlan, ReferencePlanPair

__all__ = [
    "BackendBase",
    "CallCapture",
    "BackendSetup",
    "ReferenceBackend",
    "ReferencePlan",
    "ReferencePlanPair",
    "_parse_optional_k_hint",
    "effective_k_hint",
    "iter_direction_level_pairs",
    "selector_rows_unique_from_csr_indptr",
    "warn_instrumentation_ignores_k_hint",
    "warn_k_hint_mismatch",
]
