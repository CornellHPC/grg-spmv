"""Public backend exports."""

from pygrgl_spmv.backends.base import (
    BackendBase,
    BackendSetup,
    _parse_optional_k_hint,
    _sparse_host_bytes,
    effective_k_hint,
    estimate_common_host_static_bytes,
    estimate_sparse_payload_bytes,
    iter_direction_level_pairs,
    selector_rows_unique_from_csr_indptr,
    warn_instrumentation_ignores_k_hint,
    warn_k_hint_mismatch,
)
from pygrgl_spmv.backends.memory import MemoryUsage
from pygrgl_spmv.backends.reference import ReferenceBackend, ReferencePlan, ReferencePlanPair

__all__ = [
    "BackendBase",
    "BackendSetup",
    "MemoryUsage",
    "ReferenceBackend",
    "ReferencePlan",
    "ReferencePlanPair",
    "_parse_optional_k_hint",
    "_sparse_host_bytes",
    "effective_k_hint",
    "estimate_common_host_static_bytes",
    "estimate_sparse_payload_bytes",
    "iter_direction_level_pairs",
    "selector_rows_unique_from_csr_indptr",
    "warn_instrumentation_ignores_k_hint",
    "warn_k_hint_mismatch",
]
