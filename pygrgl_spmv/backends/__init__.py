"""Public backend exports."""

from pygrgl_spmv.backends.base import (
    BackendBase,
    BackendSetup,
    _parse_optional_k_hint,
    _sparse_host_bytes,
    build_wavefront_level_stats,
    estimate_common_host_static_bytes,
    estimate_sparse_payload_bytes,
    iter_direction_level_pairs,
    log_wavefront_profile,
    selector_rows_unique_from_csr_indptr,
    warn_k_hint_mismatch,
)
from pygrgl_spmv.backends.memory import MemoryUsage
from pygrgl_spmv.backends.reference import ReferenceBackend, ReferencePlan

__all__ = [
    "BackendBase",
    "BackendSetup",
    "MemoryUsage",
    "ReferenceBackend",
    "ReferencePlan",
    "_parse_optional_k_hint",
    "_sparse_host_bytes",
    "build_wavefront_level_stats",
    "estimate_common_host_static_bytes",
    "estimate_sparse_payload_bytes",
    "iter_direction_level_pairs",
    "log_wavefront_profile",
    "selector_rows_unique_from_csr_indptr",
    "warn_k_hint_mismatch",
]
