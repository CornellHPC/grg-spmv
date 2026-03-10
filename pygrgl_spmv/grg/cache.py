"""Cache path resolution and NPZ persistence for compiled operators."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pygrgl_spmv.grg.compile import CompiledOperatorState, _invert_permutation
from pygrgl_spmv.grg.sparse import binary_csr_from_csr_parts

NPZ_SCHEMA_VERSION = 2


def cache_path_for_grg(grg_path: Path, cache_root: Path) -> Path:
    resolved = grg_path.expanduser().resolve()
    if resolved.is_absolute():
        if resolved.drive:
            drive = resolved.drive.replace(":", "")
            rel_parts = ["_drive_" + drive, *resolved.parts[1:]]
        else:
            rel_parts = ["_abs", *resolved.parts[1:]]
    else:
        rel_parts = ["_rel", *resolved.parts]
    return cache_root.joinpath(*rel_parts).with_suffix(".pygrgl_spmv.npz")


def save_operator_npz(state: CompiledOperatorState, npz_path) -> None:
    """Save compiled operator state to NPZ with the existing schema."""
    if state.A_blocks is None:
        raise RuntimeError("Compiled operator blocks are required for NPZ save")
    if state.init_vector_up_bias is None or state.init_vector_down_bias is None:
        raise RuntimeError("Init vector bias cache must be populated before NPZ save")

    save_dict = {
        "npz_schema_version": NPZ_SCHEMA_VERSION,
        "n": state.n,
        "m": state.m,
        "K": state.K,
        "ploidy": state.ploidy,
        "num_individuals": state.num_individuals,
        "level_offsets": state.level_offsets,
        "sample_perm": state.sample_perm,
        "node_perm": state.node_perm,
        "sample_to_individual": state.sample_to_individual,
        "has_individual_coals": bool(state.coalescence_counts is not None),
        "init_vector_up_bias": state.init_vector_up_bias,
        "init_vector_down_bias": state.init_vector_down_bias,
        "has_xtx_bias": bool(state.init_xtx_up_bias is not None and state.init_xtx_down_bias is not None),
        "sel_mut_indices": state.sel_mut.indices,
        "sel_mut_indptr": state.sel_mut.indptr,
        "sel_miss_indices": state.sel_miss.indices,
        "sel_miss_indptr": state.sel_miss.indptr,
    }
    if state.coalescence_counts is not None:
        save_dict["coalescence_counts"] = state.coalescence_counts
    if state.init_xtx_up_bias is not None:
        save_dict["init_xtx_up_bias"] = state.init_xtx_up_bias
    if state.init_xtx_down_bias is not None:
        save_dict["init_xtx_down_bias"] = state.init_xtx_down_bias
    for h, blocks in enumerate(state.A_blocks):
        for j, A in enumerate(blocks):
            save_dict[f"A_blocks_{h}_{j}_indices"] = A.indices
            save_dict[f"A_blocks_{h}_{j}_indptr"] = A.indptr
            save_dict[f"A_blocks_{h}_{j}_shape"] = np.array(A.shape, dtype=np.int64)
    np.savez(npz_path, **save_dict)


def load_operator_npz(npz_path, dtype, index_dtype) -> CompiledOperatorState:
    """Load a compiled operator state from the existing NPZ schema."""
    data = np.load(npz_path, allow_pickle=False)
    if "npz_schema_version" not in data.files:
        raise ValueError(
            f"Unsupported SpmvGRG NPZ schema in {npz_path}: missing npz_schema_version; "
            f"expected {NPZ_SCHEMA_VERSION}"
        )
    schema_version = data["npz_schema_version"]
    if int(schema_version) != NPZ_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported SpmvGRG NPZ schema in {npz_path}: got {int(schema_version)}, "
            f"expected {NPZ_SCHEMA_VERSION}"
        )

    n = int(data["n"])
    m = int(data["m"])
    K = int(data["K"])
    level_offsets = np.asarray(data["level_offsets"], dtype=index_dtype)
    sample_perm = np.asarray(data["sample_perm"], dtype=index_dtype)
    node_perm = np.asarray(data["node_perm"], dtype=index_dtype)
    sample_to_individual = np.asarray(data["sample_to_individual"], dtype=index_dtype)
    has_individual_coals = bool(data["has_individual_coals"])
    init_vector_up_bias = np.asarray(data["init_vector_up_bias"], dtype=dtype)
    init_vector_down_bias = np.asarray(data["init_vector_down_bias"], dtype=dtype)
    has_xtx_bias = bool(data["has_xtx_bias"])

    coalescence_counts = None
    if has_individual_coals:
        coalescence_counts = np.asarray(data["coalescence_counts"], dtype=np.int64)

    init_xtx_up_bias = None
    init_xtx_down_bias = None
    if has_xtx_bias:
        init_xtx_up_bias = np.asarray(data["init_xtx_up_bias"], dtype=dtype)
        init_xtx_down_bias = np.asarray(data["init_xtx_down_bias"], dtype=dtype)

    inv_node_perm = _invert_permutation(node_perm, index_dtype=index_dtype)
    inv_sample_perm = _invert_permutation(sample_perm, index_dtype=index_dtype)

    sel_mut = binary_csr_from_csr_parts(
        indices=data["sel_mut_indices"],
        indptr=data["sel_mut_indptr"],
        shape=(m, K),
        dtype=dtype,
        index_dtype=index_dtype,
    )
    sel_miss = binary_csr_from_csr_parts(
        indices=data["sel_miss_indices"],
        indptr=data["sel_miss_indptr"],
        shape=(m, K),
        dtype=dtype,
        index_dtype=index_dtype,
    )

    n_levels = len(level_offsets) - 1
    A_blocks: list[list[object]] = []
    for h in range(n_levels):
        level_blocks = []
        for j in range(h):
            level_blocks.append(
                binary_csr_from_csr_parts(
                    indices=data[f"A_blocks_{h}_{j}_indices"],
                    indptr=data[f"A_blocks_{h}_{j}_indptr"],
                    shape=data[f"A_blocks_{h}_{j}_shape"],
                    dtype=dtype,
                    index_dtype=index_dtype,
                )
            )
        A_blocks.append(level_blocks)

    return CompiledOperatorState(
        A_blocks=A_blocks,
        level_offsets=level_offsets,
        node_perm=node_perm,
        inv_node_perm=inv_node_perm,
        sample_perm=sample_perm,
        inv_sample_perm=inv_sample_perm,
        sel_mut=sel_mut,
        sel_miss=sel_miss,
        n=n,
        m=m,
        K=K,
        ploidy=int(data["ploidy"]),
        num_individuals=int(data["num_individuals"]),
        sample_to_individual=sample_to_individual,
        coalescence_counts=coalescence_counts,
        init_vector_up_bias=init_vector_up_bias,
        init_vector_down_bias=init_vector_down_bias,
        init_xtx_up_bias=init_xtx_up_bias,
        init_xtx_down_bias=init_xtx_down_bias,
    )


__all__ = ["NPZ_SCHEMA_VERSION", "cache_path_for_grg", "load_operator_npz", "save_operator_npz"]
