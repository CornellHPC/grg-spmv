"""I/O utilities for loading/saving SpmvGRG state."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

NPZ_SCHEMA_VERSION = 2


def save_operator_npz(op, A_blocks, npz_path):
    """
    Save SpmvGRG state to NPZ file for fast loading.

    Parameters
    ----------
    op : SpmvGRG
        The operator to save (for metadata).
    A_blocks : list of list of sp.csr_matrix
        Forward blocks: A_blocks[h][j] for j in range(h).
    npz_path : str or Path
        Path to save the NPZ file.
    """
    save_dict = {
        "npz_schema_version": NPZ_SCHEMA_VERSION,
        "n": op.n,
        "m": op.m,
        "K": op.K,
        "ploidy": op.ploidy,
        "num_individuals": op.num_individuals,
        "level_offsets": op.level_offsets,
        "sample_perm": op.sample_perm,
        "node_perm": op.node_perm,
        "sample_to_individual": op.sample_to_individual,
        "has_individual_coals": bool(op.coalescence_counts is not None),
        "init_vector_up_bias": op._init_vector_up_bias,
        "init_vector_down_bias": op._init_vector_down_bias,
        "has_xtx_bias": bool(op._init_xtx_up_bias is not None and op._init_xtx_down_bias is not None),
        "sel_mut_indices": op.sel_mut.indices,
        "sel_mut_indptr": op.sel_mut.indptr,
        "sel_miss_indices": op.sel_miss.indices,
        "sel_miss_indptr": op.sel_miss.indptr,
    }
    if op.coalescence_counts is not None:
        save_dict["coalescence_counts"] = op.coalescence_counts
    if op._init_xtx_up_bias is not None:
        save_dict["init_xtx_up_bias"] = op._init_xtx_up_bias
    if op._init_xtx_down_bias is not None:
        save_dict["init_xtx_down_bias"] = op._init_xtx_down_bias
    for h, blocks in enumerate(A_blocks):
        for j, A in enumerate(blocks):
            save_dict[f"A_blocks_{h}_{j}_indices"] = A.indices
            save_dict[f"A_blocks_{h}_{j}_indptr"] = A.indptr
            save_dict[f"A_blocks_{h}_{j}_shape"] = np.array(A.shape, dtype=np.int64)
    np.savez(npz_path, **save_dict)


def load_operator_npz(npz_path, dtype, index_dtype):
    """
    Load SpmvGRG state from NPZ file.

    Parameters
    ----------
    npz_path : str or Path
        Path to the NPZ file.
    dtype : numpy dtype
        Data type for sparse matrix values.
    index_dtype : numpy dtype
        Index dtype for permutation arrays.

    Returns
    -------
    dict
        Dictionary containing operator attributes.
    """
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

    result = {
        "n": int(data["n"]),
        "m": int(data["m"]),
        "K": int(data["K"]),
        "ploidy": int(data["ploidy"]),
        "num_individuals": int(data["num_individuals"]),
        "level_offsets": np.asarray(data["level_offsets"], dtype=index_dtype),
        "sample_perm": np.asarray(data["sample_perm"], dtype=index_dtype),
        "node_perm": np.asarray(data["node_perm"], dtype=index_dtype),
        "sample_to_individual": np.asarray(data["sample_to_individual"], dtype=index_dtype),
        "has_individual_coals": bool(data["has_individual_coals"]),
        "init_vector_up_bias": np.asarray(data["init_vector_up_bias"], dtype=dtype),
        "init_vector_down_bias": np.asarray(data["init_vector_down_bias"], dtype=dtype),
        "has_xtx_bias": bool(data["has_xtx_bias"]),
    }
    if result["has_individual_coals"]:
        result["coalescence_counts"] = np.asarray(data["coalescence_counts"], dtype=np.int64)
    else:
        result["coalescence_counts"] = None
    if result["has_xtx_bias"]:
        result["init_xtx_up_bias"] = np.asarray(data["init_xtx_up_bias"], dtype=dtype)
        result["init_xtx_down_bias"] = np.asarray(data["init_xtx_down_bias"], dtype=dtype)
    else:
        result["init_xtx_up_bias"] = None
        result["init_xtx_down_bias"] = None

    inv_node_perm = np.empty(result["K"], dtype=index_dtype)
    inv_node_perm[result["node_perm"]] = np.arange(result["K"], dtype=index_dtype)
    result["_inv_node_perm"] = inv_node_perm

    inv_sample_perm = np.empty(result["n"], dtype=index_dtype)
    inv_sample_perm[result["sample_perm"]] = np.arange(result["n"], dtype=index_dtype)
    result["_inv_sample_perm"] = inv_sample_perm

    sel_mut_nnz = len(data["sel_mut_indices"])
    result["sel_mut"] = sp.csr_matrix(
        (
            np.ones(sel_mut_nnz, dtype=dtype),
            data["sel_mut_indices"].astype(index_dtype, copy=False),
            data["sel_mut_indptr"].astype(index_dtype, copy=False),
        ),
        shape=(result["m"], result["K"]),
    )

    sel_miss_nnz = len(data["sel_miss_indices"])
    result["sel_miss"] = sp.csr_matrix(
        (
            np.ones(sel_miss_nnz, dtype=dtype),
            data["sel_miss_indices"].astype(index_dtype, copy=False),
            data["sel_miss_indptr"].astype(index_dtype, copy=False),
        ),
        shape=(result["m"], result["K"]),
    )

    # Reconstruct list-of-lists: A_blocks[h] has h blocks.
    n_levels = len(result["level_offsets"]) - 1
    result["A_blocks"] = []
    for h in range(n_levels):
        level_blocks = []
        for j in range(h):
            indices = data[f"A_blocks_{h}_{j}_indices"].astype(index_dtype, copy=False)
            indptr = data[f"A_blocks_{h}_{j}_indptr"].astype(index_dtype, copy=False)
            shape = tuple(data[f"A_blocks_{h}_{j}_shape"])
            nnz = len(indices)
            level_blocks.append(
                sp.csr_matrix(
                    (np.ones(nnz, dtype=dtype), indices, indptr),
                    shape=shape,
                )
            )
        result["A_blocks"].append(level_blocks)

    return result
