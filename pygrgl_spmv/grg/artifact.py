"""Standalone `.grg_spmv` artifact persistence for compiled operators."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pygrgl_spmv.grg.compile import (
    CompiledOperatorState,
    VALID_INTRA_BLOCK_ORDERINGS,
    VALID_ORDERINGS,
    _invert_permutation,
)
from pygrgl_spmv.grg.sparse import binary_csr_from_csr_parts

GRG_SPMV_FORMAT_MAGIC = "grg_spmv"
GRG_SPMV_FORMAT_VERSION = 3
_FORMAT_MAGIC_KEY = "grg_spmv_magic"
_FORMAT_VERSION_KEY = "grg_spmv_format_version"


def artifact_path_for_grg(
    grg_path: Path,
    artifact_root: Path,
    *,
    ordering: str,
    intra_block_ordering: str,
) -> Path:
    resolved = grg_path.expanduser().resolve()
    if resolved.is_absolute():
        if resolved.drive:
            drive = resolved.drive.replace(":", "")
            rel_parts = ["_drive_" + drive, *resolved.parts[1:]]
        else:
            rel_parts = ["_abs", *resolved.parts[1:]]
    else:
        rel_parts = ["_rel", *resolved.parts]
    suffix = f".order-{ordering}.intra-{intra_block_ordering}.grg_spmv"
    return artifact_root.joinpath(*rel_parts).with_suffix(suffix)


def save_grg_spmv(state: CompiledOperatorState, artifact_path) -> None:
    if state.A_blocks is None:
        raise RuntimeError("Compiled operator blocks are required for .grg_spmv save")
    if state.init_vector_up_bias is None or state.init_vector_down_bias is None:
        raise RuntimeError("Init vector bias cache must be populated before .grg_spmv save")

    index_dtype = np.asarray(state.level_offsets).dtype
    if index_dtype not in {np.dtype(np.int32), np.dtype(np.int64)}:
        raise ValueError(f"Unsupported artifact index dtype: {index_dtype}")

    save_dict = {
        _FORMAT_MAGIC_KEY: np.asarray(GRG_SPMV_FORMAT_MAGIC),
        _FORMAT_VERSION_KEY: np.asarray(GRG_SPMV_FORMAT_VERSION, dtype=np.int32),
        "num_samples": np.asarray(state.num_samples, dtype=np.int64),
        "num_mutations": np.asarray(state.num_mutations, dtype=np.int64),
        "num_nodes": np.asarray(state.num_nodes, dtype=np.int64),
        "ploidy": np.asarray(state.ploidy, dtype=np.int64),
        "num_individuals": np.asarray(state.num_individuals, dtype=np.int64),
        "num_edges": np.asarray(state.num_edges, dtype=np.int64),
        "has_missing_data": np.asarray(state.has_missing_data, dtype=bool),
        "ordering": np.asarray(state.ordering),
        "intra_block_ordering": np.asarray(state.intra_block_ordering),
        "level_offsets": np.asarray(state.level_offsets, dtype=index_dtype),
        "node_perm": np.asarray(state.node_perm, dtype=index_dtype),
        "sample_rows": np.asarray(state.sample_rows, dtype=index_dtype),
        "sample_to_individual": np.asarray(state.sample_to_individual, dtype=index_dtype),
        "mutation_positions": np.asarray(state.mutation_positions, dtype=np.float64),
        "mutation_times": np.asarray(state.mutation_times, dtype=np.float64),
        "mutation_alleles": np.asarray(state.mutation_alleles),
        "mutation_allele_offsets": np.asarray(state.mutation_allele_offsets),
        "mutation_ref_alleles": np.asarray(state.mutation_ref_alleles),
        "mutation_ref_allele_offsets": np.asarray(state.mutation_ref_allele_offsets),
        "has_individual_coals": np.asarray(state.coalescence_counts is not None, dtype=bool),
        "init_vector_up_bias": np.asarray(state.init_vector_up_bias),
        "init_vector_down_bias": np.asarray(state.init_vector_down_bias),
        "has_xtx_bias": np.asarray(
            state.init_xtx_up_bias is not None and state.init_xtx_down_bias is not None,
            dtype=bool,
        ),
        "sel_mut_indices": np.asarray(state.sel_mut.indices, dtype=index_dtype),
        "sel_mut_indptr": np.asarray(state.sel_mut.indptr, dtype=index_dtype),
        "sel_miss_indices": np.asarray(state.sel_miss.indices, dtype=index_dtype),
        "sel_miss_indptr": np.asarray(state.sel_miss.indptr, dtype=index_dtype),
    }
    if state.coalescence_counts is not None:
        save_dict["coalescence_counts"] = np.asarray(state.coalescence_counts, dtype=np.int64)
    if state.init_xtx_up_bias is not None:
        save_dict["init_xtx_up_bias"] = np.asarray(state.init_xtx_up_bias)
    if state.init_xtx_down_bias is not None:
        save_dict["init_xtx_down_bias"] = np.asarray(state.init_xtx_down_bias)
    for dst_level, blocks in enumerate(state.A_blocks):
        for src_level, block in enumerate(blocks):
            save_dict[f"A_blocks_{dst_level}_{src_level}_indices"] = np.asarray(block.indices, dtype=index_dtype)
            save_dict[f"A_blocks_{dst_level}_{src_level}_indptr"] = np.asarray(block.indptr, dtype=index_dtype)
            save_dict[f"A_blocks_{dst_level}_{src_level}_shape"] = np.asarray(block.shape, dtype=index_dtype)

    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(handle, **save_dict)


def _load_archive(artifact_path) -> np.lib.npyio.NpzFile:
    return np.load(artifact_path, allow_pickle=False)


def _artifact_index_dtype(data: np.lib.npyio.NpzFile) -> np.dtype:
    dtype = np.asarray(data["level_offsets"]).dtype
    if dtype not in {np.dtype(np.int32), np.dtype(np.int64)}:
        raise ValueError(f"Unsupported .grg_spmv structural dtype: {dtype}")
    required = (
        "node_perm",
        "sample_rows",
        "sample_to_individual",
        "sel_mut_indices",
        "sel_mut_indptr",
        "sel_miss_indices",
        "sel_miss_indptr",
    )
    for key in required:
        arr_dtype = np.asarray(data[key]).dtype
        if arr_dtype != dtype:
            raise ValueError(f"Inconsistent structural dtype for {key}: {arr_dtype}, expected {dtype}")
    for key in data.files:
        if key.startswith("A_blocks_") and key.endswith(("_indices", "_indptr", "_shape")):
            arr_dtype = np.asarray(data[key]).dtype
            if arr_dtype != dtype:
                raise ValueError(f"Inconsistent structural dtype for {key}: {arr_dtype}, expected {dtype}")
    return dtype


def _load_string_key(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in data.files:
        raise ValueError(f"Unsupported .grg_spmv artifact: missing {key}")
    return str(np.asarray(data[key]).item())


def load_grg_spmv(artifact_path, dtype, index_dtype) -> CompiledOperatorState:
    data = _load_archive(artifact_path)
    if _FORMAT_MAGIC_KEY not in data.files:
        raise ValueError(f"Unsupported .grg_spmv artifact in {artifact_path}: missing {_FORMAT_MAGIC_KEY}")
    magic = str(np.asarray(data[_FORMAT_MAGIC_KEY]).item())
    if magic != GRG_SPMV_FORMAT_MAGIC:
        raise ValueError(f"Unsupported .grg_spmv artifact in {artifact_path}: got magic {magic!r}")
    if _FORMAT_VERSION_KEY not in data.files:
        raise ValueError(f"Unsupported .grg_spmv artifact in {artifact_path}: missing {_FORMAT_VERSION_KEY}")
    version = int(np.asarray(data[_FORMAT_VERSION_KEY]).item())
    if version != GRG_SPMV_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported .grg_spmv artifact in {artifact_path}: got {version}, expected {GRG_SPMV_FORMAT_VERSION}"
        )

    artifact_index_dtype = _artifact_index_dtype(data)
    requested_index_dtype = np.dtype(index_dtype)
    if requested_index_dtype != artifact_index_dtype:
        raise ValueError(
            f".grg_spmv structural dtype is {artifact_index_dtype.name}, "
            f"but SpmvGRG was requested with index_dtype={requested_index_dtype.name}"
        )

    ordering = _load_string_key(data, "ordering")
    intra_block_ordering = _load_string_key(data, "intra_block_ordering")
    if ordering not in VALID_ORDERINGS:
        raise ValueError(f"Unsupported .grg_spmv ordering {ordering!r}")
    if intra_block_ordering not in VALID_INTRA_BLOCK_ORDERINGS:
        raise ValueError(f"Unsupported .grg_spmv intra_block_ordering {intra_block_ordering!r}")

    num_samples = int(np.asarray(data["num_samples"]).item())
    num_mutations = int(np.asarray(data["num_mutations"]).item())
    num_nodes = int(np.asarray(data["num_nodes"]).item())
    level_offsets = np.asarray(data["level_offsets"], dtype=artifact_index_dtype)
    node_perm = np.asarray(data["node_perm"], dtype=artifact_index_dtype)
    sample_rows = np.asarray(data["sample_rows"], dtype=artifact_index_dtype)
    sample_to_individual = np.asarray(data["sample_to_individual"], dtype=artifact_index_dtype)
    init_vector_up_bias = np.asarray(data["init_vector_up_bias"], dtype=dtype)
    init_vector_down_bias = np.asarray(data["init_vector_down_bias"], dtype=dtype)

    coalescence_counts = None
    if bool(np.asarray(data["has_individual_coals"]).item()):
        coalescence_counts = np.asarray(data["coalescence_counts"], dtype=np.int64)

    init_xtx_up_bias = None
    init_xtx_down_bias = None
    if bool(np.asarray(data["has_xtx_bias"]).item()):
        init_xtx_up_bias = np.asarray(data["init_xtx_up_bias"], dtype=dtype)
        init_xtx_down_bias = np.asarray(data["init_xtx_down_bias"], dtype=dtype)

    inv_node_perm = _invert_permutation(node_perm, index_dtype=artifact_index_dtype)

    sel_mut = binary_csr_from_csr_parts(
        indices=np.asarray(data["sel_mut_indices"], dtype=artifact_index_dtype),
        indptr=np.asarray(data["sel_mut_indptr"], dtype=artifact_index_dtype),
        shape=(num_mutations, num_nodes),
        dtype=dtype,
        index_dtype=artifact_index_dtype,
    )
    sel_miss = binary_csr_from_csr_parts(
        indices=np.asarray(data["sel_miss_indices"], dtype=artifact_index_dtype),
        indptr=np.asarray(data["sel_miss_indptr"], dtype=artifact_index_dtype),
        shape=(num_mutations, num_nodes),
        dtype=dtype,
        index_dtype=artifact_index_dtype,
    )

    num_levels = len(level_offsets) - 1
    A_blocks: list[list[object]] = []
    for dst_level in range(num_levels):
        level_blocks = []
        for src_level in range(dst_level):
            level_blocks.append(
                binary_csr_from_csr_parts(
                    indices=np.asarray(data[f"A_blocks_{dst_level}_{src_level}_indices"], dtype=artifact_index_dtype),
                    indptr=np.asarray(data[f"A_blocks_{dst_level}_{src_level}_indptr"], dtype=artifact_index_dtype),
                    shape=np.asarray(data[f"A_blocks_{dst_level}_{src_level}_shape"], dtype=artifact_index_dtype),
                    dtype=dtype,
                    index_dtype=artifact_index_dtype,
                )
            )
        A_blocks.append(level_blocks)

    return CompiledOperatorState(
        A_blocks=A_blocks,
        level_offsets=level_offsets,
        node_perm=node_perm,
        inv_node_perm=inv_node_perm,
        sample_rows=sample_rows,
        sel_mut=sel_mut,
        sel_miss=sel_miss,
        num_samples=num_samples,
        num_mutations=num_mutations,
        num_nodes=num_nodes,
        ploidy=int(np.asarray(data["ploidy"]).item()),
        num_individuals=int(np.asarray(data["num_individuals"]).item()),
        num_edges=int(np.asarray(data["num_edges"]).item()),
        has_missing_data=bool(np.asarray(data["has_missing_data"]).item()),
        ordering=ordering,
        intra_block_ordering=intra_block_ordering,
        sample_to_individual=sample_to_individual,
        mutation_positions=np.asarray(data["mutation_positions"], dtype=np.float64),
        mutation_times=np.asarray(data["mutation_times"], dtype=np.float64),
        mutation_alleles=np.asarray(data["mutation_alleles"]),
        mutation_allele_offsets=np.asarray(data["mutation_allele_offsets"]),
        mutation_ref_alleles=np.asarray(data["mutation_ref_alleles"]),
        mutation_ref_allele_offsets=np.asarray(data["mutation_ref_allele_offsets"]),
        coalescence_counts=coalescence_counts,
        init_vector_up_bias=init_vector_up_bias,
        init_vector_down_bias=init_vector_down_bias,
        init_xtx_up_bias=init_xtx_up_bias,
        init_xtx_down_bias=init_xtx_down_bias,
    )


__all__ = [
    "GRG_SPMV_FORMAT_MAGIC",
    "GRG_SPMV_FORMAT_VERSION",
    "artifact_path_for_grg",
    "load_grg_spmv",
    "save_grg_spmv",
]
