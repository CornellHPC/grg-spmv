"""GRG compilation into stable-height sparse operator state."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pygrgl
import scipy.sparse as sp

from pygrgl_spmv._rss import rss_checkpoint
from pygrgl_spmv.backends import BackendSetup
from pygrgl_spmv.grg.sparse import binary_csr_from_parts, finalize_int_array, pick_small_signed_int_dtype

_LOGGER = logging.getLogger(__name__)


@dataclass
class CompiledOperatorState:
    """Compiled operator state shared between cache I/O and runtime setup."""

    A_blocks: list[list[sp.csr_matrix]] | None
    level_offsets: np.ndarray
    node_perm: np.ndarray
    inv_node_perm: np.ndarray
    sel_mut: sp.csr_matrix
    sel_miss: sp.csr_matrix
    num_samples: int
    num_mutations: int
    num_nodes: int
    ploidy: int
    num_individuals: int
    num_edges: int
    has_missing_data: bool
    sample_to_individual: np.ndarray
    mutation_positions: np.ndarray
    mutation_times: np.ndarray
    mutation_alleles: np.ndarray
    mutation_allele_offsets: np.ndarray
    mutation_ref_alleles: np.ndarray
    mutation_ref_allele_offsets: np.ndarray
    coalescence_counts: np.ndarray | None
    init_vector_up_bias: np.ndarray | None = None
    init_vector_down_bias: np.ndarray | None = None
    init_xtx_up_bias: np.ndarray | None = None
    init_xtx_down_bias: np.ndarray | None = None

    def to_backend_setup(self, dtype: np.dtype) -> BackendSetup:
        if self.A_blocks is None:
            raise RuntimeError("Compiled operator blocks are not available for backend setup")
        return BackendSetup(
            A_blocks=self.A_blocks,
            level_offsets=self.level_offsets,
            num_samples=self.num_samples,
            num_mutations=self.num_mutations,
            num_nodes=self.num_nodes,
            sel_mut=self.sel_mut,
            sel_miss=self.sel_miss,
            coalescence_counts=self.coalescence_counts,
            dtype=dtype,
        )


def _invert_permutation(perm: np.ndarray) -> np.ndarray:
    """Build the inverse of a dense permutation array."""
    perm_arr = finalize_int_array(perm, label="perm")
    inv = np.empty(int(perm_arr.size), dtype=perm_arr.dtype)
    inv[perm_arr] = np.arange(int(perm_arr.size), dtype=perm_arr.dtype)
    return inv


def _compute_node_levels(grg, num_nodes: int, *, node_id_scratch_dtype: np.dtype) -> np.ndarray:
    """Compute node levels from down edges only."""
    levels = np.zeros(num_nodes, dtype=np.int32)
    for node_id in range(num_nodes):
        children = np.asarray(grg.get_down_edges(node_id), dtype=node_id_scratch_dtype)
        if children.size == 0:
            continue
        levels[node_id] = int(levels[children].max()) + 1
    return levels


def _build_stable_height_order(
    node_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build stable height order directly from per-height counts."""
    num_nodes = int(node_levels.shape[0])
    level_values = np.asarray(node_levels, dtype=np.int64)
    num_levels = 0 if num_nodes == 0 else int(level_values.max()) + 1

    counts = np.bincount(level_values, minlength=num_levels)
    level_offsets64 = np.zeros(num_levels + 1, dtype=np.int64)
    level_offsets64[1:] = np.cumsum(counts, dtype=np.int64)

    node_perm64 = np.empty(num_nodes, dtype=np.int64)
    next_pos = level_offsets64[:-1].copy()
    for node_id in range(num_nodes):
        level = int(level_values[node_id])
        pos = int(next_pos[level])
        node_perm64[pos] = node_id
        next_pos[level] = pos + 1

    node_perm = finalize_int_array(node_perm64, label="node_perm")
    inv_node_perm = _invert_permutation(node_perm)
    level_offsets = finalize_int_array(level_offsets64, label="level_offsets")
    return node_perm, inv_node_perm, level_offsets


def _build_level_blocks(
    grg,
    *,
    node_levels: np.ndarray,
    level_offsets: np.ndarray,
    inv_node_perm: np.ndarray,
    node_id_scratch_dtype: np.dtype,
) -> list[list[sp.csr_matrix]]:
    """Build block CSR matrices in two streamed passes with exact-sized arrays."""
    num_nodes = int(node_levels.shape[0])
    num_levels = int(level_offsets.size - 1)
    level_offsets64 = np.asarray(level_offsets, dtype=np.int64)
    block_indptr_scratch_dtype = pick_small_signed_int_dtype(max(int(grg.num_edges), 0))
    level_sizes = [int(level_offsets64[level + 1] - level_offsets64[level]) for level in range(num_levels)]
    block_index_scratch_dtypes = [
        pick_small_signed_int_dtype(max(level_size - 1, 0))
        for level_size in level_sizes
    ]
    block_indptrs: list[list[np.ndarray | None]] = [[None] * level for level in range(num_levels)]

    for parent_id in range(num_nodes):
        parent_level = int(node_levels[parent_id])
        children = np.asarray(grg.get_down_edges(parent_id), dtype=node_id_scratch_dtype)
        if children.size == 0:
            continue

        row_local = int(inv_node_perm[parent_id]) - int(level_offsets[parent_level])
        child_levels = node_levels[children]
        if np.any(child_levels >= parent_level):
            raise RuntimeError(
                f"Invalid GRG topology at node {parent_id}: expected child levels < parent level {parent_level}"
            )
        child_level_counts = np.bincount(np.asarray(child_levels, dtype=np.int64), minlength=parent_level)
        for child_level in np.flatnonzero(child_level_counts):
            indptr = block_indptrs[parent_level][child_level]
            if indptr is None:
                indptr = np.zeros(level_sizes[parent_level] + 1, dtype=block_indptr_scratch_dtype)
                block_indptrs[parent_level][child_level] = indptr
            indptr[row_local + 1] += int(child_level_counts[child_level])
    block_indices: list[list[np.ndarray | None]] = [[None] * level for level in range(num_levels)]
    for parent_level in range(num_levels):
        for child_level in range(parent_level):
            indptr = block_indptrs[parent_level][child_level]
            if indptr is None:
                continue
            np.cumsum(indptr, out=indptr)
            indptr = finalize_int_array(
                indptr,
                label=f"A_blocks[{parent_level}][{child_level}].indptr",
            )
            block_indptrs[parent_level][child_level] = indptr
            block_indices[parent_level][child_level] = np.empty(
                int(indptr[-1]),
                dtype=block_index_scratch_dtypes[child_level],
            )
    for parent_id in range(num_nodes):
        parent_level = int(node_levels[parent_id])
        children = np.asarray(grg.get_down_edges(parent_id), dtype=node_id_scratch_dtype)
        if children.size == 0:
            continue

        row_local = int(inv_node_perm[parent_id]) - int(level_offsets[parent_level])
        child_levels = node_levels[children]
        child_positions = inv_node_perm[children]
        order = np.lexsort((child_positions, child_levels))
        child_levels_sorted = child_levels[order]
        child_positions_sorted = child_positions[order]
        bounds = np.concatenate(
            [
                np.array([0], dtype=np.int64),
                np.flatnonzero(np.diff(child_levels_sorted)) + 1,
                np.array([child_levels_sorted.size], dtype=np.int64),
            ]
        )
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            child_level = int(child_levels_sorted[int(b0)])
            indptr = block_indptrs[parent_level][child_level]
            indices = block_indices[parent_level][child_level]
            if indptr is None or indices is None:
                raise RuntimeError(
                    f"Missing block allocation for A_blocks[{parent_level}][{child_level}]"
                )
            start = int(indptr[row_local])
            end = int(indptr[row_local + 1])
            if (end - start) != int(b1 - b0):
                raise RuntimeError(
                    f"Row nnz mismatch for block ({parent_level}, {child_level}) row {row_local}: "
                    f"expected {end - start}, got {int(b1 - b0)}"
                )
            cols_local = indices[start:end]
            np.subtract(
                child_positions_sorted[int(b0) : int(b1)],
                level_offsets[child_level],
                out=cols_local,
                casting="unsafe",
            )
            if cols_local.size > 1 and np.any(cols_local[1:] == cols_local[:-1]):
                raise RuntimeError(
                    f"Duplicate GRG edge detected for parent node {parent_id} and child level {child_level}"
                )
    A_blocks: list[list[sp.csr_matrix]] = []
    for parent_level in range(num_levels):
        level_blocks: list[sp.csr_matrix] = []
        for child_level in range(parent_level):
            indptr = block_indptrs[parent_level][child_level]
            indices = block_indices[parent_level][child_level]
            if indptr is None or indices is None:
                level_blocks.append(
                    binary_csr_from_parts(
                        indices=np.empty(0, dtype=np.int32),
                        indptr=np.zeros(level_sizes[parent_level] + 1, dtype=np.int32),
                        shape=(level_sizes[parent_level], level_sizes[child_level]),
                    )
                )
                continue
            level_blocks.append(
                binary_csr_from_parts(
                    indices=finalize_int_array(
                        indices,
                        label=f"A_blocks[{parent_level}][{child_level}].indices",
                    ),
                    indptr=indptr,
                    shape=(level_sizes[parent_level], level_sizes[child_level]),
                    shared_data=True,
                )
            )
        A_blocks.append(level_blocks)
    return A_blocks


def _build_selectors(
    mutation_rows,
    *,
    inv_node_perm: np.ndarray,
    num_mutations: int,
    num_nodes: int,
    node_id_scratch_dtype: np.dtype,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build selectors from sorted mutation rows, allowing contiguous repeated mutation IDs."""
    row_count = len(mutation_rows)
    if row_count < num_mutations:
        raise ValueError(
            "GRG mutation rows are incomplete: "
            f"got {row_count} row(s) from get_mutation_node_miss(), expected at least {num_mutations}"
        )

    invalid_node = int(pygrgl.INVALID_NODE)
    selector_indptr_scratch_dtype = pick_small_signed_int_dtype(max(row_count, 0))
    mut_indptr = np.zeros(num_mutations + 1, dtype=selector_indptr_scratch_dtype)
    miss_indptr = np.zeros(num_mutations + 1, dtype=selector_indptr_scratch_dtype)
    mut_indices = np.empty(row_count, dtype=node_id_scratch_dtype)
    miss_indices = np.empty(row_count, dtype=node_id_scratch_dtype)
    mut_nnz = 0
    miss_nnz = 0
    next_mut_id = 0

    for mut_id_raw, mut_node_raw, miss_node_raw in mutation_rows:
        mut_id = int(mut_id_raw)
        if mut_id < 0 or mut_id >= num_mutations:
            raise ValueError(
                f"Mutation id {mut_id} is out of range for num_mutations={num_mutations}"
            )
        if mut_id == next_mut_id:
            next_mut_id += 1
        elif mut_id != (next_mut_id - 1):
            raise ValueError(
                "GRG mutation rows must be sorted by ascending mutation id, "
                f"with repeated ids contiguous; got mutation id {mut_id} after next_mut_id={next_mut_id}"
            )

        mut_node = int(mut_node_raw)
        if mut_node != invalid_node:
            mut_indices[mut_nnz] = inv_node_perm[mut_node]
            mut_nnz += 1
        mut_indptr[mut_id + 1] = mut_nnz

        miss_node = int(miss_node_raw)
        if miss_node != invalid_node:
            miss_indices[miss_nnz] = inv_node_perm[miss_node]
            miss_nnz += 1
        miss_indptr[mut_id + 1] = miss_nnz

    if next_mut_id != num_mutations:
        raise ValueError(
            "GRG mutation rows are incomplete: "
            f"ended at mutation id {next_mut_id - 1}, expected {num_mutations - 1}"
        )

    sel_mut = binary_csr_from_parts(
        indices=finalize_int_array(mut_indices[:mut_nnz], label="sel_mut.indices"),
        indptr=finalize_int_array(mut_indptr, label="sel_mut.indptr"),
        shape=(num_mutations, num_nodes),
    )
    sel_miss = binary_csr_from_parts(
        indices=finalize_int_array(miss_indices[:miss_nnz], label="sel_miss.indices"),
        indptr=finalize_int_array(miss_indptr, label="sel_miss.indptr"),
        shape=(num_mutations, num_nodes),
    )
    if row_count > num_mutations and sel_miss.nnz > 0:
        # Repeated mutation rows can legitimately share one missingness node,
        # which produces duplicate coordinates only in the missingness selector.
        sel_miss.sum_duplicates()
        sel_miss.data.fill(True)
    return sel_mut, sel_miss


def _build_coalescence_counts(grg, *, node_perm: np.ndarray) -> np.ndarray | None:
    """Load and validate per-node coalescence counts in internal node order."""
    counts_orig = np.array(
        [grg.get_num_individual_coals(i) for i in range(grg.num_nodes)],
        dtype=np.int64,
    )
    not_set = int(pygrgl.COAL_COUNT_NOT_SET)
    has_individual_coals = getattr(grg, "has_individual_coals", None)
    if has_individual_coals is None:
        has_individual_coals = bool(np.any(counts_orig[grg.num_samples :] != not_set))
    if not bool(has_individual_coals):
        return None

    missing_mask = counts_orig == not_set
    internal_missing_count = int(np.sum(missing_mask[grg.num_samples :]))
    if internal_missing_count > 0:
        raise ValueError(
            "GRG has missing coalescence counts for "
            f"{internal_missing_count} internal nodes; init='xtx' requires complete internal counts."
        )
    sample_missing = missing_mask[: grg.num_samples]
    if np.any(sample_missing):
        counts_orig[np.flatnonzero(sample_missing)] = 0

    return counts_orig[node_perm]


_NUCLEOTIDE_ENCODE = {"A": 0b00, "T": 0b01, "C": 0b10, "G": 0b11}


def _encode_alleles(alleles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Pack variable-length allele strings into a 2-bit-per-nucleotide uint8 buffer with CSR offsets."""
    n = len(alleles)
    offsets = np.empty(n + 1, dtype=np.uint32)
    offsets[0] = 0

    total = 0
    for i, allele_str in enumerate(alleles):
        for ch in allele_str:
            if ch not in _NUCLEOTIDE_ENCODE:
                raise ValueError(f"Non-ATCG character {ch!r} in allele at index {i}: {allele_str!r}")
        total += len(allele_str)
        if total > np.iinfo(np.uint32).max:
            raise ValueError(f"Allele offset overflow at index {i}: total nucleotide count {total} exceeds uint32 max")
        offsets[i + 1] = total

    buf = np.zeros((total + 3) // 4, dtype=np.uint8)
    pos = 0
    for allele_str in alleles:
        for ch in allele_str:
            buf[pos // 4] |= np.uint8(_NUCLEOTIDE_ENCODE[ch] << ((pos % 4) * 2))
            pos += 1

    return buf, offsets


def _build_mutation_table(grg) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions: list[float] = []
    times: list[float] = []
    alleles: list[str] = []
    ref_alleles: list[str] = []
    for mutation_id in range(int(grg.num_mutations)):
        mutation = grg.get_mutation_by_id(int(mutation_id))
        positions.append(float(mutation.position))
        times.append(float(mutation.time))
        alleles.append(str(mutation.allele))
        ref_alleles.append(str(mutation.ref_allele))

    allele_data, allele_offsets = _encode_alleles(alleles)
    ref_data, ref_offsets = _encode_alleles(ref_alleles)
    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
        allele_data,
        allele_offsets,
        ref_data,
        ref_offsets,
    )


def _validate_sample_prefix(*, node_perm: np.ndarray, level_offsets: np.ndarray, num_samples: int) -> None:
    """Require stable height order to keep samples as an internal prefix."""
    expected = np.arange(num_samples, dtype=np.asarray(node_perm).dtype)
    if not np.array_equal(node_perm[:num_samples], expected):
        raise RuntimeError(
            "Stable height order must keep sample nodes as an internal prefix; "
            "this prototype no longer supports sample permutations."
        )
    if int(level_offsets[1]) < num_samples:
        raise RuntimeError(
            "Stable height order must place the full sample prefix inside level 0; "
            f"got level_offsets[1]={int(level_offsets[1])}, num_samples={num_samples}"
        )


def compile_grg(
    grg,
) -> CompiledOperatorState:
    """Compile a non-empty immutable GRG into the normalized sparse traversal layout."""
    num_samples = int(grg.num_samples)
    num_mutations = int(grg.num_mutations)
    num_nodes = int(grg.num_nodes)
    num_edges = int(grg.num_edges)
    if num_samples == 0 or num_nodes == 0:
        raise ValueError("compile_grg() only supports non-empty GRGs with at least one sample and one node.")
    if isinstance(grg, pygrgl.MutableGRG):
        raise ValueError(
            "compile_grg() only supports immutable GRGs; MutableGRG inputs may expose unsorted mutation rows."
        )
    node_id_scratch_dtype = pick_small_signed_int_dtype(max(num_nodes - 1, 0))

    prev_rss_bytes = rss_checkpoint(_LOGGER, "compile:start", None)
    node_levels = _compute_node_levels(grg, num_nodes, node_id_scratch_dtype=node_id_scratch_dtype)
    node_perm, inv_node_perm, level_offsets = _build_stable_height_order(node_levels)
    _validate_sample_prefix(node_perm=node_perm, level_offsets=level_offsets, num_samples=num_samples)

    A_blocks = _build_level_blocks(
        grg,
        node_levels=node_levels,
        level_offsets=level_offsets,
        inv_node_perm=inv_node_perm,
        node_id_scratch_dtype=node_id_scratch_dtype,
    )
    del node_levels
    prev_rss_bytes = rss_checkpoint(_LOGGER, "compile:blocks_ready", prev_rss_bytes)

    mutation_rows = grg.get_mutation_node_miss()
    prev_rss_bytes = rss_checkpoint(_LOGGER, "compile:mutation_rows_loaded", prev_rss_bytes)

    sel_mut, sel_miss = _build_selectors(
        mutation_rows,
        inv_node_perm=inv_node_perm,
        num_mutations=num_mutations,
        num_nodes=num_nodes,
        node_id_scratch_dtype=node_id_scratch_dtype,
    )
    prev_rss_bytes = rss_checkpoint(_LOGGER, "compile:selectors_built", prev_rss_bytes)
    del mutation_rows
    prev_rss_bytes = rss_checkpoint(_LOGGER, "compile:selectors_ready", prev_rss_bytes)

    sample_to_individual = finalize_int_array(
        np.arange(num_samples, dtype=node_id_scratch_dtype) // max(int(grg.ploidy), 1),
        label="sample_to_individual",
    )
    coalescence_counts = _build_coalescence_counts(grg, node_perm=node_perm)
    mutation_positions, mutation_times, mutation_alleles, mutation_allele_offsets, mutation_ref_alleles, mutation_ref_allele_offsets = _build_mutation_table(grg)
    rss_checkpoint(_LOGGER, "compile:mutation_table_ready", prev_rss_bytes)

    return CompiledOperatorState(
        A_blocks=A_blocks,
        level_offsets=level_offsets,
        node_perm=node_perm,
        inv_node_perm=inv_node_perm,
        sel_mut=sel_mut,
        sel_miss=sel_miss,
        num_samples=num_samples,
        num_mutations=num_mutations,
        num_nodes=num_nodes,
        ploidy=int(grg.ploidy),
        num_individuals=int(grg.num_individuals),
        num_edges=num_edges,
        has_missing_data=bool(grg.has_missing_data),
        sample_to_individual=sample_to_individual,
        mutation_positions=mutation_positions,
        mutation_times=mutation_times,
        mutation_alleles=mutation_alleles,
        mutation_allele_offsets=mutation_allele_offsets,
        mutation_ref_alleles=mutation_ref_alleles,
        mutation_ref_allele_offsets=mutation_ref_allele_offsets,
        coalescence_counts=coalescence_counts,
    )


__all__ = ["CompiledOperatorState", "_invert_permutation", "compile_grg"]
