"""GRG compilation into level-ordered sparse operator state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygrgl
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee

from pygrgl_spmv.backends import BackendSetup
from pygrgl_spmv.grg.sparse import binary_csr_from_coo


@dataclass
class CompiledOperatorState:
    """Compiled operator state shared between cache I/O and runtime setup."""

    A_blocks: list[list[sp.csr_matrix]] | None
    level_offsets: np.ndarray
    node_perm: np.ndarray
    inv_node_perm: np.ndarray
    sample_perm: np.ndarray
    inv_sample_perm: np.ndarray
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
    mutation_alleles: np.ndarray | None
    mutation_ref_alleles: np.ndarray | None
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
            sample_perm=self.sample_perm,
            inv_sample_perm=self.inv_sample_perm,
            coalescence_counts=self.coalescence_counts,
            dtype=dtype,
        )


def _invert_permutation(perm: np.ndarray, *, index_dtype: np.dtype) -> np.ndarray:
    """Build the inverse of a dense permutation array."""
    perm_arr = np.asarray(perm, dtype=index_dtype)
    inv = np.empty(int(perm_arr.size), dtype=index_dtype)
    inv[perm_arr] = np.arange(int(perm_arr.size), dtype=index_dtype)
    return inv


def _rcm_bipartite(A: sp.spmatrix, index_dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Bipartite reverse Cuthill-McKee for a rectangular block."""
    nrows, ncols = A.shape
    A_csr = sp.csr_matrix(A)

    top = sp.hstack([sp.csr_matrix((nrows, nrows)), A_csr], format="csr")
    bottom = sp.hstack([A_csr.T, sp.csr_matrix((ncols, ncols))], format="csr")
    B = sp.vstack([top, bottom], format="csr")

    perm_full = reverse_cuthill_mckee(B, symmetric_mode=True)

    row_indices = perm_full[perm_full < nrows]
    col_indices = perm_full[perm_full >= nrows] - nrows
    if row_indices.size != nrows or col_indices.size != ncols:
        raise RuntimeError(
            "reverse_cuthill_mckee returned incomplete bipartite permutation "
            f"(rows={row_indices.size}/{nrows}, cols={col_indices.size}/{ncols})"
        )
    if not np.array_equal(np.sort(row_indices), np.arange(nrows)):
        raise RuntimeError("Row permutation from reverse_cuthill_mckee is invalid")
    if not np.array_equal(np.sort(col_indices), np.arange(ncols)):
        raise RuntimeError("Column permutation from reverse_cuthill_mckee is invalid")
    return row_indices.astype(index_dtype, copy=False), col_indices.astype(index_dtype, copy=False)


def _extract_edges_and_heights(grg, K: int, *, index_dtype: np.dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract directed edges (parent -> child) and node heights."""
    heights = np.zeros(K, dtype=np.int64)
    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []

    for node_id in range(K):
        children = np.asarray(grg.get_down_edges(node_id), dtype=index_dtype)
        if children.size == 0:
            continue
        row_chunks.append(np.full(int(children.size), node_id, dtype=index_dtype))
        col_chunks.append(children)
        heights[node_id] = int(heights[children].max()) + 1

    if row_chunks:
        row_arr = np.concatenate(row_chunks)
        col_arr = np.concatenate(col_chunks)
    else:
        row_arr = np.empty(0, dtype=index_dtype)
        col_arr = np.empty(0, dtype=index_dtype)
    return row_arr, col_arr, heights


def _build_level_order(heights: np.ndarray, *, index_dtype: np.dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build height-sorted node order and level offsets."""
    K = int(heights.shape[0])

    perm_height = np.argsort(heights, kind="stable").astype(index_dtype, copy=False)
    inv_perm_height = np.empty(K, dtype=index_dtype)
    inv_perm_height[perm_height] = np.arange(K, dtype=index_dtype)

    sorted_heights = heights[perm_height]
    if np.any(np.diff(sorted_heights) < 0):
        raise RuntimeError("Heights must be non-decreasing after sorting")
    changes = np.flatnonzero(np.diff(sorted_heights)) + 1
    level_offsets = np.concatenate(
        [
            np.array([0], dtype=index_dtype),
            changes.astype(index_dtype, copy=False),
            np.array([K], dtype=index_dtype),
        ]
    )

    num_levels = int(level_offsets.size - 1)
    node_levels = np.empty(K, dtype=np.int32)
    for h in range(num_levels):
        lo, hi = int(level_offsets[h]), int(level_offsets[h + 1])
        node_levels[perm_height[lo:hi]] = h

    return perm_height, inv_perm_height, level_offsets, node_levels


def _build_level_perms(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    inv_perm_height: np.ndarray,
    level_offsets: np.ndarray,
    dst_levels: np.ndarray,
    src_order: np.ndarray,
    src_offsets: np.ndarray,
    index_dtype: np.dtype,
    dtype: np.dtype,
) -> list[np.ndarray]:
    """Compute always-on per-level permutations (RCM + min-column ordering)."""
    num_levels = int(level_offsets.size - 1)

    level_perms = [
        np.arange(int(level_offsets[h + 1] - level_offsets[h]), dtype=index_dtype)
        for h in range(num_levels)
    ]

    for h in range(1, num_levels):
        lo_e = int(src_offsets[h])
        hi_e = int(src_offsets[h + 1])
        if lo_e == hi_e:
            continue
        edge_idx = src_order[lo_e:hi_e]
        prev_idx = edge_idx[dst_levels[edge_idx] == (h - 1)]
        if prev_idx.size == 0:
            continue

        row_local = (inv_perm_height[rows[prev_idx]] - level_offsets[h]).astype(np.int64, copy=False)
        col_local = (inv_perm_height[cols[prev_idx]] - level_offsets[h - 1]).astype(np.int64, copy=False)

        row_size = int(level_offsets[h + 1] - level_offsets[h])
        col_size = int(level_offsets[h] - level_offsets[h - 1])

        if h == 1:
            A_block = binary_csr_from_coo(
                row_local,
                col_local,
                shape=(row_size, col_size),
                dtype=dtype,
            )
            row_perm, col_perm = _rcm_bipartite(A_block, index_dtype)
            level_perms[1] = row_perm
            level_perms[0] = col_perm
            continue

        min_col = np.full(row_size, col_size, dtype=np.int64)
        np.minimum.at(min_col, row_local, col_local)
        level_perms[h] = np.argsort(min_col, kind="stable").astype(index_dtype, copy=False)

    return level_perms


def _build_blocks_from_edges(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    inv_final_perm: np.ndarray,
    level_offsets: np.ndarray,
    dst_levels: np.ndarray,
    src_order: np.ndarray,
    src_offsets: np.ndarray,
    dtype: np.dtype,
) -> list[list[sp.csr_matrix]]:
    """Build block CSR matrices directly from edge buckets."""
    num_levels = int(level_offsets.size - 1)
    A_blocks: list[list[sp.csr_matrix]] = []

    for h in range(num_levels):
        row_size = int(level_offsets[h + 1] - level_offsets[h])
        level_blocks = [
            sp.csr_matrix((row_size, int(level_offsets[j + 1] - level_offsets[j])), dtype=dtype)
            for j in range(h)
        ]
        lo_e = int(src_offsets[h])
        hi_e = int(src_offsets[h + 1])
        if lo_e == hi_e:
            A_blocks.append(level_blocks)
            continue

        edge_idx = src_order[lo_e:hi_e]
        dst_sub = dst_levels[edge_idx]
        order_dst = np.argsort(dst_sub, kind="stable")
        edge_sorted = edge_idx[order_dst]
        dst_sorted = dst_sub[order_dst]

        splits = np.flatnonzero(np.diff(dst_sorted)) + 1
        bounds = np.concatenate([np.array([0]), splits, np.array([edge_sorted.size])])
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            j = int(dst_sorted[int(b0)])
            if j >= h:
                raise RuntimeError(
                    f"Invalid GRG edge bucket (source level {h}, target level {j}); expected target < source"
                )
            pair_idx = edge_sorted[int(b0) : int(b1)]
            if pair_idx.size == 0:
                continue
            row_local = (inv_final_perm[rows[pair_idx]] - level_offsets[h]).astype(np.int64, copy=False)
            col_local = (inv_final_perm[cols[pair_idx]] - level_offsets[j]).astype(np.int64, copy=False)
            col_size = int(level_offsets[j + 1] - level_offsets[j])
            level_blocks[j] = binary_csr_from_coo(
                row_local,
                col_local,
                shape=(row_size, col_size),
                dtype=dtype,
            )

        A_blocks.append(level_blocks)

    return A_blocks


def _build_binary_selector(
    rows: list[int],
    cols_orig: list[int],
    *,
    inv_final_perm: np.ndarray,
    shape: tuple[int, int],
    index_dtype: np.dtype,
    dtype: np.dtype,
) -> sp.csr_matrix:
    if not rows:
        return sp.csr_matrix(shape, dtype=dtype)
    row_arr = np.asarray(rows, dtype=index_dtype)
    col_arr = inv_final_perm[np.asarray(cols_orig, dtype=index_dtype)]
    return binary_csr_from_coo(row_arr, col_arr, shape=shape, dtype=dtype)


def _build_selectors(grg, *, inv_final_perm: np.ndarray, m: int, K: int, index_dtype: np.dtype, dtype: np.dtype) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build mutation and missingness selector matrices."""
    mut_rows: list[int] = []
    mut_cols_orig: list[int] = []
    miss_rows: list[int] = []
    miss_cols_orig: list[int] = []

    for mut_id, mut_node, miss_node in grg.get_mutation_node_miss():
        if mut_node != pygrgl.INVALID_NODE:
            mut_rows.append(int(mut_id))
            mut_cols_orig.append(int(mut_node))
        if miss_node != pygrgl.INVALID_NODE:
            miss_rows.append(int(mut_id))
            miss_cols_orig.append(int(miss_node))

    sel_mut = _build_binary_selector(
        mut_rows,
        mut_cols_orig,
        inv_final_perm=inv_final_perm,
        shape=(m, K),
        index_dtype=index_dtype,
        dtype=dtype,
    )
    sel_miss = _build_binary_selector(
        miss_rows,
        miss_cols_orig,
        inv_final_perm=inv_final_perm,
        shape=(m, K),
        index_dtype=index_dtype,
        dtype=dtype,
    )

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


def _build_mutation_table(grg) -> tuple[np.ndarray, np.ndarray, None, None]:
    positions: list[float] = []
    times: list[float] = []
    for mutation_id in range(int(grg.num_mutations)):
        mutation = grg.get_mutation_by_id(int(mutation_id))
        positions.append(float(mutation.position))
        times.append(float(mutation.time))

    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
        None,
        None,
    )


def compile_grg(grg, *, dtype: np.dtype, index_dtype: np.dtype) -> CompiledOperatorState:
    """Compile a GRG object into the normalized sparse traversal layout."""
    num_samples = int(grg.num_samples)
    num_mutations = int(grg.num_mutations)
    num_nodes = int(grg.num_nodes)

    rows, cols, heights = _extract_edges_and_heights(grg, num_nodes, index_dtype=index_dtype)
    perm_height, inv_perm_height, level_offsets, node_levels = _build_level_order(
        heights,
        index_dtype=index_dtype,
    )
    num_levels = int(level_offsets.size - 1)
    E = int(rows.size)

    src_levels = node_levels[rows] if E else np.empty(0, dtype=np.int32)
    dst_levels = node_levels[cols] if E else np.empty(0, dtype=np.int32)
    if E and np.any(src_levels <= dst_levels):
        raise RuntimeError("Invalid GRG topology: expected every edge to point from higher to lower level")

    src_order = np.argsort(src_levels, kind="stable") if E else np.empty(0, dtype=np.int64)
    src_counts = np.bincount(src_levels, minlength=num_levels) if E else np.zeros(num_levels, dtype=np.int64)
    src_offsets = np.zeros(num_levels + 1, dtype=np.int64)
    src_offsets[1:] = np.cumsum(src_counts, dtype=np.int64)

    level_perms = _build_level_perms(
        rows,
        cols,
        inv_perm_height=inv_perm_height,
        level_offsets=level_offsets,
        dst_levels=dst_levels,
        src_order=src_order,
        src_offsets=src_offsets,
        index_dtype=index_dtype,
        dtype=dtype,
    )

    within_perm = np.arange(num_nodes, dtype=index_dtype)
    for h in range(num_levels):
        lo, hi = int(level_offsets[h]), int(level_offsets[h + 1])
        perm = level_perms[h]
        if perm.shape[0] != (hi - lo):
            raise RuntimeError(f"Invalid level permutation for level {h}: got {perm.shape[0]}, expected {hi - lo}")
        within_perm[lo:hi] = lo + perm

    final_perm = perm_height[within_perm]
    inv_final_perm = _invert_permutation(final_perm, index_dtype=index_dtype)

    A_blocks = _build_blocks_from_edges(
        rows,
        cols,
        inv_final_perm=inv_final_perm,
        level_offsets=level_offsets,
        dst_levels=dst_levels,
        src_order=src_order,
        src_offsets=src_offsets,
        dtype=dtype,
    )

    print(f"[compile_grg] num_levels={num_levels}, num_nodes={num_nodes}, num_samples={num_samples}")
    for h in range(num_levels):
        n_nodes = int(level_offsets[h + 1] - level_offsets[h])
        print(f"  level {h:4d}: {n_nodes:8d} nodes")
    print(f"[compile_grg] A_blocks nnz (dst_level, src_level) -> nnz:")
    total_nnz = 0
    for h in range(num_levels):
        if len(A_blocks[h]) != h:
            raise RuntimeError(f"Invalid number of blocks at level {h}: got {len(A_blocks[h])}, expected {h}")
        for j, blk in enumerate(A_blocks[h]):
            expected_shape = (
                int(level_offsets[h + 1] - level_offsets[h]),
                int(level_offsets[j + 1] - level_offsets[j]),
            )
            if blk.shape != expected_shape:
                raise RuntimeError(
                    f"Invalid block shape for A_blocks[{h}][{j}]: got {blk.shape}, expected {expected_shape}"
                )
            if blk.nnz > 0:
                print(f"  A_blocks[{h}][{j}]: shape={blk.shape}, nnz={blk.nnz}")
                total_nnz += blk.nnz
    print(f"[compile_grg] total A_blocks nnz={total_nnz}")

    sel_mut, sel_miss = _build_selectors(
        grg,
        inv_final_perm=inv_final_perm,
        m=num_mutations,
        K=num_nodes,
        index_dtype=index_dtype,
        dtype=dtype,
    )

    sample_to_individual = np.arange(num_samples, dtype=index_dtype) // max(int(grg.ploidy), 1)
    coalescence_counts = _build_coalescence_counts(grg, node_perm=final_perm)
    mutation_positions, mutation_times, mutation_alleles, mutation_ref_alleles = _build_mutation_table(grg)

    sample_perm = final_perm[:num_samples].copy()
    return CompiledOperatorState(
        A_blocks=A_blocks,
        level_offsets=level_offsets,
        node_perm=final_perm.copy(),
        inv_node_perm=inv_final_perm.copy(),
        sample_perm=sample_perm,
        inv_sample_perm=_invert_permutation(sample_perm, index_dtype=index_dtype),
        sel_mut=sel_mut,
        sel_miss=sel_miss,
        num_samples=num_samples,
        num_mutations=num_mutations,
        num_nodes=num_nodes,
        ploidy=int(grg.ploidy),
        num_individuals=int(grg.num_individuals),
        num_edges=int(grg.num_edges),
        has_missing_data=bool(grg.has_missing_data),
        sample_to_individual=sample_to_individual,
        mutation_positions=mutation_positions,
        mutation_times=mutation_times,
        mutation_alleles=mutation_alleles,
        mutation_ref_alleles=mutation_ref_alleles,
        coalescence_counts=coalescence_counts,
    )

__all__ = ["CompiledOperatorState", "_invert_permutation", "compile_grg"]
