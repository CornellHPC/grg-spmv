"""GRG compilation into level-ordered sparse operator state."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np
import pygrgl
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee

from pygrgl_spmv.backends import BackendSetup
from pygrgl_spmv.grg.sparse import binary_csr_from_coo

VALID_ORDERINGS = frozenset({"height", "depth"})
VALID_INTRA_BLOCK_ORDERINGS = frozenset({"rcm_mincol", "none"})


@dataclass
class CompiledOperatorState:
    """Compiled operator state shared between cache I/O and runtime setup."""

    A_blocks: list[list[sp.csr_matrix]] | None
    level_offsets: np.ndarray
    node_perm: np.ndarray
    inv_node_perm: np.ndarray
    sample_rows: np.ndarray
    sel_mut: sp.csr_matrix
    sel_miss: sp.csr_matrix
    num_samples: int
    num_mutations: int
    num_nodes: int
    ploidy: int
    num_individuals: int
    num_edges: int
    has_missing_data: bool
    ordering: str
    intra_block_ordering: str
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
            sample_rows=self.sample_rows,
            coalescence_counts=self.coalescence_counts,
            dtype=dtype,
        )


def _validate_ordering(ordering: str) -> str:
    token = str(ordering).strip().lower()
    if token not in VALID_ORDERINGS:
        raise ValueError(f"Unknown ordering {ordering!r}; expected one of {sorted(VALID_ORDERINGS)}")
    return token


def _validate_intra_block_ordering(intra_block_ordering: str) -> str:
    token = str(intra_block_ordering).strip().lower()
    if token not in VALID_INTRA_BLOCK_ORDERINGS:
        raise ValueError(
            "Unknown intra_block_ordering "
            f"{intra_block_ordering!r}; expected one of {sorted(VALID_INTRA_BLOCK_ORDERINGS)}"
        )
    return token


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


def _extract_edges(
    grg,
    K: int,
    *,
    index_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    """Extract directed edges and child lists from the GRG."""
    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    children_by_node: list[np.ndarray] = []
    indegree = np.zeros(K, dtype=np.int64)

    for node_id in range(K):
        children = np.asarray(grg.get_down_edges(node_id), dtype=index_dtype)
        children_by_node.append(children)
        if children.size == 0:
            continue
        row_chunks.append(np.full(int(children.size), node_id, dtype=index_dtype))
        col_chunks.append(children)
        np.add.at(indegree, children.astype(np.int64, copy=False), 1)

    if row_chunks:
        row_arr = np.concatenate(row_chunks)
        col_arr = np.concatenate(col_chunks)
    else:
        row_arr = np.empty(0, dtype=index_dtype)
        col_arr = np.empty(0, dtype=index_dtype)
    return row_arr, col_arr, tuple(children_by_node), indegree


def _topological_order(children_by_node: tuple[np.ndarray, ...], indegree: np.ndarray) -> np.ndarray:
    """Return a stable topological order from sources to sinks."""
    K = len(children_by_node)
    indegree_work = np.asarray(indegree, dtype=np.int64).copy()
    frontier = [int(node_id) for node_id in range(K) if int(indegree_work[node_id]) == 0]
    heapq.heapify(frontier)

    topo = np.empty(K, dtype=np.int64)
    out_idx = 0
    while frontier:
        node_id = heapq.heappop(frontier)
        topo[out_idx] = int(node_id)
        out_idx += 1
        for child_id in children_by_node[node_id]:
            child = int(child_id)
            indegree_work[child] -= 1
            if int(indegree_work[child]) == 0:
                heapq.heappush(frontier, child)

    if out_idx != K:
        raise RuntimeError(f"Invalid GRG topology: expected a DAG, visited {out_idx}/{K} nodes in topological sort")
    return topo


def _compute_depths_and_heights(
    children_by_node: tuple[np.ndarray, ...],
    topo_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-node depths and heights from the DAG."""
    K = len(children_by_node)
    depths = np.zeros(K, dtype=np.int64)
    heights = np.zeros(K, dtype=np.int64)

    for node_id in topo_order:
        parent = int(node_id)
        child_depth = int(depths[parent]) + 1
        for child_id in children_by_node[parent]:
            child = int(child_id)
            if child_depth > int(depths[child]):
                depths[child] = child_depth

    for node_id in topo_order[::-1]:
        node = int(node_id)
        children = children_by_node[node]
        if children.size == 0:
            continue
        heights[node] = int(heights[children].max()) + 1

    return depths, heights


def _build_level_layout(
    level_values: np.ndarray,
    *,
    index_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build node order and level offsets from non-decreasing level values."""
    K = int(level_values.shape[0])
    node_order = np.argsort(level_values, kind="stable").astype(index_dtype, copy=False)
    inv_node_order = np.empty(K, dtype=index_dtype)
    inv_node_order[node_order] = np.arange(K, dtype=index_dtype)

    sorted_levels = level_values[node_order]
    if np.any(np.diff(sorted_levels) < 0):
        raise RuntimeError("Level values must be non-decreasing after sorting")
    changes = np.flatnonzero(np.diff(sorted_levels)) + 1
    level_offsets = np.concatenate(
        [
            np.array([0], dtype=index_dtype),
            changes.astype(index_dtype, copy=False),
            np.array([K], dtype=index_dtype),
        ]
    )

    num_levels = int(level_offsets.size - 1)
    node_levels = np.empty(K, dtype=np.int32)
    for level in range(num_levels):
        lo, hi = int(level_offsets[level]), int(level_offsets[level + 1])
        node_levels[node_order[lo:hi]] = level

    return node_order, inv_node_order, level_offsets, node_levels


def _build_intra_level_perms(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    inv_level_order: np.ndarray,
    level_offsets: np.ndarray,
    dst_levels: np.ndarray,
    src_order: np.ndarray,
    src_offsets: np.ndarray,
    index_dtype: np.dtype,
    dtype: np.dtype,
    intra_block_ordering: str,
) -> list[np.ndarray]:
    """Compute within-level permutations."""
    num_levels = int(level_offsets.size - 1)
    perms = [
        np.arange(int(level_offsets[level + 1] - level_offsets[level]), dtype=index_dtype)
        for level in range(num_levels)
    ]
    if intra_block_ordering == "none":
        return perms
    if intra_block_ordering != "rcm_mincol":
        raise ValueError(f"Unhandled intra_block_ordering {intra_block_ordering!r}")

    for level in range(1, num_levels):
        lo_e = int(src_offsets[level])
        hi_e = int(src_offsets[level + 1])
        if lo_e == hi_e:
            continue
        edge_idx = src_order[lo_e:hi_e]
        prev_level_edges = edge_idx[dst_levels[edge_idx] == (level - 1)]
        if prev_level_edges.size == 0:
            continue

        row_local = (inv_level_order[rows[prev_level_edges]] - level_offsets[level]).astype(np.int64, copy=False)
        col_local = (inv_level_order[cols[prev_level_edges]] - level_offsets[level - 1]).astype(np.int64, copy=False)
        row_size = int(level_offsets[level + 1] - level_offsets[level])
        col_size = int(level_offsets[level] - level_offsets[level - 1])

        if level == 1:
            block = binary_csr_from_coo(
                row_local,
                col_local,
                shape=(row_size, col_size),
                dtype=dtype,
            )
            row_perm, col_perm = _rcm_bipartite(block, index_dtype)
            perms[1] = row_perm
            perms[0] = col_perm
            continue

        min_col = np.full(row_size, col_size, dtype=np.int64)
        np.minimum.at(min_col, row_local, col_local)
        perms[level] = np.argsort(min_col, kind="stable").astype(index_dtype, copy=False)

    return perms


def _build_blocks_from_edges(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    inv_node_perm: np.ndarray,
    level_offsets: np.ndarray,
    dst_levels: np.ndarray,
    src_order: np.ndarray,
    src_offsets: np.ndarray,
    dtype: np.dtype,
) -> list[list[sp.csr_matrix]]:
    """Build block CSR matrices directly from edge buckets."""
    num_levels = int(level_offsets.size - 1)
    A_blocks: list[list[sp.csr_matrix]] = []

    for level in range(num_levels):
        row_size = int(level_offsets[level + 1] - level_offsets[level])
        level_blocks = [
            sp.csr_matrix((row_size, int(level_offsets[src + 1] - level_offsets[src])), dtype=dtype)
            for src in range(level)
        ]
        lo_e = int(src_offsets[level])
        hi_e = int(src_offsets[level + 1])
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
            src_level = int(dst_sorted[int(b0)])
            if src_level >= level:
                raise RuntimeError(
                    f"Invalid GRG edge bucket (source level {level}, target level {src_level}); expected target < source"
                )
            pair_idx = edge_sorted[int(b0) : int(b1)]
            if pair_idx.size == 0:
                continue
            row_local = (inv_node_perm[rows[pair_idx]] - level_offsets[level]).astype(np.int64, copy=False)
            col_local = (inv_node_perm[cols[pair_idx]] - level_offsets[src_level]).astype(np.int64, copy=False)
            col_size = int(level_offsets[src_level + 1] - level_offsets[src_level])
            level_blocks[src_level] = binary_csr_from_coo(
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
    inv_node_perm: np.ndarray,
    shape: tuple[int, int],
    index_dtype: np.dtype,
    dtype: np.dtype,
) -> sp.csr_matrix:
    if not rows:
        return sp.csr_matrix(shape, dtype=dtype)
    row_arr = np.asarray(rows, dtype=index_dtype)
    col_arr = inv_node_perm[np.asarray(cols_orig, dtype=index_dtype)]
    return binary_csr_from_coo(row_arr, col_arr, shape=shape, dtype=dtype)


def _build_selectors(
    grg,
    *,
    inv_node_perm: np.ndarray,
    m: int,
    K: int,
    index_dtype: np.dtype,
    dtype: np.dtype,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
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
        inv_node_perm=inv_node_perm,
        shape=(m, K),
        index_dtype=index_dtype,
        dtype=dtype,
    )
    sel_miss = _build_binary_selector(
        miss_rows,
        miss_cols_orig,
        inv_node_perm=inv_node_perm,
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


def _validate_sample_rows(sample_rows: np.ndarray, *, num_samples: int, num_nodes: int) -> np.ndarray:
    rows = np.asarray(sample_rows)
    if rows.shape != (num_samples,):
        raise RuntimeError(f"Invalid sample_rows shape: got {rows.shape}, expected ({num_samples},)")
    if rows.size == 0:
        return rows
    if np.any(rows < 0) or np.any(rows >= num_nodes):
        raise RuntimeError("sample_rows contains out-of-range compiled node rows")
    if np.unique(rows).size != rows.size:
        raise RuntimeError("sample_rows must contain unique compiled node rows")
    return rows


def compile_grg(
    grg,
    *,
    dtype: np.dtype,
    index_dtype: np.dtype,
    ordering: str = "height",
    intra_block_ordering: str = "rcm_mincol",
) -> CompiledOperatorState:
    """Compile a GRG object into the normalized sparse traversal layout."""
    ordering = _validate_ordering(ordering)
    intra_block_ordering = _validate_intra_block_ordering(intra_block_ordering)

    num_samples = int(grg.num_samples)
    num_mutations = int(grg.num_mutations)
    num_nodes = int(grg.num_nodes)

    rows, cols, children_by_node, indegree = _extract_edges(grg, num_nodes, index_dtype=index_dtype)
    topo_order = _topological_order(children_by_node, indegree)
    depths, heights = _compute_depths_and_heights(children_by_node, topo_order)
    if ordering == "height":
        level_values = heights
    else:
        level_values = int(np.max(depths, initial=0)) - depths

    level_order, inv_level_order, level_offsets, node_levels = _build_level_layout(
        level_values,
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

    intra_level_perms = _build_intra_level_perms(
        rows,
        cols,
        inv_level_order=inv_level_order,
        level_offsets=level_offsets,
        dst_levels=dst_levels,
        src_order=src_order,
        src_offsets=src_offsets,
        index_dtype=index_dtype,
        dtype=dtype,
        intra_block_ordering=intra_block_ordering,
    )

    within_perm = np.arange(num_nodes, dtype=index_dtype)
    for level in range(num_levels):
        lo, hi = int(level_offsets[level]), int(level_offsets[level + 1])
        perm = intra_level_perms[level]
        if perm.shape[0] != (hi - lo):
            raise RuntimeError(f"Invalid intra-level permutation for level {level}: got {perm.shape[0]}, expected {hi - lo}")
        within_perm[lo:hi] = lo + perm

    node_perm = level_order[within_perm]
    inv_node_perm = _invert_permutation(node_perm, index_dtype=index_dtype)

    A_blocks = _build_blocks_from_edges(
        rows,
        cols,
        inv_node_perm=inv_node_perm,
        level_offsets=level_offsets,
        dst_levels=dst_levels,
        src_order=src_order,
        src_offsets=src_offsets,
        dtype=dtype,
    )

    for level in range(num_levels):
        if len(A_blocks[level]) != level:
            raise RuntimeError(f"Invalid number of blocks at level {level}: got {len(A_blocks[level])}, expected {level}")
        for src_level, block in enumerate(A_blocks[level]):
            expected_shape = (
                int(level_offsets[level + 1] - level_offsets[level]),
                int(level_offsets[src_level + 1] - level_offsets[src_level]),
            )
            if block.shape != expected_shape:
                raise RuntimeError(
                    f"Invalid block shape for A_blocks[{level}][{src_level}]: got {block.shape}, expected {expected_shape}"
                )

    sel_mut, sel_miss = _build_selectors(
        grg,
        inv_node_perm=inv_node_perm,
        m=num_mutations,
        K=num_nodes,
        index_dtype=index_dtype,
        dtype=dtype,
    )

    sample_rows = _validate_sample_rows(
        inv_node_perm[np.arange(num_samples, dtype=index_dtype)].copy(),
        num_samples=num_samples,
        num_nodes=num_nodes,
    )
    sample_to_individual = np.arange(num_samples, dtype=index_dtype) // max(int(grg.ploidy), 1)
    coalescence_counts = _build_coalescence_counts(grg, node_perm=node_perm)
    (
        mutation_positions,
        mutation_times,
        mutation_alleles,
        mutation_allele_offsets,
        mutation_ref_alleles,
        mutation_ref_allele_offsets,
    ) = _build_mutation_table(grg)

    return CompiledOperatorState(
        A_blocks=A_blocks,
        level_offsets=level_offsets,
        node_perm=node_perm.copy(),
        inv_node_perm=inv_node_perm.copy(),
        sample_rows=sample_rows,
        sel_mut=sel_mut,
        sel_miss=sel_miss,
        num_samples=num_samples,
        num_mutations=num_mutations,
        num_nodes=num_nodes,
        ploidy=int(grg.ploidy),
        num_individuals=int(grg.num_individuals),
        num_edges=int(grg.num_edges),
        has_missing_data=bool(grg.has_missing_data),
        ordering=ordering,
        intra_block_ordering=intra_block_ordering,
        sample_to_individual=sample_to_individual,
        mutation_positions=mutation_positions,
        mutation_times=mutation_times,
        mutation_alleles=mutation_alleles,
        mutation_allele_offsets=mutation_allele_offsets,
        mutation_ref_alleles=mutation_ref_alleles,
        mutation_ref_allele_offsets=mutation_ref_allele_offsets,
        coalescence_counts=coalescence_counts,
    )


__all__ = [
    "CompiledOperatorState",
    "VALID_INTRA_BLOCK_ORDERINGS",
    "VALID_ORDERINGS",
    "_invert_permutation",
    "compile_grg",
]
