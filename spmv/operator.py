"""
SpMVOperator - Level-wise SpMV operator for GRG-based genotype matrices.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pygrgl
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import LinearOperator

from spmv.io import save_operator_npz, load_operator_npz, _extract_block
from spmv.backends import get_backend_class


def _create_backend(config: dict[str, Any]):
    """
    Create a backend from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Backend configuration with keys:
        - 'type': str — Backend type ('spsparse', 'mkl', 'cusparse')
        - Additional backend-specific parameters
    """
    backend_type = config.get('type', 'spsparse')
    verbose = config.get('verbose', False)
    cls = get_backend_class(backend_type)

    match backend_type:
        case 'spsparse':
            return cls(config.get('n_workers', 1),
                       config.get('chunk_size', 4096), verbose)
        case 'mkl':
            return cls(config.get('n_threads', 0), verbose)
        case 'cusparse':
            return cls(fmt=config.get('fmt', 'csr'), k=config.get('k'),
                       algorithm=config.get('algorithm', 'default'),
                       verbose=verbose)
        case _:
            raise ValueError(f"No config handler for backend: {backend_type!r}")


class SpMVOperator(LinearOperator):
    """
    LinearOperator for GRG-based genotype matrix G (n x m).

    Replaces a single SpTRSV with H level-wise SpMVs, where H is the
    maximum node height.

    Parameters
    ----------
    path : str or Path
        Path to a .grg file.
    backend_config : dict
        Backend configuration with keys:
        - 'type': str — Backend type ('spsparse', 'mkl', 'cusparse')
        - Additional backend-specific parameters (e.g., 'n_workers', 'n_threads')
    dtype : numpy dtype
        Data type for computation.
    index_dtype : numpy dtype
        Index data type for permutation arrays and indices.
    use_rcm : bool
        Apply Reverse Cuthill-McKee reordering for cache locality.

    Public Attributes
    -----------------
    n : int - Number of samples (rows of G).
    m : int - Number of mutations (columns of G).
    K : int - Total number of nodes in the GRG.
    level_offsets : ndarray - Boundary indices for height levels.
    sample_perm : ndarray - Maps new sample indices to original.
    sel : csr_matrix - Selector matrix (m x K).
    """

    def __init__(self, path, backend_config: dict[str, Any], dtype, index_dtype, use_rcm: bool = True):
        self._use_rcm = use_rcm
        self._dtype = np.dtype(dtype)
        self._index_dtype = np.dtype(index_dtype)
        self._backend = _create_backend(backend_config)

        path = Path(path)
        assert path.suffix == ".grg", f"Not a GRG file: {path}"
        algo = "rcm" if use_rcm else "none"
        npz_path = path.with_suffix(f".spmv.{algo}.npz")

        if npz_path.exists():
            print(f"Loading cached SpMVOperator from {npz_path}")
            self._load_from_npz(npz_path)
        else:
            print(f"Building SpMVOperator with {path}")
            grg = pygrgl.load_immutable_grg(str(path))
            self._build_from_grg(grg)
            save_operator_npz(self, self.A_blocks, npz_path, algo)

        # Initialize backend with block-wise matrices and permutations
        self._backend.setup(
            self.A_blocks, self.level_offsets,
            self.n, self.K, self.sel,
            self.sample_perm, self._inv_sample_perm, self._dtype,
        )
        del self.A_blocks

    def _build_from_grg(self, grg):
        """Build SpMVOperator from a GRG object."""
        n, m, K = grg.num_samples, grg.num_mutations, grg.num_nodes
        dtype = self._dtype
        index_dtype = self._index_dtype

        # 1. Extract edges and compute heights
        rows, cols = [], []
        heights = np.zeros(K, dtype=index_dtype)
        for i in range(K):
            children = grg.get_down_edges(i)
            for c in children:
                rows.append(i)
                cols.append(c)
            if children:
                heights[i] = max(heights[c] for c in children) + 1
        E = len(rows)
        rows = np.array(rows, dtype=index_dtype)
        cols = np.array(cols, dtype=index_dtype)

        # 2. Height-sorted permutation
        perm_height = np.argsort(heights, kind='stable')
        sorted_heights = heights[perm_height]
        assert np.all(np.diff(sorted_heights) >= 0), "Heights must be non-decreasing!"

        changes = np.flatnonzero(np.diff(sorted_heights)) + 1
        self.level_offsets = np.concatenate([[0], changes, [K]])
        num_levels = len(self.level_offsets) - 1

        # 3. Build height-sorted adjacency matrix
        inv_perm_height = np.empty(K, dtype=index_dtype)
        inv_perm_height[perm_height] = np.arange(K)
        rows_h = inv_perm_height[rows]
        cols_h = inv_perm_height[cols]
        ones_E = np.ones(E, dtype=dtype)
        A_h = sp.csr_matrix((ones_E, (rows_h, cols_h)), shape=(K, K))

        # 4. Compute RCM permutations per level (optional)
        level_perms = [np.arange(self.level_offsets[h+1] - self.level_offsets[h])
                       for h in range(num_levels)]

        if self._use_rcm:
            for h in range(1, num_levels):
                row_lo, row_hi = self.level_offsets[h], self.level_offsets[h + 1]
                col_lo, col_hi = self.level_offsets[h - 1], self.level_offsets[h]

                # Extract block A_{h,h-1}
                A_block = A_h[row_lo:row_hi, col_lo:col_hi]
                if A_block.nnz == 0:
                    continue
                if h == 1:
                    # Level 1: jointly permute rows (level 1) AND columns (level 0)
                    # via bipartite RCM for optimal cache locality
                    row_perm, col_perm = self._rcm_bipartite(A_block, index_dtype)
                    level_perms[1] = row_perm
                    level_perms[0] = col_perm
                else:
                    # Levels 2+: columns already fixed; sort rows by min column index
                    level_perms[h] = self._min_col_perm(A_block, index_dtype)

        # 5. Compose permutations
        within_perm = np.arange(K, dtype=index_dtype)
        for h in range(num_levels):
            lo, hi = self.level_offsets[h], self.level_offsets[h + 1]
            within_perm[lo:hi] = lo + level_perms[h]

        final_perm = perm_height[within_perm]
        inv_final_perm = np.empty(K, dtype=index_dtype)
        inv_final_perm[final_perm] = np.arange(K)

        self.sample_perm = final_perm[:n].copy()
        self._inv_sample_perm = inv_final_perm[:n].copy()

        new_rows = inv_final_perm[rows]
        new_cols = inv_final_perm[cols]
        del rows, cols, A_h

        # 6. Build A and extract block-wise matrices
        off = self.level_offsets
        A = sp.csr_matrix((ones_E, (new_rows, new_cols)), shape=(K, K))

        self.A_blocks = []
        for h in range(num_levels):
            lo, hi = off[h], off[h + 1]
            self.A_blocks.append([
                _extract_block(A, lo, hi, off[j], off[j + 1], dtype, index_dtype)
                for j in range(h)
            ])

        # Assert block shapes
        for h in range(num_levels):
            assert len(self.A_blocks[h]) == h
            for j, blk in enumerate(self.A_blocks[h]):
                assert blk.shape == (off[h+1]-off[h], off[j+1]-off[j])

        # 7. Selector matrix
        pairs = grg.get_mutation_node_pairs()
        pair_mut_ids = []
        pair_node_ids_orig = []
        for mid, nid in pairs:
            if nid != pygrgl.INVALID_NODE:
                pair_mut_ids.append(mid)
                pair_node_ids_orig.append(nid)

        pair_mut_ids = np.array(pair_mut_ids, dtype=index_dtype)
        pair_node_ids = inv_final_perm[np.array(pair_node_ids_orig, dtype=index_dtype)]
        del pair_node_ids_orig

        sel_ones = np.ones(len(pair_mut_ids), dtype=dtype)
        self.sel = sp.csr_matrix((sel_ones, (pair_mut_ids, pair_node_ids)), shape=(m, K))

        self.n = n
        self.m = m
        self.K = K
        super().__init__(dtype=dtype, shape=(n, m))

    @staticmethod
    def _rcm_bipartite(A, index_dtype):
        """
        Bipartite RCM for a rectangular block A_{h,h-1}.

        Builds the augmented symmetric matrix B = [[0, A], [A^T, 0]] and
        applies Reverse Cuthill-McKee.  Returns *both* row and column
        permutations so that the caller can reorder both levels jointly.
        """
        nrows, ncols = A.shape
        A_csr = sp.csr_matrix(A)

        top = sp.hstack([sp.csr_matrix((nrows, nrows)), A_csr], format='csr')
        bottom = sp.hstack([A_csr.T, sp.csr_matrix((ncols, ncols))], format='csr')
        B = sp.vstack([top, bottom], format='csr')

        perm_full = reverse_cuthill_mckee(B, symmetric_mode=True)

        row_indices = []
        col_indices = []
        for idx in perm_full:
            if idx < nrows:
                row_indices.append(idx)
            else:
                col_indices.append(idx - nrows)

        # Handle disconnected components (nodes missing from RCM output)
        row_set = set(row_indices)
        col_set = set(col_indices)
        for i in range(nrows):
            if i not in row_set:
                row_indices.append(i)
        for i in range(ncols):
            if i not in col_set:
                col_indices.append(i)

        return np.array(row_indices, dtype=index_dtype), np.array(col_indices, dtype=index_dtype)

    @staticmethod
    def _min_col_perm(A, index_dtype):
        """Sort rows by their minimum column index (cache-friendly ordering)."""
        A_csr = sp.csr_matrix(A)
        nrows, ncols = A_csr.shape
        min_col = np.full(nrows, ncols, dtype=np.int64)
        for i in range(nrows):
            s, e = A_csr.indptr[i], A_csr.indptr[i + 1]
            if e > s:
                min_col[i] = A_csr.indices[s:e].min()
        return np.argsort(min_col, kind='stable').astype(index_dtype)

    def _load_from_npz(self, npz_path):
        """Load operator state from NPZ file."""
        data = load_operator_npz(npz_path, self._dtype, self._index_dtype)
        self.n = data['n']
        self.m = data['m']
        self.K = data['K']
        self.level_offsets = data['level_offsets']
        self.sample_perm = data['sample_perm']
        self._inv_sample_perm = data['_inv_sample_perm']
        self.sel = data['sel']
        self.A_blocks = data['A_blocks']
        super().__init__(dtype=self._dtype, shape=(self.n, self.m))

    def _matmat(self, X):
        """G @ X for matrix X (m x k)."""
        return self._backend.backward_matmat(np.atleast_2d(X))

    def _rmatmat(self, X):
        """G^T @ X for matrix X (n x k)."""
        return self._backend.forward_matmat(np.atleast_2d(X))
