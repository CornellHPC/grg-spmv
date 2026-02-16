"""MKL Backend — Intel MKL-accelerated SpMM via sparse_dot_mkl."""

from typing import List

import numpy as np
import scipy.sparse as sp
from sparse_dot_mkl import dot_product_mkl, mkl_set_num_threads

from spmv.backends import Backend


class MklBackend(Backend):
    """
    MKL-accelerated SpMV backend.

    Uses sparse_dot_mkl for all SpMM calls. MKL handles parallelism
    internally — no ThreadPoolExecutor needed.

    Parameters
    ----------
    n_threads : int
        Number of MKL threads. 0 = use MKL default.
    verbose : bool
        Print verbose info.
    """

    def __init__(self, n_threads: int = 0, verbose: bool = False):
        self._n_threads = n_threads
        self._verbose = verbose

    @property
    def n_threads(self) -> int:
        return self._n_threads

    def setup(
        self,
        A_blocks: List[List[sp.csr_matrix]],
        level_offsets: np.ndarray,
        n: int,
        K: int,
        sel: sp.csr_matrix,
        sample_perm: np.ndarray,
        inv_sample_perm: np.ndarray,
        dtype: np.dtype,
    ) -> None:
        if self._n_threads > 0:
            mkl_set_num_threads(self._n_threads)

        self._A_blocks = A_blocks
        self._level_offsets = level_offsets
        self._n = n
        self._K = K
        self._sel = sel
        self._sel_T = sel.T.tocsr()
        self._sample_perm = sample_perm
        self._inv_sample_perm = inv_sample_perm
        self._dtype = dtype

        # Compute AT_blocks internally from A_blocks
        num_levels = len(level_offsets) - 1
        self._AT_blocks = []
        for h in range(num_levels):
            row = []
            for j in range(num_levels - 1 - h):
                src = h + 1 + j
                row.append(A_blocks[src][h].T.tocsr())
            self._AT_blocks.append(row)

        if self._verbose:
            total_nnz_fwd = sum(blk.nnz for blocks in A_blocks for blk in blocks)
            total_nnz_bwd = sum(blk.nnz for blocks in self._AT_blocks for blk in blocks)
            print(f"MklBackend setup: {len(A_blocks)} levels, "
                  f"n_threads={self._n_threads}")
            print(f"  A_blocks total nnz: {total_nnz_fwd:,}")
            print(f"  AT_blocks total nnz: {total_nnz_bwd:,}")
            for h in range(len(A_blocks)):
                fwd_shapes = [blk.shape for blk in A_blocks[h]]
                bwd_shapes = [blk.shape for blk in self._AT_blocks[h]]
                print(f"  Level {h}: A_blocks={fwd_shapes}, AT_blocks={bwd_shapes}")

    def forward_matmat(self, X: np.ndarray) -> np.ndarray:
        """
        Compute G^T @ X: (n x k) -> (m x k).

        1. U = zeros(K, k); U[:n] = X[sample_perm]
        2. For h=1..H-1: for j<h: U[off[h]:off[h+1]] += A_blocks[h][j] @ U[off[j]:off[j+1]]
        3. Return sel @ U
        """
        off = self._level_offsets
        X = np.atleast_2d(X)
        k = X.shape[1]
        U = np.zeros((self._K, k), dtype=self._dtype)
        U[:self._n] = X[self._sample_perm]

        for h in range(1, len(off) - 1):
            lo, hi = off[h], off[h + 1]
            for j, blk in enumerate(self._A_blocks[h]):
                jlo, jhi = off[j], off[j + 1]
                if blk.nnz == 0:
                    continue
                U[lo:hi] += dot_product_mkl(blk, U[jlo:jhi])

        return dot_product_mkl(self._sel, U)

    def backward_matmat(self, X: np.ndarray) -> np.ndarray:
        """
        Compute G @ X: (m x k) -> (n x k).

        1. V = sel_T @ X
        2. For h=H-2..0: for j, block in AT_blocks[h]: V[off[h]:off[h+1]] += block @ V[off[h+1+j]:off[h+2+j]]
        3. Return V[inv_sample_perm]
        """
        off = self._level_offsets
        X = np.atleast_2d(X)
        V = dot_product_mkl(self._sel_T, X)

        for h in range(len(off) - 2, -1, -1):
            lo, hi = off[h], off[h + 1]
            for j, blk in enumerate(self._AT_blocks[h]):
                src_lo, src_hi = off[h + 1 + j], off[h + 2 + j]
                if blk.nnz == 0:
                    continue
                V[lo:hi] += dot_product_mkl(blk, V[src_lo:src_hi])

        return V[self._inv_sample_perm]
