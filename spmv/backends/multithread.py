"""
Multithread Backend - Multi-threaded SpMV using dynamic chunk scheduling.

Each thread processes chunks of rows on-demand using CSR views (no copying).
"""

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import List
import multiprocessing

import numpy as np
import scipy.sparse as sp

from spmv.backends import Backend


class MultithreadBackend(Backend):
    """
    Multithread-based SpMV backend with multi-threading.

    Uses dynamic scheduling where each thread processes row chunks on-demand.
    CSR views are created on-the-fly to avoid memory copying.

    Parameters
    ----------
    n_workers : int
        Number of parallel workers. 0 = auto (use all CPUs), 1 = sequential.
    chunk_size : int
        Number of rows per chunk for parallel execution.
    verbose : bool
        Print verbose information (matrix sizes, nnz, timing).
    """

    def __init__(self, n_workers: int, chunk_size: int, verbose: bool = False):
        self._n_workers = multiprocessing.cpu_count() if n_workers == 0 else n_workers
        self._chunk_size = chunk_size
        self._verbose = verbose

    @property
    def n_workers(self) -> int:
        return self._n_workers

    def setup(
        self,
        A_blocks: List[List[sp.csr_matrix]],
        AT_blocks: List[List[sp.csr_matrix]],
        level_offsets: np.ndarray,
        n: int,
        K: int,
        sel: sp.csr_matrix,
        sel_T: sp.csr_matrix,
        sample_perm: np.ndarray,
        inv_sample_perm: np.ndarray,
        dtype: np.dtype,
    ) -> None:
        """Store references to block-wise sparse matrices and permutations."""
        self._A_blocks = A_blocks
        self._AT_blocks = AT_blocks
        self._level_offsets = level_offsets
        self._n = n
        self._K = K
        self._sel = sel
        self._sel_T = sel_T
        self._sample_perm = sample_perm
        self._inv_sample_perm = inv_sample_perm
        self._dtype = dtype
        if self._verbose:
            total_nnz_fwd = sum(blk.nnz for blocks in A_blocks for blk in blocks)
            total_nnz_bwd = sum(blk.nnz for blocks in AT_blocks for blk in blocks)
            print(f"MultithreadBackend setup: {len(A_blocks)} levels")
            print(f"  A_blocks total nnz: {total_nnz_fwd:,}")
            print(f"  AT_blocks total nnz: {total_nnz_bwd:,}")
            for h in range(len(A_blocks)):
                lo, hi = level_offsets[h], level_offsets[h + 1]
                fwd_shapes = [blk.shape for blk in A_blocks[h]]
                bwd_shapes = [blk.shape for blk in AT_blocks[h]]
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

        if self._verbose:
            print(f"Forward pass: X.shape={X.shape}")

        for h in range(1, len(off) - 1):
            lo, hi = off[h], off[h + 1]
            for j, blk in enumerate(self._A_blocks[h]):
                jlo, jhi = off[j], off[j + 1]
                if blk.nnz == 0:
                    continue
                if self._verbose:
                    t0 = perf_counter()
                U[lo:hi] += self._spmv(blk, U[jlo:jhi])
                if self._verbose:
                    elapsed = (perf_counter() - t0) * 1e3
                    print(f"  Level {h}, block {j}: shape={blk.shape}, "
                          f"nnz={blk.nnz:,}, time={elapsed:.2f}ms")

        return self._spmv(self._sel, U)

    def backward_matmat(self, X: np.ndarray) -> np.ndarray:
        """
        Compute G @ X: (m x k) -> (n x k).

        1. V = sel_T @ X
        2. For h=H-2..0: for j, block in AT_blocks[h]: V[off[h]:off[h+1]] += block @ V[off[h+1+j]:off[h+2+j]]
        3. Return V[inv_sample_perm]
        """
        off = self._level_offsets
        X = np.atleast_2d(X)
        V = self._spmv(self._sel_T, X)

        if self._verbose:
            print(f"Backward pass: X.shape={X.shape}")

        for h in range(len(off) - 2, -1, -1):
            lo, hi = off[h], off[h + 1]
            for j, blk in enumerate(self._AT_blocks[h]):
                src_lo, src_hi = off[h + 1 + j], off[h + 2 + j]
                if blk.nnz == 0:
                    continue
                if self._verbose:
                    t0 = perf_counter()
                V[lo:hi] += self._spmv(blk, V[src_lo:src_hi])
                if self._verbose:
                    elapsed = (perf_counter() - t0) * 1e3
                    print(f"  Level {h}, block {j}: shape={blk.shape}, "
                          f"nnz={blk.nnz:,}, time={elapsed:.2f}ms")

        return V[self._inv_sample_perm]

    @staticmethod
    def _csr_row_slice(A: sp.csr_matrix, start: int, end: int) -> sp.csr_matrix:
        """Create a CSR view of rows [start:end] without copying data."""
        return sp.csr_matrix(
            (A.data[A.indptr[start]:A.indptr[end]],
             A.indices[A.indptr[start]:A.indptr[end]],
             A.indptr[start:end+1] - A.indptr[start]),
            shape=(end - start, A.shape[1])
        )

    def _spmv(self, A: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
        """Compute A @ x using dynamic chunk scheduling."""
        nrows = A.shape[0]

        if self._n_workers <= 1 or nrows < self._n_workers:
            return A @ x

        is_1d = x.ndim == 1
        chunk_size = self._chunk_size

        if is_1d:
            result = np.zeros(nrows, dtype=x.dtype)
        else:
            result = np.zeros((nrows, x.shape[1]), dtype=x.dtype)

        def process_chunk(start: int) -> None:
            """Process a chunk of rows."""
            end = min(start + chunk_size, nrows)
            A_chunk = self._csr_row_slice(A, start, end)
            chunk_result = A_chunk @ x
            if is_1d:
                result[start:end] = chunk_result
            else:
                result[start:end, :] = chunk_result

        chunks = list(range(0, nrows, chunk_size))

        with ThreadPoolExecutor(max_workers=self._n_workers) as executor:
            list(executor.map(process_chunk, chunks))

        return result
