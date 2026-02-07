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
        self._A_fwd = None
        self._AT_bwd = None
        self._level_offsets = None

    @property
    def n_workers(self) -> int:
        return self._n_workers

    def setup(
        self,
        A_fwd: List[sp.csr_matrix],
        AT_bwd: List[sp.csr_matrix],
        level_offsets: np.ndarray
    ) -> None:
        """Store references to the sparse matrices and level offsets."""
        self._A_fwd = A_fwd
        self._AT_bwd = AT_bwd
        self._level_offsets = level_offsets
        if self._verbose:
            total_nnz_fwd = sum(A.nnz for A in A_fwd)
            total_nnz_bwd = sum(A.nnz for A in AT_bwd)
            print(f"MultithreadBackend setup: {len(A_fwd)} levels")
            print(f"  A_fwd total nnz: {total_nnz_fwd:,}")
            print(f"  AT_bwd total nnz: {total_nnz_bwd:,}")
            for h in range(len(A_fwd)):
                lo, hi = level_offsets[h], level_offsets[h + 1]
                print(f"  Level {h}: A_fwd={A_fwd[h].shape}, AT_bwd={AT_bwd[h].shape}")

    def forward_matmat(self, U: np.ndarray) -> np.ndarray:
        """
        Compute forward pass: U[lo:hi] = A_fwd[h] @ U[:lo] for each level h.
        
        A_fwd[h] has shape (hi-lo) × lo, so multiply by U[:lo].
        """
        off = self._level_offsets
        if self._verbose:
            print(f"Forward pass: U.shape={U.shape}")
        for h in range(1, len(off) - 1):
            lo, hi = off[h], off[h + 1]
            if self._verbose:
                t0 = perf_counter()
            # A_fwd[h] has columns [0:lo], so multiply by U[:lo]
            U[lo:hi] = self._spmv(self._A_fwd[h], U[:lo])
            if self._verbose:
                elapsed = (perf_counter() - t0) * 1e3
                print(f"  Level {h}: A_fwd[{h}].shape={self._A_fwd[h].shape}, "
                      f"nnz={self._A_fwd[h].nnz:,}, time={elapsed:.2f}ms")
        return U

    def backward_matmat(self, V: np.ndarray) -> np.ndarray:
        """
        Compute backward pass: V[lo:hi] += AT_bwd[h] @ V[hi:] for each level h in reverse.
        
        AT_bwd[h] has shape (hi-lo) × (K-hi), so multiply by V[hi:].
        """
        off = self._level_offsets
        K = off[-1]
        if self._verbose:
            print(f"Backward pass: V.shape={V.shape}")
        # Process levels from H-1 down to 0 (inclusive)
        for h in range(len(off) - 2, -1, -1):
            lo, hi = off[h], off[h + 1]
            if self._verbose:
                t0 = perf_counter()
            # AT_bwd[h] has columns [hi:K], so multiply by V[hi:]
            V[lo:hi] += self._spmv(self._AT_bwd[h], V[hi:])
            if self._verbose:
                elapsed = (perf_counter() - t0) * 1e3
                print(f"  Level {h}: AT_bwd[{h}].shape={self._AT_bwd[h].shape}, "
                      f"nnz={self._AT_bwd[h].nnz:,}, time={elapsed:.2f}ms")
        return V

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
