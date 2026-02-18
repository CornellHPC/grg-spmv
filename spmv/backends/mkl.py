"""MKL Backend — Intel MKL-accelerated SpMM via ctypes Inspector-Executor API."""

import os
from functools import reduce
from math import gcd
from typing import List

import numpy as np
import scipy.sparse as sp

from spmv.backends import Backend
from spmv.backends.mkl_utils import (
    MklSparseHandle,
    mkl_get_max_threads,
    mkl_set_num_threads,
)


def _estimate_bsr_blocksize(level_offsets):
    """Find largest power-of-2 block size that divides all level sizes."""
    level_sizes = np.diff(level_offsets)
    g = reduce(gcd, level_sizes.tolist())
    bs = 1
    for candidate in [2, 4, 8]:
        if g % candidate == 0:
            bs = candidate
    return bs


class MklBackend(Backend):
    """
    MKL-accelerated SpMV backend using the Inspector-Executor Sparse BLAS API.

    Persistent MKL handles are created once in setup() with optimization hints.
    Forward/backward matmat loops contain only the actual SpMV/SpMM calls.

    Parameters
    ----------
    n_threads : int
        Number of MKL threads. 0 = use os.cpu_count().
    fmt : str
        Sparse format: 'csr', 'csc', 'coo', 'bsr'.
    k_hint : int
        Expected number of dense columns for SpMM hint optimization.
        MKL pre-optimizes for this k.  SpMM still works correctly for
        any k, but performance is best when k == k_hint.
    blocksize : int or None
        Block size for BSR format. Required when fmt='bsr'.
        When None and fmt='bsr', auto-detected from GCD of level sizes.
    verbose : bool
        Print diagnostics.
    """

    def __init__(self, n_threads: int = 0, fmt: str = 'csr',
                 k_hint: int = 1, blocksize=None, verbose: bool = False):
        self._n_threads = os.cpu_count() if n_threads == 0 else n_threads
        self._fmt = fmt
        self._k_hint = k_hint
        self._blocksize = blocksize
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
        # 1. Set thread count (always — fixes the state-leak bug)
        mkl_set_num_threads(self._n_threads)

        # 2. Store metadata
        self._level_offsets = level_offsets
        self._n = n
        self._K = K
        self._sample_perm = sample_perm
        self._inv_sample_perm = inv_sample_perm
        self._dtype = dtype
        self._m = sel.shape[0]

        # 3. Determine BSR blocksize if needed
        fmt = self._fmt
        bs = self._blocksize
        if fmt == 'bsr' and bs is None:
            bs = _estimate_bsr_blocksize(level_offsets)
            self._blocksize = bs

        # 4. Create persistent handles for A_blocks
        num_levels = len(level_offsets) - 1
        self._A_handles = [[] for _ in range(num_levels)]
        for h in range(num_levels):
            for j in range(h):
                blk = A_blocks[h][j]
                if blk.nnz == 0:
                    self._A_handles[h].append(None)
                else:
                    self._A_handles[h].append(
                        MklSparseHandle(blk, fmt, blocksize=bs))

        # 5. Create persistent handles for AT_blocks
        self._AT_handles = [[] for _ in range(num_levels)]
        for h in range(num_levels):
            for j in range(num_levels - 1 - h):
                src = h + 1 + j
                AT = A_blocks[src][h].T.tocsr()
                if AT.nnz == 0:
                    self._AT_handles[h].append(None)
                else:
                    self._AT_handles[h].append(
                        MklSparseHandle(AT, fmt, blocksize=bs))

        # 6. Create handles for sel and sel_T
        self._sel_handle = MklSparseHandle(sel, fmt, blocksize=bs)
        self._sel_T_handle = MklSparseHandle(sel.T.tocsr(), fmt, blocksize=bs)

        # 7. Set optimization hints on ALL handles and optimize
        k = self._k_hint
        expected = 1000
        all_handles = []
        for row in self._A_handles:
            all_handles.extend(h for h in row if h is not None)
        for row in self._AT_handles:
            all_handles.extend(h for h in row if h is not None)
        all_handles.append(self._sel_handle)
        all_handles.append(self._sel_T_handle)

        for handle in all_handles:
            handle.set_mv_hint(transpose=False, expected_calls=expected)
            if k > 1:
                handle.set_mm_hint(k, transpose=False, expected_calls=expected)
            handle.optimize()

        # 8. Verbose diagnostics
        if self._verbose:
            total_nnz_fwd = sum(
                h.nnz for row in self._A_handles for h in row if h is not None)
            total_nnz_bwd = sum(
                h.nnz for row in self._AT_handles for h in row if h is not None)
            print(f"MklBackend setup: fmt={fmt}, k_hint={k}, "
                  f"n_threads={self._n_threads} (actual={mkl_get_max_threads()})")
            print(f"  {num_levels} levels, "
                  f"A_blocks total nnz: {total_nnz_fwd:,}, "
                  f"AT_blocks total nnz: {total_nnz_bwd:,}")

    def forward_matmat(self, X: np.ndarray) -> np.ndarray:
        """Compute G^T @ X: (n x k) -> (m x k)."""
        mkl_set_num_threads(self._n_threads)  # defensive
        off = self._level_offsets
        X = np.atleast_2d(X)
        k = X.shape[1]
        U = np.zeros((self._K, k), dtype=self._dtype)
        U[:self._n] = X[self._sample_perm]

        for h in range(1, len(off) - 1):
            lo, hi = int(off[h]), int(off[h + 1])
            for j, handle in enumerate(self._A_handles[h]):
                if handle is None:
                    continue
                jlo, jhi = int(off[j]), int(off[j + 1])
                if k == 1:
                    handle.mv(U[jlo:jhi, 0], U[lo:hi, 0],
                              alpha=1.0, beta=1.0)
                else:
                    handle.mm(U[jlo:jhi], U[lo:hi],
                              alpha=1.0, beta=1.0)

        result = np.empty((self._m, k), dtype=self._dtype)
        if k == 1:
            self._sel_handle.mv(U[:, 0], result[:, 0],
                                alpha=1.0, beta=0.0)
        else:
            self._sel_handle.mm(U, result, alpha=1.0, beta=0.0)
        return result

    def backward_matmat(self, X: np.ndarray) -> np.ndarray:
        """Compute G @ X: (m x k) -> (n x k)."""
        mkl_set_num_threads(self._n_threads)  # defensive
        off = self._level_offsets
        X = np.atleast_2d(X)
        k = X.shape[1]
        V = np.zeros((self._K, k), dtype=self._dtype)

        if k == 1:
            self._sel_T_handle.mv(X[:, 0], V[:, 0],
                                  alpha=1.0, beta=0.0)
        else:
            self._sel_T_handle.mm(X, V, alpha=1.0, beta=0.0)

        for h in range(len(off) - 2, -1, -1):
            lo, hi = int(off[h]), int(off[h + 1])
            for j, handle in enumerate(self._AT_handles[h]):
                if handle is None:
                    continue
                src_lo = int(off[h + 1 + j])
                src_hi = int(off[h + 2 + j])
                if k == 1:
                    handle.mv(V[src_lo:src_hi, 0], V[lo:hi, 0],
                              alpha=1.0, beta=1.0)
                else:
                    handle.mm(V[src_lo:src_hi], V[lo:hi],
                              alpha=1.0, beta=1.0)

        return V[self._inv_sample_perm]
