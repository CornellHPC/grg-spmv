from time import perf_counter
from typing import List

import numpy as np
import scipy.sparse as sp

from spmv.backends import Backend


class ThreadBackend(Backend):
    def __init__(self, verbose: bool = False):
        self._verbose = verbose
        self._A_fwd = None
        self._AT_bwd = None
        self._level_offsets = None

    def setup(
        self,
        A_fwd: List[sp.csr_matrix],
        AT_bwd: List[sp.csr_matrix],
        level_offsets: np.ndarray
    ) -> None:
        self._A_fwd = A_fwd
        self._AT_bwd = AT_bwd
        self._level_offsets = level_offsets
        if self._verbose:
            total_nnz_fwd = sum(A.nnz for A in A_fwd)
            total_nnz_bwd = sum(A.nnz for A in AT_bwd)
            print(f"ThreadBackend setup: {len(A_fwd)} levels")
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
            U[lo:hi] = self._A_fwd[h] @ U[:lo]
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
            V[lo:hi] += self._AT_bwd[h] @ V[hi:]
            if self._verbose:
                elapsed = (perf_counter() - t0) * 1e3
                print(f"  Level {h}: AT_bwd[{h}].shape={self._AT_bwd[h].shape}, "
                      f"nnz={self._AT_bwd[h].nnz:,}, time={elapsed:.2f}ms")
        return V
