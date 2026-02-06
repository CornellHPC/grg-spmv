"""
Backends - Backend implementations for sparse matrix-vector multiplication.

Abstract base class for backends that compute level-wise SpMV operations.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import scipy.sparse as sp


class Backend(ABC):
    """
    Abstract base class for SpMV backends.

    A backend computes level-wise forward and backward passes for SpMV.
    Implementations may use CPU threads, GPU, or distributed systems.
    
    Block extraction optimization:
    - A_fwd[h] has shape (level_size) × (level_offsets[h]) 
      - Contains columns [0:lo], multiplied by U[:lo]
    - AT_bwd[h] has shape (level_size) × (K - level_offsets[h+1])
      - Contains columns [hi:K], multiplied by V[hi:]
    """

    @abstractmethod
    def setup(
        self,
        A_fwd: List[sp.csr_matrix],
        AT_bwd: List[sp.csr_matrix],
        level_offsets: np.ndarray
    ) -> None:
        """
        Initialize the backend with the sparse matrices.
        
        Parameters
        ----------
        A_fwd : List[csr_matrix]
            Forward matrices for each level.
        AT_bwd : List[csr_matrix]
            Backward matrices for each level.
        level_offsets : np.ndarray
            Level boundary indices.
        """
        pass

    @abstractmethod
    def forward_matmat(self, U: np.ndarray) -> np.ndarray:
        """
        Compute forward pass: U[lo:hi] = A_fwd[h] @ U[:lo] for each level h.

        Parameters
        ----------
        U : np.ndarray
            Input/output matrix (K × k) to update in-place.

        Returns
        -------
        np.ndarray
            Updated U matrix.
        """
        pass

    @abstractmethod
    def backward_matmat(self, V: np.ndarray) -> np.ndarray:
        """
        Compute backward pass: V[lo:hi] += AT_bwd[h] @ V[hi:] for each level h in reverse.

        Parameters
        ----------
        V : np.ndarray
            Input/output matrix (K × k) to update in-place.

        Returns
        -------
        np.ndarray
            Updated V matrix.
        """
        pass


__all__ = ['Backend']
