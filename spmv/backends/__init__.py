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

    A backend computes the full G^T @ X (forward) and G @ X (backward) pipelines,
    including pre/post-permutation and the selector multiply.

    Block structure:
    - A_blocks[h][j] has shape (off[h+1]-off[h]) x (off[j+1]-off[j]) for j < h
    """

    @abstractmethod
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
        """
        Initialize the backend with block-wise sparse matrices and permutations.

        Parameters
        ----------
        A_blocks : list of list of csr_matrix
            Forward blocks: A_blocks[h][j] for j in range(h).
        level_offsets : np.ndarray
            Level boundary indices.
        n : int
            Number of samples.
        K : int
            Total number of nodes.
        sel : csr_matrix
            Selector matrix (m x K).
        sample_perm : np.ndarray
            Maps new sample indices to original.
        inv_sample_perm : np.ndarray
            Inverse of sample_perm.
        dtype : np.dtype
            Data type for computation.
        """
        pass

    @abstractmethod
    def forward_matmat(self, X: np.ndarray) -> np.ndarray:
        """
        Compute G^T @ X: (n x k) -> (m x k).

        Full pipeline: permute samples, level-wise forward SpMM, selector multiply.
        """
        pass

    @abstractmethod
    def backward_matmat(self, X: np.ndarray) -> np.ndarray:
        """
        Compute G @ X: (m x k) -> (n x k).

        Full pipeline: selector^T multiply, level-wise backward SpMM, inverse permute.
        """
        pass


BACKEND_REGISTRY: dict[str, tuple[str, str]] = {
    'spsparse':  ('spmv.backends.spsparse',  'SpsparseBackend'),
    'mkl':       ('spmv.backends.mkl',       'MklBackend'),
    'cusparse':  ('spmv.backends.cusparse',  'CusparseBackend'),
}


def get_backend_class(backend_type: str) -> type[Backend]:
    """Lazy-import and return the backend class for the given type name."""
    if backend_type not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend type: {backend_type!r}. "
                         f"Available: {sorted(BACKEND_REGISTRY)}")
    module_path, class_name = BACKEND_REGISTRY[backend_type]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


__all__ = ['Backend', 'BACKEND_REGISTRY', 'get_backend_class']
