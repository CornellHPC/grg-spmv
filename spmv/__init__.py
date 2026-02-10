"""
spmv - Level-wise SpMV operator for GRG-based genotype matrices.

This package provides efficient sparse matrix-vector multiplication
for computing G @ x and G^T @ x where G is the genotype matrix
represented as a GRG (Genotype Representation Graph).
"""

import numpy as np

INDEX_DTYPE = np.uint32
DATA_DTYPE = np.float64

from spmv.operator import SpMVOperator

__all__ = ["SpMVOperator", "INDEX_DTYPE", "DATA_DTYPE"]
