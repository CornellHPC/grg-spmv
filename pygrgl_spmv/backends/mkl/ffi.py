"""
MKL utility layer — constants, ctypes signatures, and MklSparseHandle.

Isolates all ctypes boilerplate so that mkl.py only deals with
high-level MKL Inspector-Executor calls.  Mirrors the pattern
used by cuda_utils.py for cuSPARSE.
"""

import ctypes
import ctypes.util
import os
from ctypes import (
    POINTER, Structure, byref, c_double, c_int, c_void_p,
)

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# MKL sparse constants  (from mkl_spblas.h)
# ---------------------------------------------------------------------------
SPARSE_STATUS_SUCCESS = 0
SPARSE_STATUS_NOT_INITIALIZED = 1
SPARSE_STATUS_ALLOC_FAILED = 2
SPARSE_STATUS_INVALID_VALUE = 3
SPARSE_STATUS_EXECUTION_FAILED = 4
SPARSE_STATUS_INTERNAL_ERROR = 5
SPARSE_STATUS_NOT_SUPPORTED = 6

SPARSE_INDEX_BASE_ZERO = 0
SPARSE_INDEX_BASE_ONE = 1

SPARSE_OPERATION_NON_TRANSPOSE = 10
SPARSE_OPERATION_TRANSPOSE = 11
SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12

SPARSE_MATRIX_TYPE_GENERAL = 20
SPARSE_MATRIX_TYPE_SYMMETRIC = 21
SPARSE_MATRIX_TYPE_HERMITIAN = 22
SPARSE_MATRIX_TYPE_TRIANGULAR = 23
SPARSE_MATRIX_TYPE_DIAGONAL = 24
SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR = 25
SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL = 26

SPARSE_FILL_MODE_LOWER = 40
SPARSE_FILL_MODE_UPPER = 41

SPARSE_DIAG_NON_UNIT = 50
SPARSE_DIAG_UNIT = 51

SPARSE_LAYOUT_ROW_MAJOR = 101
SPARSE_LAYOUT_COLUMN_MAJOR = 102

_STATUS_NAMES = {
    0: 'SUCCESS',
    1: 'NOT_INITIALIZED',
    2: 'ALLOC_FAILED',
    3: 'INVALID_VALUE',
    4: 'EXECUTION_FAILED',
    5: 'INTERNAL_ERROR',
    6: 'NOT_SUPPORTED',
}


# ---------------------------------------------------------------------------
# matrix_descr struct
# ---------------------------------------------------------------------------

class MatrixDescr(Structure):
    """MKL struct matrix_descr { sparse_matrix_type_t type; ... }."""
    _fields_ = [
        ('type', c_int),
        ('mode', c_int),    # sparse_fill_mode_t
        ('diag', c_int),    # sparse_diag_type_t
    ]


GENERAL_DESCR = MatrixDescr(
    type=SPARSE_MATRIX_TYPE_GENERAL,
    mode=SPARSE_FILL_MODE_LOWER,
    diag=SPARSE_DIAG_NON_UNIT,
)


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _load_mkl():
    """Load libmkl_rt and return the ctypes handle."""
    # 1. Explicit env var
    path = os.environ.get('MKL_RT')
    if path:
        return ctypes.cdll.LoadLibrary(path)

    # 2. ctypes.util.find_library
    found = ctypes.util.find_library('mkl_rt')
    if found:
        return ctypes.cdll.LoadLibrary(found)

    # 3. Direct name (relies on LD_LIBRARY_PATH / RPATH)
    return ctypes.cdll.LoadLibrary('libmkl_rt.so')


# ---------------------------------------------------------------------------
# Integer type detection (LP64 vs ILP64)
# ---------------------------------------------------------------------------

def _detect_mkl_int(lib):
    """Detect whether MKL uses 32-bit (LP64) or 64-bit (ILP64) integers.

    Creates a tiny 1×1 CSR matrix and exports it.  If the export with
    int32 index arrays succeeds, we're LP64; otherwise ILP64.

    Returns numpy dtype (np.int32 or np.int64).
    """
    # Try int32 (LP64) first — the common case
    for np_int, ct_int in [(np.int32, ctypes.c_int), (np.int64, ctypes.c_long)]:
        try:
            rows_start = np.array([0, 1], dtype=np_int)
            rows_end = np.array([1], dtype=np_int)
            col_idx = np.array([0], dtype=np_int)
            values = np.array([1.0], dtype=np.float64)

            handle = c_void_p()
            status = lib.mkl_sparse_d_create_csr(
                byref(handle),
                c_int(SPARSE_INDEX_BASE_ZERO),
                ct_int(1), ct_int(1),
                rows_start.ctypes.data_as(POINTER(ct_int)),
                rows_end.ctypes.data_as(POINTER(ct_int)),
                col_idx.ctypes.data_as(POINTER(ct_int)),
                values.ctypes.data_as(POINTER(c_double)),
            )
            if status == SPARSE_STATUS_SUCCESS:
                lib.mkl_sparse_destroy(handle)
                return np_int
        except Exception:
            continue

    raise RuntimeError("Could not detect MKL integer type (LP64 or ILP64)")


# ---------------------------------------------------------------------------
# ctypes signature setup
# ---------------------------------------------------------------------------

def _setup_mkl_signatures(lib, ct_int):
    """Set argtypes/restype on all MKL functions we use."""
    INT_P = POINTER(ct_int)
    DBL_P = POINTER(c_double)

    # mkl_sparse_d_create_csr(handle, indexing, nrows, ncols, rows_start, rows_end, col_idx, values)
    lib.mkl_sparse_d_create_csr.argtypes = [
        POINTER(c_void_p), c_int, ct_int, ct_int, INT_P, INT_P, INT_P, DBL_P,
    ]
    lib.mkl_sparse_d_create_csr.restype = c_int

    # mkl_sparse_d_create_csc(handle, indexing, nrows, ncols, cols_start, cols_end, row_idx, values)
    lib.mkl_sparse_d_create_csc.argtypes = [
        POINTER(c_void_p), c_int, ct_int, ct_int, INT_P, INT_P, INT_P, DBL_P,
    ]
    lib.mkl_sparse_d_create_csc.restype = c_int

    # mkl_sparse_d_create_coo(handle, base, nrows, ncols, nnz, row_idx, col_idx, values)
    lib.mkl_sparse_d_create_coo.argtypes = [
        POINTER(c_void_p), c_int, ct_int, ct_int, ct_int, INT_P, INT_P, DBL_P,
    ]
    lib.mkl_sparse_d_create_coo.restype = c_int

    # mkl_sparse_d_create_bsr(handle, base, layout, nrows, ncols, block_size,
    #                          rows_start, rows_end, col_idx, values)
    lib.mkl_sparse_d_create_bsr.argtypes = [
        POINTER(c_void_p), c_int, c_int, ct_int, ct_int, ct_int,
        INT_P, INT_P, INT_P, DBL_P,
    ]
    lib.mkl_sparse_d_create_bsr.restype = c_int

    # mkl_sparse_destroy(handle)
    lib.mkl_sparse_destroy.argtypes = [c_void_p]
    lib.mkl_sparse_destroy.restype = c_int

    # mkl_sparse_d_mv(op, alpha, A, descr, x, beta, y)
    lib.mkl_sparse_d_mv.argtypes = [
        c_int, c_double, c_void_p, MatrixDescr, DBL_P, c_double, DBL_P,
    ]
    lib.mkl_sparse_d_mv.restype = c_int

    # mkl_sparse_d_mm(op, alpha, A, descr, layout, x, columns, ldx, beta, y, ldy)
    lib.mkl_sparse_d_mm.argtypes = [
        c_int, c_double, c_void_p, MatrixDescr, c_int,
        DBL_P, ct_int, ct_int,
        c_double, DBL_P, ct_int,
    ]
    lib.mkl_sparse_d_mm.restype = c_int

    # mkl_sparse_set_mv_hint(A, op, descr, expected_calls)
    lib.mkl_sparse_set_mv_hint.argtypes = [c_void_p, c_int, MatrixDescr, ct_int]
    lib.mkl_sparse_set_mv_hint.restype = c_int

    # mkl_sparse_set_mm_hint(A, op, descr, layout, dense_matrix_columns, expected_calls)
    lib.mkl_sparse_set_mm_hint.argtypes = [
        c_void_p, c_int, MatrixDescr, c_int, ct_int, ct_int,
    ]
    lib.mkl_sparse_set_mm_hint.restype = c_int

    # mkl_sparse_optimize(A)
    lib.mkl_sparse_optimize.argtypes = [c_void_p]
    lib.mkl_sparse_optimize.restype = c_int

    # MKL_Set_Num_Threads / MKL_Get_Max_Threads
    lib.MKL_Set_Num_Threads.argtypes = [c_int]
    lib.MKL_Set_Num_Threads.restype = None

    lib.MKL_Get_Max_Threads.argtypes = []
    lib.MKL_Get_Max_Threads.restype = c_int


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_mkl_lib = None
_mkl_int_dtype = None
_ct_int = None


def _ensure_loaded():
    """Load MKL library and detect integer type (once)."""
    global _mkl_lib, _mkl_int_dtype, _ct_int
    if _mkl_lib is None:
        _mkl_lib = _load_mkl()
        _mkl_int_dtype = _detect_mkl_int(_mkl_lib)
        _ct_int = ctypes.c_int if _mkl_int_dtype == np.int32 else ctypes.c_long
        _setup_mkl_signatures(_mkl_lib, _ct_int)
    return _mkl_lib, _mkl_int_dtype, _ct_int


# ---------------------------------------------------------------------------
# Status checking
# ---------------------------------------------------------------------------

def _check(status, func_name):
    """Raise RuntimeError on MKL error."""
    if status != SPARSE_STATUS_SUCCESS:
        name = _STATUS_NAMES.get(status, f'UNKNOWN({status})')
        raise RuntimeError(f"MKL {func_name} failed: status {status} ({name})")


# ---------------------------------------------------------------------------
# MklSparseHandle — persistent inspector-executor handle
# ---------------------------------------------------------------------------

class MklSparseHandle:
    """Persistent MKL sparse handle wrapping a scipy sparse matrix.

    The MKL Inspector-Executor API borrows array pointers, so we must
    keep the underlying scipy matrix alive for the handle's lifetime.

    Parameters
    ----------
    mat : scipy sparse matrix
        Input matrix (will be converted to the requested format).
    fmt : str
        Target format: 'csr', 'csc', 'coo'.
    """

    def __init__(self, mat, fmt='csr'):
        lib, int_dtype, ct_int = _ensure_loaded()
        self._lib = lib
        self._ct_int = ct_int
        self._handle = c_void_p()
        self._fmt = fmt

        # Convert to target format and keep reference
        self._mat = _scipy_to_fmt(mat, fmt)
        self._nnz = self._mat.nnz
        self._shape = self._mat.shape

        # Ensure index arrays use the correct MKL integer type
        if fmt in ('csr', 'csc'):
            self._mat.indptr = self._mat.indptr.astype(int_dtype, copy=False)
            self._mat.indices = self._mat.indices.astype(int_dtype, copy=False)
        elif fmt == 'coo':
            self._mat.row = self._mat.row.astype(int_dtype, copy=False)
            self._mat.col = self._mat.col.astype(int_dtype, copy=False)

        # Ensure data is float64 C-contiguous
        self._mat.data = np.ascontiguousarray(self._mat.data, dtype=np.float64)

        # Create the MKL handle
        INT_P = POINTER(ct_int)
        DBL_P = POINTER(c_double)
        m, n = self._mat.shape

        if fmt == 'csr':
            indptr = self._mat.indptr
            _check(lib.mkl_sparse_d_create_csr(
                byref(self._handle),
                c_int(SPARSE_INDEX_BASE_ZERO),
                ct_int(m), ct_int(n),
                indptr[:-1].ctypes.data_as(INT_P),
                indptr[1:].ctypes.data_as(INT_P),
                self._mat.indices.ctypes.data_as(INT_P),
                self._mat.data.ctypes.data_as(DBL_P),
            ), 'mkl_sparse_d_create_csr')

        elif fmt == 'csc':
            indptr = self._mat.indptr
            _check(lib.mkl_sparse_d_create_csc(
                byref(self._handle),
                c_int(SPARSE_INDEX_BASE_ZERO),
                ct_int(m), ct_int(n),
                indptr[:-1].ctypes.data_as(INT_P),
                indptr[1:].ctypes.data_as(INT_P),
                self._mat.indices.ctypes.data_as(INT_P),
                self._mat.data.ctypes.data_as(DBL_P),
            ), 'mkl_sparse_d_create_csc')

        elif fmt == 'coo':
            nnz = self._mat.nnz
            _check(lib.mkl_sparse_d_create_coo(
                byref(self._handle),
                ct_int(SPARSE_INDEX_BASE_ZERO),
                ct_int(m), ct_int(n), ct_int(nnz),
                self._mat.row.ctypes.data_as(INT_P),
                self._mat.col.ctypes.data_as(INT_P),
                self._mat.data.ctypes.data_as(DBL_P),
            ), 'mkl_sparse_d_create_coo')

        else:
            raise ValueError(f"Unsupported format: {fmt!r}")

    @property
    def nnz(self):
        return self._nnz

    @property
    def shape(self):
        return self._shape

    def set_mv_hint(self, transpose=False, expected_calls=1000):
        """Set SpMV optimization hint.  Silently ignored for unsupported formats."""
        op = SPARSE_OPERATION_TRANSPOSE if transpose else SPARSE_OPERATION_NON_TRANSPOSE
        status = self._lib.mkl_sparse_set_mv_hint(
            self._handle, c_int(op), GENERAL_DESCR, self._ct_int(expected_calls),
        )
        if status not in (SPARSE_STATUS_SUCCESS, SPARSE_STATUS_NOT_SUPPORTED):
            _check(status, 'mkl_sparse_set_mv_hint')

    def set_mm_hint(self, k, transpose=False, layout=SPARSE_LAYOUT_ROW_MAJOR,
                    expected_calls=1000):
        """Set SpMM optimization hint.  Silently ignored for unsupported formats."""
        op = SPARSE_OPERATION_TRANSPOSE if transpose else SPARSE_OPERATION_NON_TRANSPOSE
        status = self._lib.mkl_sparse_set_mm_hint(
            self._handle, c_int(op), GENERAL_DESCR, c_int(layout),
            self._ct_int(k), self._ct_int(expected_calls),
        )
        if status not in (SPARSE_STATUS_SUCCESS, SPARSE_STATUS_NOT_SUPPORTED):
            _check(status, 'mkl_sparse_set_mm_hint')

    def optimize(self):
        """Trigger MKL internal optimization.  Silently ignored for unsupported formats."""
        status = self._lib.mkl_sparse_optimize(self._handle)
        if status not in (SPARSE_STATUS_SUCCESS, SPARSE_STATUS_NOT_SUPPORTED):
            _check(status, 'mkl_sparse_optimize')

    def mv(self, x, y, alpha=1.0, beta=1.0, transpose=False):
        """y = alpha * op(A) * x + beta * y.

        x and y must be contiguous float64 1-D arrays.
        """
        op = SPARSE_OPERATION_TRANSPOSE if transpose else SPARSE_OPERATION_NON_TRANSPOSE
        DBL_P = POINTER(c_double)
        _check(self._lib.mkl_sparse_d_mv(
            c_int(op), c_double(alpha), self._handle, GENERAL_DESCR,
            x.ctypes.data_as(DBL_P),
            c_double(beta),
            y.ctypes.data_as(DBL_P),
        ), 'mkl_sparse_d_mv')

    def mm(self, B, C, alpha=1.0, beta=1.0, transpose=False,
           layout=SPARSE_LAYOUT_ROW_MAJOR):
        """C = alpha * op(A) * B + beta * C.

        B and C must be C-contiguous float64 2-D arrays (row-major).
        """
        op = SPARSE_OPERATION_TRANSPOSE if transpose else SPARSE_OPERATION_NON_TRANSPOSE
        k = B.shape[1]
        ldb = B.shape[1]  # row-major: leading dimension = #columns
        ldc = C.shape[1]
        DBL_P = POINTER(c_double)
        _check(self._lib.mkl_sparse_d_mm(
            c_int(op), c_double(alpha), self._handle, GENERAL_DESCR,
            c_int(layout),
            B.ctypes.data_as(DBL_P), self._ct_int(k), self._ct_int(ldb),
            c_double(beta),
            C.ctypes.data_as(DBL_P), self._ct_int(ldc),
        ), 'mkl_sparse_d_mm')

    def destroy(self):
        """Destroy the MKL handle."""
        if self._handle:
            self._lib.mkl_sparse_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.destroy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scipy_to_fmt(mat, fmt):
    """Convert a scipy sparse matrix to the target format."""
    if fmt == 'csr':
        return mat.tocsr()
    elif fmt == 'csc':
        return mat.tocsc()
    elif fmt == 'coo':
        return mat.tocoo()
    else:
        raise ValueError(f"Unsupported format: {fmt!r}")


def mkl_set_num_threads(n):
    """Set the number of MKL threads (process-global)."""
    lib, _, _ = _ensure_loaded()
    lib.MKL_Set_Num_Threads(c_int(n))


def mkl_get_max_threads():
    """Query the current MKL thread count."""
    lib, _, _ = _ensure_loaded()
    return lib.MKL_Get_Max_Threads()
