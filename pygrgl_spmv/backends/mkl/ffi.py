"""Lean MKL ctypes helpers and persistent sparse handles."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from ctypes import POINTER, Structure, byref, c_double, c_float, c_int, c_long, c_size_t, c_void_p
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

SPARSE_STATUS_SUCCESS = 0
SPARSE_STATUS_NOT_SUPPORTED = 6

SPARSE_INDEX_BASE_ZERO = 0
SPARSE_OPERATION_NON_TRANSPOSE = 10
SPARSE_OPERATION_TRANSPOSE = 11
SPARSE_MATRIX_TYPE_GENERAL = 20
SPARSE_FILL_MODE_LOWER = 40
SPARSE_DIAG_NON_UNIT = 50
SPARSE_LAYOUT_ROW_MAJOR = 101

_STATUS_NAMES = {
    0: "SUCCESS",
    1: "NOT_INITIALIZED",
    2: "ALLOC_FAILED",
    3: "INVALID_VALUE",
    4: "EXECUTION_FAILED",
    5: "INTERNAL_ERROR",
    6: "NOT_SUPPORTED",
}
_INT32_MAX = int(np.iinfo(np.int32).max)
_MAP_FAILED = ctypes.c_void_p(-1).value
_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.mmap.argtypes = [c_void_p, c_size_t, c_int, c_int, c_int, c_long]
_LIBC.mmap.restype = c_void_p
_LIBC.munmap.argtypes = [c_void_p, c_size_t]
_LIBC.munmap.restype = c_int


class MatrixDescr(Structure):
    _fields_ = [
        ("type", c_int),
        ("mode", c_int),
        ("diag", c_int),
    ]


GENERAL_DESCR = MatrixDescr(
    type=SPARSE_MATRIX_TYPE_GENERAL,
    mode=SPARSE_FILL_MODE_LOWER,
    diag=SPARSE_DIAG_NON_UNIT,
)


@dataclass(frozen=True)
class _ScalarSpec:
    dtype: np.dtype
    c_scalar: type[ctypes._SimpleCData]
    create_csr: str
    create_csc: str
    create_coo: str
    mv: str
    mm: str


_SCALAR_SPECS = {
    np.dtype(np.float32): _ScalarSpec(
        dtype=np.dtype(np.float32),
        c_scalar=c_float,
        create_csr="mkl_sparse_s_create_csr",
        create_csc="mkl_sparse_s_create_csc",
        create_coo="mkl_sparse_s_create_coo",
        mv="mkl_sparse_s_mv",
        mm="mkl_sparse_s_mm",
    ),
    np.dtype(np.float64): _ScalarSpec(
        dtype=np.dtype(np.float64),
        c_scalar=c_double,
        create_csr="mkl_sparse_d_create_csr",
        create_csc="mkl_sparse_d_create_csc",
        create_coo="mkl_sparse_d_create_coo",
        mv="mkl_sparse_d_mv",
        mm="mkl_sparse_d_mm",
    ),
}


def _scalar_spec(dtype) -> _ScalarSpec:
    dt = np.dtype(dtype)
    if dt not in _SCALAR_SPECS:
        raise ValueError(f"MKL runtime supports only float32/float64, got {dt}")
    return _SCALAR_SPECS[dt]


def _load_mkl():
    path = os.environ.get("MKL_RT")
    if path:
        return ctypes.cdll.LoadLibrary(path)
    found = ctypes.util.find_library("mkl_rt")
    if found:
        return ctypes.cdll.LoadLibrary(found)
    return ctypes.cdll.LoadLibrary("libmkl_rt.so")


def _detect_mkl_int(lib):
    for np_int, ct_int in ((np.int32, ctypes.c_int), (np.int64, ctypes.c_long)):
        try:
            rows_start = np.array([0, 1], dtype=np_int)
            rows_end = np.array([1], dtype=np_int)
            col_idx = np.array([0], dtype=np_int)
            values = np.array([1.0], dtype=np.float64)
            handle = c_void_p()
            status = lib.mkl_sparse_d_create_csr(
                byref(handle),
                c_int(SPARSE_INDEX_BASE_ZERO),
                ct_int(1),
                ct_int(1),
                rows_start.ctypes.data_as(POINTER(ct_int)),
                rows_end.ctypes.data_as(POINTER(ct_int)),
                col_idx.ctypes.data_as(POINTER(ct_int)),
                values.ctypes.data_as(POINTER(c_double)),
            )
            if status == SPARSE_STATUS_SUCCESS:
                lib.mkl_sparse_destroy(handle)
                return np.dtype(np_int)
        except Exception:
            continue
    raise RuntimeError("Could not detect MKL integer type (LP64 or ILP64)")


def _setup_scalar_signatures(lib, ct_int, spec: _ScalarSpec) -> None:
    int_p = POINTER(ct_int)
    scalar = spec.c_scalar
    scalar_p = POINTER(scalar)

    getattr(lib, spec.create_csr).argtypes = [
        POINTER(c_void_p),
        c_int,
        ct_int,
        ct_int,
        int_p,
        int_p,
        int_p,
        scalar_p,
    ]
    getattr(lib, spec.create_csr).restype = c_int

    getattr(lib, spec.create_csc).argtypes = [
        POINTER(c_void_p),
        c_int,
        ct_int,
        ct_int,
        int_p,
        int_p,
        int_p,
        scalar_p,
    ]
    getattr(lib, spec.create_csc).restype = c_int

    getattr(lib, spec.create_coo).argtypes = [
        POINTER(c_void_p),
        c_int,
        ct_int,
        ct_int,
        ct_int,
        int_p,
        int_p,
        scalar_p,
    ]
    getattr(lib, spec.create_coo).restype = c_int

    getattr(lib, spec.mv).argtypes = [
        c_int,
        scalar,
        c_void_p,
        MatrixDescr,
        scalar_p,
        scalar,
        scalar_p,
    ]
    getattr(lib, spec.mv).restype = c_int

    getattr(lib, spec.mm).argtypes = [
        c_int,
        scalar,
        c_void_p,
        MatrixDescr,
        c_int,
        scalar_p,
        ct_int,
        ct_int,
        scalar,
        scalar_p,
        ct_int,
    ]
    getattr(lib, spec.mm).restype = c_int


def _setup_mkl_signatures(lib, ct_int) -> None:
    for spec in _SCALAR_SPECS.values():
        _setup_scalar_signatures(lib, ct_int, spec)

    lib.mkl_sparse_destroy.argtypes = [c_void_p]
    lib.mkl_sparse_destroy.restype = c_int

    lib.mkl_sparse_set_mv_hint.argtypes = [c_void_p, c_int, MatrixDescr, ct_int]
    lib.mkl_sparse_set_mv_hint.restype = c_int

    lib.mkl_sparse_set_mm_hint.argtypes = [c_void_p, c_int, MatrixDescr, c_int, ct_int, ct_int]
    lib.mkl_sparse_set_mm_hint.restype = c_int

    lib.mkl_sparse_optimize.argtypes = [c_void_p]
    lib.mkl_sparse_optimize.restype = c_int

    lib.MKL_Set_Num_Threads.argtypes = [c_int]
    lib.MKL_Set_Num_Threads.restype = None


_mkl_lib = None
_mkl_int_dtype = None
_ct_int = None


def _ensure_loaded():
    global _mkl_lib, _mkl_int_dtype, _ct_int
    if _mkl_lib is None:
        _mkl_lib = _load_mkl()
        _mkl_int_dtype = _detect_mkl_int(_mkl_lib)
        _ct_int = ctypes.c_int if _mkl_int_dtype == np.dtype(np.int32) else ctypes.c_long
        _setup_mkl_signatures(_mkl_lib, _ct_int)
    return _mkl_lib, _mkl_int_dtype, _ct_int


def _check(status, func_name: str) -> None:
    if status != SPARSE_STATUS_SUCCESS:
        name = _STATUS_NAMES.get(status, f"UNKNOWN({status})")
        raise RuntimeError(f"MKL {func_name} failed: status {status} ({name})")


def _validate_lp64_matrix(mat, fmt: str) -> None:
    m, n = (int(v) for v in mat.shape)
    if m > _INT32_MAX:
        raise ValueError(f"LP64 MKL requires nrows <= {_INT32_MAX}, got {m}")
    if n > _INT32_MAX:
        raise ValueError(f"LP64 MKL requires ncols <= {_INT32_MAX}, got {n}")
    if fmt in {"csr", "csc"}:
        for name in ("indptr", "indices"):
            arr = np.asarray(getattr(mat, name))
            if arr.dtype == np.dtype(np.int64):
                raise ValueError(f"LP64 MKL requires {fmt.upper()} {name} to use int32, got int64")
        return
    if fmt == "coo":
        if int(mat.nnz) > _INT32_MAX:
            raise ValueError(f"LP64 MKL requires COO nnz <= {_INT32_MAX}, got {int(mat.nnz)}")
        for name in ("row", "col"):
            arr = np.asarray(getattr(mat, name))
            if arr.dtype == np.dtype(np.int64):
                raise ValueError(f"LP64 MKL requires COO {name} to use int32, got int64")
        return
    raise ValueError(f"Unsupported format: {fmt!r}")


def _scipy_to_fmt(mat, fmt: str):
    if fmt == "csr":
        return mat.tocsr()
    if fmt == "csc":
        return mat.tocsc()
    if fmt == "coo":
        return mat.tocoo()
    raise ValueError(f"Unsupported format: {fmt!r}")


def _require_dense_vector(values, *, dtype: np.dtype, label: str) -> np.ndarray:
    arr = np.asarray(values)
    itemsize = int(np.dtype(dtype).itemsize)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a 1-D dense vector, got ndim={arr.ndim}")
    if arr.dtype != np.dtype(dtype):
        raise TypeError(f"{label} must have dtype {np.dtype(dtype)}, got {arr.dtype}")
    if arr.size and arr.strides[0] != itemsize:
        raise ValueError(f"{label} must be contiguous, got strides={arr.strides}")
    return arr


def _mmap(addr: int | None, length: int, prot: int, flags: int, fd: int, offset: int) -> int:
    target = None if addr is None else c_void_p(int(addr))
    result = _LIBC.mmap(target, c_size_t(int(length)), c_int(int(prot)), c_int(int(flags)), c_int(int(fd)), c_long(int(offset)))
    ptr = ctypes.cast(result, c_void_p).value
    if ptr == _MAP_FAILED:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), "mmap")
    assert ptr is not None
    return int(ptr)


def _munmap(addr: int, length: int) -> None:
    if not addr or not length:
        return
    status = _LIBC.munmap(c_void_p(int(addr)), c_size_t(int(length)))
    if status != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), "munmap")


def _address_array(addr: int, length: int, dtype: np.dtype) -> np.ndarray:
    dt = np.dtype(dtype)
    c_scalar = ctypes.c_float if dt == np.dtype(np.float32) else ctypes.c_double
    return np.ctypeslib.as_array((c_scalar * int(length)).from_address(int(addr)))


def _dense_leading_dimension(values, *, dtype: np.dtype, label: str) -> tuple[np.ndarray, int]:
    arr = np.asarray(values)
    itemsize = int(np.dtype(dtype).itemsize)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2-D dense matrix, got ndim={arr.ndim}")
    if arr.dtype != np.dtype(dtype):
        raise TypeError(f"{label} must have dtype {np.dtype(dtype)}, got {arr.dtype}")
    if arr.shape[1] and arr.strides[1] != itemsize:
        raise ValueError(f"{label} must be row-major with contiguous columns, got strides={arr.strides}")
    if arr.strides[0] % itemsize != 0:
        raise ValueError(f"{label} has invalid row stride {arr.strides[0]}")
    return arr, int(arr.strides[0] // itemsize) if arr.shape[0] else max(int(arr.shape[1]), 1)


class MklSparseHandle:
    """Persistent MKL sparse handle wrapping one SciPy sparse matrix."""

    def __init__(self, mat, fmt="csr", *, dtype=np.float64):
        lib, mkl_int_dtype, mkl_ct_int = _ensure_loaded()
        spec = _scalar_spec(dtype)
        self._lib = lib
        self._ct_int = mkl_ct_int
        self._dtype = spec.dtype
        self._spec = spec
        self._handle = c_void_p()
        self._fmt = fmt

        self._mat = _scipy_to_fmt(mat, fmt)
        self._nnz = int(self._mat.nnz)
        self._shape = tuple(int(v) for v in self._mat.shape)

        if mkl_int_dtype == np.dtype(np.int32):
            _validate_lp64_matrix(self._mat, fmt)

        if fmt in {"csr", "csc"}:
            self._mat.indptr = self._mat.indptr.astype(mkl_int_dtype, copy=False)
            self._mat.indices = self._mat.indices.astype(mkl_int_dtype, copy=False)
        elif fmt == "coo":
            self._mat.row = self._mat.row.astype(mkl_int_dtype, copy=False)
            self._mat.col = self._mat.col.astype(mkl_int_dtype, copy=False)

        data = np.asarray(self._mat.data, dtype=self._dtype)
        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)
        self._mat.data = data

        int_p = POINTER(mkl_ct_int)
        scalar_p = POINTER(spec.c_scalar)
        m, n = self._shape

        if fmt == "csr":
            indptr = self._mat.indptr
            create = getattr(lib, spec.create_csr)
            _check(
                create(
                    byref(self._handle),
                    c_int(SPARSE_INDEX_BASE_ZERO),
                    mkl_ct_int(m),
                    mkl_ct_int(n),
                    indptr[:-1].ctypes.data_as(int_p),
                    indptr[1:].ctypes.data_as(int_p),
                    self._mat.indices.ctypes.data_as(int_p),
                    self._mat.data.ctypes.data_as(scalar_p),
                ),
                spec.create_csr,
            )
        elif fmt == "csc":
            indptr = self._mat.indptr
            create = getattr(lib, spec.create_csc)
            _check(
                create(
                    byref(self._handle),
                    c_int(SPARSE_INDEX_BASE_ZERO),
                    mkl_ct_int(m),
                    mkl_ct_int(n),
                    indptr[:-1].ctypes.data_as(int_p),
                    indptr[1:].ctypes.data_as(int_p),
                    self._mat.indices.ctypes.data_as(int_p),
                    self._mat.data.ctypes.data_as(scalar_p),
                ),
                spec.create_csc,
            )
        elif fmt == "coo":
            create = getattr(lib, spec.create_coo)
            _check(
                create(
                    byref(self._handle),
                    c_int(SPARSE_INDEX_BASE_ZERO),
                    mkl_ct_int(m),
                    mkl_ct_int(n),
                    mkl_ct_int(self._nnz),
                    self._mat.row.ctypes.data_as(int_p),
                    self._mat.col.ctypes.data_as(int_p),
                    self._mat.data.ctypes.data_as(scalar_p),
                ),
                spec.create_coo,
            )
        else:
            raise ValueError(f"Unsupported format: {fmt!r}")

    @property
    def nnz(self):
        return self._nnz

    @property
    def shape(self):
        return self._shape

    def set_mv_hint(self, transpose=False, expected_calls=1000):
        op = SPARSE_OPERATION_TRANSPOSE if transpose else SPARSE_OPERATION_NON_TRANSPOSE
        status = self._lib.mkl_sparse_set_mv_hint(self._handle, c_int(op), GENERAL_DESCR, self._ct_int(expected_calls))
        if status not in (SPARSE_STATUS_SUCCESS, SPARSE_STATUS_NOT_SUPPORTED):
            _check(status, "mkl_sparse_set_mv_hint")

    def set_mm_hint(self, k, transpose=False, expected_calls=1000):
        op = SPARSE_OPERATION_TRANSPOSE if transpose else SPARSE_OPERATION_NON_TRANSPOSE
        status = self._lib.mkl_sparse_set_mm_hint(
            self._handle,
            c_int(op),
            GENERAL_DESCR,
            c_int(SPARSE_LAYOUT_ROW_MAJOR),
            self._ct_int(k),
            self._ct_int(expected_calls),
        )
        if status not in (SPARSE_STATUS_SUCCESS, SPARSE_STATUS_NOT_SUPPORTED):
            _check(status, "mkl_sparse_set_mm_hint")

    def optimize(self):
        status = self._lib.mkl_sparse_optimize(self._handle)
        if status not in (SPARSE_STATUS_SUCCESS, SPARSE_STATUS_NOT_SUPPORTED):
            _check(status, "mkl_sparse_optimize")

    def mv(self, x, y, alpha=1.0, beta=1.0, transpose=False):
        x_arr = _require_dense_vector(x, dtype=self._dtype, label="x")
        y_arr = _require_dense_vector(y, dtype=self._dtype, label="y")
        op = SPARSE_OPERATION_TRANSPOSE if transpose else SPARSE_OPERATION_NON_TRANSPOSE
        scalar = self._spec.c_scalar
        scalar_p = POINTER(self._spec.c_scalar)
        _check(
            getattr(self._lib, self._spec.mv)(
                c_int(op),
                scalar(alpha),
                self._handle,
                GENERAL_DESCR,
                x_arr.ctypes.data_as(scalar_p),
                scalar(beta),
                y_arr.ctypes.data_as(scalar_p),
            ),
            self._spec.mv,
        )

    def mm(self, B, C, alpha=1.0, beta=1.0, transpose=False):
        b_arr, ldb = _dense_leading_dimension(B, dtype=self._dtype, label="B")
        c_arr, ldc = _dense_leading_dimension(C, dtype=self._dtype, label="C")
        if b_arr.shape[1] != c_arr.shape[1]:
            raise ValueError(f"B and C must have the same number of columns, got {b_arr.shape[1]} and {c_arr.shape[1]}")
        op = SPARSE_OPERATION_TRANSPOSE if transpose else SPARSE_OPERATION_NON_TRANSPOSE
        scalar = self._spec.c_scalar
        scalar_p = POINTER(self._spec.c_scalar)
        _check(
            getattr(self._lib, self._spec.mm)(
                c_int(op),
                scalar(alpha),
                self._handle,
                GENERAL_DESCR,
                c_int(SPARSE_LAYOUT_ROW_MAJOR),
                b_arr.ctypes.data_as(scalar_p),
                self._ct_int(int(b_arr.shape[1])),
                self._ct_int(ldb),
                scalar(beta),
                c_arr.ctypes.data_as(scalar_p),
                self._ct_int(ldc),
            ),
            self._spec.mm,
        )

    def destroy(self):
        if self._handle:
            self._lib.mkl_sparse_destroy(self._handle)
            self._handle = None


def mkl_set_num_threads(n):
    lib, _, _ = _ensure_loaded()
    lib.MKL_Set_Num_Threads(c_int(n))
