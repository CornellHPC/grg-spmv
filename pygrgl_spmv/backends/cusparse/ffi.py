"""
CUDA / cuSPARSE utility layer — constants, ctypes signatures, and helpers.

Isolates all ctypes boilerplate so that backend modules only deal with
high-level cuSPARSE calls.
"""

import ctypes
import logging
from ctypes import c_int, c_int64, c_size_t, c_void_p, byref, POINTER

import numpy as np

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# cuSPARSE enum constants  (values from cusparse.h / library_types.h)
# ---------------------------------------------------------------------------
CUSPARSE_STATUS_SUCCESS = 0
CUSPARSE_OPERATION_NON_TRANSPOSE = 0
CUSPARSE_OPERATION_TRANSPOSE = 1
CUSPARSE_ORDER_COL = 1              # column-major dense layout
CUSPARSE_ORDER_ROW = 2              # row-major (C-contiguous) dense layout
CUSPARSE_INDEX_32I = 2              # int32 index type
CUSPARSE_INDEX_BASE_ZERO = 0
CUSPARSE_POINTER_MODE_DEVICE = 1    # alpha/beta are device pointers

# cudaDataType  (library_types.h)
CUDA_R_32F = 0   # float32
CUDA_R_64F = 1   # float64

# cusparseSpMMAlg_t - SpMM algorithm selection
CUSPARSE_SPMM_ALG_DEFAULT = 0
CUSPARSE_SPMM_COO_ALG1 = 1
CUSPARSE_SPMM_COO_ALG2 = 2
CUSPARSE_SPMM_COO_ALG3 = 3
CUSPARSE_SPMM_COO_ALG4 = 5
CUSPARSE_SPMM_CSR_ALG1 = 4
CUSPARSE_SPMM_CSR_ALG2 = 6
CUSPARSE_SPMM_CSR_ALG3 = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_status(status, func_name):
    """Raise RuntimeError on cuSPARSE error."""
    if status != CUSPARSE_STATUS_SUCCESS:
        raise RuntimeError(f"cuSPARSE {func_name} failed with status {status}")


def cuda_dtype(np_dtype):
    """Map numpy dtype to CUDA datatype constant."""
    dt = np.dtype(np_dtype)
    if dt == np.float32:
        return CUDA_R_32F
    elif dt == np.float64:
        return CUDA_R_64F
    else:
        raise ValueError(f"Unsupported dtype for cuSPARSE: {dt}")


def parse_algo(algo: str):
    """Parse algo string to cuSPARSE algorithm constant."""
    algo_map = {
        'default': CUSPARSE_SPMM_ALG_DEFAULT,
        'coo_alg1': CUSPARSE_SPMM_COO_ALG1,
        'coo_alg2': CUSPARSE_SPMM_COO_ALG2,
        'coo_alg3': CUSPARSE_SPMM_COO_ALG3,
        'coo_alg4': CUSPARSE_SPMM_COO_ALG4,
        'csr_alg1': CUSPARSE_SPMM_CSR_ALG1,
        'csr_alg2': CUSPARSE_SPMM_CSR_ALG2,
        'csr_alg3': CUSPARSE_SPMM_CSR_ALG3,
    }
    if not isinstance(algo, str):
        raise TypeError(f"algo must be a string, got {type(algo).__name__}")
    algo_key = algo.lower()
    if algo_key not in algo_map:
        raise ValueError(f"Unknown algo: {algo}. Valid options: {list(algo_map.keys())}")
    return algo_map[algo_key]


# ---------------------------------------------------------------------------
# ctypes signature setup
# ---------------------------------------------------------------------------

def _setup_cusparse_signatures(lib):
    """Set argtypes/restype on all cuSPARSE functions we use."""

    # cusparseGetVersion
    lib.cusparseGetVersion.argtypes = [c_void_p, POINTER(c_int)]
    lib.cusparseGetVersion.restype = c_int

    # cusparseCreate / cusparseDestroy
    lib.cusparseCreate.argtypes = [POINTER(c_void_p)]
    lib.cusparseCreate.restype = c_int

    lib.cusparseDestroy.argtypes = [c_void_p]
    lib.cusparseDestroy.restype = c_int

    # cusparseSetPointerMode
    lib.cusparseSetPointerMode.argtypes = [c_void_p, c_int]
    lib.cusparseSetPointerMode.restype = c_int

    # cusparseSetStream
    lib.cusparseSetStream.argtypes = [c_void_p, c_size_t]
    lib.cusparseSetStream.restype = c_int

    # cusparseCreateConstCsr
    lib.cusparseCreateConstCsr.argtypes = [
        POINTER(c_void_p),
        c_int64, c_int64, c_int64,
        c_void_p, c_void_p, c_void_p,
        c_int, c_int, c_int, c_int
    ]
    lib.cusparseCreateConstCsr.restype = c_int

    # cusparseCreateConstCsc
    lib.cusparseCreateConstCsc.argtypes = [
        POINTER(c_void_p),
        c_int64, c_int64, c_int64,
        c_void_p, c_void_p, c_void_p,
        c_int, c_int, c_int, c_int
    ]
    lib.cusparseCreateConstCsc.restype = c_int

    # cusparseCreateConstCoo
    lib.cusparseCreateConstCoo.argtypes = [
        POINTER(c_void_p),
        c_int64, c_int64, c_int64,
        c_void_p, c_void_p, c_void_p,
        c_int, c_int, c_int
    ]
    lib.cusparseCreateConstCoo.restype = c_int

    # cusparseDestroySpMat / cusparseDestroyDnMat
    lib.cusparseDestroySpMat.argtypes = [c_void_p]
    lib.cusparseDestroySpMat.restype = c_int

    lib.cusparseDestroyDnMat.argtypes = [c_void_p]
    lib.cusparseDestroyDnMat.restype = c_int

    # cusparseCreateDnMat
    lib.cusparseCreateDnMat.argtypes = [
        POINTER(c_void_p),
        c_int64, c_int64, c_int64,
        c_void_p, c_int, c_int
    ]
    lib.cusparseCreateDnMat.restype = c_int

    # cusparseSpMM_bufferSize
    _spmm_args = [
        c_void_p,
        c_int, c_int,
        c_void_p, c_void_p, c_void_p,
        c_void_p, c_void_p,
        c_int, c_int,
    ]
    lib.cusparseSpMM_bufferSize.argtypes = _spmm_args + [POINTER(c_size_t)]
    lib.cusparseSpMM_bufferSize.restype = c_int

    # cusparseSpMM_preprocess
    lib.cusparseSpMM_preprocess.argtypes = _spmm_args + [c_void_p]
    lib.cusparseSpMM_preprocess.restype = c_int

    # cusparseSpMM
    lib.cusparseSpMM.argtypes = _spmm_args + [c_void_p]
    lib.cusparseSpMM.restype = c_int


def _load_cusparse():
    """Load libcusparse.so and configure all ctypes signatures."""
    lib = ctypes.cdll.LoadLibrary('libcusparse.so')
    _setup_cusparse_signatures(lib)
    return lib


# ---------------------------------------------------------------------------
# CuSparseLib — high-level wrapper around cuSPARSE ctypes FFI
# ---------------------------------------------------------------------------

class CuSparseLib:
    """High-level wrapper around cuSPARSE ctypes FFI.

    Owns both the ctypes library handle and the cuSPARSE handle.
    All methods call _check_status internally — callers never touch
    raw ctypes or status checking.
    """

    def __init__(self):
        self._lib = _load_cusparse()
        self._handle = c_void_p()
        _check_status(
            self._lib.cusparseCreate(byref(self._handle)),
            'cusparseCreate',
        )
        _check_status(
            self._lib.cusparseSetPointerMode(self._handle, CUSPARSE_POINTER_MODE_DEVICE),
            'cusparseSetPointerMode',
        )

    # --- Descriptor creation ------------------------------------------------

    def create_csr(self, nrows, ncols, nnz, indptr_ptr, indices_ptr, data_ptr, cdt):
        """Create a const CSR sparse matrix descriptor. Returns c_void_p."""
        desc = c_void_p()
        _check_status(self._lib.cusparseCreateConstCsr(
            byref(desc),
            c_int64(nrows), c_int64(ncols), c_int64(nnz),
            c_void_p(indptr_ptr), c_void_p(indices_ptr), c_void_p(data_ptr),
            c_int(CUSPARSE_INDEX_32I), c_int(CUSPARSE_INDEX_32I),
            c_int(CUSPARSE_INDEX_BASE_ZERO), c_int(cdt),
        ), 'cusparseCreateConstCsr')
        return desc

    def create_csc(self, nrows, ncols, nnz, indptr_ptr, indices_ptr, data_ptr, cdt):
        """Create a const CSC sparse matrix descriptor. Returns c_void_p."""
        desc = c_void_p()
        _check_status(self._lib.cusparseCreateConstCsc(
            byref(desc),
            c_int64(nrows), c_int64(ncols), c_int64(nnz),
            c_void_p(indptr_ptr), c_void_p(indices_ptr), c_void_p(data_ptr),
            c_int(CUSPARSE_INDEX_32I), c_int(CUSPARSE_INDEX_32I),
            c_int(CUSPARSE_INDEX_BASE_ZERO), c_int(cdt),
        ), 'cusparseCreateConstCsc')
        return desc

    def create_coo(self, nrows, ncols, nnz, row_ptr, col_ptr, data_ptr, cdt):
        """Create a const COO sparse matrix descriptor. Returns c_void_p."""
        desc = c_void_p()
        _check_status(self._lib.cusparseCreateConstCoo(
            byref(desc),
            c_int64(nrows), c_int64(ncols), c_int64(nnz),
            c_void_p(row_ptr), c_void_p(col_ptr), c_void_p(data_ptr),
            c_int(CUSPARSE_INDEX_32I), c_int(CUSPARSE_INDEX_BASE_ZERO),
            c_int(cdt),
        ), 'cusparseCreateConstCoo')
        return desc

    def create_dnmat(self, nrows, ncols, ld, buf_ptr, cdt, order):
        """Create a dense matrix descriptor. Returns c_void_p."""
        desc = c_void_p()
        _check_status(self._lib.cusparseCreateDnMat(
            byref(desc),
            c_int64(nrows), c_int64(ncols), c_int64(ld),
            c_void_p(buf_ptr),
            c_int(cdt), c_int(order),
        ), 'cusparseCreateDnMat')
        return desc

    def destroy_sp_mat(self, desc):
        """Destroy a sparse matrix descriptor."""
        _check_status(self._lib.cusparseDestroySpMat(desc), 'cusparseDestroySpMat')

    def destroy_dn_mat(self, desc):
        """Destroy a dense matrix descriptor."""
        _check_status(self._lib.cusparseDestroyDnMat(desc), 'cusparseDestroyDnMat')

    # --- Stream management --------------------------------------------------

    def set_stream(self, stream_ptr):
        """Set the CUDA stream for subsequent cuSPARSE calls."""
        _check_status(
            self._lib.cusparseSetStream(self._handle, c_size_t(stream_ptr)),
            'cusparseSetStream',
        )

    # --- SpMM operations ----------------------------------------------------

    def spmm_buffer_size(self, cp, algo, op_a, op_b,
                         alpha_ptr, sp_desc, B_desc, beta_ptr, C_desc, cdt):
        """Query SpMM buffer size and allocate workspace. Returns cupy array."""
        buf_size = c_size_t(0)
        _check_status(self._lib.cusparseSpMM_bufferSize(
            self._handle, c_int(op_a), c_int(op_b),
            c_void_p(alpha_ptr), sp_desc, B_desc,
            c_void_p(beta_ptr), C_desc,
            c_int(cdt), c_int(algo), byref(buf_size),
        ), 'cusparseSpMM_bufferSize')
        return cp.zeros(max(buf_size.value, 4), dtype=cp.uint8)

    def spmm_preprocess(self, algo, op_a, op_b,
                        alpha_ptr, sp_desc, B_desc, beta_ptr, C_desc,
                        cdt, ext_buf_ptr):
        """Run cusparseSpMM_preprocess on an already-allocated workspace."""
        _check_status(self._lib.cusparseSpMM_preprocess(
            self._handle, c_int(op_a), c_int(op_b),
            c_void_p(alpha_ptr), sp_desc, B_desc,
            c_void_p(beta_ptr), C_desc,
            c_int(cdt), c_int(algo), c_void_p(ext_buf_ptr),
        ), 'cusparseSpMM_preprocess')

    def spmm(self, algo, op_a, op_b,
             alpha_ptr, sp_desc, B_desc, beta_ptr, C_desc,
             cdt, ext_buf_ptr):
        """Launch a single cusparseSpMM kernel."""
        _check_status(self._lib.cusparseSpMM(
            self._handle, c_int(op_a), c_int(op_b),
            c_void_p(alpha_ptr), sp_desc, B_desc,
            c_void_p(beta_ptr), C_desc,
            c_int(cdt), c_int(algo), c_void_p(ext_buf_ptr),
        ), 'cusparseSpMM')

    # --- Cleanup ------------------------------------------------------------

    def destroy(self):
        """Destroy the cuSPARSE handle."""
        if self._handle:
            self._lib.cusparseDestroy(self._handle)
            self._handle = None

    @property
    def version(self):
        """Return cuSPARSE version string (e.g. '12.6.1')."""
        ver = c_int(0)
        _check_status(
            self._lib.cusparseGetVersion(self._handle, byref(ver)),
            'cusparseGetVersion',
        )
        v = ver.value
        return f"{v // 10000}.{(v % 10000) // 100}.{v % 100}"


# ---------------------------------------------------------------------------
# GPU timing
# ---------------------------------------------------------------------------

class GpuTimer:
    """CUDA event-based GPU timer. Zero overhead when not created.

    Records CUDA events on a stream and reports elapsed time between them.
    Uses GPU-side timestamps for accurate kernel timing; H2D/D2H phases
    measure wall-clock time between event recordings (accurate because
    the stream is idle during synchronous memcpy).

    Usage::

        timer = GpuTimer(cp, stream) if verbose else None
        if timer: timer.mark()
        do_h2d()
        if timer: timer.mark('H2D')
        launch_kernel()
        if timer: timer.mark('kernel')
        stream.synchronize()
        do_d2h()
        if timer:
            timer.mark('D2H')
            timer.report('forward (graph)')
    """
    __slots__ = ('_cp', '_s', '_ev')

    def __init__(self, cp, stream):
        self._cp = cp
        self._s = stream
        self._ev = []

    def mark(self, label=''):
        ev = self._cp.cuda.Event()
        ev.record(self._s)
        self._ev.append((ev, label))

    def report(self, prefix):
        self._ev[-1][0].synchronize()
        parts = []
        total = 0.0
        for i in range(len(self._ev) - 1):
            dt = self._cp.cuda.get_elapsed_time(self._ev[i][0], self._ev[i + 1][0])
            parts.append(f"{self._ev[i + 1][1]}={dt:.2f}ms")
            total += dt
        _LOGGER.info("%s: %s total=%.2fms", prefix, " ".join(parts), total)
