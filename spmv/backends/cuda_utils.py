"""
CUDA / cuSPARSE utility layer — constants, ctypes signatures, and helpers.

Isolates all ctypes boilerplate so that backend modules only deal with
high-level cuSPARSE calls.
"""

import ctypes
from ctypes import c_int, c_int64, c_size_t, c_void_p, byref, POINTER

import numpy as np

# ---------------------------------------------------------------------------
# cuSPARSE enum constants  (values from cusparse.h / library_types.h)
# ---------------------------------------------------------------------------
CUSPARSE_STATUS_SUCCESS = 0
CUSPARSE_OPERATION_NON_TRANSPOSE = 0
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

def check_status(status, func_name):
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


def parse_algorithm(alg_str):
    """Parse algorithm string to cuSPARSE algorithm constant."""
    alg_map = {
        'default': CUSPARSE_SPMM_ALG_DEFAULT,
        'coo_alg1': CUSPARSE_SPMM_COO_ALG1,
        'coo_alg2': CUSPARSE_SPMM_COO_ALG2,
        'coo_alg3': CUSPARSE_SPMM_COO_ALG3,
        'coo_alg4': CUSPARSE_SPMM_COO_ALG4,
        'csr_alg1': CUSPARSE_SPMM_CSR_ALG1,
        'csr_alg2': CUSPARSE_SPMM_CSR_ALG2,
        'csr_alg3': CUSPARSE_SPMM_CSR_ALG3,
    }
    if isinstance(alg_str, int):
        return alg_str
    alg_lower = alg_str.lower()
    if alg_lower not in alg_map:
        raise ValueError(f"Unknown algorithm: {alg_str}. Valid options: {list(alg_map.keys())}")
    return alg_map[alg_lower]


# ---------------------------------------------------------------------------
# ctypes signature setup
# ---------------------------------------------------------------------------

def setup_cusparse_signatures(lib):
    """Set argtypes/restype on all cuSPARSE functions we use."""

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

    # cusparseCreateCsr
    lib.cusparseCreateCsr.argtypes = [
        POINTER(c_void_p),
        c_int64, c_int64, c_int64,
        c_void_p, c_void_p, c_void_p,
        c_int, c_int, c_int, c_int
    ]
    lib.cusparseCreateCsr.restype = c_int

    # cusparseCreateCsc
    lib.cusparseCreateCsc.argtypes = [
        POINTER(c_void_p),
        c_int64, c_int64, c_int64,
        c_void_p, c_void_p, c_void_p,
        c_int, c_int, c_int, c_int
    ]
    lib.cusparseCreateCsc.restype = c_int

    # cusparseCreateCoo
    lib.cusparseCreateCoo.argtypes = [
        POINTER(c_void_p),
        c_int64, c_int64, c_int64,
        c_void_p, c_void_p, c_void_p,
        c_int, c_int, c_int
    ]
    lib.cusparseCreateCoo.restype = c_int

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


def load_cusparse():
    """Load libcusparse.so and configure all ctypes signatures."""
    lib = ctypes.cdll.LoadLibrary('libcusparse.so')
    setup_cusparse_signatures(lib)
    return lib


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
        print(f"{prefix}: {' '.join(parts)} total={total:.2f}ms")
