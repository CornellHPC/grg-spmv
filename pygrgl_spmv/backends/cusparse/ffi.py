"""
CUDA / cuSPARSE utility layer — constants, ctypes signatures, and helpers.

Isolates all ctypes boilerplate so that backend modules only deal with
high-level cuSPARSE calls.
"""

import ctypes
import logging
import os
from ctypes import (
    POINTER,
    byref,
    c_char_p,
    c_int,
    c_int64,
    c_size_t,
    c_uint,
    c_ulonglong,
    c_ubyte,
    c_ushort,
    c_void_p,
)
from pathlib import Path

import numpy as np

_LOGGER = logging.getLogger(__name__)
_CUDA_ROOT_ENV_VARS = ("CUDA_HOME", "NVHPC_CUDA_HOME", "CUDATOOLKIT_HOME", "CRAY_CUDATOOLKIT_DIR")
_RTLD_NOW = getattr(os, "RTLD_NOW", 0)
_RTLD_GLOBAL = getattr(os, "RTLD_GLOBAL", 0)
_RTLD_DEEPBIND = getattr(os, "RTLD_DEEPBIND", 0)

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

# CUDA Driver API constants used by the VMM-backed shared ones source.
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_status(status, func_name):
    """Raise RuntimeError on cuSPARSE error."""
    if status != CUSPARSE_STATUS_SUCCESS:
        raise RuntimeError(f"cuSPARSE {func_name} failed with status {status}")


def _check_driver_status(lib, status, func_name):
    """Raise RuntimeError on CUDA Driver API error."""
    if status != CUDA_SUCCESS:
        err_str = c_char_p()
        try:
            lib.cuGetErrorString(status, byref(err_str))
        except Exception:
            pass
        message = None if err_str.value is None else err_str.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"CUDA driver {func_name} failed with status {status}: {message or '<unknown>'}")


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


def _iter_unique_paths(paths):
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        yield path


def _cuda_roots() -> list[Path]:
    roots = [Path(raw) for name in _CUDA_ROOT_ENV_VARS if (raw := os.environ.get(name))]
    return [path for path in _iter_unique_paths(roots) if path.exists()]


def _ld_library_dirs() -> list[Path]:
    raw = os.environ.get("LD_LIBRARY_PATH", "")
    return [Path(token) for token in raw.split(":") if token]


def _nvhpc_math_lib_dir(cuda_root: Path) -> Path | None:
    if cuda_root.parent.name != "cuda":
        return None
    return cuda_root.parent.parent / "math_libs" / cuda_root.name / "lib64"


def _candidate_library_dirs(*, for_cusparse: bool) -> list[Path]:
    dirs: list[Path] = []
    for cuda_root in _cuda_roots():
        if for_cusparse:
            math_lib = _nvhpc_math_lib_dir(cuda_root)
            if math_lib is not None:
                dirs.append(math_lib)
        dirs.append(cuda_root / "lib64")
        if not for_cusparse:
            dirs.append(cuda_root / "nvvm" / "lib64")
    dirs.extend(_ld_library_dirs())
    return [path for path in _iter_unique_paths(dirs) if path.exists()]


def _resolve_library_path(name: str, dirs: list[Path]) -> Path | None:
    for directory in dirs:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


class CUmemLocation(ctypes.Structure):
    _fields_ = [
        ("type", c_int),
        ("id", c_int),
    ]


class _CUmemAllocationPropFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", c_ubyte),
        ("gpuDirectRDMACapable", c_ubyte),
        ("usage", c_ushort),
        ("reserved", c_ubyte * 4),
    ]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", c_int),
        ("requestedHandleTypes", c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", c_void_p),
        ("allocFlags", _CUmemAllocationPropFlags),
    ]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("location", CUmemLocation),
        ("flags", c_ulonglong),
    ]


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

    # cusparseSpMM_preprocess (kept for diagnostics / standalone repro harnesses)
    lib.cusparseSpMM_preprocess.argtypes = _spmm_args + [c_void_p]
    lib.cusparseSpMM_preprocess.restype = c_int

    # cusparseSpMM
    lib.cusparseSpMM.argtypes = _spmm_args + [c_void_p]
    lib.cusparseSpMM.restype = c_int


def _setup_cuda_driver_signatures(lib):
    """Set argtypes/restype on the CUDA Driver API functions used for VMM."""

    lib.cuInit.argtypes = [c_uint]
    lib.cuInit.restype = c_int

    lib.cuGetErrorString.argtypes = [c_int, POINTER(c_char_p)]
    lib.cuGetErrorString.restype = c_int

    lib.cuDeviceGet.argtypes = [POINTER(c_int), c_int]
    lib.cuDeviceGet.restype = c_int

    lib.cuDeviceGetAttribute.argtypes = [POINTER(c_int), c_int, c_int]
    lib.cuDeviceGetAttribute.restype = c_int

    lib.cuCtxGetCurrent.argtypes = [POINTER(c_void_p)]
    lib.cuCtxGetCurrent.restype = c_int

    lib.cuMemGetAllocationGranularity.argtypes = [POINTER(c_size_t), POINTER(CUmemAllocationProp), c_int]
    lib.cuMemGetAllocationGranularity.restype = c_int

    lib.cuMemCreate.argtypes = [POINTER(c_ulonglong), c_size_t, POINTER(CUmemAllocationProp), c_ulonglong]
    lib.cuMemCreate.restype = c_int

    lib.cuMemAddressReserve.argtypes = [POINTER(c_ulonglong), c_size_t, c_size_t, c_ulonglong, c_ulonglong]
    lib.cuMemAddressReserve.restype = c_int

    lib.cuMemMap.argtypes = [c_ulonglong, c_size_t, c_size_t, c_ulonglong, c_ulonglong]
    lib.cuMemMap.restype = c_int

    lib.cuMemSetAccess.argtypes = [c_ulonglong, c_size_t, POINTER(CUmemAccessDesc), c_size_t]
    lib.cuMemSetAccess.restype = c_int

    lib.cuMemUnmap.argtypes = [c_ulonglong, c_size_t]
    lib.cuMemUnmap.restype = c_int

    lib.cuMemRelease.argtypes = [c_ulonglong]
    lib.cuMemRelease.restype = c_int

    lib.cuMemAddressFree.argtypes = [c_ulonglong, c_size_t]
    lib.cuMemAddressFree.restype = c_int


def _load_cusparse_library():
    """Load libcusparse.so and configure all ctypes signatures."""
    nvjit_path = _resolve_library_path("libnvJitLink.so.12", _candidate_library_dirs(for_cusparse=False))
    if nvjit_path is not None:
        ctypes.CDLL(str(nvjit_path), mode=_RTLD_GLOBAL | _RTLD_NOW)
    else:
        try:
            ctypes.CDLL("libnvJitLink.so.12", mode=_RTLD_GLOBAL | _RTLD_NOW)
        except OSError:
            pass

    cusparse_path = _resolve_library_path("libcusparse.so", _candidate_library_dirs(for_cusparse=True))
    cusparse_ref = "libcusparse.so" if cusparse_path is None else str(cusparse_path)
    cusparse_mode = _RTLD_GLOBAL | _RTLD_NOW
    if _RTLD_DEEPBIND:
        cusparse_mode |= _RTLD_DEEPBIND
    lib = ctypes.CDLL(cusparse_ref, mode=cusparse_mode)
    _setup_cusparse_signatures(lib)
    _LOGGER.debug(
        "Loaded cuSPARSE runtime nvJitLink=%s libcusparse=%s",
        "<default>" if nvjit_path is None else str(nvjit_path),
        cusparse_ref,
    )
    return lib


def _load_cuda_driver_library():
    """Load libcuda.so.1 and configure the Driver API signatures needed for VMM."""
    driver_path = _resolve_library_path("libcuda.so.1", _candidate_library_dirs(for_cusparse=False))
    driver_ref = "libcuda.so.1" if driver_path is None else str(driver_path)
    driver_mode = _RTLD_GLOBAL | _RTLD_NOW
    if _RTLD_DEEPBIND:
        driver_mode |= _RTLD_DEEPBIND
    lib = ctypes.CDLL(driver_ref, mode=driver_mode)
    _setup_cuda_driver_signatures(lib)
    _LOGGER.debug("Loaded CUDA driver runtime libcuda=%s", driver_ref)
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
        self._lib = _load_cusparse_library()
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

    def spmm_buffer_size(self, algo, op_a, op_b,
                         alpha_ptr, sp_desc, B_desc, beta_ptr, C_desc, cdt):
        """Query SpMM workspace size in bytes."""
        buf_size = c_size_t(0)
        _check_status(self._lib.cusparseSpMM_bufferSize(
            self._handle, c_int(op_a), c_int(op_b),
            c_void_p(alpha_ptr), sp_desc, B_desc,
            c_void_p(beta_ptr), C_desc,
            c_int(cdt), c_int(algo), byref(buf_size),
        ), 'cusparseSpMM_bufferSize')
        return int(buf_size.value)

    def spmm_preprocess(self, algo, op_a, op_b,
                        alpha_ptr, sp_desc, B_desc, beta_ptr, C_desc,
                        cdt, ext_buf_ptr):
        """Run cusparseSpMM_preprocess on an already-allocated workspace.

        Retained for diagnostics and standalone repro harnesses; the Python
        cuSPARSE backend runtime does not currently call this.
        """
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


class CudaVmmDriver:
    """Small CUDA Driver API wrapper for VMM-backed shared ones."""

    def __init__(self):
        self._lib = _load_cuda_driver_library()
        _check_driver_status(self._lib, self._lib.cuInit(0), "cuInit")

    def current_context(self) -> int | None:
        ctx = c_void_p()
        _check_driver_status(self._lib, self._lib.cuCtxGetCurrent(byref(ctx)), "cuCtxGetCurrent")
        return None if ctx.value is None else int(ctx.value)

    def device_handle(self, device_id: int) -> int:
        handle = c_int()
        _check_driver_status(self._lib, self._lib.cuDeviceGet(byref(handle), c_int(int(device_id))), "cuDeviceGet")
        return int(handle.value)

    def vmm_supported(self, device_id: int) -> bool:
        value = c_int()
        _check_driver_status(
            self._lib,
            self._lib.cuDeviceGetAttribute(
                byref(value),
                c_int(CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED),
                c_int(self.device_handle(device_id)),
            ),
            "cuDeviceGetAttribute",
        )
        return bool(value.value)

    def allocation_granularity(self, device_id: int, *, recommended: bool) -> int:
        prop = CUmemAllocationProp()
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = 0
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = int(self.device_handle(device_id))
        granularity = c_size_t()
        _check_driver_status(
            self._lib,
            self._lib.cuMemGetAllocationGranularity(
                byref(granularity),
                byref(prop),
                c_int(CU_MEM_ALLOC_GRANULARITY_RECOMMENDED if recommended else CU_MEM_ALLOC_GRANULARITY_MINIMUM),
            ),
            "cuMemGetAllocationGranularity",
        )
        return int(granularity.value)

    def mem_create(self, device_id: int, size_bytes: int) -> int:
        prop = CUmemAllocationProp()
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = 0
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = int(self.device_handle(device_id))
        handle = c_ulonglong()
        _check_driver_status(
            self._lib,
            self._lib.cuMemCreate(byref(handle), c_size_t(int(size_bytes)), byref(prop), c_ulonglong(0)),
            "cuMemCreate",
        )
        return int(handle.value)

    def address_reserve(self, size_bytes: int, *, alignment_bytes: int = 0) -> int:
        addr = c_ulonglong()
        _check_driver_status(
            self._lib,
            self._lib.cuMemAddressReserve(
                byref(addr),
                c_size_t(int(size_bytes)),
                c_size_t(int(alignment_bytes)),
                c_ulonglong(0),
                c_ulonglong(0),
            ),
            "cuMemAddressReserve",
        )
        return int(addr.value)

    def mem_map(self, addr: int, size_bytes: int, handle: int) -> None:
        _check_driver_status(
            self._lib,
            self._lib.cuMemMap(
                c_ulonglong(int(addr)),
                c_size_t(int(size_bytes)),
                c_size_t(0),
                c_ulonglong(int(handle)),
                c_ulonglong(0),
            ),
            "cuMemMap",
        )

    def mem_set_access(self, addr: int, size_bytes: int, device_id: int) -> None:
        access = CUmemAccessDesc()
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = int(self.device_handle(device_id))
        access.flags = c_ulonglong(CU_MEM_ACCESS_FLAGS_PROT_READWRITE)
        _check_driver_status(
            self._lib,
            self._lib.cuMemSetAccess(
                c_ulonglong(int(addr)),
                c_size_t(int(size_bytes)),
                byref(access),
                c_size_t(1),
            ),
            "cuMemSetAccess",
        )

    def mem_unmap(self, addr: int, size_bytes: int) -> None:
        _check_driver_status(
            self._lib,
            self._lib.cuMemUnmap(c_ulonglong(int(addr)), c_size_t(int(size_bytes))),
            "cuMemUnmap",
        )

    def mem_release(self, handle: int) -> None:
        _check_driver_status(self._lib, self._lib.cuMemRelease(c_ulonglong(int(handle))), "cuMemRelease")

    def address_free(self, addr: int, size_bytes: int) -> None:
        _check_driver_status(
            self._lib,
            self._lib.cuMemAddressFree(c_ulonglong(int(addr)), c_size_t(int(size_bytes))),
            "cuMemAddressFree",
        )
