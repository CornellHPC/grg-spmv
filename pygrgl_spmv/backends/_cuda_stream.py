"""Small CUDA stream/device helpers shared by GPU backends."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from ctypes import POINTER, byref, c_char_p, c_int, c_void_p
from functools import cache
from numbers import Integral
from pathlib import Path

_CUDA_ROOT_ENV_VARS = ("CUDA_HOME", "NVHPC_CUDA_HOME", "CUDATOOLKIT_HOME", "CRAY_CUDATOOLKIT_DIR")
_RTLD_NOW = getattr(os, "RTLD_NOW", 0)
_RTLD_GLOBAL = getattr(os, "RTLD_GLOBAL", 0)


class CudaStreamToken:
    """Tiny CUDA stream protocol wrapper for a raw ``cudaStream_t``."""

    def __init__(self, ptr: int) -> None:
        self._ptr = int(ptr)

    def __cuda_stream__(self) -> tuple[int, int]:
        return 0, self._ptr


def _iter_unique_refs(refs: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        unique.append(ref)
    return unique


def _candidate_cudart_refs() -> list[str]:
    refs: list[str] = []
    for name in _CUDA_ROOT_ENV_VARS:
        raw = os.environ.get(name)
        if not raw:
            continue
        root = Path(raw)
        refs.append(str(root / "lib64" / "libcudart.so"))
        refs.append(str(root / "lib64" / "libcudart.so.12"))
    for raw in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if not raw:
            continue
        refs.append(str(Path(raw) / "libcudart.so"))
        refs.append(str(Path(raw) / "libcudart.so.12"))
    found = ctypes.util.find_library("cudart")
    if found:
        refs.append(found)
    refs.extend(("libcudart.so", "libcudart.so.12"))
    return _iter_unique_refs(refs)


def _load_cuda_runtime_library():
    mode = _RTLD_GLOBAL | _RTLD_NOW
    last_error: OSError | None = None
    for ref in _candidate_cudart_refs():
        try:
            lib = ctypes.CDLL(ref, mode=mode)
            break
        except OSError as exc:
            last_error = exc
    else:
        raise OSError("Could not load CUDA runtime library libcudart") from last_error

    lib.cudaGetErrorString.argtypes = [c_int]
    lib.cudaGetErrorString.restype = c_char_p
    lib.cudaGetDeviceCount.argtypes = [POINTER(c_int)]
    lib.cudaGetDeviceCount.restype = c_int
    try:
        lib.cudaStreamGetDevice.argtypes = [c_void_p, POINTER(c_int)]
        lib.cudaStreamGetDevice.restype = c_int
    except AttributeError as exc:
        raise RuntimeError("CUDA runtime does not expose cudaStreamGetDevice") from exc
    return lib


@cache
def _cuda_runtime():
    return _load_cuda_runtime_library()


def _runtime_error_message(lib, status: int) -> str:
    try:
        raw = lib.cudaGetErrorString(int(status))
    except Exception:
        raw = None
    if raw is None:
        return f"status {status}"
    return raw.decode("utf-8", errors="replace")


def _check_runtime_status(status: int, func_name: str) -> None:
    if int(status) == 0:
        return
    lib = _cuda_runtime()
    raise RuntimeError(
        f"CUDA runtime {func_name} failed with status {int(status)}: {_runtime_error_message(lib, int(status))}"
    )


def parse_cuda_device(device: object) -> int:
    """Validate and normalize a CUDA device ordinal."""
    if isinstance(device, bool) or not isinstance(device, Integral):
        raise TypeError(f"CUDA device must be a non-bool int ordinal, got {type(device).__name__}")
    device_id = int(device)
    if device_id < 0:
        raise ValueError(f"CUDA device ordinal must be non-negative, got {device_id}")
    count = _cuda_device_count()
    if device_id >= count:
        raise ValueError(f"CUDA device ordinal {device_id} is out of range for {count} visible CUDA devices")
    return device_id


def _cuda_device_count() -> int:
    count = c_int()
    _check_runtime_status(int(_cuda_runtime().cudaGetDeviceCount(byref(count))), "cudaGetDeviceCount")
    return int(count.value)


def _cuda_stream_device(ptr: int) -> int:
    """Resolve the owning CUDA device of a non-null ``cudaStream_t`` handle."""
    stream_ptr = int(ptr)
    if stream_ptr == 0:
        raise ValueError("cudaStreamGetDevice requires a non-null CUDA stream handle")
    device = c_int()
    _check_runtime_status(
        int(_cuda_runtime().cudaStreamGetDevice(c_void_p(stream_ptr), byref(device))),
        "cudaStreamGetDevice",
    )
    return int(device.value)


def parse_cuda_stream(stream: object) -> tuple[int, object | None]:
    """Return ``(cudaStream_t, owner_ref)`` from an int or CUDA-stream-protocol object."""
    if isinstance(stream, bool):
        raise TypeError("CUDA stream must be a non-bool int handle or an object implementing __cuda_stream__()")
    if isinstance(stream, Integral):
        ptr = int(stream)
        if ptr < 0:
            raise ValueError(f"CUDA stream handle must be non-negative, got {ptr}")
        return ptr, None

    method = getattr(stream, "__cuda_stream__", None)
    if method is None:
        raise TypeError(
            "CUDA stream must be a non-bool int handle or an object implementing __cuda_stream__()"
        )

    token = method()
    if isinstance(token, bool):
        raise TypeError("__cuda_stream__() must return an int handle or a (version, handle) pair")
    if isinstance(token, Integral):
        ptr = int(token)
    elif isinstance(token, (tuple, list)):
        if len(token) != 2:
            raise TypeError("__cuda_stream__() tuple/list return must be exactly (version, handle)")
        version, ptr = token
        version = int(version)
        if version != 0:
            raise ValueError(f"Unsupported CUDA stream protocol version {version}; expected 0")
        ptr = int(ptr)
    else:
        raise TypeError("__cuda_stream__() must return an int handle or a (version, handle) pair")

    if ptr < 0:
        raise ValueError(f"CUDA stream handle must be non-negative, got {ptr}")
    return ptr, stream


__all__ = ["CudaStreamToken", "parse_cuda_device", "parse_cuda_stream"]
