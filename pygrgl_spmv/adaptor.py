"""High-level convenience layer for loading and using GRG SpMV backends."""

from __future__ import annotations

import contextlib
import concurrent.futures
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pygrgl_spmv.grg import RuntimeRequirements
from pygrgl_spmv.backends.mkl import MklPlan, MklPlanPair, MklRuntime, plan_mkl_layout
from pygrgl_spmv.backends.cusparse import (
    CusparsePlan,
    CusparsePlanPair,
    CusparseRuntime,
    plan_cusparse_layout,
)


# ---------------------------------------------------------------------------
# CUDA-graph wrappers
# ---------------------------------------------------------------------------

class CapturedBoundGRG:
    """BoundGRG variant backed by pre-captured CUDA graphs.

    All BoundGRG attributes are delegated to the wrapped *grg* via __getattr__.

    matmul() performs graph replay. When *native=False* (default), input is a
    numpy array copied host→device and output is returned as a numpy array.
    When *native=True*, input is a GPU tensor or cupy array and output is a
    cupy.ndarray on device — no host transfers occur.
    """

    def __init__(self, grg, prepared_ops, graphs, src_tensors, init_tensors, miss_tensors, capture_stream, *, native=False):
        self._grg = grg
        self._prepared_ops = prepared_ops  # dict[key, PreparedOp]
        self._graphs = graphs              # dict[key, CUDAGraph]
        self._src_tensors = src_tensors    # dict[key, torch.Tensor]
        self._init_tensors = init_tensors  # dict[key, torch.Tensor]
        self._miss_tensors = miss_tensors  # dict[key, torch.Tensor]
        self._capture_stream = capture_stream
        self._native = native

    @property
    def use_cupy(self):
        return self._native

    def __getattr__(self, name):
        if name == "_grg":
            raise AttributeError(name)
        return getattr(self._grg, name)

    def _key(self, direction, by_individual, init_mode="none", use_miss=False):
        k = (direction, by_individual, init_mode, use_miss)
        return k if k in self._src_tensors else (direction, by_individual, "none", False)

    @staticmethod
    def _to_np_dtype(d):
        if isinstance(d, np.dtype):
            return d
        mod = getattr(type(d), "__module__", "")
        if mod.startswith("torch"):
            import torch
            _map = {
                torch.float32: np.float32,
                torch.float64: np.float64,
                torch.int32:   np.int32,
            }
            if d not in _map:
                raise TypeError(f"unsupported torch dtype {d!r}")
            return np.dtype(_map[d])
        return np.dtype(d)

    def _validate_array(self, arr, *, name, role, expected_shape, expected_dtype, expected_device):
        """Validate one input array against expected shape/dtype/device.

        expected_device: torch.device (native mode) or None (copy mode → numpy).
        role: 'input' | 'init' | 'miss-in' | 'miss-out' (for error wording).
        """
        if arr is None:
            raise ValueError(f"matmul(): {name} ({role}) is required but got None")

        import torch
        try:
            import cupy as cp
            _cp_ndarray = cp.ndarray
        except Exception:
            _cp_ndarray = ()

        exp_np_dtype = self._to_np_dtype(expected_dtype)

        if expected_device is not None:
            if isinstance(arr, torch.Tensor):
                if not arr.is_cuda:
                    raise TypeError(
                        f"matmul(): {name} ({role}) expected GPU tensor on {expected_device}, "
                        f"got CPU torch.Tensor"
                    )
                if arr.device != expected_device:
                    raise TypeError(
                        f"matmul(): {name} ({role}) expected device {expected_device}, "
                        f"got torch tensor on {arr.device}"
                    )
                actual_np_dtype = self._to_np_dtype(arr.dtype)
            elif _cp_ndarray and isinstance(arr, _cp_ndarray):
                if arr.device.id != expected_device.index:
                    raise TypeError(
                        f"matmul(): {name} ({role}) expected device {expected_device}, "
                        f"got cupy array on cuda:{arr.device.id}"
                    )
                actual_np_dtype = np.dtype(arr.dtype)
            else:
                raise TypeError(
                    f"matmul(): {name} ({role}) expected torch.Tensor or cupy.ndarray on "
                    f"{expected_device} (native mode), got {type(arr).__name__}"
                )
        else:
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"matmul(): {name} ({role}) expected numpy.ndarray (copy mode), "
                    f"got {type(arr).__name__}"
                )
            actual_np_dtype = arr.dtype

        allow_int32 = (
            exp_np_dtype == np.dtype(np.float64)
            and actual_np_dtype == np.dtype(np.int32)
        )
        if actual_np_dtype != exp_np_dtype and not allow_int32:
            raise TypeError(
                f"matmul(): {name} ({role}) dtype mismatch: expected {exp_np_dtype}, got {actual_np_dtype}"
            )

        actual_shape = tuple(arr.shape)
        if actual_shape != tuple(expected_shape):
            raise ValueError(
                f"matmul(): {name} ({role}) shape mismatch: expected {tuple(expected_shape)}, got {actual_shape}"
            )

    def _maybe_cast_to_capture_dtype(self, arr, target_dtype):
        """If arr is int32 and target is fp64, cast to fp64; otherwise pass through.

        Used to support int32 inputs to an fp64 capture. In native mode the
        result is a torch.Tensor (works for torch and cupy inputs); in copy
        mode the result is a numpy.ndarray.
        """
        import torch
        actual = self._to_np_dtype(arr.dtype)
        if actual == np.dtype(np.int32) and target_dtype == torch.float64:
            if self._native:
                return torch.as_tensor(arr).to(torch.float64)
            return np.asarray(arr).astype(np.float64, copy=False)
        return arr

    def matmul(self, input, direction, emit_all_nodes=False, by_individual=False,
               init=None, miss=None):
        """Run the captured matmul graph and return the result.

        Parameters
        ----------
        input : array
            Source matrix of shape ``(k, input_cols)``, dtype matching the runtime
            (float32 or float64). Shape must equal ``self._src_tensors[key].shape``.
            Exception: when the capture dtype is float64, int32 input/init/miss
            arrays are also accepted; they are cast to float64 before the kernel
            runs and the returned ``result`` is cast back to int32.
        direction : {"up", "down"}
        init : None | "xtx" | 1-D array | 2-D array
            Initialization payload; must match the init_mode of the captured spec
            (vector / matrix / xtx). Raises ValueError if the captured key needs
            init and ``init`` is None.
        miss : None | array
            DOWN with use_miss: miss INPUT, shape ``self._miss_tensors[key].shape``.
            UP   with use_miss: miss OUTPUT accumulator, shape
            ``self._prepared_ops[key].miss_output.shape``; updated in place (+=).

        Mode-specific contract
        ----------------------
        Native mode (self._native=True):
            input / init / miss MUST be GPU arrays — ``torch.Tensor`` on, or
            ``cupy.ndarray`` on, the same CUDA device the graph was captured on.
            CPU (numpy) arrays are rejected. Returned ``result`` is a fresh
            ``cupy.ndarray`` on that device. miss (UP+use_miss) is updated in
            place via a cupy ``+=`` on cupy's current stream.

        Copy mode (self._native=False):
            input / init / miss MUST be ``numpy.ndarray`` on host. GPU arrays are
            rejected. Host↔device transfers happen internally on
            ``self._capture_stream``. Returned ``result`` is a fresh
            ``numpy.ndarray`` on host. miss (UP+use_miss) is updated in place via
            a numpy ``+=`` on the CPU.

        Output readiness
        ----------------
        When matmul returns, ``result`` and the in-place update to ``miss`` are
        fully materialized in their memory space (host or device). The caller may
        consume them from any stream / library / CPU without further sync.

        Two ``torch.cuda.synchronize(device)`` calls inside matmul enforce this:
          1. After graph_replay, before the cupy ``.copy()`` of the output —
             orders the kernel on capture_stream ahead of cupy reads on cupy's
             current stream.
          2. After the cupy ``.copy()`` and ``miss +=`` — drains cupy's current
             stream so caller-side reads see ready data.

        Both syncs are device-wide barriers (cudaDeviceSynchronize), so they
        also drain any other streams active on the device.

        Raises
        ------
        TypeError
            Array type / device / dtype does not match the mode's contract.
        ValueError
            Array shape mismatch, or required ``init``/``miss`` missing for the
            captured key.
        """
        assert emit_all_nodes is False, "emit_all_nodes=True is not supported for CapturedBoundGRG.matmul()"
        #TODO: support emit all nodes

        if init is None:
            init_mode, init_payload = "none", None
        elif isinstance(init, str):
            if init != "xtx":
                raise ValueError(f"unexpected init value: {init!r}")
            init_mode, init_payload = "xtx", None
        elif hasattr(init, "ndim") and init.ndim == 1:
            init_mode, init_payload = "vector", init
        else:
            init_mode, init_payload = "matrix", init
        use_miss = miss is not None

        import torch
        nvtx = torch.cuda.nvtx
        key = self._key(direction, by_individual, init_mode, use_miss)
        src = self._src_tensors[key]

        expected_dtype = src.dtype
        expected_device = src.device if self._native else None

        # int32 → fp64 coercion: only when capture dtype is fp64 AND input is int32.
        input_is_int32 = (
            src.dtype == torch.float64
            and hasattr(input, "dtype")
            and self._to_np_dtype(input.dtype) == np.dtype(np.int32)
        )

        self._validate_array(
            input, name="input", role="input",
            expected_shape=tuple(src.shape),
            expected_dtype=expected_dtype,
            expected_device=expected_device,
        )
        if key in self._init_tensors:
            self._validate_array(
                init_payload, name="init", role="init",
                expected_shape=tuple(self._init_tensors[key].shape),
                expected_dtype=expected_dtype,
                expected_device=expected_device,
            )
        if key in self._miss_tensors:
            self._validate_array(
                miss, name="miss", role="miss-in",
                expected_shape=tuple(self._miss_tensors[key].shape),
                expected_dtype=expected_dtype,
                expected_device=expected_device,
            )
        elif use_miss:
            op_miss_out = self._prepared_ops[key].miss_output
            self._validate_array(
                miss, name="miss", role="miss-out",
                expected_shape=tuple(op_miss_out.shape),
                expected_dtype=expected_dtype,
                expected_device=expected_device,
            )

        nvtx.range_push(f"matmul_{direction}")

        nvtx.range_push("input")
        if self._native:
            if not isinstance(input, torch.Tensor):
                import cupy as cp
                cp.cuda.get_current_stream().synchronize()
                input = torch.as_tensor(input)
            input = self._maybe_cast_to_capture_dtype(input, src.dtype)
            with torch.cuda.stream(self._capture_stream):
                src.copy_(input)
        else:
            input_cast = self._maybe_cast_to_capture_dtype(input, src.dtype)
            with torch.cuda.stream(self._capture_stream):
                src.copy_(torch.from_numpy(np.ascontiguousarray(input_cast)))
        nvtx.range_pop()

        nvtx.range_push("init")
        if key in self._init_tensors:
            if init_payload is None:
                raise ValueError(f"matmul() requires init data for captured graph key={key!r}")
            init_cast = self._maybe_cast_to_capture_dtype(init_payload, src.dtype)
            with torch.cuda.stream(self._capture_stream):
                if self._native:
                    self._init_tensors[key].copy_(torch.as_tensor(init_cast))
                else:
                    self._init_tensors[key].copy_(torch.from_numpy(np.ascontiguousarray(init_cast)))
        nvtx.range_pop()

        nvtx.range_push("miss_input")
        if key in self._miss_tensors:
            if miss is None:
                raise ValueError(f"matmul() requires miss for captured graph key={key!r}")
            miss_cast = self._maybe_cast_to_capture_dtype(miss, src.dtype)
            with torch.cuda.stream(self._capture_stream):
                if self._native:
                    self._miss_tensors[key].copy_(torch.as_tensor(miss_cast))
                else:
                    self._miss_tensors[key].copy_(torch.from_numpy(np.ascontiguousarray(miss_cast)))
        nvtx.range_pop()

        nvtx.range_push("graph_replay")
        self._graphs[key].replay()
        nvtx.range_pop()

        # Sync #1: kernel on capture_stream → cupy reads on cupy's current stream.
        nvtx.range_push("sync")
        torch.cuda.synchronize(src.device)
        nvtx.range_pop()

        # Native: cp.copy() is async on cupy's current stream; sync #2 drains it.
        nvtx.range_push("output")
        if self._native:
            import cupy as cp
            result = cp.asarray(self._prepared_ops[key].output).copy()
            if input_is_int32:
                result = result.astype(cp.int32)
        else:
            result = self._prepared_ops[key].output.cpu().numpy().copy()
            if input_is_int32:
                result = result.astype(np.int32)
        nvtx.range_pop()

        # Native: cp.copy() and miss += are async on cupy's current stream; sync #2 drains them.
        nvtx.range_push("miss_output")
        op = self._prepared_ops[key]
        if use_miss and hasattr(op, "miss_output") and miss is not None:
            if self._native:
                import cupy as cp
                miss_data = cp.asarray(op.miss_output).copy()
            else:
                miss_data = op.miss_output.cpu().numpy().copy()
            miss += miss_data.astype(miss.dtype, copy=False)
        nvtx.range_pop()

        # Sync #2: cupy copy/accumulate → caller-side reads on any stream.
        nvtx.range_push("sync")
        torch.cuda.synchronize(src.device)
        nvtx.range_pop()

        nvtx.range_pop()
        return result


# ---------------------------------------------------------------------------
# Capture spec and run configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaptureSpec:
    """Describes one prepare_matmul_cuda call to be captured as a CUDA graph."""
    direction: str
    k: int
    by_individual: bool = False
    init_mode: str = "none"   # "none" | "vector" | "matrix" | "xtx"
    use_miss: bool = False


@dataclass(frozen=True)
class RunConfigs:
    """Pairs a RuntimeRequirements with the set of CUDA graph captures to perform.

    req: controls backend buffer allocation (passed to plan_*_layout).
    capture_ops: sequence of CaptureSpec describing every graph to capture when
        cfg.capture=True. If empty, two default ops (up + down) are derived from req.
    """
    req: RuntimeRequirements
    capture_ops: tuple[CaptureSpec, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Backend config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MklBackendConfig:
    """Configuration for the MKL CPU backend.

    n_threads: int or dict mapping file stem to {"mkl_threads": [n_up, n_down]}.
    A value of 0 (or a per-direction 0) triggers auto-detection:
    physical_cores // n_files, minimum 1.
    """
    n_threads: object   # int | dict[str, {"mkl_threads": [int, int]}]
    optimize: bool = False


@dataclass(frozen=True)
class CusparseBackendConfig:
    """Configuration for the cuSPARSE GPU backend.

    device: int or dict mapping file stem to {"cuda_device": int}.
    Files on the same device share a single layout+runtime; different devices
    are loaded in parallel.
    allow_residency: if True, resident mode is preferred. Streaming mode is enabled if set to false.
    capture: if True, capture CUDA graphs after loading and return CapturedBoundGRG.
    native: if True, CapturedBoundGRG uses GPU-native I/O (matmul_native).
    """
    device: object      # int | dict[str, {"cuda_device": int}]
    allow_residency: bool = True
    capture: bool = False
    native: bool = False


# ---------------------------------------------------------------------------
# Backend factory functions
# ---------------------------------------------------------------------------

def make_backend_mkl(n_threads=0, optimize=False, **kwargs) -> MklBackendConfig:
    """Create an MKL backend config.

    n_threads: int (same for all files, 0 = auto-detect) or dict of the form
        {"<file_stem>": {"mkl_threads": [n_up, n_down]}, ...}
    """
    if kwargs:
        raise TypeError(f"make_backend_mkl() got unexpected keyword arguments: {sorted(kwargs)}")
    if not isinstance(n_threads, (int, dict)):
        raise TypeError(f"n_threads must be int or dict, got {type(n_threads).__name__}")
    if isinstance(n_threads, int) and n_threads < 0:
        raise ValueError(f"n_threads must be >= 0, got {n_threads}")
    return MklBackendConfig(n_threads=n_threads, optimize=bool(optimize))


def make_backend_cusparse(device=0, allow_residency=True, capture=False, native=False, **kwargs) -> CusparseBackendConfig:
    """Create a cuSPARSE backend config.

    device: int (same device for all files) or dict of the form
        {"<file_stem>": {"cuda_device": int}, ...}
    capture: if True, CUDA graphs are captured after loading.
    native: if True, matmul uses GPU-native I/O (requires capture=True).
    """
    if kwargs:
        raise TypeError(f"make_backend_cusparse() got unexpected keyword arguments: {sorted(kwargs)}")
    if not isinstance(device, (int, dict)):
        raise TypeError(f"device must be int or dict, got {type(device).__name__}")
    if isinstance(device, int) and device < 0:
        raise ValueError(f"device must be >= 0, got {device}")
    return CusparseBackendConfig(
        device=device,
        allow_residency=bool(allow_residency),
        capture=bool(capture),
        native=bool(native),
    )


# ---------------------------------------------------------------------------
# RunConfig factory functions
# ---------------------------------------------------------------------------

def make_runconfig_matmul(k_up, k_down=None, need_miss=False, **kwargs) -> RunConfigs:
    """Create a RunConfigs for a plain matmul workload."""
    assert False, "UNCHECKED"
    if kwargs:
        raise TypeError(f"make_runconfig_matmul() got unexpected keyword arguments: {sorted(kwargs)}")
    if k_down is None:
        k_down = k_up
    return RunConfigs(
        req=RuntimeRequirements(
            max_k_up=int(k_up),
            max_k_down=int(k_down),
            need_down_miss_input=bool(need_miss),
            need_up_miss_output=bool(need_miss),
            need_init_vector=False,
            need_init_matrix=False,
            need_init_xtx=False,
        ),
        capture_ops=(
            CaptureSpec("up",   int(k_up),   by_individual=False, use_miss=bool(need_miss)),
            CaptureSpec("down", int(k_down), by_individual=False, use_miss=bool(need_miss)),
            CaptureSpec("up",   int(k_up),   by_individual=True,  use_miss=bool(need_miss)),
            CaptureSpec("down", int(k_down), by_individual=True,  use_miss=bool(need_miss)),
        ),
    )


def make_runconfig_pca(**kwargs) -> RunConfigs:
    """Create a RunConfigs for a PCA workload (init_vector enabled, k=20).

    Captures all five graph variants used by PCA:
      up/down at k=20 (main matmul), up/down at k=1 by_individual=True
      (individual eigsh blocks), and up at k=1 by_individual=False
      (allele_counts single pass).
    """
    assert False, "UNCHECKED"
    if kwargs:
        raise TypeError(f"make_runconfig_pca() got unexpected keyword arguments: {sorted(kwargs)}")
    return RunConfigs(
        req=RuntimeRequirements(
            max_k_up=1,
            max_k_down=1,
            need_down_miss_input=False,
            need_up_miss_output=False,
            need_init_vector=True,
            need_init_matrix=False,
            need_init_xtx=False,
        ),

        capture_ops=(
            CaptureSpec("up",   1, by_individual=False, init_mode="vector"),
            CaptureSpec("down", 1, by_individual=False, init_mode="vector"),
            CaptureSpec("up",   1, by_individual=True,  init_mode="vector"),
            CaptureSpec("down", 1, by_individual=True,  init_mode="vector"),
        ),
    )

def make_runconfig_bolt(**kwargs) -> RunConfigs:
    """Create a RunConfigs for a BoltLMM workload (init_vector enabled, k=20).

    Captures all five graph variants used by BoltLMM:
      up/down at k=20 (main matmul), up/down at k=1 by_individual=True
      (individual eigsh blocks), and up at k=1 by_individual=False
      (allele_counts single pass).
    """
    if kwargs:
        raise TypeError(f"make_runconfig_boltlmm() got unexpected keyword arguments: {sorted(kwargs)}")
    return RunConfigs(
        req=RuntimeRequirements(
            max_k_up=1,
            max_k_down=1,
            need_down_miss_input=True,
            need_up_miss_output=True,
            need_init_vector=False,
            need_init_matrix=False,
            need_init_xtx=True,
        ),

        capture_ops=(
            # all others
            CaptureSpec("up",   1, by_individual=True),
            CaptureSpec("up",   1, by_individual=True, init_mode="xtx"),
            CaptureSpec("down", 1, by_individual=True),
            CaptureSpec("up",   1, by_individual=True, use_miss=True),
            CaptureSpec("down", 1, by_individual=True, use_miss=True),
            CaptureSpec("up",   1, by_individual=False),
            CaptureSpec("down", 1, by_individual=False),
            CaptureSpec("up",   1, by_individual=False, use_miss=True),
            CaptureSpec("down", 1, by_individual=False, use_miss=True),
        ),
    )

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _bare_req(req) -> RuntimeRequirements:
    """Extract the RuntimeRequirements from a RunConfigs or pass through."""
    return req.req if isinstance(req, RunConfigs) else req


def _validate_filename(path: Path) -> None:
    if path.suffix != ".grg_spmv":
        raise ValueError(f"expected a .grg_spmv file, got {path}")
    if not path.exists():
        raise FileNotFoundError(path)


def _validate_dtype(dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError(f"dtype must be float32 or float64, got {dtype}")
    return dtype


def _validate_backend(backend) -> None:
    if not isinstance(backend, (MklBackendConfig, CusparseBackendConfig)):
        raise TypeError(
            f"backend must be MklBackendConfig or CusparseBackendConfig, "
            f"got {type(backend).__name__}"
        )


def _validate_req(req) -> None:
    if not isinstance(req, (RunConfigs, RuntimeRequirements)):
        raise TypeError(
            f"req must be RunConfigs or RuntimeRequirements, got {type(req).__name__}"
        )


def _validate_stack(stack) -> None:
    if not isinstance(stack, contextlib.ExitStack):
        raise TypeError(f"stack must be contextlib.ExitStack, got {type(stack).__name__}")


def _physical_cores() -> int:
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
        if cores:
            return cores
    except ImportError:
        pass
    return os.cpu_count() or 1


def _auto_threads(n_files: int) -> int:
    import warnings
    cores = _physical_cores()
    t = cores // n_files
    if t < 1:
        warnings.warn(
            f"physical cores ({cores}) < number of files ({n_files}); "
            f"using 1 thread per file",
            RuntimeWarning,
            stacklevel=4,
        )
        return 1
    return t


def _resolve_mkl_threads(cfg: MklBackendConfig, path: Path, n_files: int) -> tuple[int, int]:
    """Return (n_threads_up, n_threads_down) for this file."""
    if isinstance(cfg.n_threads, int):
        t = _auto_threads(n_files) if cfg.n_threads == 0 else cfg.n_threads
        return (t, t)
    if path.stem not in cfg.n_threads:
        raise KeyError(
            f"file stem {path.stem!r} not found in n_threads config; "
            f"available stems: {sorted(cfg.n_threads)}"
        )
    entry = cfg.n_threads[path.stem]
    if "mkl_threads" not in entry:
        raise KeyError(
            f"n_threads config for {path.stem!r} is missing 'mkl_threads' key; "
            f"got: {entry!r}"
        )
    up, down = entry["mkl_threads"]
    return (up, down)


def _available_cuda_devices() -> int:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def _check_device(device_id: int) -> None:
    n = _available_cuda_devices()
    if device_id >= n:
        raise RuntimeError(
            f"CUDA device {device_id} is not available; "
            f"system has {n} device(s) (0–{max(0, n - 1)})"
        )


def _resolve_device(cfg: CusparseBackendConfig, path: Path) -> int:
    if isinstance(cfg.device, int):
        _check_device(cfg.device)
        return cfg.device
    if path.stem not in cfg.device:
        raise KeyError(
            f"file stem {path.stem!r} not found in device config; "
            f"available stems: {sorted(cfg.device)}"
        )
    entry = cfg.device[path.stem]
    if "cuda_device" not in entry:
        raise KeyError(
            f"device config for {path.stem!r} is missing 'cuda_device' key; "
            f"got: {entry!r}"
        )
    device_id = entry["cuda_device"]
    _check_device(device_id)
    return device_id


def _make_mkl_pair(n_threads_up: int, n_threads_down: int, optimize: bool) -> MklPlanPair:
    return MklPlanPair(
        plan_up=MklPlan(store="N", fmt="CSR", n_threads=n_threads_up, optimize=optimize),
        plan_down=MklPlan(store="T", fmt="CSC", n_threads=n_threads_down, optimize=optimize),
    )


def _make_cusparse_standard_pair() -> CusparsePlanPair:
    return CusparsePlanPair(
        plan_up=CusparsePlan(
            store="N", fmt="CSR", op_a="N", op_b="N",
            order_b="ROW", order_c="ROW", algo="DEFAULT", scratch="none",
        ),
        plan_down=CusparsePlan(
            store="T", fmt="CSC", op_a="N", op_b="N",
            order_b="ROW", order_c="ROW", algo="DEFAULT", scratch="none",
        ),
    )


_VALID_INIT_MODES = {"none", "vector", "matrix", "xtx"}


def _validate_capture_spec(spec: CaptureSpec, req: RuntimeRequirements) -> None:
    if spec.init_mode not in _VALID_INIT_MODES:
        raise ValueError(
            f"CaptureSpec.init_mode {spec.init_mode!r} invalid; expected one of {sorted(_VALID_INIT_MODES)}"
        )
    if spec.init_mode != "none" and spec.use_miss:
        raise ValueError(
            f"CaptureSpec direction={spec.direction!r}: init_mode and use_miss are mutually exclusive"
        )
    if spec.use_miss:
        if spec.direction == "up" and not req.need_up_miss_output:
            raise ValueError("CaptureSpec use_miss=True for 'up' but need_up_miss_output is False")
        if spec.direction == "down" and not req.need_down_miss_input:
            raise ValueError("CaptureSpec use_miss=True for 'down' but need_down_miss_input is False")
    if spec.init_mode == "vector" and not req.need_init_vector:
        raise ValueError("CaptureSpec init_mode='vector' but need_init_vector is False")
    if spec.init_mode == "matrix" and not req.need_init_matrix:
        raise ValueError("CaptureSpec init_mode='matrix' but need_init_matrix is False")
    if spec.init_mode == "xtx" and not req.need_init_xtx:
        raise ValueError("CaptureSpec init_mode='xtx' but need_init_xtx is False")


def _capture_grg(grg, capture_stream, req, cfg, stack) -> CapturedBoundGRG:
    """Capture CUDA graphs for all ops in req.capture_ops and return a CapturedBoundGRG.

    All prepare_matmul_cuda contexts are entered into the caller's stack so they
    live as long as the runtime.

    A single shared src buffer and a single shared init/miss buffer are allocated
    (each sized to the maximum across all ops) and aliased per op. This is safe
    because ops of the same GRG never run concurrently.
    """
    import torch

    if isinstance(req, RunConfigs) and req.capture_ops:
        ops_to_capture = req.capture_ops
    else:
        bare = _bare_req(req)
        ops_to_capture = (
            CaptureSpec("up",   bare.max_k_up),
            CaptureSpec("down", bare.max_k_down),
        )

    bare_req = _bare_req(req)

    # ---- Phase 1: enter all contexts and record buffer sizes ----
    prepared_ops: dict = {}
    input_numels: list[int] = []
    init_numels:  list[int] = []
    miss_numels:  list[int] = []

    for spec in ops_to_capture:
        _validate_capture_spec(spec, bare_req)
        key = (spec.direction, spec.by_individual, spec.init_mode, spec.use_miss)
        if key in prepared_ops:
            raise ValueError(f"Duplicate CaptureSpec key {key!r} in capture_ops")
        op = stack.enter_context(
            grg.prepare_matmul_cuda(
                direction=spec.direction,
                k=spec.k,
                by_individual=spec.by_individual,
                init_mode=spec.init_mode,
                use_miss=spec.use_miss,
            )
        )
        prepared_ops[key] = op
        input_numels.append(op.input.numel())
        init_numels.append(
            op.init_vector.numel() if spec.init_mode == "vector" else
            op.init_matrix.numel() if spec.init_mode == "matrix" else 0
        )
        miss_numels.append(
            op.miss_input.numel()  if (spec.use_miss and spec.direction == "down") else
            op.miss_output.numel() if (spec.use_miss and spec.direction == "up")   else 0
        )

    # ---- Phase 2: allocate shared staging buffers ----
    any_op = next(iter(prepared_ops.values()))
    dtype, device = any_op.input.dtype, any_op.input.device

    shared_src  = torch.zeros(max(input_numels), dtype=dtype, device=device)
    shared_init = torch.zeros(max(init_numels),  dtype=dtype, device=device) if max(init_numels) > 0 else None
    shared_miss = torch.zeros(max(miss_numels),  dtype=dtype, device=device) if max(miss_numels) > 0 else None

    # ---- Phase 3: capture graphs with aliased buffer views ----
    graphs:       dict = {}
    src_tensors:  dict = {}
    init_tensors: dict = {}
    miss_tensors: dict = {}

    for i, spec in enumerate(ops_to_capture):
        key = (spec.direction, spec.by_individual, spec.init_mode, spec.use_miss)
        op  = prepared_ops[key]
        k, input_cols = op.input.shape  # op.input is (k, input_cols), column-major

        # Alias a column-major view of shared_src that matches op.input's layout.
        # op.input is _io0_torch[:input_cols, :k].T, so strides are (1, k).
        src = torch.as_strided(shared_src, size=(k, input_cols), stride=(1, k))
        graph = torch.cuda.CUDAGraph()

        if spec.init_mode == "vector":
            init = shared_init[:k]  # shape (k,), stride (1,)
            with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="thread_local"):
                op.input.copy_(src)
                op.init_vector.copy_(init)
                op()
            init_tensors[key] = init

        elif spec.init_mode == "matrix":
            num_nodes = op.init_matrix.shape[1]  # op.init_matrix is (k, num_nodes), column-major
            init = torch.as_strided(shared_init, size=(k, num_nodes), stride=(1, k))
            with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="thread_local"):
                op.input.copy_(src)
                op.init_matrix.copy_(init)
                op()
            init_tensors[key] = init

        elif spec.use_miss and spec.direction == "down":
            num_mut = op.miss_input.shape[1]  # op.miss_input is (k, num_mutations), column-major
            miss = torch.as_strided(shared_miss, size=(k, num_mut), stride=(1, k))
            with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="thread_local"):
                op.input.copy_(src)
                op.miss_input.copy_(miss)
                op()
            miss_tensors[key] = miss

        else:  # "none", "xtx", or UP+use_miss (miss_output written by kernel)
            with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="thread_local"):
                op.input.copy_(src)
                op()

        src_tensors[key] = src
        graphs[key] = graph

    return CapturedBoundGRG(
        grg, prepared_ops, graphs, src_tensors, init_tensors, miss_tensors,
        capture_stream, native=cfg.native,
    )


# ---------------------------------------------------------------------------
# Internal load implementations
# ---------------------------------------------------------------------------

def _load_mkl_multi(paths, cfg, req, stack, dtype):
    result = []
    for path in paths:
        n_up, n_down = _resolve_mkl_threads(cfg, path, len(paths))
        pair = _make_mkl_pair(n_up, n_down, cfg.optimize)
        layout = plan_mkl_layout(artifacts=[path], pair=pair, dtype=dtype, requirements=_bare_req(req))
        runtime = stack.enter_context(MklRuntime(layout))
        result.extend(runtime.grgs)
    return result


def _load_cusparse_multi(paths, cfg, req, stack, dtype):
    device_groups: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for i, path in enumerate(paths):
        device_groups[_resolve_device(cfg, path)].append((i, path))

    grg_by_index: dict[int, object] = {}
    lock = threading.Lock()
    pair = _make_cusparse_standard_pair()

    def _load_one_device(device_id, indexed_paths):
        import torch
        # Pin this thread to its device. torch.cuda.Stream(device=...) sets the
        # stream's device but NOT the thread-local current device, which stays at
        # the default (0). torch.cuda.graph.__enter__ calls torch.cuda.synchronize()
        # with no argument, so without this every worker would synchronize device 0
        # — and a stray sync of device 0 while another thread is mid-capture there
        # raises a CUDA error. Pinning the device makes the sync hit the right one.
        with torch.cuda.device(device_id):
            capture_stream = torch.cuda.Stream(device=device_id) if cfg.capture else 0

            group_paths = [p for _, p in indexed_paths]
            layout = plan_cusparse_layout(
                artifacts=group_paths,
                pair=pair,
                dtype=dtype,
                requirements=_bare_req(req),
                vram_budget_bytes=0,
                ring_buffer_size=0,
                allow_residency=cfg.allow_residency,
                device=device_id,
                stream=capture_stream,
            )
            # Heavy I/O: load from disk to GPU — done outside the global stack lock.
            device_stack = contextlib.ExitStack()
            try:
                runtime = device_stack.enter_context(CusparseRuntime(layout))
            except Exception:
                device_stack.close()
                raise
            # Transfer cleanup ownership to the caller's ExitStack (fast, needs lock).
            with lock:
                stack.enter_context(device_stack.pop_all())

            for (original_idx, _), grg in zip(indexed_paths, runtime.grgs):
                if cfg.capture:
                    grg = _capture_grg(grg, capture_stream, req, cfg, stack)
                grg_by_index[original_idx] = grg

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_load_one_device, dev, grps)
            for dev, grps in device_groups.items()
        ]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    return [grg_by_index[i] for i in range(len(paths))]


# ---------------------------------------------------------------------------
# Public load functions
# ---------------------------------------------------------------------------

def load_grg_spmv_single(filename, backend, req, stack, dtype=np.float64):
    """Load a single .grg_spmv file and return a BoundGRG (or CapturedBoundGRG).

    The runtime is entered into *stack*; callers must keep the stack alive for
    as long as the returned GRG is in use.
    """
    return load_grg_spmv_multi([filename], backend, req, stack, dtype)[0]


def load_grg_spmv_multi(filenames, backend, req, stack, dtype=np.float64):
    """Load one or more .grg_spmv files and return a list of BoundGRGs.

    req may be a RunConfigs or a plain RuntimeRequirements.

    For the MKL backend each file gets its own runtime (entered into *stack*
    sequentially). For the cuSPARSE backend, files are grouped by device and
    each device group is loaded in parallel. When cfg.capture=True, each GRG
    is wrapped in a CapturedBoundGRG with pre-captured CUDA graphs.
    """
    paths = [Path(f) for f in filenames]
    if not paths:
        raise ValueError("filenames must be non-empty")
    for p in paths:
        _validate_filename(p)
    _validate_backend(backend)
    _validate_req(req)
    _validate_stack(stack)
    dtype = _validate_dtype(dtype)

    if isinstance(backend, MklBackendConfig):
        return _load_mkl_multi(paths, backend, req, stack, dtype)
    else:
        return _load_cusparse_multi(paths, backend, req, stack, dtype)


# ---------------------------------------------------------------------------

__all__ = [
    "CapturedBoundGRG",
    "CaptureSpec",
    "RunConfigs",
    "MklBackendConfig",
    "CusparseBackendConfig",
    "make_backend_mkl",
    "make_backend_cusparse",
    "make_runconfig_matmul",
    "make_runconfig_pca",
    "make_runconfig_bolt",
    "load_grg_spmv_single",
    "load_grg_spmv_multi",
]
