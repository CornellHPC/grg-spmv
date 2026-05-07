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

    def __init__(self, grg, prepared_ops, graphs, src_tensors, init_tensors, capture_stream, *, native=False):
        self._grg = grg
        self._prepared_ops = prepared_ops  # dict[(direction, by_individual), PreparedOp]
        self._graphs = graphs              # dict[(direction, by_individual), CUDAGraph]
        self._src_tensors = src_tensors    # dict[(direction, by_individual), torch.Tensor]
        self._init_tensors = init_tensors  # dict[(direction, by_individual), torch.Tensor]
        self._capture_stream = capture_stream
        self._native = native

    def __getattr__(self, name):
        if name == "_grg":
            raise AttributeError(name)
        return getattr(self._grg, name)

    def _key(self, direction, by_individual):
        k = (direction, by_individual)
        return k if k in self._src_tensors else direction

    def matmul(self, input, direction, emit_all_nodes=False, by_individual=False, init=None, miss=None):
        """Graph replay. Input/output are numpy arrays (copy mode) or GPU tensors/cupy arrays (native mode)."""
        import torch
        nvtx = torch.cuda.nvtx
        key = self._key(direction, by_individual)
        src = self._src_tensors[key]

        nvtx.range_push(f"matmul_{direction}")

        nvtx.range_push("input")
        if self._native:
            if not isinstance(input, torch.Tensor):
                import cupy as cp
                cp.cuda.get_current_stream().synchronize()
                input = torch.as_tensor(input)
            with torch.cuda.stream(self._capture_stream):
                src.copy_(input)
        else:
            with torch.cuda.stream(self._capture_stream):
                src.copy_(torch.from_numpy(np.ascontiguousarray(input)))
        nvtx.range_pop()

        nvtx.range_push("graph_replay")
        self._graphs[key].replay()
        nvtx.range_pop()

        nvtx.range_push("sync")
        torch.cuda.synchronize(src.device)
        nvtx.range_pop()

        nvtx.range_push("output")
        if self._native:
            import cupy as cp
            result = cp.asarray(self._prepared_ops[key].output).copy()
        else:
            result = self._prepared_ops[key].output.cpu().numpy().copy()
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
    #TODO: support more runtime requirement params?
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
            CaptureSpec("up",   int(k_up),  by_individual=False),
            CaptureSpec("down", int(k_down), by_individual=False),
            CaptureSpec("up",   int(k_up),  by_individual=True),
            CaptureSpec("down", int(k_down), by_individual=True),
        ),
    )


def make_runconfig_pca(**kwargs) -> RunConfigs:
    """Create a RunConfigs for a PCA workload (init_vector enabled, k=20).

    Captures all five graph variants used by PCA:
      up/down at k=20 (main matmul), up/down at k=1 by_individual=True
      (individual eigsh blocks), and up at k=1 by_individual=False
      (allele_counts single pass).
    """
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

        #TODO: check what is actually required
        capture_ops=(
            CaptureSpec("up",   1, by_individual=False),  
            CaptureSpec("down", 1, by_individual=False),  
            CaptureSpec("up",    1, by_individual=True),   
            CaptureSpec("down",  1, by_individual=True),   
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


def _init_mode_from_req(req) -> str:
    r = _bare_req(req)
    if r.need_init_xtx:    return "xtx"
    if r.need_init_matrix: return "matrix"
    if r.need_init_vector: return "vector"
    return "none"


def _capture_grg(grg, capture_stream, req, cfg, stack) -> CapturedBoundGRG:
    """Capture CUDA graphs for all ops in req.capture_ops and return a CapturedBoundGRG.

    All prepare_matmul_cuda contexts are entered into the caller's stack so they
    live as long as the runtime.
    """
    import torch
    init_mode = _init_mode_from_req(req)

    if isinstance(req, RunConfigs) and req.capture_ops:
        ops_to_capture = req.capture_ops
    else:
        bare = _bare_req(req)
        ops_to_capture = (
            CaptureSpec("up",   bare.max_k_up),
            CaptureSpec("down", bare.max_k_down),
        )

    prepared_ops: dict = {}
    graphs: dict = {}
    src_tensors: dict = {}
    init_tensors: dict = {}

    for spec in ops_to_capture:
        key = (spec.direction, spec.by_individual)
        op = stack.enter_context(
            grg.prepare_matmul_cuda(
                direction=spec.direction,
                k=spec.k,
                init_mode=init_mode,
                by_individual=spec.by_individual,
            )
        )
        src = torch.zeros_like(op.input)
        graph = torch.cuda.CUDAGraph()

        if init_mode == "vector":
            init = torch.zeros_like(op.init_vector)
            with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="thread_local"):
                op.input.copy_(src)
                op.init_vector.copy_(init)
                op()
            init_tensors[key] = init
        elif init_mode == "matrix":
            init = torch.zeros_like(op.init_matrix)
            with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="thread_local"):
                op.input.copy_(src)
                op.init_matrix.copy_(init)
                op()
            init_tensors[key] = init
        else:  # "none" or "xtx"
            with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="thread_local"):
                op.input.copy_(src)
                op()

        prepared_ops[key] = op
        graphs[key] = graph
        src_tensors[key] = src

    return CapturedBoundGRG(
        grg, prepared_ops, graphs, src_tensors, init_tensors,
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
        capture_stream = torch.cuda.Stream(device=device_id) if cfg.capture else 0

        group_paths = [p for _, p in indexed_paths]
        layout = plan_cusparse_layout(
            artifacts=group_paths,
            pair=pair,
            dtype=dtype,
            requirements=_bare_req(req),
            vram_budget_bytes=0,
            ring_buffer_size=2,
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
    "load_grg_spmv_single",
    "load_grg_spmv_multi",
]
