"""Benchmark PCA eigensolvers on `.grg_spmv` artifacts using cuSPARSE."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any
import warnings

import cupy as cp
import numpy as np
from cupyx.scipy.sparse.linalg import LinearOperator as CupyxLinearOperator
from cupyx.scipy.sparse.linalg import eigsh as cupyx_eigsh
from cupyx.scipy.sparse.linalg import lobpcg as cupyx_lobpcg
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import eigsh as scipy_eigsh
from scipy.sparse.linalg import lobpcg as scipy_lobpcg

from pygrgl_spmv import RuntimeRequirements
from pygrgl_spmv.backends.cusparse import CusparseRuntime, plan_cusparse_layout
from pygrgl_spmv.grg.artifact import ArtifactScan, scan_grg_spmv
from scripts.bench.cusparse import DEFAULT_CUSPARSE_PLAN_NAME, parse_cusparse_plan


_DTYPE = np.dtype(np.float64)
_METHOD_ORDER = ("scipy-eigsh", "scipy-lobpcg", "cupyx-eigsh", "cupyx-lobpcg", "randomized-rr")

_ROW_WEIGHTED_SUM = cp.RawKernel(
    r"""
extern "C" __global__
void row_weighted_sum(
    const double* x,
    const double* weights,
    double* out,
    const long long ncols,
    const long long row_stride,
    const long long col_stride)
{
    extern __shared__ double scratch[];
    const int row = blockIdx.x;
    double value = 0.0;
    for (long long col = threadIdx.x; col < ncols; col += blockDim.x) {
        value += x[row * row_stride + col * col_stride] * weights[col];
    }
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[row] = scratch[0];
    }
}
""",
    "row_weighted_sum",
)


@dataclass
class Bucket:
    calls: int = 0
    total_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        return 0.0 if self.calls == 0 else self.total_ms / float(self.calls)

    def add(self, elapsed_ms: float) -> None:
        self.calls += 1
        self.total_ms += float(elapsed_ms)


@dataclass
class RunStats:
    setup: dict[str, float] = field(default_factory=dict)
    hot: dict[int, Bucket] = field(default_factory=dict)
    h2d: dict[int, Bucket] = field(default_factory=dict)
    d2h: dict[int, Bucket] = field(default_factory=dict)
    prepared_setup_by_q: dict[int, float] = field(default_factory=dict)
    graph_capture_by_q: dict[int, float] = field(default_factory=dict)


@dataclass
class PreparedXTXState:
    q: int
    managers: tuple[object, ...]
    ops: dict[str, object]
    cupy_views: dict[str, Any]
    work_buffers: dict[str, Any]
    graph: Any | None = None


@dataclass
class MethodResult:
    name: str
    elapsed_seconds: float
    params: dict[str, Any]
    warnings: list[str]
    skip_reason: str | None = None
    error: str | None = None
    eigenvalues: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    quality_failed: bool = False


def _stats_add_setup(stats: RunStats, category: str, elapsed_seconds: float) -> None:
    stats.setup[str(category)] = stats.setup.get(str(category), 0.0) + float(elapsed_seconds)


def _stats_add_prepared_setup(stats: RunStats, q: int, elapsed_seconds: float) -> None:
    key = int(q)
    stats.prepared_setup_by_q[key] = stats.prepared_setup_by_q.get(key, 0.0) + float(elapsed_seconds)


def _stats_bucket(target: dict[int, Bucket], q: int) -> Bucket:
    key = int(q)
    bucket = target.get(key)
    if bucket is None:
        bucket = Bucket()
        target[key] = bucket
    return bucket


def _snapshot_buckets(source: dict[int, Bucket]) -> dict[int, Bucket]:
    return {int(key): Bucket(value.calls, value.total_ms) for key, value in source.items()}


def _reset_method_stats(stats: RunStats) -> None:
    stats.hot.clear()
    stats.h2d.clear()
    stats.d2h.clear()


def _row_weighted_sum(x: cp.ndarray, weights: cp.ndarray, out: cp.ndarray) -> None:
    threads = 256
    itemsize = int(x.dtype.itemsize)
    _ROW_WEIGHTED_SUM(
        (int(x.shape[0]),),
        (threads,),
        (
            x,
            weights,
            out,
            np.int64(x.shape[1]),
            np.int64(x.strides[0] // itemsize),
            np.int64(x.strides[1] // itemsize),
        ),
        shared_mem=threads * itemsize,
    )


def _cuda_event_ms(stream, fn: Callable[[], Any]) -> tuple[Any, float]:
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    with stream:
        start.record(stream)
        result = fn()
        end.record(stream)
    end.synchronize()
    return result, float(cp.cuda.get_elapsed_time(start, end))


def _synchronized_wall_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    cp.cuda.runtime.deviceSynchronize()
    start = perf_counter()
    result = fn()
    cp.cuda.runtime.deviceSynchronize()
    return result, (perf_counter() - start) * 1000.0


class PreparedXTX:
    def __init__(self, grg, *, runtime: CusparseRuntime, stats: RunStats, graph: bool, stream=None) -> None:
        self.grg = grg
        self.runtime = runtime
        self.stream = stream if stream is not None else runtime.stream
        self.stats = stats
        self.use_graph = bool(graph)
        self.has_missing = bool(grg.has_missing_data)
        self.shape = (int(grg.num_mutations), int(grg.num_mutations))
        self.freqs: cp.ndarray | None = None
        self.center: cp.ndarray | None = None
        self.sigma: cp.ndarray | None = None
        self.inv_sigma: cp.ndarray | None = None
        self.center_over_sigma: cp.ndarray | None = None
        self._state: PreparedXTXState | None = None

    def close(self) -> None:
        self._close_state()

    def close_active(self) -> None:
        self._close_state()

    def initialize(self) -> None:
        if self.freqs is not None:
            return
        setup_start = perf_counter()
        prep_start = perf_counter()
        manager = self.grg.prepare_matmul_cuda(direction="up", k=1, by_individual=True, use_miss=self.has_missing)
        up = manager.__enter__()
        _stats_add_prepared_setup(self.stats, 1, perf_counter() - prep_start)
        try:
            views = {
                "up_input": cp.from_dlpack(up.input),
                "up_output": cp.from_dlpack(up.output),
            }
            if self.has_missing:
                views["up_miss_output"] = cp.from_dlpack(up.miss_output)

            def run() -> None:
                views["up_input"].fill(1.0)
                miss_output = views.get("up_miss_output")
                if miss_output is not None:
                    miss_output.fill(0.0)
                up()

            _cuda_event_ms(self.stream, run)
            counts = views["up_output"][0]
            miss_counts = cp.zeros_like(counts) if views.get("up_miss_output") is None else views["up_miss_output"][0]
            denominator = float(self.grg.num_samples) - miss_counts
            valid = denominator != 0.0
            safe_denominator = cp.where(valid, denominator, 1.0)
            freqs = cp.where(valid, counts / safe_denominator, 0.0)
            center = float(self.grg.ploidy) * freqs
            sigma = cp.sqrt(float(self.grg.ploidy) * freqs * (1.0 - freqs))
            self.freqs = freqs
            self.center = center
            self.sigma = cp.where(sigma == 0.0, 1.0, sigma)
            self.inv_sigma = 1.0 / self.sigma
            self.center_over_sigma = center * self.inv_sigma
            cp.cuda.runtime.deviceSynchronize()
            _stats_add_setup(self.stats, "allele_frequency_setup", perf_counter() - setup_start)
        finally:
            manager.__exit__(None, None, None)

    def copy_host_to_device(self, values: np.ndarray) -> cp.ndarray:
        arr, elapsed_ms = _cuda_event_ms(self.stream, lambda: cp.asarray(values, dtype=_DTYPE))
        q = 1 if arr.ndim == 1 else int(arr.shape[1])
        _stats_bucket(self.stats.h2d, q).add(elapsed_ms)
        return arr

    def copy_device_to_host(self, values: cp.ndarray) -> np.ndarray:
        q = 1 if values.ndim == 1 else int(values.shape[1])
        host, elapsed_ms = _synchronized_wall_ms(lambda: cp.asnumpy(values))
        _stats_bucket(self.stats.d2h, q).add(elapsed_ms)
        return np.asarray(host, dtype=_DTYPE)

    def apply_device(self, values, *, record: bool = True):
        if self.freqs is None:
            raise RuntimeError("PreparedXTX.initialize() must be called before use")
        arr = cp.asarray(values, dtype=_DTYPE)
        was_vector = arr.ndim == 1
        if was_vector:
            arr = arr.reshape((self.shape[1], 1))
        if arr.ndim != 2 or int(arr.shape[0]) != self.shape[1]:
            raise ValueError(f"X.T @ X operand has shape {tuple(arr.shape)}, expected ({self.shape[1]}, q)")
        # CuPy eigensolver/randomized-RR producers may run on the ambient stream before self.stream consumes.
        producer_stream = cp.cuda.get_current_stream()
        if int(producer_stream.ptr) != int(self.stream.ptr):
            input_ready = cp.cuda.Event()
            input_ready.record(producer_stream)
            self.stream.wait_event(input_ready)
        state = self._ensure_state(int(arr.shape[1]))
        if record and self.use_graph:
            result = self._apply_graph(state, arr)
        elif record:
            result = self._apply_direct(state, arr)
        else:
            result = self._apply_untimed(state, arr)
        return result[:, 0] if was_vector else result

    def _close_state(self) -> None:
        state = self._state
        if state is None:
            return
        for manager in reversed(state.managers):
            manager.__exit__(None, None, None)
        self._state = None

    def _ensure_state(self, q: int) -> PreparedXTXState:
        width = int(q)
        state = self._state
        if state is not None and state.q == width:
            return state
        self._close_state()
        setup_start = perf_counter()
        managers: list[object] = []
        try:
            down_manager = self.grg.prepare_matmul_cuda(direction="down", k=width, by_individual=True, use_miss=self.has_missing)
            down = down_manager.__enter__()
            managers.append(down_manager)
            up_manager = self.grg.prepare_matmul_cuda(direction="up", k=width, by_individual=True, use_miss=self.has_missing)
            up = up_manager.__enter__()
            managers.append(up_manager)
            cupy_views = {
                "down_input": cp.from_dlpack(down.input),
                "down_output": cp.from_dlpack(down.output),
                "up_input": cp.from_dlpack(up.input),
                "up_output": cp.from_dlpack(up.output),
            }
            if self.has_missing:
                cupy_views["down_miss_input"] = cp.from_dlpack(down.miss_input)
                cupy_views["up_miss_output"] = cp.from_dlpack(up.miss_output)
            work_buffers = {
                "output_mq": cp.empty((self.shape[0], width), dtype=_DTYPE),
                "constants_down": cp.empty((width,), dtype=_DTYPE),
                "constants_up": cp.empty((width,), dtype=_DTYPE),
                "temp_qm": cp.empty((width, self.shape[0]), dtype=_DTYPE),
            }
            if self.use_graph:
                work_buffers["input_mq"] = cp.empty((self.shape[0], width), dtype=_DTYPE)
            state = PreparedXTXState(
                q=width,
                managers=tuple(managers),
                ops={"down": down, "up": up},
                cupy_views=cupy_views,
                work_buffers=work_buffers,
            )
            self._state = state
            _stats_add_prepared_setup(self.stats, width, perf_counter() - setup_start)
            return state
        except Exception:
            for manager in reversed(managers):
                manager.__exit__(None, None, None)
            raise

    def _body(self, state: PreparedXTXState, source_mq: cp.ndarray) -> None:
        assert self.freqs is not None
        assert self.center is not None
        assert self.inv_sigma is not None
        assert self.center_over_sigma is not None
        views = state.cupy_views
        work = state.work_buffers
        cp.multiply(source_mq.T, self.inv_sigma[None, :], out=views["down_input"])
        if "down_miss_input" in views:
            cp.multiply(views["down_input"], self.freqs[None, :], out=views["down_miss_input"])
        _row_weighted_sum(views["down_input"], self.center, work["constants_down"])
        state.ops["down"]()

        cp.subtract(views["down_output"], work["constants_down"][:, None], out=views["up_input"])
        cp.sum(views["up_input"], axis=1, out=work["constants_up"])
        if "up_miss_output" in views:
            views["up_miss_output"].fill(0.0)
        state.ops["up"]()

        if "up_miss_output" in views:
            cp.multiply(views["up_miss_output"], self.freqs[None, :], out=work["temp_qm"])
            cp.add(views["up_output"], work["temp_qm"], out=views["up_output"])
        cp.multiply(views["up_output"], self.inv_sigma[None, :], out=views["up_output"])
        cp.multiply(work["constants_up"][:, None], self.center_over_sigma[None, :], out=work["temp_qm"])
        cp.subtract(views["up_output"], work["temp_qm"], out=views["up_output"])
        cp.copyto(work["output_mq"], views["up_output"].T)

    def _capture(self, state: PreparedXTXState) -> None:
        if state.graph is not None:
            return
        if self.has_missing:
            raise RuntimeError("--graph is supported only when has_missing_data=False")
        capture_start = perf_counter()
        work = state.work_buffers
        with self.stream:
            work["input_mq"].fill(0.0)
            self._body(state, work["input_mq"])
        self.stream.synchronize()
        cp.cuda.runtime.deviceSynchronize()
        with self.stream:
            self.stream.begin_capture()
            self._body(state, work["input_mq"])
            state.graph = self.stream.end_capture()
        self.stream.synchronize()
        elapsed = perf_counter() - capture_start
        self.stats.graph_capture_by_q[state.q] = self.stats.graph_capture_by_q.get(state.q, 0.0) + elapsed

    def _apply_graph(self, state: PreparedXTXState, arr: cp.ndarray) -> cp.ndarray:
        self._capture(state)
        assert state.graph is not None
        work = state.work_buffers

        def run():
            cp.copyto(work["input_mq"], arr)
            state.graph.launch(stream=self.stream)
            return work["output_mq"].copy()

        result, elapsed_ms = _cuda_event_ms(self.stream, run)
        _stats_bucket(self.stats.hot, state.q).add(elapsed_ms)
        return result

    def _apply_direct(self, state: PreparedXTXState, arr: cp.ndarray) -> cp.ndarray:
        work = state.work_buffers

        def run():
            self._body(state, arr)
            return work["output_mq"].copy()

        result, elapsed_ms = _cuda_event_ms(self.stream, run)
        _stats_bucket(self.stats.hot, state.q).add(elapsed_ms)
        return result

    def _apply_untimed(self, state: PreparedXTXState, arr: cp.ndarray) -> cp.ndarray:
        work = state.work_buffers
        if self.use_graph:
            self._capture(state)
            assert state.graph is not None
            with self.stream:
                cp.copyto(work["input_mq"], arr)
                state.graph.launch(stream=self.stream)
                result = work["output_mq"].copy()
            self.stream.synchronize()
            return result
        with self.stream:
            self._body(state, arr)
            result = work["output_mq"].copy()
        self.stream.synchronize()
        return result


class CupyXTXOperator(CupyxLinearOperator):
    def __init__(self, hot: PreparedXTX) -> None:
        self.hot = hot
        super().__init__(dtype=cp.dtype(_DTYPE), shape=hot.shape)

    def _matmat(self, values):
        return self.hot.apply_device(values)

    def _matvec(self, values):
        return self.hot.apply_device(values)


class HostXTXOperator(ScipyLinearOperator):
    def __init__(self, hot: PreparedXTX) -> None:
        self.hot = hot
        super().__init__(dtype=_DTYPE, shape=hot.shape)

    def _matmat(self, values):
        device_values = self.hot.copy_host_to_device(np.asarray(values, dtype=_DTYPE))
        device_result = self.hot.apply_device(device_values)
        return self.hot.copy_device_to_host(device_result)

    def _matvec(self, values):
        device_values = self.hot.copy_host_to_device(np.asarray(values, dtype=_DTYPE))
        device_result = self.hot.apply_device(device_values)
        return self.hot.copy_device_to_host(device_result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PCA eigensolvers on a .grg_spmv artifact using cuSPARSE.")
    parser.add_argument("--artifact", required=True, help="Path to a .grg_spmv artifact")
    parser.add_argument("--pcs", required=True, type=int, help="Number of principal components")
    parser.add_argument("--tol", type=float, default=1e-6, help="Eigensolver tolerance")
    parser.add_argument("--eigsh-maxiter", type=int, default=200, help="eigsh max iterations")
    parser.add_argument("--eigsh-ncv", type=int, default=None, help="Optional eigsh Lanczos subspace size")
    parser.add_argument("--lobpcg-maxiter", type=int, default=200, help="LOBPCG max iterations")
    parser.add_argument("--rr-oversample", type=int, default=40, help="Randomized Rayleigh-Ritz oversampling width")
    parser.add_argument("--rr-power-iters", type=int, default=10, help="Randomized Rayleigh-Ritz power iterations")
    parser.add_argument("--seed", type=int, default=2026, help="Initial-vector RNG seed")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ordinal")
    parser.add_argument("--stream", type=int, default=0, help="CUDA stream handle")
    parser.add_argument("--ring-buffer-size", type=int, default=0, help="cuSPARSE streamed ring-buffer size")
    parser.add_argument("--vram-budget-bytes", type=int, default=None, help="Owned device memory budget")
    parser.add_argument("--graph", action="store_true", help="Capture hot X.T @ X calls in a CUDA graph")
    args = parser.parse_args()
    if int(args.pcs) < 1:
        parser.error(f"--pcs must be >= 1, got {args.pcs}")
    if float(args.tol) < 0.0:
        parser.error(f"--tol must be non-negative, got {args.tol}")
    if args.eigsh_maxiter is not None and int(args.eigsh_maxiter) < 1:
        parser.error(f"--eigsh-maxiter must be >= 1, got {args.eigsh_maxiter}")
    if args.eigsh_ncv is not None and int(args.eigsh_ncv) < 2:
        parser.error(f"--eigsh-ncv must be >= 2, got {args.eigsh_ncv}")
    if int(args.lobpcg_maxiter) < 1:
        parser.error(f"--lobpcg-maxiter must be >= 1, got {args.lobpcg_maxiter}")
    if int(args.rr_oversample) < 0:
        parser.error(f"--rr-oversample must be >= 0, got {args.rr_oversample}")
    if int(args.rr_power_iters) < 0:
        parser.error(f"--rr-power-iters must be >= 0, got {args.rr_power_iters}")
    if int(args.ring_buffer_size) < 0:
        parser.error(f"--ring-buffer-size must be >= 0, got {args.ring_buffer_size}")
    if args.vram_budget_bytes is not None and int(args.vram_budget_bytes) < 1:
        parser.error(f"--vram-budget-bytes must be >= 1, got {args.vram_budget_bytes}")
    if args.graph and int(args.stream) != 0:
        parser.error("--graph requires --stream 0 because the benchmark owns the capture stream")
    return args


def _validate_artifact(path: Path, *, pcs: int, rr_oversample: int, graph: bool) -> ArtifactScan:
    if path.suffix != ".grg_spmv":
        raise ValueError(f"expected a .grg_spmv artifact, got {path}")
    if not path.exists():
        raise FileNotFoundError(path)
    scan = scan_grg_spmv(path)
    if int(pcs) >= int(scan.num_mutations):
        raise ValueError(f"--pcs must be < num_mutations; got pcs={pcs}, num_mutations={scan.num_mutations}")
    if int(pcs) + int(rr_oversample) >= int(scan.num_mutations):
        raise ValueError(
            "--pcs + --rr-oversample must be < num_mutations; "
            f"got pcs={pcs}, rr_oversample={rr_oversample}, num_mutations={scan.num_mutations}"
        )
    if graph and bool(scan.has_missing_data):
        raise ValueError("--graph is supported only for artifacts with has_missing_data=False")
    return scan


def _requirements(scan: ArtifactScan, *, max_dense_width: int) -> RuntimeRequirements:
    return RuntimeRequirements(
        max_k_up=int(max_dense_width),
        max_k_down=int(max_dense_width),
        need_down_miss_input=bool(scan.has_missing_data),
        need_up_miss_output=bool(scan.has_missing_data),
        need_init_vector=False,
        need_init_matrix=False,
        need_init_xtx=False,
    )


def _free_memory_bytes(device: int) -> int:
    with cp.cuda.Device(int(device)):
        free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
    return int(free_bytes)


def _plan_layout(
    *,
    artifact: Path,
    scan: ArtifactScan,
    max_dense_width: int,
    device: int,
    stream,
    ring_buffer_size: int,
    vram_budget_bytes: int | None,
    stats: RunStats,
):
    pair = parse_cusparse_plan(DEFAULT_CUSPARSE_PLAN_NAME)
    reqs = _requirements(scan, max_dense_width=max_dense_width)
    if vram_budget_bytes is None:
        start = perf_counter()
        budget = _free_memory_bytes(device)
        _stats_add_setup(stats, "free_memory_probe", perf_counter() - start)
    else:
        budget = int(vram_budget_bytes)

    start = perf_counter()
    layout = plan_cusparse_layout(
        artifacts=[artifact],
        pair=pair,
        dtype=_DTYPE,
        requirements=reqs,
        vram_budget_bytes=int(budget),
        ring_buffer_size=int(ring_buffer_size),
        allow_residency=True,
        device=int(device),
        stream=stream,
    )
    _stats_add_setup(stats, "layout_planning", perf_counter() - start)
    return layout, int(budget)


def _reported_eigsh_ncv(method: str, n: int, k: int, requested: int | None) -> int:
    n = int(n)
    k = int(k)
    if method == "cupyx-eigsh":
        if requested is None:
            return min(max(2 * k, k + 32), n - 1)
        return min(max(int(requested), k + 2), n - 1)
    if requested is None:
        return min(max(2 * k + 1, 20), n)
    return min(int(requested), n)


def _method_params(name: str, ctx: SimpleNamespace) -> dict[str, Any]:
    if name.endswith("eigsh"):
        n = int(ctx.cupy_operator.shape[0])
        requested_ncv = None if ctx.eigsh_ncv is None else int(ctx.eigsh_ncv)
        return {
            "k": int(ctx.pcs),
            "which": "LA",
            "tol": float(ctx.tol),
            "maxiter": None if ctx.eigsh_maxiter is None else int(ctx.eigsh_maxiter),
            "ncv": requested_ncv,
            "effective_ncv": _reported_eigsh_ncv(name, n, int(ctx.pcs), requested_ncv),
            "seed": int(ctx.seed),
        }
    if name.endswith("lobpcg"):
        return {"largest": True, "tol": float(ctx.tol), "maxiter": int(ctx.lobpcg_maxiter), "seed": int(ctx.seed)}
    return {
        "width": int(ctx.pcs + ctx.rr_oversample),
        "oversample": int(ctx.rr_oversample),
        "power_iters": int(ctx.rr_power_iters),
        "seed": int(ctx.seed),
    }


def _run_scipy_eigsh(ctx: SimpleNamespace):
    rng = np.random.default_rng(int(ctx.seed))
    v0 = rng.standard_normal((ctx.host_operator.shape[0],), dtype=np.float64)
    return scipy_eigsh(
        ctx.host_operator,
        k=int(ctx.pcs),
        which="LA",
        tol=float(ctx.tol),
        ncv=ctx.eigsh_ncv,
        maxiter=ctx.eigsh_maxiter,
        v0=v0,
    )


def _run_scipy_lobpcg(ctx: SimpleNamespace):
    rng = np.random.default_rng(int(ctx.seed))
    x0 = rng.standard_normal((ctx.host_operator.shape[0], int(ctx.pcs)), dtype=np.float64)
    return scipy_lobpcg(
        ctx.host_operator,
        x0,
        largest=True,
        tol=float(ctx.tol),
        maxiter=int(ctx.lobpcg_maxiter),
    )


def _run_cupyx_eigsh(ctx: SimpleNamespace):
    rng = cp.random.default_rng(int(ctx.seed))
    v0 = rng.standard_normal((ctx.cupy_operator.shape[0],), dtype=cp.float64)
    return cupyx_eigsh(
        ctx.cupy_operator,
        k=int(ctx.pcs),
        which="LA",
        tol=float(ctx.tol),
        ncv=ctx.eigsh_ncv,
        maxiter=ctx.eigsh_maxiter,
        v0=v0,
    )


def _run_cupyx_lobpcg(ctx: SimpleNamespace):
    rng = cp.random.default_rng(int(ctx.seed))
    x0 = rng.standard_normal((ctx.cupy_operator.shape[0], int(ctx.pcs)), dtype=cp.float64)
    return cupyx_lobpcg(
        ctx.cupy_operator,
        x0,
        largest=True,
        tol=float(ctx.tol),
        maxiter=int(ctx.lobpcg_maxiter),
    )


def _run_randomized_rr(ctx: SimpleNamespace):
    width = int(ctx.pcs + ctx.rr_oversample)
    rng = cp.random.default_rng(int(ctx.seed))
    omega = rng.standard_normal((ctx.cupy_operator.shape[0], width), dtype=cp.float64)
    y = ctx.hot_operator.apply_device(omega)
    for _ in range(int(ctx.rr_power_iters)):
        q, _ = cp.linalg.qr(y, mode="reduced")
        y = ctx.hot_operator.apply_device(q)
    q, _ = cp.linalg.qr(y, mode="reduced")
    aq = ctx.hot_operator.apply_device(q)
    small = q.T @ aq
    values, vectors_small = cp.linalg.eigh(small)
    order = cp.flip(cp.argsort(values))[: int(ctx.pcs)]
    return values[order], q @ vectors_small[:, order]


_RUNNERS: dict[str, Callable[[SimpleNamespace], tuple[Any, Any]]] = {
    "scipy-eigsh": _run_scipy_eigsh,
    "scipy-lobpcg": _run_scipy_lobpcg,
    "cupyx-eigsh": _run_cupyx_eigsh,
    "cupyx-lobpcg": _run_cupyx_lobpcg,
    "randomized-rr": _run_randomized_rr,
}


def _sort_eigenpairs(eigenvalues, eigenvectors) -> tuple[np.ndarray, np.ndarray]:
    values = cp.asnumpy(eigenvalues) if isinstance(eigenvalues, cp.ndarray) else np.asarray(eigenvalues)
    vectors = cp.asnumpy(eigenvectors) if isinstance(eigenvectors, cp.ndarray) else np.asarray(eigenvectors)
    values = np.asarray(values).real.astype(_DTYPE, copy=False)
    vectors = np.asarray(vectors).real.astype(_DTYPE, copy=False)
    if vectors.ndim == 1:
        vectors = vectors.reshape((vectors.shape[0], 1))
    order = np.flip(np.argsort(values))
    return values[order], vectors[:, order]


def _warning_messages(caught: list[warnings.WarningMessage]) -> list[str]:
    return [f"{item.category.__name__}: {' '.join(str(item.message).split())}" for item in caught]


def _method_timing_metrics(stats: RunStats) -> dict[str, Any]:
    return {
        "hot": _snapshot_buckets(stats.hot),
        "h2d": _snapshot_buckets(stats.h2d),
        "d2h": _snapshot_buckets(stats.d2h),
    }


def _run_method(name: str, ctx: SimpleNamespace, stats: RunStats) -> MethodResult:
    _reset_method_stats(stats)
    params = _method_params(name, ctx)
    pcs = int(ctx.pcs)
    caught_messages: list[str] = []
    start = perf_counter()
    try:
        if name.endswith("lobpcg"):
            n = int(ctx.cupy_operator.shape[0])
            if n < 5 * pcs:
                return MethodResult(
                    name=name,
                    elapsed_seconds=perf_counter() - start,
                    params=params,
                    warnings=[],
                    skip_reason=(
                        f"LOBPCG dense fallback skipped because num_mutations={n} "
                        f"< 5 * pcs={5 * pcs}; dense fallback would apply an n-column identity"
                    ),
                    metrics=_method_timing_metrics(stats),
                )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                eigenvalues, eigenvectors = _RUNNERS[name](ctx)
                cp.cuda.runtime.deviceSynchronize()
            finally:
                caught_messages = _warning_messages(caught)
        elapsed = perf_counter() - start
        values, vectors = _sort_eigenpairs(eigenvalues, eigenvectors)
        if values.size < pcs or vectors.ndim != 2 or vectors.shape[1] < pcs:
            return MethodResult(
                name=name,
                elapsed_seconds=elapsed,
                params=params,
                warnings=caught_messages,
                error=(
                    "solver returned fewer eigenpairs than requested: "
                    f"values={values.size}, vectors_shape={tuple(vectors.shape)}, requested={ctx.pcs}"
                ),
                metrics=_method_timing_metrics(stats),
            )
        selected_values = values[:pcs]
        selected_vectors = vectors[:, :pcs]
        nonfinite_error = None
        if not np.isfinite(selected_values).all():
            nonfinite_error = "solver returned non-finite eigenvalues"
        elif not np.isfinite(selected_vectors).all():
            nonfinite_error = "solver returned non-finite eigenvectors"
        if nonfinite_error is not None:
            return MethodResult(
                name=name,
                elapsed_seconds=elapsed,
                params=params,
                warnings=caught_messages,
                error=nonfinite_error,
                metrics=_method_timing_metrics(stats),
            )
        return MethodResult(
            name=name,
            elapsed_seconds=elapsed,
            params=params,
            warnings=caught_messages,
            eigenvalues=selected_values,
            eigenvectors=selected_vectors,
            metrics=_method_timing_metrics(stats),
        )
    except Exception as exc:
        try:
            cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass
        return MethodResult(
            name=name,
            elapsed_seconds=perf_counter() - start,
            params=params,
            warnings=caught_messages,
            error=f"{type(exc).__name__}: {exc}",
            metrics=_method_timing_metrics(stats),
        )
    finally:
        ctx.hot_operator.close_active()


def _successful(result: MethodResult) -> bool:
    return (
        result.skip_reason is None
        and result.error is None
        and result.eigenvalues is not None
        and result.eigenvectors is not None
    )


def _select_reference(results: list[MethodResult]) -> MethodResult | None:
    for name in ("scipy-eigsh", "cupyx-eigsh"):
        result = next((item for item in results if item.name == name and _successful(item)), None)
        if result is not None:
            return result
    return next((item for item in results if _successful(item)), None)


def _evaluate_quality(results: list[MethodResult], reference: MethodResult | None, ctx: SimpleNamespace) -> None:
    threshold = max(1e-5, 100.0 * float(ctx.tol))
    ref_values = None if reference is None else reference.eigenvalues
    ref_vectors = None if reference is None else reference.eigenvectors
    for result in results:
        if not _successful(result):
            continue
        assert result.eigenvalues is not None and result.eigenvectors is not None
        vectors_gpu = cp.asarray(result.eigenvectors, dtype=_DTYPE)
        values_gpu = cp.asarray(result.eigenvalues, dtype=_DTYPE)
        applied = ctx.hot_operator.apply_device(vectors_gpu, record=False)
        residual = applied - vectors_gpu * values_gpu[None, :]
        residual_norms = cp.linalg.norm(residual, axis=0)
        relative = residual_norms / cp.maximum(cp.abs(values_gpu), 1.0)
        result.metrics["max_residual_norm"] = float(cp.asnumpy(cp.max(residual_norms)))
        result.metrics["max_relative_residual"] = float(cp.asnumpy(cp.max(relative)))
        if ref_values is not None and ref_vectors is not None:
            diff = np.abs(ref_values - result.eigenvalues)
            denom = np.maximum(np.abs(ref_values), 1.0)
            singular_values = np.linalg.svd(ref_vectors.T @ result.eigenvectors, compute_uv=False)
            clipped = np.clip(singular_values, 0.0, 1.0)
            min_singular = float(np.min(clipped))
            max_angle_sin = float(np.sqrt(max(0.0, 1.0 - min_singular * min_singular)))
            result.metrics["max_abs_eigenvalue_diff_vs_reference"] = float(np.max(diff))
            result.metrics["max_relative_eigenvalue_diff_vs_reference"] = float(np.max(diff / denom))
            result.metrics["min_singular_value_vs_reference"] = min_singular
            result.metrics["max_principal_angle_sin"] = max_angle_sin
        result.quality_failed = bool(result.metrics["max_relative_residual"] > threshold)
        if ref_vectors is not None and "max_principal_angle_sin" in result.metrics:
            result.quality_failed = result.quality_failed or bool(result.metrics["max_principal_angle_sin"] > 1e-2)


def _print_buckets(prefix: str, buckets: dict[int, Bucket]) -> None:
    if not buckets:
        print(f"{prefix}none")
        return
    for q in sorted(buckets):
        bucket = buckets[q]
        print(f"{prefix}q={q} calls={bucket.calls} total_ms={bucket.total_ms:.6f} avg_ms={bucket.avg_ms:.6f}")


def _print_setup(stats: RunStats) -> None:
    print("setup_seconds:")
    for key in ("artifact_scan", "free_memory_probe", "layout_planning", "runtime_initialization", "allele_frequency_setup"):
        print(f"  {key}={stats.setup.get(key, 0.0):.6f}")
    print("prepared_setup_by_q:")
    if stats.prepared_setup_by_q:
        for q in sorted(stats.prepared_setup_by_q):
            print(f"  q={q} seconds={stats.prepared_setup_by_q[q]:.6f}")
    else:
        print("  none")
    print("graph_capture_by_q:")
    if stats.graph_capture_by_q:
        for q in sorted(stats.graph_capture_by_q):
            print(f"  q={q} seconds={stats.graph_capture_by_q[q]:.6f}")
    else:
        print("  none")


def _print_layout(layout) -> None:
    print("layout_bytes:")
    for key in sorted(layout.bytes_by_category):
        print(f"  {key}={layout.bytes_by_category[key]}")
    print(f"  total={layout.bytes_total}")
    print(f"  required_full_residency={layout.required_budget_for_full_residency}")


def _print_result(result: MethodResult) -> None:
    if result.skip_reason is not None:
        status = "skipped"
    elif result.error is not None:
        status = "failed"
    elif result.quality_failed:
        status = "quality_failed"
    else:
        status = "ok"
    print(f"\nmethod={result.name}")
    print(f"  status={status}")
    if result.skip_reason is not None:
        print(f"  skip_reason={result.skip_reason}")
    print(f"  elapsed_seconds={result.elapsed_seconds:.6f}")
    print(f"  params={result.params}")
    print("  eigenvalues=" + ("none" if result.eigenvalues is None else np.array2string(result.eigenvalues, precision=10, separator=", ")))
    print("  hot_xtx_stats:")
    _print_buckets("    ", result.metrics.get("hot", {}))
    if result.metrics.get("h2d"):
        print("  h2d_stats:")
        _print_buckets("    ", result.metrics["h2d"])
    if result.metrics.get("d2h"):
        print("  d2h_stats:")
        _print_buckets("    ", result.metrics["d2h"])
    for key in (
        "max_residual_norm",
        "max_relative_residual",
        "max_abs_eigenvalue_diff_vs_reference",
        "max_relative_eigenvalue_diff_vs_reference",
        "min_singular_value_vs_reference",
        "max_principal_angle_sin",
    ):
        if key in result.metrics:
            print(f"  {key}={result.metrics[key]:.10g}")
    if result.warnings:
        print(f"  warnings={len(result.warnings)}")
        for message in result.warnings:
            print(f"    {message}")
    else:
        print("  warnings=none")
    if result.error is not None:
        print(f"  error={result.error}")


def main() -> None:
    args = _parse_args()
    artifact = Path(args.artifact).expanduser()
    stats = RunStats()
    start = perf_counter()
    scan = _validate_artifact(
        artifact,
        pcs=int(args.pcs),
        rr_oversample=int(args.rr_oversample),
        graph=bool(args.graph),
    )
    _stats_add_setup(stats, "artifact_scan", perf_counter() - start)
    max_dense_width = int(args.pcs) + int(args.rr_oversample)
    device_id = int(args.device)

    with cp.cuda.Device(device_id):
        stream: object = int(args.stream)
        capture_stream = None
        if args.graph:
            capture_stream = cp.cuda.Stream(non_blocking=True)
            stream = capture_stream

        layout, budget = _plan_layout(
            artifact=artifact,
            scan=scan,
            max_dense_width=max_dense_width,
            device=device_id,
            stream=stream,
            ring_buffer_size=int(args.ring_buffer_size),
            vram_budget_bytes=args.vram_budget_bytes,
            stats=stats,
        )

        print(f"artifact={scan.path}")
        print(
            "metadata "
            f"pcs={args.pcs} samples={scan.num_samples} individuals={scan.num_individuals} "
            f"mutations={scan.num_mutations} nodes={scan.num_nodes} edges={scan.num_edges} "
            f"ploidy={scan.ploidy} levels={scan.num_levels} has_missing_data={scan.has_missing_data}"
        )
        print(
            "runtime_config "
            f"backend=cusparse plan={DEFAULT_CUSPARSE_PLAN_NAME} device={args.device} "
            f"vram_budget_bytes={budget} ring_buffer_size={args.ring_buffer_size} graph={args.graph} "
            f"max_dense_width={max_dense_width}"
        )

        results: list[MethodResult] = []
        reference: MethodResult | None = None
        runtime_start = perf_counter()
        with CusparseRuntime(layout) as runtime:
            _stats_add_setup(stats, "runtime_initialization", perf_counter() - runtime_start)
            (grg,) = runtime.grgs
            hot = PreparedXTX(grg, runtime=runtime, stats=stats, graph=bool(args.graph), stream=capture_stream)
            try:
                hot.initialize()
                ctx = SimpleNamespace(
                    pcs=int(args.pcs),
                    tol=float(args.tol),
                    eigsh_maxiter=None if args.eigsh_maxiter is None else int(args.eigsh_maxiter),
                    eigsh_ncv=None if args.eigsh_ncv is None else int(args.eigsh_ncv),
                    lobpcg_maxiter=int(args.lobpcg_maxiter),
                    rr_oversample=int(args.rr_oversample),
                    rr_power_iters=int(args.rr_power_iters),
                    seed=int(args.seed),
                    hot_operator=hot,
                    cupy_operator=CupyXTXOperator(hot),
                    host_operator=HostXTXOperator(hot),
                )
                results = [_run_method(name, ctx, stats) for name in _METHOD_ORDER]
                reference = _select_reference(results)
                _evaluate_quality(results, reference, ctx)
            finally:
                hot.close()

        _print_setup(stats)
        _print_layout(layout)
        print(f"reference_method={reference.name if reference is not None else 'none'}")
        for result in results:
            _print_result(result)


if __name__ == "__main__":
    main()
