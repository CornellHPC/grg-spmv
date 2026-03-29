"""Tests for CusparseBackend swap_mode (deferred async block upload)."""

from __future__ import annotations

import gc
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.backends.cusparse import CusparseBackend, CusparsePlanPair
from pygrgl_spmv.tests.conftest import (
    DATA_DTYPE,
    INDEX_DTYPE,
    make_cusparse_plan,
    tol,
)

cp = pytest.importorskip("cupy")
pytestmark = [pytest.mark.gpu, pytest.mark.cusparse]

_CACHE_DIR = Path(".pytest_cache") / "pygrgl_spmv_npz"

_N_GRGS = 5
_GRG_ENV = "GRG_SPMV_TEST_GRG"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLAN_UP = make_cusparse_plan(store="N", fmt="CSR", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT")
_PLAN_DOWN = make_cusparse_plan(store="T", fmt="CSC", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT")


def _make_swap_backend(**kwargs) -> CusparseBackend:
    return CusparseBackend(
        pair=CusparsePlanPair.from_dicts(_PLAN_UP, _PLAN_DOWN),
        swap_mode=True,
        **kwargs,
    )


def _make_eager_backend(**kwargs) -> CusparseBackend:
    return CusparseBackend(
        pair=CusparsePlanPair.from_dicts(_PLAN_UP, _PLAN_DOWN),
        swap_mode=False,
        **kwargs,
    )


def _make_op(grg_path, backend: CusparseBackend) -> SpmvGRG:
    return SpmvGRG(grg_path, backend, DATA_DTYPE, INDEX_DTYPE, artifact_dir=_CACHE_DIR)


def _reset_gpu_pool() -> int:
    """Sync device, free unreferenced CuPy pool blocks, return live bytes."""
    cp.cuda.runtime.deviceSynchronize()
    cp.get_default_memory_pool().free_all_blocks()
    return cp.get_default_memory_pool().used_bytes()


def _device_mem_used() -> int:
    """Return bytes currently in use on the GPU device (all allocations, not just CuPy pool).

    Uses memGetInfo(free, total); used = total - free. More reliable than pool.used_bytes()
    because it captures VMM-backed and non-pool CUDA allocations, and is not affected by
    CuPy's retained-epoch GC that can make pool bytes decrease during prepare().
    """
    cp.cuda.runtime.deviceSynchronize()
    cp.get_default_memory_pool().free_all_blocks()
    free, total = cp.cuda.runtime.memGetInfo()
    return total - free


def _random_input(op: SpmvGRG, direction: str, k: int, seed: int) -> np.ndarray:
    """Return an (n_rows, k) input matrix in the shape expected by matmul()."""
    rng = np.random.default_rng(seed)
    n = op.num_samples if direction == "up" else op.num_mutations
    return rng.standard_normal((n, k)).astype(DATA_DTYPE)


# ---------------------------------------------------------------------------
# 1. Correctness: swap mode output matches eager mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 4, 8])
def test_swap_mode_up_matches_eager(primary_grg_path, k):
    backend_eager = _make_eager_backend()
    op_eager = _make_op(primary_grg_path, backend_eager)
    X = _random_input(op_eager, "up", k, seed=5001 + k)

    Y_eager = op_eager.matmul(X.T, "up").T

    backend_swap = _make_swap_backend()
    op_swap = _make_op(primary_grg_path, backend_swap)
    backend_swap.prepare()
    Y_swap = op_swap.matmul(X.T, "up").T
    backend_swap.finish()

    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(Y_swap, Y_eager, atol=atol, rtol=rtol)


@pytest.mark.parametrize("k", [1, 4, 8])
def test_swap_mode_down_matches_eager(primary_grg_path, k):
    backend_eager = _make_eager_backend()
    op_eager = _make_op(primary_grg_path, backend_eager)
    X = _random_input(op_eager, "down", k, seed=5101 + k)

    Y_eager = op_eager.matmul(X.T, "down").T

    backend_swap = _make_swap_backend()
    op_swap = _make_op(primary_grg_path, backend_swap)
    backend_swap.prepare()
    Y_swap = op_swap.matmul(X.T, "down").T
    backend_swap.finish()

    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(Y_swap, Y_eager, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# 2. Error: compute before prepare() raises
# ---------------------------------------------------------------------------

def test_run_without_prepare_raises(primary_grg_path):
    backend = _make_swap_backend()
    op = _make_op(primary_grg_path, backend)
    X = _random_input(op, "up", k=4, seed=5200)
    with pytest.raises(RuntimeError, match="prepare()"):
        op.matmul(X.T, "up")


# ---------------------------------------------------------------------------
# 3. Prepare / finish cycle is repeatable
# ---------------------------------------------------------------------------

def test_cycle_repeatable(primary_grg_path):
    backend_eager = _make_eager_backend()
    op_eager = _make_op(primary_grg_path, backend_eager)
    X = _random_input(op_eager, "up", k=4, seed=5300)
    Y_eager = op_eager.matmul(X.T, "up").T
    atol, rtol = tol(DATA_DTYPE)

    backend_swap = _make_swap_backend()
    op_swap = _make_op(primary_grg_path, backend_swap)
    for _ in range(3):
        backend_swap.prepare()
        Y = op_swap.matmul(X.T, "up").T
        backend_swap.finish()
        np.testing.assert_allclose(Y, Y_eager, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# 4. prepare() is idempotent
# ---------------------------------------------------------------------------

def test_double_prepare_is_safe(primary_grg_path):
    backend_eager = _make_eager_backend()
    op_eager = _make_op(primary_grg_path, backend_eager)
    X = _random_input(op_eager, "up", k=4, seed=5400)
    Y_eager = op_eager.matmul(X.T, "up").T
    atol, rtol = tol(DATA_DTYPE)

    backend_swap = _make_swap_backend()
    op_swap = _make_op(primary_grg_path, backend_swap)
    backend_swap.prepare()
    backend_swap.prepare()  # second call must be a no-op
    Y = op_swap.matmul(X.T, "up").T
    backend_swap.finish()

    np.testing.assert_allclose(Y, Y_eager, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# 5. prepare() / finish() are no-ops in eager mode
# ---------------------------------------------------------------------------

def test_prepare_finish_noop_in_eager_mode(primary_grg_path):
    backend = _make_eager_backend()
    op = _make_op(primary_grg_path, backend)
    X = _random_input(op, "up", k=4, seed=5500)

    Y_before = op.matmul(X.T, "up").T
    backend.prepare()   # should not raise or change anything
    backend.finish()    # should not raise or change anything
    Y_after = op.matmul(X.T, "up").T  # backend still works after no-op finish()

    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(Y_after, Y_before, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# 6. k_hint with swap_mode emits a logger warning (no graph capture)
# ---------------------------------------------------------------------------

def test_k_hint_warning_in_swap_mode(primary_grg_path, caplog):
    plan_with_hint = make_cusparse_plan(
        k_hint=4, store="N", fmt="CSR", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"
    )
    backend = CusparseBackend(
        pair=CusparsePlanPair.from_dicts(plan_with_hint, None),
        swap_mode=True,
    )
    with caplog.at_level(logging.WARNING):
        _make_op(primary_grg_path, backend)
    assert any("graph capture is disabled" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 7. GPU memory freed after finish()
# ---------------------------------------------------------------------------

def test_gpu_memory_freed_after_finish(primary_grg_path):
    pool = cp.get_default_memory_pool()
    cp.cuda.runtime.deviceSynchronize()
    pool.free_all_blocks()

    backend = _make_swap_backend()
    _make_op(primary_grg_path, backend)
    cp.cuda.runtime.deviceSynchronize()
    pool.free_all_blocks()
    after_setup = pool.used_bytes()

    backend.prepare()
    cp.cuda.runtime.deviceSynchronize()
    after_prepare = pool.used_bytes()

    backend.finish()
    cp.cuda.runtime.deviceSynchronize()
    pool.free_all_blocks()
    after_finish = pool.used_bytes()

    assert after_prepare > after_setup, "prepare() should allocate GPU memory for blocks"
    assert after_finish <= after_setup, "finish() should free block GPU memory"


# ---------------------------------------------------------------------------
# 8. Two backends: async copy of B overlaps with compute of A
# ---------------------------------------------------------------------------

def test_two_grgs_concurrent(primary_grg_path):
    """prepare() on both backends before either computes — B's H2D copy overlaps A's SpMM."""
    b_ref = _make_eager_backend()
    op_ref = _make_op(primary_grg_path, b_ref)
    X = _random_input(op_ref, "up", k=4, seed=5800)
    Y_expected = op_ref.matmul(X.T, "up").T
    atol, rtol = tol(DATA_DTYPE)

    b1 = _make_swap_backend()
    b2 = _make_swap_backend()
    op1 = _make_op(primary_grg_path, b1)
    op2 = _make_op(primary_grg_path, b2)

    b1.prepare()
    b2.prepare()  # async copy of b2 runs concurrently with b1's future compute

    Y1 = op1.matmul(X.T, "up").T
    b1.finish()

    Y2 = op2.matmul(X.T, "up").T
    b2.finish()

    np.testing.assert_allclose(Y1, Y_expected, atol=atol, rtol=rtol)
    np.testing.assert_allclose(Y2, Y_expected, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# 9. Memory efficiency: swap mode uses O(1) GPU block memory vs O(N)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env_grg_path():
    """GRG file from GRG_SPMV_TEST_GRG env var; skip if not set."""
    path = os.environ.get(_GRG_ENV)
    if not path:
        pytest.skip(f"Set {_GRG_ENV}=<path/to/file.grg> to run GPU memory tests")
    p = Path(path)
    if not p.exists():
        pytest.skip(f"GRG file not found: {path}")
    return str(p)


def test_swap_mode_peak_memory(env_grg_path, capsys):
    """Block index buffers stay on CPU between prepare()/finish() cycles.
    Compares peak GPU memory for:
      (1) 1 eager backend               — minimum baseline
      (2) N eager backends simultaneously — worst-case O(N) usage
      (3) N swap backends sequential    — should remain O(1)
    Requires: GRG_SPMV_TEST_GRG env var pointing to a real (non-trivial) GRG.
    """
    K = 4

    # ── Baseline: no operators (CUDA context + driver overhead) ─────────────
    mem_base = _device_mem_used()

    # ── Case 1: single eager backend ────────────────────────────────────────
    b1 = _make_eager_backend()
    op1 = _make_op(env_grg_path, b1)
    rng = np.random.default_rng(9001)
    X = rng.standard_normal((op1.num_samples, K)).astype(DATA_DTYPE)
    m_after_setup = _device_mem_used() - mem_base
    op1.matmul(X.T, "up")
    m_after_matmul = _device_mem_used() - mem_base
    mem_1_eager = max(m_after_setup, m_after_matmul)
    del b1, op1
    gc.collect()

    # ── Case 2: N eager backends all setup simultaneously ───────────────────
    b_eager = [_make_eager_backend() for _ in range(_N_GRGS)]
    op_eager = [_make_op(env_grg_path, b) for b in b_eager]
    mem_n_eager_setup = _device_mem_used() - mem_base
    for op in op_eager:
        op.matmul(X.T, "up")
    mem_n_eager_post = _device_mem_used() - mem_base
    mem_n_eager = max(mem_n_eager_setup, mem_n_eager_post)
    del b_eager, op_eager
    gc.collect()

    # ── Case 3: N swap backends, sequential prepare → compute → finish ──────
    b_swap = [_make_swap_backend() for _ in range(_N_GRGS)]
    op_swap = [_make_op(env_grg_path, b) for b in b_swap]
    cp.cuda.runtime.deviceSynchronize()
    gc.collect()
    mem_swap_setup = _device_mem_used() - mem_base  # blocks on CPU → should be near 0

    peak_swap_per_step: list[int] = []
    for b, op in zip(b_swap, op_swap):
        b.prepare()
        cp.cuda.runtime.deviceSynchronize()
        m_after_prepare = _device_mem_used() - mem_base
        op.matmul(X.T, "up")
        m_after_matmul = _device_mem_used() - mem_base
        peak_swap_per_step.append(max(m_after_prepare, m_after_matmul))
        b.finish()
    peak_swap = max(peak_swap_per_step)

    del b_swap, op_swap
    gc.collect()

    # ── Case 4: pipelined — prepare(i+1) while computing (i) ────────────────
    # Pattern: prepare(0), prepare(1), compute(0), finish(0),
    #                       prepare(2), compute(1), finish(1), ...
    # Peak = 2 backends' blocks resident simultaneously (one computing, one uploading).
    b_pipe = [_make_swap_backend() for _ in range(_N_GRGS)]
    op_pipe = [_make_op(env_grg_path, b) for b in b_pipe]

    peak_pipe_per_step: list[int] = []
    b_pipe[0].prepare()
    for i in range(_N_GRGS):
        if i + 1 < _N_GRGS:
            b_pipe[i + 1].prepare()  # start next H2D copy while i computes
        # Both i and i+1 blocks are now allocated on GPU; _device_mem_used syncs internally
        m_after_prepare = _device_mem_used() - mem_base
        op_pipe[i].matmul(X.T, "up")
        m_after_matmul = _device_mem_used() - mem_base
        peak_pipe_per_step.append(max(m_after_prepare, m_after_matmul))
        b_pipe[i].finish()
    peak_pipe = max(peak_pipe_per_step)

    del b_pipe, op_pipe
    gc.collect()

    # ── Report ───────────────────────────────────────────────────────────────
    MB = 1 << 20
    with capsys.disabled():
        print(f"\n{'─'*60}")
        print(f"GPU memory benchmark: {_N_GRGS}× {Path(env_grg_path).name}  (baseline={mem_base/MB:.0f} MB subtracted)")
        print(f"  1 eager (baseline)         : {mem_1_eager / MB:8.2f} MB")
        print(f"  {_N_GRGS} eager simultaneously      : {mem_n_eager / MB:8.2f} MB"
              f"  ({mem_n_eager / max(mem_1_eager, 1):.1f}×)")
        print(f"  {_N_GRGS} swap — after setup (CPU)  : {mem_swap_setup / MB:8.2f} MB  (blocks on CPU)")
        print(f"  {_N_GRGS} swap — sequential peak    : {peak_swap / MB:8.2f} MB"
              f"  ({peak_swap / max(mem_1_eager, 1):.1f}×)")
        print(f"  {_N_GRGS} swap — pipelined peak     : {peak_pipe / MB:8.2f} MB"
              f"  ({peak_pipe / max(mem_1_eager, 1):.1f}×, expected ≈2×)")
        print(f"{'─'*60}")

    # N eager must use substantially more than 1 (otherwise GRG is trivially small)
    assert mem_n_eager >= 2 * mem_1_eager, (
        f"{_N_GRGS} eager backends ({mem_n_eager / MB:.2f} MB) should use ≥2× "
        f"a single backend ({mem_1_eager / MB:.2f} MB); "
        f"use a larger GRG via {_GRG_ENV}"
    )
    # Swap peak must be lower than half of N-eager (the whole point)
    assert peak_swap < mem_n_eager // 2, (
        f"Swap sequential peak ({peak_swap / MB:.2f} MB) should be < ½ of "
        f"N-eager ({mem_n_eager / MB:.2f} MB)"
    )
    # After swap setup, GPU usage should be less than 1 eager backend (blocks are on CPU)
    if mem_swap_setup >= mem_1_eager:
        import warnings
        warnings.warn(
            f"Swap setup GPU usage ({mem_swap_setup / MB:.2f} MB) is not less than "
            f"eager ({mem_1_eager / MB:.2f} MB); blocks may not be fully on CPU",
            RuntimeWarning,
            stacklevel=2,
        )
    # Pipelined peak must be between 1× and N×: more than sequential (2 backends resident)
    # but less than all N at once
    assert peak_swap <= peak_pipe < mem_n_eager, (
        f"Pipelined peak ({peak_pipe / MB:.2f} MB) should be between "
        f"sequential ({peak_swap / MB:.2f} MB) and N-eager ({mem_n_eager / MB:.2f} MB)"
    )


# ---------------------------------------------------------------------------
# 10. Timing: swap+pipelined vs sequential eager
# ---------------------------------------------------------------------------

@contextmanager
def _nvtx_range(label: str):
    """NVTX range annotation visible in Nsight Systems / Nsight Compute timelines."""
    cp.cuda.nvtx.RangePush(label)
    try:
        yield
    finally:
        cp.cuda.nvtx.RangePop()


def _run_eager_sequential(op_eager, X):
    for op in op_eager:
        op.matmul(X.T, "up")
    cp.cuda.runtime.deviceSynchronize()


def _run_swap_sequential(b_swap, op_swap, X):
    for b, op in zip(b_swap, op_swap):
        b.prepare()
        op.matmul(X.T, "up")
        b.finish()
    cp.cuda.runtime.deviceSynchronize()


def _run_swap_pipelined(b_pipe, op_pipe, X):
    b_pipe[0].prepare()
    for i in range(len(b_pipe)):
        if i + 1 < len(b_pipe):
            b_pipe[i + 1].prepare()
        op_pipe[i].matmul(X.T, "up")
        b_pipe[i].finish()
    cp.cuda.runtime.deviceSynchronize()


def test_swap_mode_timing(env_grg_path, capsys):
    """Compare wall-clock time for N operators across three execution strategies:
      (1) Sequential eager  — blocks always on GPU, pure compute time
      (2) Sequential swap   — prepare/compute/finish serially, H2D is on the critical path
      (3) Pipelined swap    — H2D of backend i+1 overlaps with compute of backend i

    One warmup pass is performed before each timed run to prime GPU caches and JIT.
    NVTX ranges are added only to the timed runs so the Nsight timeline is uncluttered.

    To profile with Nsight Systems:
        GRG_SPMV_TEST_GRG=/path/to/large.grg \\
        nsys profile --trace=cuda,nvtx --output=swap_timing \\
            python -m pytest <path>/test_cusparse_swap.py::test_swap_mode_timing \\
            --backend=cusparse -s
        nsys stats swap_timing.nsys-rep
    """
    K = 4

    b_ref = _make_eager_backend()
    op_ref = _make_op(env_grg_path, b_ref)
    rng = np.random.default_rng(8001)
    X = rng.standard_normal((op_ref.num_samples, K)).astype(DATA_DTYPE)
    del b_ref, op_ref
    gc.collect()

    # ── Sequential eager ─────────────────────────────────────────────────────
    b_eager = [_make_eager_backend() for _ in range(_N_GRGS)]
    op_eager = [_make_op(env_grg_path, b) for b in b_eager]
    _run_eager_sequential(op_eager, X)          # warmup (no NVTX)
    with _nvtx_range("eager_sequential"):
        t0 = time.perf_counter()
        for op in op_eager:
            with _nvtx_range("matmul"):
                op.matmul(X.T, "up")
        cp.cuda.runtime.deviceSynchronize()
        t_eager = time.perf_counter() - t0
    del b_eager, op_eager
    gc.collect()

    # ── Sequential swap ──────────────────────────────────────────────────────
    b_swap = [_make_swap_backend() for _ in range(_N_GRGS)]
    op_swap = [_make_op(env_grg_path, b) for b in b_swap]
    _run_swap_sequential(b_swap, op_swap, X)    # warmup (no NVTX)
    with _nvtx_range("swap_sequential"):
        t0 = time.perf_counter()
        for b, op in zip(b_swap, op_swap):
            with _nvtx_range("prepare"):
                b.prepare()
            with _nvtx_range("matmul"):
                op.matmul(X.T, "up")
            with _nvtx_range("finish"):
                b.finish()
        cp.cuda.runtime.deviceSynchronize()
        t_swap_seq = time.perf_counter() - t0
    del b_swap, op_swap
    gc.collect()

    # ── Pipelined swap ───────────────────────────────────────────────────────
    b_pipe = [_make_swap_backend() for _ in range(_N_GRGS)]
    op_pipe = [_make_op(env_grg_path, b) for b in b_pipe]
    _run_swap_pipelined(b_pipe, op_pipe, X)     # warmup (no NVTX)
    with _nvtx_range("swap_pipelined"):
        t0 = time.perf_counter()
        with _nvtx_range("prepare_0"):
            b_pipe[0].prepare()
        for i in range(_N_GRGS):
            if i + 1 < _N_GRGS:
                with _nvtx_range(f"prepare_{i + 1}"):
                    b_pipe[i + 1].prepare()
            with _nvtx_range(f"matmul_{i}"):
                op_pipe[i].matmul(X.T, "up")
            with _nvtx_range(f"finish_{i}"):
                b_pipe[i].finish()
        cp.cuda.runtime.deviceSynchronize()
        t_pipe = time.perf_counter() - t0
    del b_pipe, op_pipe
    gc.collect()

    # ── Report ───────────────────────────────────────────────────────────────
    with capsys.disabled():
        print(f"\n{'─'*60}")
        print(f"Timing benchmark: {_N_GRGS}× {Path(env_grg_path).name}")
        print(f"  sequential eager  : {t_eager   * 1e3:8.1f} ms  (1.00×  reference)")
        print(f"  sequential swap   : {t_swap_seq * 1e3:8.1f} ms  ({t_swap_seq / max(t_eager, 1e-9):.2f}×  H2D on critical path)")
        print(f"  pipelined swap    : {t_pipe     * 1e3:8.1f} ms  ({t_pipe     / max(t_eager, 1e-9):.2f}×  H2D overlapped)")
        print(f"{'─'*60}")
