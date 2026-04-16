"""Minimal runtime-centric benchmark runner."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from pygrgl_spmv import RuntimeRequirements


def benchmark_runtime(*, runtime_cls, layout, direction: str, k: int, dtype: np.dtype, warmup: int, trials: int) -> None:
    with runtime_cls(layout) as runtime:
        (grg,) = runtime.grgs
        cols = grg.num_samples if direction == "up" else grg.num_mutations
        rng = np.random.default_rng(2026)
        matrix = rng.standard_normal((int(k), cols), dtype=dtype)
        for _ in range(int(warmup)):
            _ = grg.matmul(matrix, direction)
        times = []
        for _ in range(int(trials)):
            t0 = perf_counter()
            _ = grg.matmul(matrix, direction)
            times.append((perf_counter() - t0) * 1000.0)
        arr = np.asarray(times, dtype=np.float64)
        print(f"backend={runtime_cls.__name__} direction={direction} k={k} mean_ms={arr.mean():.4f} std_ms={arr.std():.4f}")


def baseline_requirements(*, direction: str, k: int) -> RuntimeRequirements:
    return RuntimeRequirements(
        max_k_up=max(int(k), 1) if direction == "up" else 1,
        max_k_down=max(int(k), 1) if direction == "down" else 1,
        need_down_miss_input=False,
        need_up_miss_output=False,
        need_init_vector=False,
        need_init_matrix=False,
        need_init_xtx=False,
    )
