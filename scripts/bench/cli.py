"""Minimal CLI helpers for runtime-centric benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BenchArgs:
    artifact: str
    direction: str
    k: int
    trials: int
    warmup: int
    dtype: np.dtype
    device: int
    stream: int
    ring_buffer_size: int
    vram_budget_bytes: int


def add_common_args(parser: argparse.ArgumentParser, *, gpu: bool) -> None:
    parser.add_argument("--artifact", required=True, help="Path to a .grg_spmv artifact")
    parser.add_argument("--direction", choices=["up", "down"], default="up")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    if gpu:
        parser.add_argument("--device", type=int, default=0)
        parser.add_argument("--stream", type=int, default=0)
        parser.add_argument("--ring-buffer-size", type=int, default=2)
        parser.add_argument("--vram-budget-bytes", type=int, required=True)


def parse_args(args: argparse.Namespace, *, gpu: bool) -> BenchArgs:
    dtype = np.dtype(np.float32 if args.dtype == "float32" else np.float64)
    if int(args.k) < 1:
        raise ValueError(f"--k must be >= 1, got {args.k}")
    if int(args.trials) < 1:
        raise ValueError(f"--trials must be >= 1, got {args.trials}")
    if int(args.warmup) < 0:
        raise ValueError(f"--warmup must be >= 0, got {args.warmup}")
    return BenchArgs(
        artifact=str(args.artifact),
        direction=str(args.direction),
        k=int(args.k),
        trials=int(args.trials),
        warmup=int(args.warmup),
        dtype=dtype,
        device=0 if not gpu else int(args.device),
        stream=0 if not gpu else int(args.stream),
        ring_buffer_size=0 if not gpu else int(args.ring_buffer_size),
        vram_budget_bytes=0 if not gpu else int(args.vram_budget_bytes),
    )
