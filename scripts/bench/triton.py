"""Benchmark the canonical Triton runtime configuration."""

from __future__ import annotations

import argparse

from pygrgl_spmv.backends.triton import TritonPlan, TritonPlanPair, TritonRuntime, plan_triton_layout

from .cli import add_common_args, parse_args
from .run import baseline_requirements, benchmark_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TritonRuntime")
    add_common_args(parser, gpu=True)
    args = parse_args(parser.parse_args(), gpu=True)
    layout = plan_triton_layout(
        artifacts=[args.artifact],
        pair=TritonPlanPair(
            plan_up=TritonPlan(store="N", fmt="CSR", scratch="none"),
            plan_down=TritonPlan(store="T", fmt="CSC", scratch="none"),
        ),
        dtype=args.dtype,
        requirements=baseline_requirements(direction=args.direction, k=args.k),
        vram_budget_bytes=args.vram_budget_bytes,
        ring_buffer_size=args.ring_buffer_size,
        device=args.device,
        stream=args.stream,
    )
    benchmark_runtime(
        runtime_cls=TritonRuntime,
        layout=layout,
        direction=args.direction,
        k=args.k,
        dtype=args.dtype,
        warmup=args.warmup,
        trials=args.trials,
    )


if __name__ == "__main__":
    main()
