"""Benchmark the canonical cuSPARSE runtime configuration."""

from __future__ import annotations

import argparse

from pygrgl_spmv.backends.cusparse import (
    CusparsePlan,
    CusparsePlanPair,
    CusparseRuntime,
    plan_cusparse_layout,
)

from .cli import add_common_args, parse_args
from .run import baseline_requirements, benchmark_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CusparseRuntime")
    add_common_args(parser, gpu=True)
    args = parse_args(parser.parse_args(), gpu=True)
    layout = plan_cusparse_layout(
        artifacts=[args.artifact],
        pair=CusparsePlanPair(
            plan_up=CusparsePlan(store="N", fmt="CSR", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT", scratch="none"),
            plan_down=CusparsePlan(store="T", fmt="CSC", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT", scratch="none"),
        ),
        dtype=args.dtype,
        requirements=baseline_requirements(direction=args.direction, k=args.k),
        vram_budget_bytes=args.vram_budget_bytes,
        ring_buffer_size=args.ring_buffer_size,
        device=args.device,
        stream=args.stream,
    )
    benchmark_runtime(
        runtime_cls=CusparseRuntime,
        layout=layout,
        direction=args.direction,
        k=args.k,
        dtype=args.dtype,
        warmup=args.warmup,
        trials=args.trials,
    )


if __name__ == "__main__":
    main()
