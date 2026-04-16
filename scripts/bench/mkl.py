"""Benchmark the canonical MKL runtime configuration."""

from __future__ import annotations

import argparse

from pygrgl_spmv.backends.mkl import MklPlan, MklPlanPair, MklRuntime, plan_mkl_layout

from .cli import add_common_args, parse_args
from .run import baseline_requirements, benchmark_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MklRuntime")
    add_common_args(parser, gpu=False)
    args = parse_args(parser.parse_args(), gpu=False)
    layout = plan_mkl_layout(
        artifacts=[args.artifact],
        pair=MklPlanPair(
            plan_up=MklPlan(store="N", fmt="CSR", n_threads=0),
            plan_down=MklPlan(store="T", fmt="CSC", n_threads=0),
        ),
        dtype=args.dtype,
        requirements=baseline_requirements(direction=args.direction, k=args.k),
    )
    benchmark_runtime(
        runtime_cls=MklRuntime,
        layout=layout,
        direction=args.direction,
        k=args.k,
        dtype=args.dtype,
        warmup=args.warmup,
        trials=args.trials,
    )


if __name__ == "__main__":
    main()
