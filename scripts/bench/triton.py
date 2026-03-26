"""Benchmark SpmvGRG matmul for the Triton backend."""

from __future__ import annotations

import argparse

from .cli import (
    add_common_bench_args,
    configure_logging,
    parse_common_bench_args,
)
from .configs import expand_triton_configs, format_dry_run_line
from .run import run_benchmark_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SpmvGRG matmul for the Triton backend")
    add_common_bench_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(str(args.log_level))

    try:
        common = parse_common_bench_args(args)
    except ValueError as exc:
        raise SystemExit(f"Argument error: {exc}") from exc

    configs = expand_triton_configs(
        common.plan_pair_specs,
        common.orderings,
        common.intra_block_orderings,
        common.log_level,
        common.instrumentation,
    )
    if common.dry_run:
        for entry in configs:
            print(
                format_dry_run_line(
                    entry,
                    common.ks,
                    common.options,
                    dtype=common.dtype,
                    index_dtype=common.index_dtype,
                )
            )
        return

    run_benchmark_suite(
        grg_path=common.grg,
        configs=configs,
        ks=common.ks,
        options=common.options,
        n_trials=common.n_trials,
        n_warmup=common.n_warmup,
        dtype=common.dtype,
        index_dtype=common.index_dtype,
        output_atol=common.output_atol,
        output_rtol=common.output_rtol,
        skip_note=common.skip_note,
    )


if __name__ == "__main__":
    main()
