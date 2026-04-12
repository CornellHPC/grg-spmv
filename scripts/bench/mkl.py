"""Benchmark SpmvGRG matmul for the MKL backend."""

from __future__ import annotations

import argparse

from .cli import (
    add_common_bench_args,
    configure_logging,
    parse_common_bench_args,
    parse_csv_ints,
)
from .configs import BenchConfig, PlanPairSpec, expand_mkl_configs, format_dry_run_line
from .run import run_benchmark_suite


def _expand_sweep_configs(
    threads: list[int],
    optimize_variants: list[bool],
    log_level: str,
    instrumentation: bool,
) -> list[BenchConfig]:
    specs: list[PlanPairSpec] = []
    for n_threads in threads:
        for opt in optimize_variants:
            opt_str = "true" if opt else "false"
            up_spec = {"store": "N", "fmt": "CSR", "n_threads": str(n_threads), "k_hint": "none", "optimize": opt_str}
            down_spec = {"store": "T", "fmt": "CSC", "n_threads": str(n_threads), "k_hint": "none", "optimize": opt_str}
            specs.append((up_spec, down_spec))
    return expand_mkl_configs(specs, log_level, instrumentation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SpmvGRG matmul for MKL backend")
    add_common_bench_args(parser)
    parser.add_argument(
        "--threads",
        type=str,
        default=None,
        help="Comma-separated thread counts to sweep (e.g. 1,4,8,16). 0 = all cores.",
    )
    parser.add_argument(
        "--optimize",
        type=str,
        default="on",
        choices=["on", "off", "both"],
        help="Which optimize variants to include: on, off, or both (default: on)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(str(args.log_level))

    use_sweep = args.threads is not None or args.optimize != "on"

    try:
        common = parse_common_bench_args(args) if not use_sweep else parse_common_bench_args(args, require_plan=False)
    except ValueError as exc:
        raise SystemExit(f"Argument error: {exc}") from exc

    if use_sweep:
        threads = parse_csv_ints(args.threads, "--threads") if args.threads else [0]
        optimize_variants = (
            [True, False] if args.optimize == "both"
            else [False] if args.optimize == "off"
            else [True]
        )
        configs = _expand_sweep_configs(threads, optimize_variants, common.log_level, common.instrumentation)
    else:
        configs = expand_mkl_configs(common.plan_pair_specs, common.log_level, common.instrumentation)

    if common.dry_run:
        for entry in configs:
            print(format_dry_run_line(entry, common.ks, common.options, dtype=common.dtype))
        return

    run_benchmark_suite(
        grg_path=common.grg,
        grg_ref_path=common.grg_ref,
        configs=configs,
        ks=common.ks,
        options=common.options,
        n_trials=common.n_trials,
        n_warmup=common.n_warmup,
        dtype=common.dtype,
        output_atol=common.output_atol,
        output_rtol=common.output_rtol,
        skip_note=common.skip_note,
    )


if __name__ == "__main__":
    main()
