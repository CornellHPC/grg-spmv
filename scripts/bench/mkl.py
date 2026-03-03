"""Benchmark SpmvGRG matmul for the MKL backend."""

from __future__ import annotations

import argparse
import gc
import shutil
import tempfile
from pathlib import Path
from time import perf_counter

from . import (
    DTYPE,
    INDEX_DTYPE,
    assert_output_equivalence,
    benchmark_config,
    configure_logging,
    expand_mkl_configs,
    format_dry_run_line,
    parse_csv_ints,
    parse_fmt_up_down,
    parse_k_hints,
    parse_matmul_options,
    print_summary_table,
    progress,
)
from pygrgl_spmv import SpmvGRG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SpmvGRG matmul for MKL backend")
    parser.add_argument(
        "--grg",
        default="/pscratch/sd/q/qys/grg/simulation-mutation-200m.trees.v4.igd.final.grg",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="32",
        help="Comma-separated runtime-k values (input rows); e.g. 1,4,16",
    )
    parser.add_argument("--trials", type=int, default=10, help="Number of timed trials")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument(
        "--k-hints",
        type=str,
        default="none,4",
        help="Comma-separated k_hint values (use 'none' to disable hint)",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default="0,1,4,16",
        help="Comma-separated MKL thread counts",
    )
    parser.add_argument(
        "--fmt-up-down",
        type=str,
        default="csr,none",
        help="Zipped fmt pairs: fmt_up,fmt_down/fmt_up,fmt_down (none allowed)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level passed to SpmvGRG and backend",
    )
    parser.add_argument(
        "--matmul-options",
        type=str,
        default="all",
        help="Comma-separated: baseline,by_individual,init_xtx,init_vector,init_matrix,miss,all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expanded backend configs and exit without running benchmarks",
    )
    parser.add_argument(
        "--skip-note",
        action="store_true",
        help="Hide the Note column in the summary table",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    try:
        ks = parse_csv_ints(args.ks, "--ks")
        k_hints = parse_k_hints(args.k_hints)
        thread_counts = parse_csv_ints(args.threads, "--threads")
        fmt_pairs = parse_fmt_up_down(args.fmt_up_down)
        options = parse_matmul_options(args.matmul_options)
    except ValueError as exc:
        raise SystemExit(f"Argument error: {exc}") from exc

    configs = expand_mkl_configs(thread_counts, fmt_pairs, k_hints, args.log_level)
    if args.dry_run:
        for entry in configs:
            print(format_dry_run_line(entry, ks, options))
        return

    progress(f"GRG file: {args.grg}")
    all_summary_rows: list[dict[str, object]] = []
    all_outputs: list[dict[str, object]] = []
    output_dir = Path(tempfile.mkdtemp(prefix="pygrgl_spmv_bench_refs_"))
    keep_output_dir = False

    try:
        for entry in configs:
            label = entry["label"]
            cfg = entry["config"]
            assert isinstance(label, str)
            assert isinstance(cfg, dict)

            progress(f"{'=' * 72}")
            progress(f"loading operator {label}")
            t_load = perf_counter()
            op = SpmvGRG(args.grg, cfg, DTYPE, INDEX_DTYPE)
            progress(f"{label}: operator ready in {(perf_counter() - t_load) * 1000.0:.2f} ms")

            cfg_rows, cfg_outputs = benchmark_config(
                op=op,
                label=label,
                ks=ks,
                options=options,
                n_trials=args.trials,
                n_warmup=args.warmup,
                seed_base=2026,
                output_dir=output_dir,
            )
            all_summary_rows.extend(cfg_rows)
            all_outputs.extend(cfg_outputs)

            del op
            gc.collect()

        print_summary_table(all_summary_rows, skip_note=args.skip_note)
        assert_output_equivalence(all_outputs)
    except Exception:
        keep_output_dir = True
        print(f"\nSaved benchmark reference outputs to: {output_dir}")
        raise
    finally:
        if not keep_output_dir:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
