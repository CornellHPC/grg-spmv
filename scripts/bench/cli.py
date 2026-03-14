"""CLI parsing and common benchmark arguments."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np

from scripts.bench.configs import PlanPairSpec, parse_plan_pair_literal

DTYPE = np.float64
INDEX_DTYPE = np.int32
DEFAULT_MATMUL_OPTIONS = (
    "baseline",
    "by_individual",
    "init_xtx",
    "init_vector",
    "init_matrix",
    "miss",
)
LOGGER = logging.getLogger("scripts.bench")
OUTPUT_ATOL = 1e-8
OUTPUT_RTOL = 1e-5
OUTPUT_ATOL_FLOAT32 = 1e-2
OUTPUT_RTOL_FLOAT32 = 1e-1
DEFAULT_GRG_PATH = "/pscratch/sd/q/qys/grg/simulation-mutation-200m.trees.v4.igd.final.grg"
LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR")


@dataclass(frozen=True)
class CommonBenchArgs:
    grg: str
    ks: list[int]
    plan_pair_specs: list[PlanPairSpec]
    options: list[str]
    n_trials: int
    n_warmup: int
    dtype: np.dtype
    index_dtype: np.dtype
    output_atol: float
    output_rtol: float
    log_level: str
    instrumentation: bool
    dry_run: bool
    skip_note: bool


def add_common_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--grg", default=DEFAULT_GRG_PATH)
    parser.add_argument(
        "--ks",
        type=str,
        default="32",
        help="Comma-separated runtime-k values (input rows); e.g. 1,4,16",
    )
    parser.add_argument("--trials", type=int, default=10, help="Number of timed trials")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument(
        "--plan-up-down",
        action="append",
        type=parse_plan_pair_literal,
        default=None,
        help="Repeated explicit plan-pair literal [..][..]",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=list(LOG_LEVEL_CHOICES),
        help="Log level passed to SpmvGRG and backend",
    )
    parser.add_argument(
        "--instrumentation",
        action="store_true",
        help="Enable profiling/instrumentation mode even when it reduces absolute performance",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=np.dtype(DTYPE).name,
        choices=["float32", "float64"],
        help="Input/output floating dtype",
    )
    parser.add_argument(
        "--index-dtype",
        type=str,
        default=np.dtype(INDEX_DTYPE).name,
        choices=["int32", "int64"],
        help="Index dtype used by SpmvGRG",
    )
    parser.add_argument(
        "--matmul-options",
        type=str,
        default="all",
        help="Comma-separated: baseline,by_individual,init_xtx,init_vector,init_matrix,miss,all",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print expanded backend configs and exit")
    parser.add_argument("--skip-note", action="store_true", help="Hide the Note column in the summary table")


def parse_csv_ints(raw: str, field_name: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        tok = token.strip()
        if not tok:
            continue
        value = int(tok)
        if value < 0:
            raise ValueError(f"{field_name} must contain non-negative integers, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{field_name} must contain at least one value")
    return values


def parse_dtype(raw: str) -> np.dtype:
    token = str(raw).strip().lower()
    if token == "float32":
        return np.dtype(np.float32)
    if token == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"--dtype must be one of float32,float64; got {raw!r}")


def parse_index_dtype(raw: str) -> np.dtype:
    token = str(raw).strip().lower()
    if token == "int32":
        return np.dtype(np.int32)
    if token == "int64":
        return np.dtype(np.int64)
    raise ValueError(f"--index-dtype must be one of int32,int64; got {raw!r}")


def tolerances_for_dtype(dtype: np.dtype) -> tuple[float, float]:
    dt = np.dtype(dtype)
    if dt == np.float32:
        return OUTPUT_ATOL_FLOAT32, OUTPUT_RTOL_FLOAT32
    if dt == np.float64:
        return OUTPUT_ATOL, OUTPUT_RTOL
    raise ValueError(f"Unsupported benchmark dtype for tolerance: {dt}")


def parse_matmul_options(raw: str) -> list[str]:
    tokens = [tok.strip().lower() for tok in raw.split(",") if tok.strip()]
    if not tokens or "all" in tokens:
        return list(DEFAULT_MATMUL_OPTIONS)
    valid = set(DEFAULT_MATMUL_OPTIONS)
    unknown = sorted(tok for tok in tokens if tok not in valid)
    if unknown:
        raise ValueError(f"Unknown matmul option(s): {', '.join(unknown)}")
    return tokens


def parse_common_bench_args(args: argparse.Namespace) -> CommonBenchArgs:
    ks = parse_csv_ints(args.ks, "--ks")
    plan_pair_specs = list(args.plan_up_down or [])
    if not plan_pair_specs:
        raise ValueError("--plan-up-down must be provided at least once")
    options = parse_matmul_options(args.matmul_options)
    dtype = parse_dtype(args.dtype)
    index_dtype = parse_index_dtype(args.index_dtype)
    output_atol, output_rtol = tolerances_for_dtype(dtype)
    return CommonBenchArgs(
        grg=str(args.grg),
        ks=ks,
        plan_pair_specs=plan_pair_specs,
        options=options,
        n_trials=int(args.trials),
        n_warmup=int(args.warmup),
        dtype=dtype,
        index_dtype=index_dtype,
        output_atol=output_atol,
        output_rtol=output_rtol,
        log_level=str(args.log_level),
        instrumentation=bool(args.instrumentation),
        dry_run=bool(args.dry_run),
        skip_note=bool(args.skip_note),
    )


def configure_logging(log_level: str) -> None:
    level_name = str(log_level).upper()
    level = getattr(logging, level_name, logging.WARNING)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    else:
        root.setLevel(level)
    LOGGER.setLevel(level)


def progress(msg: str) -> None:
    LOGGER.info("[bench] %s", msg)


__all__ = [
    "CommonBenchArgs",
    "DEFAULT_GRG_PATH",
    "DEFAULT_MATMUL_OPTIONS",
    "DTYPE",
    "INDEX_DTYPE",
    "LOG_LEVEL_CHOICES",
    "OUTPUT_ATOL",
    "OUTPUT_ATOL_FLOAT32",
    "OUTPUT_RTOL",
    "OUTPUT_RTOL_FLOAT32",
    "add_common_bench_args",
    "configure_logging",
    "parse_common_bench_args",
    "parse_csv_ints",
    "parse_dtype",
    "parse_index_dtype",
    "parse_matmul_options",
    "progress",
    "tolerances_for_dtype",
]
