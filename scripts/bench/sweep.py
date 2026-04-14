"""Sweep all .grg_spmv files in a folder across multiple backends and configs.

Produces a single JSON report (human + machine readable) containing per-file
benchmark results, file metadata, and hardware info.

Usage:
    python -m scripts.bench.sweep --folder /path/to/folder --output results.json
    python -m scripts.bench.sweep --folder /path/to/folder --output results.json --no-cusparse
    python -m scripts.bench.sweep --folder /path/to/folder --output results.json \\
        --threads 1,4,8 --optimize both --ks 1,32 --trials 20 --warmup 5 --device 1
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pathlib
import re
import subprocess
import sys

import numpy as np

from .cli import (
    configure_logging,
    parse_csv_ints,
    parse_dtype,
    parse_matmul_options,
    tolerances_for_dtype,
)
from .configs import BenchConfig, PlanPairSpec, expand_mkl_configs, format_dry_run_line
from .run import run_benchmark_suite
from pygrgl_spmv.grg.artifact import load_grg_spmv

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def collect_cpu_info() -> dict[str, object]:
    model = "unknown"
    logical_cores = os.cpu_count() or 0
    physical_cores = logical_cores

    try:
        seen_cores: set[tuple[str, str]] = set()
        current_physical_id = "0"
        with open("/proc/cpuinfo") as f:
            for line in f:
                line = line.strip()
                if line.startswith("model name") and model == "unknown":
                    model = line.split(":", 1)[1].strip()
                elif line.startswith("physical id"):
                    current_physical_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                    seen_cores.add((current_physical_id, core_id))
        if seen_cores:
            physical_cores = len(seen_cores)
    except OSError:
        pass

    return {
        "model": model,
        "logical_cores": logical_cores,
        "physical_cores": physical_cores,
    }


def collect_memory_info() -> dict[str, object]:
    total_bytes = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    total_bytes = kb * 1024
                    break
    except OSError:
        pass

    total_gb = round(total_bytes / (1024 ** 3), 2) if total_bytes else 0.0
    return {"total_bytes": total_bytes, "total_gb": total_gb}


def collect_gpu_info() -> list[dict[str, object]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        result.check_returncode()
        gpus: list[dict[str, object]] = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",", 2)]
            if len(parts) != 3:
                continue
            idx, name, mem_mib_str = parts
            vram_mb = int(mem_mib_str)
            gpus.append(
                {
                    "index": int(idx),
                    "name": name,
                    "vram_mb": vram_mb,
                    "vram_gb": round(vram_mb / 1024, 2),
                }
            )
        return gpus
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return []


def collect_hardware_info() -> dict[str, object]:
    return {
        "cpu": collect_cpu_info(),
        "memory": collect_memory_info(),
        "gpus": collect_gpu_info(),
    }


# ---------------------------------------------------------------------------
# File metadata
# ---------------------------------------------------------------------------

def file_metadata(path: str) -> dict[str, object]:
    size_bytes = os.path.getsize(path)
    return {
        "path": os.path.abspath(path),
        "filename": os.path.basename(path),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 4),
    }


# ---------------------------------------------------------------------------
# Row serialization
# ---------------------------------------------------------------------------

def _serialize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Convert benchmark row dicts to JSON-serializable form."""
    out: list[dict[str, object]] = []
    for row in rows:
        clean: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, np.floating):
                clean[key] = None if math.isnan(float(value)) else float(value)
            elif isinstance(value, np.integer):
                clean[key] = int(value)
            elif isinstance(value, float) and math.isnan(value):
                clean[key] = None
            elif isinstance(value, np.ndarray):
                clean[key] = "<ndarray>"
            else:
                clean[key] = value
        out.append(clean)
    return out


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def build_mkl_sweep_configs(
    threads: list[int],
    optimize_variants: list[bool],
    log_level: str,
    instrumentation: bool = False,
) -> list[BenchConfig]:
    """Build MKL BenchConfigs for the Cartesian product of threads × optimize_variants."""
    specs: list[PlanPairSpec] = []
    for n_threads in threads:
        for opt in optimize_variants:
            opt_str = "true" if opt else "false"
            up_spec: dict[str, str] = {
                "store": "N",
                "fmt": "CSR",
                "n_threads": str(n_threads),
                "k_hint": "none",
                "optimize": opt_str,
            }
            down_spec: dict[str, str] = {
                "store": "T",
                "fmt": "CSC",
                "n_threads": str(n_threads),
                "k_hint": "none",
                "optimize": opt_str,
            }
            specs.append((up_spec, down_spec))
    return expand_mkl_configs(specs, log_level, instrumentation)


def build_cusparse_sweep_configs(
    device: int,
    ring_buffer_size: int,
    log_level: str,
    instrumentation: bool = False,
) -> list[BenchConfig]:
    """Build one cuSPARSE BenchConfig using a safe concrete plan pair."""
    from .configs import expand_cusparse_configs  # deferred to avoid GPU import on CPU nodes

    up_spec: dict[str, str] = {
        "k_hint": "none",
        "store": "N",
        "fmt": "CSR",
        "opA": "N",
        "opB": "N",
        "orderB": "COL",
        "orderC": "COL",
        "algo": "DEFAULT",
    }
    down_spec: dict[str, str] = {
        "k_hint": "none",
        "store": "T",
        "fmt": "CSC",
        "opA": "N",
        "opB": "N",
        "orderB": "COL",
        "orderC": "COL",
        "algo": "DEFAULT",
    }
    specs: list[PlanPairSpec] = [(up_spec, down_spec)]
    return expand_cusparse_configs(specs, device, ring_buffer_size, log_level, instrumentation)


# ---------------------------------------------------------------------------
# Per-file runner
# ---------------------------------------------------------------------------

def run_file(
    grg_path: str,
    *,
    mkl_configs: list[BenchConfig],
    cusparse_configs: list[BenchConfig],
    ks: list[int],
    options: list[str],
    n_trials: int,
    n_warmup: int,
    dtype: np.dtype,
    output_atol: float,
    output_rtol: float,
    skip_note: bool,
) -> dict[str, object]:
    """Run all configs for one .grg_spmv file; return a file result dict."""
    meta = file_metadata(grg_path)
    all_rows: list[dict[str, object]] = []
    errors: list[str] = []

    # Load compiled state once — reused across all backend configs.
    try:
        compiled = load_grg_spmv(grg_path, dtype)
    except Exception as exc:
        return {"file": meta, "errors": [f"load: {exc}"], "rows": []}

    common_kwargs: dict[str, object] = dict(
        grg_path=grg_path,
        grg_ref_path=None,
        compiled_state=compiled,
        ks=ks,
        options=options,
        n_trials=n_trials,
        n_warmup=n_warmup,
        dtype=dtype,
        output_atol=output_atol,
        output_rtol=output_rtol,
        skip_note=skip_note,
        skip_memory_table=True,
    )

    if mkl_configs:
        print(f"  [mkl] {len(mkl_configs)} config(s):")
        for cfg in mkl_configs:
            print(f"    {cfg.label}")
        try:
            rows = run_benchmark_suite(configs=mkl_configs, **common_kwargs)  # type: ignore[arg-type]
            all_rows.extend(rows)
        except Exception as exc:
            errors.append(f"mkl: {exc}")

    if cusparse_configs:
        print(f"  [cusparse] {len(cusparse_configs)} config(s):")
        for cfg in cusparse_configs:
            print(f"    {cfg.label}")
        try:
            rows = run_benchmark_suite(configs=cusparse_configs, **common_kwargs)  # type: ignore[arg-type]
            all_rows.extend(rows)
        except Exception as exc:
            errors.append(f"cusparse: {exc}")

    compiled.A_blocks = None  # all backends set up; release sparse blocks

    return {
        "file": meta,
        "errors": errors,
        "rows": _serialize_rows(all_rows),
    }


# ---------------------------------------------------------------------------
# Markdown summary table
# ---------------------------------------------------------------------------

def _config_shorthand(label: str) -> tuple[str, str]:
    """Return (backend, shorthand) from a BenchConfig label string.

    Examples:
        "mkl-up=[..., n_threads=4, ..., optimize=true]-down=[...]"
            → ("mkl", "t=4 opt=T")
        "cusparse-up=[...]-down=[...]"
            → ("cusparse", "dev=?")  (device not in label; caller fills in)
    """
    backend = label.split("-up=")[0] if "-up=" in label else label.split("-")[0]

    if backend == "mkl":
        m_threads = re.search(r"n_threads=(\d+)", label)
        m_opt = re.search(r"optimize=(true|false)", label, re.IGNORECASE)
        t = m_threads.group(1) if m_threads else "?"
        opt = "T" if (m_opt and m_opt.group(1).lower() == "true") else "F"
        return backend, f"t={t} opt={opt}"

    if backend == "cusparse":
        return backend, "cusparse"

    return backend, backend


def _print_markdown_table(file_results: list[dict[str, object]]) -> None:
    """Print a pivot GFM markdown table: rows = (file, scenario, dir, k), columns = configs."""
    # First pass: collect ordered config columns and build pivot data.
    col_order: list[str] = []
    col_seen: set[str] = set()
    # pivot[(filename, scenario, direction, k)][col] = cell_str
    pivot: dict[tuple[str, str, str, str], dict[str, str]] = {}

    for file_result in file_results:
        filename = str(file_result["file"]["filename"])  # type: ignore[index]
        for row in file_result["rows"]:  # type: ignore[index]
            if "skip" in row:
                continue
            backend, shorthand = _config_shorthand(str(row.get("config", "")))
            col = f"{backend} {shorthand}" if backend != shorthand else backend
            if col not in col_seen:
                col_order.append(col)
                col_seen.add(col)

            key = (
                filename,
                str(row.get("scenario", "")),
                str(row.get("direction", "")),
                str(row.get("k", "")),
            )
            mean_ms = row.get("call_ms_mean")
            std_ms = row.get("call_ms_std")
            intra_errors = row.get("intra_errors")

            if isinstance(mean_ms, (int, float)) and mean_ms is not None:
                cell = (
                    f"{mean_ms:.3f}\u00b1{std_ms:.3f}"
                    if isinstance(std_ms, (int, float))
                    else f"{mean_ms:.3f}"
                )
                if intra_errors:
                    cell += "*"
            else:
                cell = "-"

            pivot.setdefault(key, {})[col] = cell

    if not pivot:
        print("\n## Benchmark Summary\n\n(no results)\n")
        return

    fixed_headers = ["File", "Scenario", "k"]
    headers = fixed_headers + col_order

    def _sorted_rows(direction: str) -> list[list[str]]:
        rows = []
        for (filename, scenario, d, k), cells in sorted(
            pivot.items(),
            key=lambda item: (item[0][0], item[0][1], item[0][3]),
        ):
            if d != direction:
                continue
            rows.append([filename, scenario, k] + [cells.get(col, "-") for col in col_order])
        return rows

    def _print_section(title: str, display_rows: list[list[str]]) -> None:
        if not display_rows:
            return
        col_widths = [len(h) for h in headers]
        for row in display_rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))
        fmt = lambda cells: "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"
        sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
        has_errors = any("*" in cell for row in display_rows for cell in row[len(fixed_headers):])
        print(f"\n{title}\n")
        if has_errors:
            print("(`*` = intra-trial errors > 0)\n")
        print(fmt(headers))
        print(sep)
        for row in display_rows:
            print(fmt(row))
        print()

    _print_section("## Benchmark Summary — Up", _sorted_rows("up"))
    _print_section("## Benchmark Summary — Down", _sorted_rows("down"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep all .grg_spmv files in a folder across MKL and cuSPARSE backends.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--folder", required=True, help="Directory containing .grg_spmv files")
    parser.add_argument("--output", required=True, help="Path to write JSON results file")

    # MKL
    parser.add_argument(
        "--threads",
        type=str,
        default="1,2,4",
        help="Comma-separated MKL thread counts. 0 = all cores.",
    )
    parser.add_argument(
        "--optimize",
        type=str,
        default="both",
        choices=["on", "off", "both"],
        help="MKL optimize variants to include.",
    )

    # cuSPARSE
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device ordinal.",
    )
    parser.add_argument(
        "--no-mkl",
        action="store_true",
        help="Skip MKL benchmarks entirely.",
    )
    parser.add_argument(
        "--no-cusparse",
        action="store_true",
        help="Skip cuSPARSE benchmarks entirely",
    )
    parser.add_argument("--ring-buffer-size", type=int, default=2, help="GPU sparse slot ring size.")

    # Benchmark parameters
    parser.add_argument("--ks", type=str, default="32", help="Comma-separated runtime-k values.")
    parser.add_argument("--trials", type=int, default=10, help="Number of timed trials per config.")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs before timing.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="all",
        help="Scenarios to benchmark: 'all' or a comma-separated subset of "
             "baseline,by_individual,init_xtx,init_vector,init_matrix,miss.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Root log level for benchmark, operator, and backend loggers.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print expanded configs and exit")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show Note column in per-file tables.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    configure_logging(str(args.log_level))

    folder = pathlib.Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"--folder {folder!r} is not a directory")

    grg_files = sorted(folder.glob("*.grg_spmv"))
    if not grg_files:
        raise SystemExit(f"No .grg_spmv files found in {folder}")

    # Parse benchmark settings
    dtype = parse_dtype(args.dtype)
    output_atol, output_rtol = tolerances_for_dtype(dtype)
    ks = parse_csv_ints(args.ks, "--ks")
    options = parse_matmul_options(args.scenarios)
    skip_note = not args.verbose

    # Build configs once (reused across all files)
    threads = parse_csv_ints(args.threads, "--threads")
    optimize_variants: list[bool] = (
        [True, False] if args.optimize == "both"
        else [False] if args.optimize == "off"
        else [True]
    )
    mkl_configs: list[BenchConfig] = []
    if not args.no_mkl:
        mkl_configs = build_mkl_sweep_configs(threads, optimize_variants, args.log_level)

    cusparse_configs: list[BenchConfig] = []
    if not args.no_cusparse:
        try:
            cusparse_configs = build_cusparse_sweep_configs(
                args.device, args.ring_buffer_size, args.log_level
            )
        except Exception as exc:
            print(f"[sweep] Warning: could not build cuSPARSE configs: {exc}", file=sys.stderr)

    # Dry run
    if args.dry_run:
        for entry in mkl_configs + cusparse_configs:
            print(format_dry_run_line(entry, ks, options, dtype=dtype))
        return

    # Collect hardware info
    print("[sweep] Collecting hardware info...")
    hardware = collect_hardware_info()

    # Per-file loop
    file_results: list[dict[str, object]] = []
    for grg_path in grg_files:
        size_mb = grg_path.stat().st_size / (1024 * 1024)
        print(f"\n[sweep] {grg_path.name}  ({size_mb:.2f} MB)")
        result = run_file(
            str(grg_path),
            mkl_configs=mkl_configs,
            cusparse_configs=cusparse_configs,
            ks=ks,
            options=options,
            n_trials=args.trials,
            n_warmup=args.warmup,
            dtype=dtype,
            output_atol=output_atol,
            output_rtol=output_rtol,
            skip_note=skip_note,
        )
        file_results.append(result)
        if result["errors"]:
            print(f"  [ERRORS] {result['errors']}", file=sys.stderr)

    # Assemble output
    output: dict[str, object] = {
        "sweep_version": 1,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": hardware,
        "sweep_config": {
            "ks": ks,
            "options": options,
            "n_trials": args.trials,
            "n_warmup": args.warmup,
            "dtype": np.dtype(dtype).name,
            "output_atol": output_atol,
            "output_rtol": output_rtol,
            "threads": threads,
            "optimize": args.optimize,
            "device": None if args.no_cusparse else args.device,
        },
        "files": file_results,
    }

    out_path = pathlib.Path(args.output)
    out_path.write_text(json.dumps(output, indent=2))
    total_rows = sum(len(f["rows"]) for f in file_results)  # type: ignore[arg-type]
    print(
        f"\n[sweep] Results written to {out_path}  "
        f"({len(file_results)} files, {total_rows} rows)"
    )

    _print_markdown_table(file_results)


if __name__ == "__main__":
    main()
