"""Merge one or more sweep benchmark JSON result files into a single Markdown table.

Each input file is the output of `scripts.bench.sweep`. Results from different
hardware are placed in separate columns using symbol suffixes (-, △, □, ◇).
Identical hardware is deduplicated to one column. Rows are sorted by chromosome
number (chr1, chr2, ..., chr10, chr11, ...).

Usage:
    python -m scripts.utils.merge_sweep results_A.json results_B.json
    python -m scripts.utils.merge_sweep results*.json --output merged.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYMBOLS = ["", "-", "\u25b3", "\u25a1", "\u25c7"]  # "", "-", "△", "□", "◇"


# ---------------------------------------------------------------------------
# Config shorthand (mirrors scripts/bench/sweep.py)
# ---------------------------------------------------------------------------

def _config_shorthand(label: str) -> tuple[str, str]:
    """Return (backend, shorthand) from a BenchConfig label string."""
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


# ---------------------------------------------------------------------------
# Hardware key extraction
# ---------------------------------------------------------------------------

def _hw_key(config_label: str, sweep_hw: dict) -> tuple[str, str]:
    """Return (hw_type, hw_name) for a row given the enclosing file's hardware info.

    hw_type is "cpu" for MKL rows, "gpu" for cuSPARSE rows.
    """
    backend = config_label.split("-up=")[0] if "-up=" in config_label else config_label.split("-")[0]
    if backend == "mkl":
        model = sweep_hw.get("cpu", {}).get("model", "unknown")
        return "cpu", model
    if backend == "cusparse":
        gpus = sweep_hw.get("gpus", [])
        name = gpus[0]["name"] if gpus else "unknown"
        return "gpu", name
    model = sweep_hw.get("cpu", {}).get("model", "unknown")
    return "cpu", model


# ---------------------------------------------------------------------------
# Symbol assignment
# ---------------------------------------------------------------------------

def _assign_symbols(unique_hw: list[str]) -> dict[str, str]:
    """Assign display symbols to unique hardware names in insertion order.

    One unique value  → maps to "" (no suffix).
    Two or more       → maps to "-", "△", "□", "◇", ... in order.
    """
    if len(unique_hw) <= 1:
        return {hw: "" for hw in unique_hw}
    return {hw: _SYMBOLS[i + 1] for i, hw in enumerate(unique_hw)}


# ---------------------------------------------------------------------------
# Natural sort key for filenames
# ---------------------------------------------------------------------------

def _natural_sort_key(filename: str) -> tuple:
    """Sort key that orders chr1, chr2, ..., chr9, chr10, chr11 correctly."""
    nums = tuple(int(n) for n in re.findall(r"chr(\d+)", filename))
    if nums:
        return nums + (filename,)
    # Fallback: mixed alphanumeric
    parts: list = []
    for chunk in re.split(r"(\d+)", filename):
        parts.append(int(chunk) if chunk.isdigit() else chunk.lower())
    return tuple(parts)


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_sweep_file(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    if "files" not in data or "hardware" not in data:
        raise ValueError(f"{path!r} does not look like a sweep result file (missing 'files' or 'hardware')")
    return data


# ---------------------------------------------------------------------------
# Pivot builder
# ---------------------------------------------------------------------------

def build_pivot(sweep_files: list[dict]) -> tuple[
    dict[tuple, dict[str, str]],   # pivot[row_key][col_header] = cell
    list[str],                      # col_order
    dict[str, str],                 # cpu_hw_legend: symbol → hw_name
    dict[str, str],                 # gpu_hw_legend: symbol → hw_name
]:
    # Pass 1: collect all (config_shorthand, hw_type, hw_name) triples
    # keeping insertion order for unique hw per type.
    cpu_hw_seen: list[str] = []   # unique CPU models in order of first appearance
    gpu_hw_seen: list[str] = []   # unique GPU names

    for sweep in sweep_files:
        hw = sweep["hardware"]
        for file_entry in sweep["files"]:
            for row in file_entry["rows"]:
                if "skip" in row or "call_ms_mean" not in row:
                    continue
                hw_type, hw_name = _hw_key(str(row.get("config", "")), hw)
                if hw_type == "cpu" and hw_name not in cpu_hw_seen:
                    cpu_hw_seen.append(hw_name)
                elif hw_type == "gpu" and hw_name not in gpu_hw_seen:
                    gpu_hw_seen.append(hw_name)

    cpu_symbols = _assign_symbols(cpu_hw_seen)
    gpu_symbols = _assign_symbols(gpu_hw_seen)

    def hw_symbol(hw_type: str, hw_name: str) -> str:
        return cpu_symbols.get(hw_name, "?") if hw_type == "cpu" else gpu_symbols.get(hw_name, "?")

    # Pass 2: build pivot and col_order
    col_seen: set[str] = set()
    col_order: list[str] = []
    # pivot[row_key][col_header] = cell_str
    pivot: dict[tuple, dict[str, str]] = {}

    for sweep in sweep_files:
        hw = sweep["hardware"]
        for file_entry in sweep["files"]:
            filename = str(file_entry["file"]["filename"])
            for row in file_entry["rows"]:
                if "skip" in row or "call_ms_mean" not in row:
                    continue
                config_label = str(row.get("config", ""))
                backend, shorthand = _config_shorthand(config_label)
                hw_type, hw_name = _hw_key(config_label, hw)
                sym = hw_symbol(hw_type, hw_name)

                base_col = f"{backend} {shorthand}" if backend != shorthand else backend
                col_header = f"{base_col} {sym}".strip() if sym else base_col
                if col_header not in col_seen:
                    col_order.append(col_header)
                    col_seen.add(col_header)

                row_key = (
                    filename,
                    str(row.get("scenario", "")),
                    str(row.get("direction", "")),
                    str(row.get("k", "")),
                )

                mean_ms = row.get("call_ms_mean")
                std_ms = row.get("call_ms_std")
                intra_errors = row.get("intra_errors")

                if isinstance(mean_ms, (int, float)):
                    cell = (
                        f"{mean_ms:.3f}\u00b1{std_ms:.3f}"
                        if isinstance(std_ms, (int, float))
                        else f"{mean_ms:.3f}"
                    )
                    if intra_errors:
                        cell += "*"
                else:
                    cell = "-"

                # Keep the first value if the same col/row appears in multiple files
                # with identical hardware (deduplication).
                pivot.setdefault(row_key, {}).setdefault(col_header, cell)

    # Build legends (only for types with 2+ distinct values)
    cpu_legend: dict[str, str] = {}
    if len(cpu_hw_seen) >= 2:
        for hw_name in cpu_hw_seen:
            sym = cpu_symbols[hw_name]
            if sym:
                cpu_legend[sym] = hw_name

    gpu_legend: dict[str, str] = {}
    if len(gpu_hw_seen) >= 2:
        for hw_name in gpu_hw_seen:
            sym = gpu_symbols[hw_name]
            if sym:
                gpu_legend[sym] = hw_name

    return pivot, col_order, cpu_legend, gpu_legend


# ---------------------------------------------------------------------------
# Markdown table printer
# ---------------------------------------------------------------------------

def print_markdown_table(
    pivot: dict[tuple, dict[str, str]],
    col_order: list[str],
    cpu_legend: dict[str, str],
    gpu_legend: dict[str, str],
    out=None,
) -> None:
    if out is None:
        out = sys.stdout

    if not pivot:
        print("(no results)", file=out)
        return

    fixed_headers = ["File", "Scenario", "k"]
    headers = fixed_headers + col_order

    def _sorted_rows(direction: str) -> list[list[str]]:
        rows = []
        for (filename, scenario, d, k) in sorted(
            pivot.keys(),
            key=lambda rk: _natural_sort_key(rk[0]) + (rk[1], rk[3]),
        ):
            if d != direction:
                continue
            cells = pivot[(filename, scenario, d, k)]
            rows.append([filename, scenario, str(k)] + [cells.get(col, "-") for col in col_order])
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
        print(f"{title}\n", file=out)
        if has_errors:
            print("(`*` = intra-trial errors > 0)\n", file=out)
        print(fmt(headers), file=out)
        print(sep, file=out)
        for row in display_rows:
            print(fmt(row), file=out)
        print(file=out)

    _print_section("## Benchmark Summary — Up", _sorted_rows("up"))
    _print_section("## Benchmark Summary — Down", _sorted_rows("down"))

    # Legend
    if cpu_legend or gpu_legend:
        print("**Hardware legend:**\n", file=out)
        for sym in _SYMBOLS[1:]:
            if sym in cpu_legend:
                print(f"- CPU `{sym}`  {cpu_legend[sym]}", file=out)
        for sym in _SYMBOLS[1:]:
            if sym in gpu_legend:
                print(f"- GPU `{sym}`  {gpu_legend[sym]}", file=out)
        print(file=out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge sweep JSON result files into a single Markdown pivot table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", metavar="FILE", help="Sweep JSON result files to merge.")
    parser.add_argument("--output", type=str, default=None, help="Write Markdown to this file (default: stdout).")
    args = parser.parse_args()

    sweep_files: list[dict] = []
    for path in args.inputs:
        try:
            sweep_files.append(load_sweep_file(path))
        except Exception as exc:
            print(f"[merge_sweep] Warning: skipping {path!r}: {exc}", file=sys.stderr)

    if not sweep_files:
        raise SystemExit("No valid sweep files loaded.")

    pivot, col_order, cpu_legend, gpu_legend = build_pivot(sweep_files)

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w") as f:
            print_markdown_table(pivot, col_order, cpu_legend, gpu_legend, out=f)
        print(f"[merge_sweep] Written to {out_path}", file=sys.stderr)
    else:
        print_markdown_table(pivot, col_order, cpu_legend, gpu_legend)


if __name__ == "__main__":
    main()
