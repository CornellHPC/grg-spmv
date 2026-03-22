"""Analyze cuSPARSE benchmark timing from full benchmark logs.

This prototype accepts full benchmark logs emitted by ``scripts.bench.cusparse``.
It extracts the embedded benchmark summary table, parses baseline runtime rows,
and fits compact plan-derived timing models per direction.

Typical usage:

    uv run python -m scripts.analyze_cusparse_summary \
      --input-up output_up_2.txt \
      --input-down output_down_2.txt
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


_SUMMARY_HEADER = "BENCHMARK RUNTIME SUMMARY (time + correctness diagnostics)"
_SUMMARY_END = "Correctness diagnostics:"
_RUNTIME_ROW_RE = re.compile(
    r"^(?P<config>.+?)\s+baseline\s+(?P<direction>up|down)\s+(?P<k>\d+)\s+"
    r"(?P<mean_ms>[0-9.]+)\+/-[0-9.]+\s+"
    r"(?P<err_trials>\d+/\d+)\s+"
    r"(?P<abs_err>\S+)\s+"
    r"(?P<rel_err>\S+)"
    r"(?:\s+(?P<note>.*))?$"
)
_LABELED_PLAN_RE = re.compile(r"(?:^|-)(up|down)=(\[[^\]]+\]|<unspecified>)")

_OLD_MODEL_FORMULA = (
    "log_ms ~ logical_row_compressed * C(algo)"
    " + logical_row_compressed * C(orderC)"
    " + C(dense_case)"
)
_FINAL_MODEL_FORMULA = (
    "log_ms ~ logical_row_compressed * C(algo) * C(orderC)"
    " + C(dense_case)"
)
_RAW_DENSE_MODEL_FORMULA = (
    "log_ms ~ logical_row_compressed * C(algo)"
    " + C(opB) * C(orderB) * C(orderC)"
)


@dataclass(frozen=True)
class ParsedRow:
    config: str
    direction: str
    k: int
    mean_ms: float
    err_trials: str
    note: str
    plan_fields: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze cuSPARSE timing from full benchmark logs")
    parser.add_argument("--input-up", type=Path, required=True, help="Full benchmark log for UP-only runs")
    parser.add_argument("--input-down", type=Path, required=True, help="Full benchmark log for DOWN-only runs")
    parser.add_argument("--top-n", type=int, default=10, help="Number of fastest plans to print per direction")
    return parser.parse_args()


def _extract_summary_section(path: Path) -> str:
    text = path.read_text()
    if _SUMMARY_HEADER not in text:
        raise ValueError(f"{path}: full benchmark log required; missing summary header {_SUMMARY_HEADER!r}")
    start = text.index(_SUMMARY_HEADER)
    try:
        end = text.index(_SUMMARY_END, start)
    except ValueError as exc:
        raise ValueError(f"{path}: missing summary terminator {_SUMMARY_END!r}") from exc
    return text[start:end]


def _parse_plan_literal(raw: str) -> dict[str, str]:
    literal = raw.strip()
    if not (literal.startswith("[") and literal.endswith("]")):
        raise ValueError(f"Expected bracketed plan literal, got {raw!r}")
    fields: dict[str, str] = {}
    for part in literal[1:-1].split(","):
        token = part.strip()
        if not token:
            continue
        key, value = token.split("=", 1)
        fields[key.strip()] = value.strip()
    required = {"k_hint", "store", "fmt", "opA", "opB", "orderB", "orderC", "algo"}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"Missing plan field(s) {missing} in {raw!r}")
    return fields


def _extract_plan_literal(config: str, direction: str) -> str:
    labeled = {match.group(1): match.group(2) for match in _LABELED_PLAN_RE.finditer(config)}
    value = labeled.get(direction)
    if value is None:
        raise ValueError(f"Could not determine {direction} plan from config {config!r}")
    if value == "<unspecified>":
        raise ValueError(f"Config {config!r} does not contain a concrete {direction} plan")
    return value


def _extract_mode(note: str) -> str:
    for token in str(note).split():
        if token.startswith("mode="):
            return token.split("=", 1)[1]
    return "unknown"


def _dense_case(row: pd.Series) -> str:
    op_b = str(row["opB"])
    order_b = str(row["orderB"])
    order_c = str(row["orderC"])
    if op_b == "N" and order_b == order_c:
        return "alias"
    if op_b == "T" and order_b != order_c:
        return "reinterpret"
    return "repack"


def parse_full_log(path: Path, *, expected_direction: str) -> pd.DataFrame:
    section = _extract_summary_section(path)
    rows: list[ParsedRow] = []
    for line in section.splitlines():
        match = _RUNTIME_ROW_RE.match(line)
        if match is None:
            continue
        direction = match.group("direction")
        if direction != expected_direction:
            continue
        rows.append(
            ParsedRow(
                config=match.group("config"),
                direction=direction,
                k=int(match.group("k")),
                mean_ms=float(match.group("mean_ms")),
                err_trials=match.group("err_trials"),
                note=(match.group("note") or "").strip(),
                plan_fields=_parse_plan_literal(_extract_plan_literal(match.group("config"), direction)),
            )
        )
    if not rows:
        raise ValueError(f"{path}: no baseline runtime rows found for direction={expected_direction}")

    frame = pd.DataFrame(
        [
            {
                "config": row.config,
                "direction": row.direction,
                "k": row.k,
                "mean_ms": row.mean_ms,
                "err_trials": row.err_trials,
                "note": row.note,
                **row.plan_fields,
            }
            for row in rows
        ]
    )
    return derive_features(frame)


def derive_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["log_ms"] = np.log(df["mean_ms"])
    df["row_errors"] = df["err_trials"].map(lambda value: int(str(value).split("/", 1)[0]))
    df["mode"] = df["note"].map(_extract_mode)
    df["logical_row_compressed"] = df.apply(
        lambda row: (row["store"] == "N" and row["fmt"] == "CSR")
        or (row["store"] == "T" and row["fmt"] == "CSC"),
        axis=1,
    )
    df["dense_case"] = df.apply(_dense_case, axis=1)
    return df


def validate_design(df: pd.DataFrame, *, label: str) -> None:
    if df["k"].nunique() != 1:
        raise ValueError(f"{label}: expected a single runtime-k, saw {sorted(df['k'].unique())}")
    modes = sorted(set(df["mode"]))
    if len(modes) != 1:
        raise ValueError(f"{label}: expected a single execution mode, saw {modes}")
    if int(df["row_errors"].sum()) != 0:
        raise ValueError(f"{label}: expected zero runtime errors, saw {int(df['row_errors'].sum())}")


def fit_model(df: pd.DataFrame, formula: str):
    return smf.ols(formula, data=df).fit(cov_type="HC3")


def percent_effect(beta: float) -> float:
    return 100.0 * (math.exp(float(beta)) - 1.0)


def summarize_model(model) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, value in model.params.items():
        if name == "Intercept":
            continue
        rows.append(
            {
                "term": name,
                "coef_log": float(value),
                "pct_effect": percent_effect(float(value)),
                "pvalue": float(model.pvalues[name]),
            }
        )
    return pd.DataFrame(rows).sort_values("coef_log")


def top_configs(df: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    cols = [
        "mean_ms",
        "store",
        "fmt",
        "opA",
        "opB",
        "orderB",
        "orderC",
        "algo",
        "dense_case",
        "logical_row_compressed",
    ]
    return df.sort_values("mean_ms")[cols].head(limit).reset_index(drop=True)


def _format_float(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _frame_to_string(df: pd.DataFrame, *, float_digits: int = 4) -> str:
    if df.empty:
        return "<empty>"

    def _fmt(value):
        if isinstance(value, (float, np.floating)):
            return _format_float(float(value), digits=float_digits)
        return str(value)

    rendered = df.copy()
    for column in rendered.columns:
        rendered[column] = rendered[column].map(_fmt)
    return rendered.to_string(index=False)


def _print_direction_report(*, label: str, df: pd.DataFrame, top_n: int) -> None:
    validate_design(df, label=label)
    old_model = fit_model(df, _OLD_MODEL_FORMULA)
    final_model = fit_model(df, _FINAL_MODEL_FORMULA)
    raw_dense_model = fit_model(df, _RAW_DENSE_MODEL_FORMULA)

    print()
    print("=" * 88)
    print(f"{label.upper()} ANALYSIS")
    print("=" * 88)
    print(
        f"Rows={len(df)} runtime_k={int(df['k'].iloc[0])} "
        f"mode={df['mode'].iloc[0]} errors={int(df['row_errors'].sum())}"
    )
    print(
        "Old model: "
        f"R2={_format_float(old_model.rsquared)} "
        f"adj_R2={_format_float(old_model.rsquared_adj)}"
    )
    print(
        "Final model: "
        f"R2={_format_float(final_model.rsquared)} "
        f"adj_R2={_format_float(final_model.rsquared_adj)} "
        f"AIC={_format_float(final_model.aic, digits=2)} "
        f"BIC={_format_float(final_model.bic, digits=2)}"
    )
    print(
        "Raw dense-layout model: "
        f"R2={_format_float(raw_dense_model.rsquared)} "
        f"adj_R2={_format_float(raw_dense_model.rsquared_adj)}"
    )
    print()
    print("Final model terms:")
    print(_frame_to_string(summarize_model(final_model), float_digits=6))
    print()
    print(f"Top {top_n} overall:")
    print(_frame_to_string(top_configs(df, limit=top_n), float_digits=6))


def main() -> None:
    args = parse_args()
    up = parse_full_log(args.input_up, expected_direction="up")
    down = parse_full_log(args.input_down, expected_direction="down")

    print("cuSPARSE full-log timing analysis")
    print(f"  up:   {args.input_up}")
    print(f"  down: {args.input_down}")
    _print_direction_report(label="up", df=up, top_n=int(args.top_n))
    _print_direction_report(label="down", df=down, top_n=int(args.top_n))


if __name__ == "__main__":
    main()
