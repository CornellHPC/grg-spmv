"""Benchmark reporting and output-equivalence helpers."""

from __future__ import annotations

from dataclasses import fields
import numpy as np

from scripts.bench.cli import OUTPUT_ATOL, OUTPUT_RTOL


def bytes_to_gib(value: int | float) -> float:
    return float(value) / float(1024 ** 3)


def format_static_note(prefix: str, values) -> str:
    parts = []
    for f in fields(type(values)):
        value = int(getattr(values, f.name))
        if value > 0:
            parts.append((f.name, value))
    if not parts:
        return f"{prefix}: none"
    body = ", ".join(f"{name}={bytes_to_gib(value):.6f}GiB" for name, value in parts)
    return f"{prefix}: {body}"


def skip_summary_row(*, label: str, scenario: str, direction: str, k: int, reason: str) -> dict[str, object]:
    return {
        "config": label,
        "scenario": scenario,
        "direction": direction,
        "k": int(k),
        "skip": reason,
        "call_ms_mean": None,
        "call_ms_std": None,
        "host_gib": None,
        "device_gib": None,
        "note": f"skip: {reason}",
    }


def compare_outputs(
    ref: np.ndarray,
    arr: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, bool, float | None, float | None]:
    if ref.shape != arr.shape:
        return False, False, None, None
    abs_diff = np.abs(arr - ref)
    max_abs = float(np.max(abs_diff))
    denom = np.maximum(np.abs(ref), 1e-30)
    max_rel = float(np.max(abs_diff / denom))
    return bool(np.allclose(arr, ref, atol=atol, rtol=rtol)), True, max_abs, max_rel


def print_summary_table(rows: list[dict[str, object]], *, skip_note: bool = False) -> None:
    if not rows:
        print("No benchmark results.")
        return

    title = "BENCHMARK SUMMARY (time + memory + correctness diagnostics)"
    rendered: list[tuple[str, str, str, str, str, str, str, str, str, str, str]] = []
    for row in rows:
        config = str(row["config"])
        scenario = str(row["scenario"])
        direction = str(row["direction"])
        k_value = row.get("k")
        k_cell = "-" if k_value is None else str(int(k_value))
        if scenario in {"static", "static_est"}:
            call_cell = "-"
            err_cell = "-"
            abs_err_cell = "-"
            rel_err_cell = "-"
        elif "skip" in row:
            call_cell = f"SKIP ({row['skip']})"
            err_cell = "-"
            abs_err_cell = "-"
            rel_err_cell = "-"
        else:
            call_cell = f"{float(row['call_ms_mean']):.4f}+/-{float(row['call_ms_std']):.4f}"
            intra_errors = int(row.get("intra_errors", 0))
            intra_trials = int(row.get("intra_trials", 0))
            err_cell = f"{intra_errors}/{intra_trials}"
            abs_avg = row.get("abs_err_avg")
            abs_max = row.get("abs_err_max")
            rel_avg = row.get("rel_err_avg")
            rel_max = row.get("rel_err_max")
            abs_err_cell = "-" if abs_avg is None or abs_max is None else f"{float(abs_avg):.3e}/{float(abs_max):.3e}"
            rel_err_cell = "-" if rel_avg is None or rel_max is None else f"{float(rel_avg):.3e}/{float(rel_max):.3e}"
        host = row.get("host_gib")
        device = row.get("device_gib")
        host_cell = "SKIP" if host is None else f"{float(host):.6f}"
        device_cell = "SKIP" if device is None else f"{float(device):.6f}"
        note = str(row.get("note", ""))
        intra_fail = row.get("intra_fail_indices", [])
        tags: list[str] = []
        if isinstance(intra_fail, list) and intra_fail:
            tags.append(f"intra_fail={','.join(str(v) for v in intra_fail)}")
        if tags:
            note = f"{note} | {' | '.join(tags)}" if note else " | ".join(tags)
        rendered.append(
            (config, scenario, direction, k_cell, call_cell, err_cell, abs_err_cell, rel_err_cell, host_cell, device_cell, note)
        )

    config_w = max(len("Config"), max(len(item[0]) for item in rendered))
    scenario_w = max(len("Scenario"), max(len(item[1]) for item in rendered))
    direction_w = max(len("Direction"), max(len(item[2]) for item in rendered))
    k_w = max(len("k"), max(len(item[3]) for item in rendered))
    call_w = max(len("Call ms"), max(len(item[4]) for item in rendered))
    err_w = max(len("Err/Trials"), max(len(item[5]) for item in rendered))
    abs_w = max(len("Abs Err (avg/max)"), max(len(item[6]) for item in rendered))
    rel_w = max(len("Rel Err (avg/max)"), max(len(item[7]) for item in rendered))
    host_w = max(len("Host GiB"), max(len(item[8]) for item in rendered))
    device_w = max(len("Device GiB"), max(len(item[9]) for item in rendered))

    header = f"{'Config':<{config_w}} {'Scenario':<{scenario_w}} "
    header += f"{'Direction':<{direction_w}} {'k':>{k_w}} {'Call ms':>{call_w}} "
    header += f"{'Err/Trials':>{err_w}} {'Abs Err (avg/max)':>{abs_w}} {'Rel Err (avg/max)':>{rel_w}} "
    header += f"{'Host GiB':>{host_w}} {'Device GiB':>{device_w}}"
    if not skip_note:
        header += " Note"
    width = max(len(title), len(header))

    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(header)
    print("-" * width)

    for config, scenario, direction, k_cell, call_cell, err_cell, abs_err_cell, rel_err_cell, host_cell, device_cell, note in rendered:
        row = (
            f"{config:<{config_w}} {scenario:<{scenario_w}} "
            f"{direction:<{direction_w}} {k_cell:>{k_w}} {call_cell:>{call_w}} "
            f"{err_cell:>{err_w}} {abs_err_cell:>{abs_w}} {rel_err_cell:>{rel_w}} "
            f"{host_cell:>{host_w}} {device_cell:>{device_w}}"
        )
        if not skip_note:
            row += f" {note}"
        print(row)


def evaluate_output_equivalence(
    rows: list[dict[str, object]],
    *,
    atol: float = OUTPUT_ATOL,
    rtol: float = OUTPUT_RTOL,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    classes: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        scenario = str(row["scenario"])
        direction = str(row["direction"])
        k = int(row["k"])
        key = ("baseline" if scenario == "miss" and direction == "up" else scenario, direction, k)
        classes.setdefault(key, []).append(row)

    compared = 0
    checked_classes = 0
    errors = 0
    per_config: dict[str, dict[str, int]] = {}
    for key in sorted(classes):
        points = classes[key]
        if len(points) < 2:
            continue
        checked_classes += 1
        for i in range(len(points) - 1):
            row_i = points[i]
            cfg_i = str(row_i["config"])
            if cfg_i not in per_config:
                per_config[cfg_i] = {"failures": 0, "trials": 0}
            arr_i = np.asarray(row_i["output"])
            for j in range(i + 1, len(points)):
                row_j = points[j]
                cfg_j = str(row_j["config"])
                if cfg_j not in per_config:
                    per_config[cfg_j] = {"failures": 0, "trials": 0}
                arr_j = np.asarray(row_j["output"])
                compared += 1
                per_config[cfg_i]["trials"] += 1
                per_config[cfg_j]["trials"] += 1
                ok_ab, _, _, _ = compare_outputs(arr_i, arr_j, atol=atol, rtol=rtol)
                ok_ba, _, _, _ = compare_outputs(arr_j, arr_i, atol=atol, rtol=rtol)
                if not (ok_ab and ok_ba):
                    errors += 1
                    per_config[cfg_i]["failures"] += 1
                    per_config[cfg_j]["failures"] += 1

    print(
        "\nOutput equivalence diagnostics: "
        f"classes={checked_classes}, comparisons={compared}, errors={errors}, atol={atol}, rtol={rtol}"
    )
    return {"classes": checked_classes, "comparisons": compared, "errors": errors}, per_config


def summarize_intra_diagnostics(rows: list[dict[str, object]]) -> tuple[int, int]:
    errors = 0
    trials = 0
    for row in rows:
        scenario = str(row.get("scenario", ""))
        if scenario in {"static", "static_est"} or "skip" in row:
            continue
        errors += int(row.get("intra_errors", 0))
        trials += int(row.get("intra_trials", 0))
    return errors, trials


__all__ = [
    "bytes_to_gib",
    "compare_outputs",
    "evaluate_output_equivalence",
    "format_static_note",
    "print_summary_table",
    "skip_summary_row",
    "summarize_intra_diagnostics",
]
