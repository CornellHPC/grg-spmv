"""Benchmark reporting and output-equivalence helpers."""

from __future__ import annotations

import numpy as np

from scripts.bench.cli import OUTPUT_ATOL, OUTPUT_RTOL
from scripts.bench.configs import BenchConfig


def bytes_to_gib(value: int | float) -> float:
    return float(value) / float(1024 ** 3)


def _common_value(values: list[object]) -> object | None:
    if not values:
        return None
    head = values[0]
    if all(value == head for value in values[1:]):
        return head
    return None


def _parse_plan_text(plan_text: str | None) -> tuple[tuple[str, str], ...] | None:
    if plan_text is None:
        return None
    text = str(plan_text).strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"Unexpected benchmark plan text {plan_text!r}")
    body = text[1:-1]
    if not body:
        return tuple()
    parts: list[tuple[str, str]] = []
    for token in body.split(","):
        key, value = token.split("=", 1)
        parts.append((key, value))
    return tuple(parts)


def _format_plan_fields(fields: tuple[tuple[str, str], ...]) -> str:
    return "[" + ",".join(f"{key}={value}" for key, value in fields) + "]"


def _format_side(side: str, fields: tuple[tuple[str, str], ...] | None) -> str:
    if fields is None:
        return f"{side}=<unspecified>"
    return f"{side}{_format_plan_fields(fields)}"


def summarize_config_display(configs: list[BenchConfig]) -> tuple[list[str], dict[str, str]]:
    if not configs:
        return [], {}
    label_counts: dict[str, int] = {}
    for entry in configs:
        label = str(entry.label)
        label_counts[label] = label_counts.get(label, 0) + 1
    duplicates = sorted(label for label, count in label_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Benchmark config labels must be unique, got duplicates: {duplicates}")

    backend_common = _common_value([entry.backend_name for entry in configs])
    ordering_common = _common_value([entry.ordering for entry in configs])
    intra_common = _common_value([entry.intra_block_ordering for entry in configs])
    instrumentation_common = _common_value([bool(entry.instrumentation) for entry in configs])
    parsed_up = [_parse_plan_text(entry.plan_up_text) for entry in configs]
    parsed_down = [_parse_plan_text(entry.plan_down_text) for entry in configs]

    def _common_side(plans: list[tuple[tuple[str, str], ...] | None]) -> tuple[tuple[str, str], ...] | None:
        if all(plan is None for plan in plans):
            return None
        if any(plan is None for plan in plans):
            return ()
        first = plans[0]
        assert first is not None
        common: list[tuple[str, str]] = []
        lookups = [dict(plan) for plan in plans if plan is not None]
        for key, value in first:
            if all(lookup.get(key) == value for lookup in lookups):
                common.append((key, value))
        return tuple(common)

    common_up = _common_side(parsed_up)
    common_down = _common_side(parsed_down)

    common_lines: list[str] = []
    if backend_common is not None:
        common_lines.append(f"backend={backend_common}")
    if ordering_common is not None:
        common_lines.append(f"ordering={ordering_common}")
    if intra_common is not None:
        common_lines.append(f"intra_block_ordering={intra_common}")
    if instrumentation_common is not None:
        common_lines.append(f"instrumentation={'on' if bool(instrumentation_common) else 'off'}")
    if all(plan is None for plan in parsed_up):
        common_lines.append("up=<unspecified>")
    elif common_up:
        common_lines.append(_format_side("up", common_up))
    if all(plan is None for plan in parsed_down):
        common_lines.append("down=<unspecified>")
    elif common_down:
        common_lines.append(_format_side("down", common_down))

    display_by_label: dict[str, str] = {}
    for entry, up_fields, down_fields in zip(configs, parsed_up, parsed_down, strict=True):
        parts: list[str] = []
        if backend_common is None:
            parts.append(f"backend={entry.backend_name}")
        if ordering_common is None:
            parts.append(f"ordering={entry.ordering}")
        if intra_common is None:
            parts.append(f"intra_block_ordering={entry.intra_block_ordering}")
        if instrumentation_common is None:
            parts.append(f"instr={'on' if entry.instrumentation else 'off'}")

        if not all(plan is None for plan in parsed_up):
            if up_fields is None:
                if common_up != up_fields:
                    parts.append("up=<unspecified>")
            else:
                common_keys = set() if common_up is None else {key for key, _ in common_up}
                diff_fields = tuple((key, value) for key, value in up_fields if key not in common_keys)
                if common_up == () or diff_fields:
                    parts.append(_format_side("up", diff_fields if diff_fields else up_fields))

        if not all(plan is None for plan in parsed_down):
            if down_fields is None:
                if common_down != down_fields:
                    parts.append("down=<unspecified>")
            else:
                common_keys = set() if common_down is None else {key for key, _ in common_down}
                diff_fields = tuple((key, value) for key, value in down_fields if key not in common_keys)
                if common_down == () or diff_fields:
                    parts.append(_format_side("down", diff_fields if diff_fields else down_fields))

        display_by_label[entry.label] = " ".join(parts) if parts else "-"

    if len(configs) > 1:
        display_counts: dict[str, int] = {}
        for entry in configs:
            display = str(display_by_label[entry.label])
            display_counts[display] = display_counts.get(display, 0) + 1
        for entry in configs:
            display = str(display_by_label[entry.label])
            if display == "-" or display_counts[display] > 1:
                display_by_label[entry.label] = str(entry.label)

    return common_lines, display_by_label


def print_common_config(common_lines: list[str]) -> None:
    if not common_lines:
        return
    print("\nCommon config:")
    for line in common_lines:
        print(f"  {line}")


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


def print_runtime_table(
    rows: list[dict[str, object]],
    *,
    config_display: dict[str, str] | None = None,
    skip_note: bool = False,
) -> None:
    if not rows:
        print("No benchmark runtime results.")
        return

    title = "BENCHMARK RUNTIME SUMMARY (time + correctness diagnostics)"
    rendered: list[tuple[str, str, str, str, str, str, str, str, str]] = []
    for row in rows:
        raw_config = str(row["config"])
        config = raw_config if config_display is None else str(config_display.get(raw_config, raw_config))
        scenario = str(row["scenario"])
        direction = str(row["direction"])
        k_value = row.get("k")
        k_cell = "-" if k_value is None else str(int(k_value))
        if "skip" in row:
            call_cell = f"SKIP ({row['skip']})"
            err_cell = "-"
            abs_err_cell = "-"
            rel_err_cell = "-"
        else:
            call_cell = f"{float(row['call_ms_mean']):.4f}+/-{float(row['call_ms_std']):.4f}"
            err_cell = f"{int(row.get('intra_errors', 0))}/{int(row.get('intra_trials', 0))}"
            abs_avg = row.get("abs_err_avg")
            abs_max = row.get("abs_err_max")
            rel_avg = row.get("rel_err_avg")
            rel_max = row.get("rel_err_max")
            abs_err_cell = "-" if abs_avg is None or abs_max is None else f"{float(abs_avg):.3e}/{float(abs_max):.3e}"
            rel_err_cell = "-" if rel_avg is None or rel_max is None else f"{float(rel_avg):.3e}/{float(rel_max):.3e}"
        note = str(row.get("note", ""))
        intra_fail = row.get("intra_fail_indices", [])
        if isinstance(intra_fail, list) and intra_fail:
            tag = f"intra_fail={','.join(str(v) for v in intra_fail)}"
            note = f"{note} | {tag}" if note else tag
        rendered.append((config, scenario, direction, k_cell, call_cell, err_cell, abs_err_cell, rel_err_cell, note))

    config_w = max(len("Config"), max(len(item[0]) for item in rendered))
    scenario_w = max(len("Scenario"), max(len(item[1]) for item in rendered))
    direction_w = max(len("Direction"), max(len(item[2]) for item in rendered))
    k_w = max(len("k"), max(len(item[3]) for item in rendered))
    call_w = max(len("Call ms"), max(len(item[4]) for item in rendered))
    err_w = max(len("Err/Trials"), max(len(item[5]) for item in rendered))
    abs_w = max(len("Abs Err (avg/max)"), max(len(item[6]) for item in rendered))
    rel_w = max(len("Rel Err (avg/max)"), max(len(item[7]) for item in rendered))

    header = f"{'Config':<{config_w}} {'Scenario':<{scenario_w}} {'Direction':<{direction_w}} {'k':>{k_w}} "
    header += f"{'Call ms':>{call_w}} {'Err/Trials':>{err_w}} {'Abs Err (avg/max)':>{abs_w}} {'Rel Err (avg/max)':>{rel_w}}"
    if not skip_note:
        header += " Note"
    width = max(len(title), len(header))

    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(header)
    print("-" * width)
    for config, scenario, direction, k_cell, call_cell, err_cell, abs_err_cell, rel_err_cell, note in rendered:
        line = (
            f"{config:<{config_w}} {scenario:<{scenario_w}} {direction:<{direction_w}} {k_cell:>{k_w}} "
            f"{call_cell:>{call_w}} {err_cell:>{err_w}} {abs_err_cell:>{abs_w}} {rel_err_cell:>{rel_w}}"
        )
        if not skip_note:
            line += f" {note}"
        print(line)


def print_memory_table(
    rows: list[dict[str, object]],
    *,
    config_display: dict[str, str] | None = None,
    skip_note: bool = False,
) -> None:
    if not rows:
        print("No benchmark memory results.")
        return

    title = "BENCHMARK MEMORY SUMMARY (compact live allocation tree)"
    rendered: list[tuple[int, str, str, str, str, str, str, str, str, str, str, str, str, str]] = []
    for row in rows:
        raw_config = str(row["config"])
        config = raw_config if config_display is None else str(config_display.get(raw_config, raw_config))
        scenario = str(row["scenario"])
        case_direction = str(row["case_direction"])
        case_k_value = row.get("case_k")
        case_k_cell = "-" if case_k_value is None else str(int(case_k_value))
        node = str(row["node"])
        parent = str(row["parent"])
        level = int(row.get("level", 0))
        if "skip" in row:
            gib_cell = "SKIP"
            space = "-"
        else:
            gib_cell = f"{float(row['gib']):.6f}"
            space = str(row["space"])
        rendered.append(
            (
                level,
                config,
                scenario,
                case_direction,
                case_k_cell,
                node,
                parent,
                gib_cell,
                space,
                str(row.get("owner", "")),
                str(row.get("active", "")),
                str(row.get("retention", "")),
                str(row.get("kinds", "")),
                str(row.get("note", "")),
            )
        )

    config_w = max(len("Config"), max(len(item[1]) for item in rendered))
    scenario_w = max(len("Scenario"), max(len(item[2]) for item in rendered))
    direction_w = max(len("CaseDir"), max(len(item[3]) for item in rendered))
    k_w = max(len("CaseK"), max(len(item[4]) for item in rendered))
    node_w = max(len("Node"), max(len(item[5]) for item in rendered))
    parent_w = max(len("Parent"), max(len(item[6]) for item in rendered))
    gib_w = max(len("GiB"), max(len(item[7]) for item in rendered))
    space_w = max(len("Space"), max(len(item[8]) for item in rendered))
    owner_w = max(len("Owner"), max(len(item[9]) for item in rendered))
    active_w = max(len("Active"), max(len(item[10]) for item in rendered))
    retention_w = max(len("Retention"), max(len(item[11]) for item in rendered))
    kinds_w = max(len("Kinds"), max(len(item[12]) for item in rendered))

    header = f"{'Config':<{config_w}} {'Scenario':<{scenario_w}} {'CaseDir':<{direction_w}} {'CaseK':>{k_w}} "
    header += (
        f"{'Node':<{node_w}} {'Parent':<{parent_w}} {'GiB':>{gib_w}} {'Space':<{space_w}} "
        f"{'Owner':<{owner_w}} {'Active':<{active_w}} {'Retention':<{retention_w}} {'Kinds':<{kinds_w}}"
    )
    if not skip_note:
        header += " Note"
    width = max(len(title), len(header))

    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(header)
    print("-" * width)
    current_level: int | None = None
    for level, config, scenario, case_direction, case_k_cell, node, parent, gib_cell, space, owner, active, retention, kinds, note in rendered:
        if current_level is not None and level != current_level:
            print(f" LEVEL {level} ".center(width, "-"))
        current_level = level
        line = (
            f"{config:<{config_w}} {scenario:<{scenario_w}} {case_direction:<{direction_w}} {case_k_cell:>{k_w}} "
            f"{node:<{node_w}} {parent:<{parent_w}} {gib_cell:>{gib_w}} {space:<{space_w}} "
            f"{owner:<{owner_w}} {active:<{active_w}} {retention:<{retention_w}} {kinds:<{kinds_w}}"
        )
        if not skip_note:
            line += f" {note}"
        print(line)


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
        if "skip" in row:
            continue
        errors += int(row.get("intra_errors", 0))
        trials += int(row.get("intra_trials", 0))
    return errors, trials


__all__ = [
    "bytes_to_gib",
    "compare_outputs",
    "evaluate_output_equivalence",
    "print_common_config",
    "print_memory_table",
    "print_runtime_table",
    "summarize_config_display",
    "summarize_intra_diagnostics",
]
