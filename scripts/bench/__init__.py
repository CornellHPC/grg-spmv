"""Shared benchmark helpers for backend-specific benchmark scripts."""

from __future__ import annotations

import logging
from dataclasses import fields
from pathlib import Path
from time import perf_counter

import numpy as np

DTYPE = np.float64
INDEX_DTYPE = np.uintp
DEFAULT_MATMUL_OPTIONS = (
    "baseline",
    "by_individual",
    "init_xtx",
    "init_vector",
    "init_matrix",
    "miss",
)
_VALID_FMTS = {"csr", "csc", "coo", "none"}
LOGGER = logging.getLogger("scripts.bench")
OUTPUT_ATOL = 1e-8
OUTPUT_RTOL = 1e-5


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


def parse_k_hints(raw: str) -> list[int | None]:
    out: list[int | None] = []
    for token in raw.split(","):
        tok = token.strip().lower()
        if not tok:
            continue
        if tok == "none":
            out.append(None)
            continue
        value = int(tok)
        if value <= 0:
            raise ValueError(f"--k-hints values must be positive or 'none', got {value}")
        out.append(value)
    if not out:
        raise ValueError("--k-hints must contain at least one value")
    return out


def parse_matmul_options(raw: str) -> list[str]:
    tokens = [tok.strip().lower() for tok in raw.split(",") if tok.strip()]
    if not tokens or "all" in tokens:
        return list(DEFAULT_MATMUL_OPTIONS)
    valid = set(DEFAULT_MATMUL_OPTIONS)
    unknown = sorted(tok for tok in tokens if tok not in valid)
    if unknown:
        raise ValueError(f"Unknown matmul option(s): {', '.join(unknown)}")
    return tokens


def parse_fmt_up_down(raw: str) -> list[tuple[str | None, str | None]]:
    pairs: list[tuple[str | None, str | None]] = []
    tokens = [tok.strip() for tok in raw.split("/") if tok.strip()]
    if not tokens:
        raise ValueError("--fmt-up-down must contain at least one pair")

    for token in tokens:
        parts = [part.strip().lower() for part in token.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid format pair {token!r}; expected '<fmt_up>,<fmt_down>'")
        up_raw, down_raw = parts
        if up_raw not in _VALID_FMTS:
            raise ValueError(f"Unsupported fmt_up {up_raw!r}; expected one of {sorted(_VALID_FMTS)}")
        if down_raw not in _VALID_FMTS:
            raise ValueError(f"Unsupported fmt_down {down_raw!r}; expected one of {sorted(_VALID_FMTS)}")

        up = None if up_raw == "none" else up_raw
        down = None if down_raw == "none" else down_raw
        if up is None and down is None:
            raise ValueError("Invalid format pair 'none,none'; at least one side must be non-none")
        pairs.append((up, down))

    return pairs


def expand_mkl_configs(
    thread_counts: list[int],
    fmt_pairs: list[tuple[str | None, str | None]],
    k_hints: list[int | None],
    log_level: str,
) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for fmt_up, fmt_down in fmt_pairs:
        for n_threads in thread_counts:
            for k_hint in k_hints:
                label = (
                    f"mkl-fu={'none' if fmt_up is None else fmt_up}-"
                    f"fd={'none' if fmt_down is None else fmt_down}-"
                    f"t={n_threads}-kh={'none' if k_hint is None else k_hint}"
                )
                cfg = {
                    "type": "mkl",
                    "n_threads": n_threads,
                    "fmt_up": fmt_up,
                    "fmt_down": fmt_down,
                    "k_hint": k_hint,
                    "log_level": str(log_level).upper(),
                }
                configs.append({"label": label, "config": cfg})
    return configs


def expand_cusparse_configs(
    fmt_pairs: list[tuple[str | None, str | None]],
    k_hints: list[int | None],
    log_level: str,
) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for fmt_up, fmt_down in fmt_pairs:
        for k_hint in k_hints:
            label = (
                f"cusparse-fu={'none' if fmt_up is None else fmt_up}-"
                f"fd={'none' if fmt_down is None else fmt_down}-"
                f"kh={'none' if k_hint is None else k_hint}"
            )
            cfg = {
                "type": "cusparse",
                "fmt_up": fmt_up,
                "fmt_down": fmt_down,
                "k_hint": k_hint,
                "algo_up": "default",
                "algo_down": "default",
                "log_level": str(log_level).upper(),
            }
            configs.append({"label": label, "config": cfg})
    return configs


def format_dry_run_line(config_entry: dict[str, object], ks: list[int], options: list[str]) -> str:
    cfg = config_entry["config"]
    assert isinstance(cfg, dict)

    items = [
        f"backend={cfg['type']}",
        f"fmt_up={'none' if cfg.get('fmt_up') is None else cfg.get('fmt_up')}",
        f"fmt_down={'none' if cfg.get('fmt_down') is None else cfg.get('fmt_down')}",
        f"k_hint={'none' if cfg.get('k_hint') is None else cfg.get('k_hint')}",
        f"log_level={cfg.get('log_level')}",
    ]
    if cfg.get("type") == "mkl":
        items.append(f"threads={cfg.get('n_threads')}")
    items.append("ks=" + ",".join(str(k) for k in ks))
    items.append("scenarios=" + ",".join(options))
    return " ".join(items)


def _bytes_to_gib(value: int) -> float:
    return float(int(value) / (1024.0**3))


def _format_static_note(prefix: str, static_bytes) -> str:
    parts: list[tuple[str, int]] = []
    for f in fields(static_bytes):
        value = int(getattr(static_bytes, f.name))
        if value > 0:
            parts.append((f.name, value))
    if not parts:
        return f"{prefix}: none"
    body = ", ".join(f"{name}={_bytes_to_gib(value):.6f}GiB" for name, value in parts)
    return f"{prefix}: {body}"


def _assert_outputs_allclose(
    ref: np.ndarray,
    arr: np.ndarray,
    *,
    config: str,
    scenario: str,
    direction: str,
    k: int,
    call_label: str,
    atol: float,
    rtol: float,
) -> None:
    if ref.shape != arr.shape:
        raise AssertionError(
            f"Output shape mismatch for {config} scenario={scenario} direction={direction} k={k} "
            f"at {call_label}: expected {ref.shape}, got {arr.shape}"
        )
    if np.allclose(arr, ref, atol=atol, rtol=rtol):
        return

    abs_diff = np.abs(arr - ref)
    max_abs = float(np.max(abs_diff))
    denom = np.maximum(np.abs(ref), 1e-30)
    max_rel = float(np.max(abs_diff / denom))
    raise AssertionError(
        f"Output mismatch for {config} scenario={scenario} direction={direction} k={k} at {call_label}: "
        f"max_abs={max_abs} max_rel={max_rel} atol={atol} rtol={rtol}"
    )


def _validate_and_extract_runtime_memory(
    *,
    call_slice,
    direction: str,
    k: int,
    n_warmup: int,
    n_trials: int,
) -> tuple[int, int, str]:
    expected = n_warmup + n_trials
    if len(call_slice) != expected:
        raise AssertionError(
            f"Expected {expected} memory records for direction={direction}, k={k}, got {len(call_slice)}"
        )

    expected_stage = "run_up" if direction == "up" else "run_down"
    for idx, record in enumerate(call_slice):
        if record.stage != expected_stage:
            raise AssertionError(
                f"Unexpected stage in memory record {idx}: expected {expected_stage}, got {record.stage}"
            )
        if int(record.runtime_k) != int(k):
            raise AssertionError(
                f"Unexpected runtime_k in memory record {idx}: expected {k}, got {record.runtime_k}"
            )
        rec_direction = str(record.meta.get("direction", ""))
        if rec_direction != direction:
            raise AssertionError(
                f"Unexpected direction in memory record {idx}: expected {direction}, got {rec_direction}"
            )

    timed_records = call_slice[n_warmup:]
    if len(timed_records) != n_trials:
        raise AssertionError(
            f"Expected {n_trials} timed memory records for direction={direction}, k={k}, got {len(timed_records)}"
        )

    host_values = [int(record.host.total()) for record in timed_records]
    device_values = [int(record.device.total()) for record in timed_records]
    modes = [str(record.meta.get("mode", "n/a")) for record in timed_records]

    if len(set(host_values)) != 1 or len(set(device_values)) != 1:
        raise AssertionError(
            f"Memory totals vary across timed trials for direction={direction}, k={k}: "
            f"host={host_values}, device={device_values}"
        )
    if len(set(modes)) != 1:
        raise AssertionError(f"Execution mode varies across timed trials for direction={direction}, k={k}: {modes}")

    return host_values[0], device_values[0], modes[0]


def benchmark_config(
    *,
    op,
    label: str,
    ks: list[int],
    options: list[str],
    n_trials: int,
    n_warmup: int,
    seed_base: int,
    output_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs_by_k: dict[int, dict[str, np.ndarray | None]] = {}
    for k in ks:
        rng = np.random.default_rng(seed_base + int(k))
        inputs: dict[str, np.ndarray | None] = {
            "up_sample": rng.standard_normal((k, op.n), dtype=DTYPE),
            "down": rng.standard_normal((k, op.m), dtype=DTYPE),
            "init_vec": rng.standard_normal(k, dtype=DTYPE),
            "init_mat": rng.standard_normal((k, op.K), dtype=DTYPE),
            "miss_down": rng.standard_normal((k, op.m), dtype=DTYPE),
            "miss_up": np.zeros((k, op.m), dtype=DTYPE),
            "up_indiv": None,
        }
        if op.num_individuals != op.n:
            inputs["up_indiv"] = rng.standard_normal((k, op.num_individuals), dtype=DTYPE)
        inputs_by_k[int(k)] = inputs

    summary_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []

    host_static = op._backend.mem_usage.host_static
    device_static = op._backend.mem_usage.device_static
    summary_rows.append(
        {
            "config": label,
            "scenario": "static",
            "direction": "-",
            "k": None,
            "call_ms_mean": None,
            "call_ms_std": None,
            "host_gib": _bytes_to_gib(host_static.total()),
            "device_gib": _bytes_to_gib(device_static.total()),
            "note": _format_static_note("actual_host", host_static)
            + " | "
            + _format_static_note("actual_device", device_static),
        }
    )

    est_host, est_device = op._backend.estimate_static_bytes()
    summary_rows.append(
        {
            "config": label,
            "scenario": "static_est",
            "direction": "-",
            "k": None,
            "call_ms_mean": None,
            "call_ms_std": None,
            "host_gib": _bytes_to_gib(est_host.total()),
            "device_gib": _bytes_to_gib(est_device.total()),
            "note": _format_static_note("est_host", est_host) + " | " + _format_static_note("est_device", est_device),
        }
    )

    for scenario in options:
        progress(f"{label}: scenario={scenario} start")
        for direction in ("up", "down"):
            for k in ks:
                inputs = inputs_by_k[int(k)]
                matrix: np.ndarray | None = None
                skip_reason: str | None = None

                if scenario == "baseline":
                    matrix = inputs["up_sample"] if direction == "up" else inputs["down"]
                    kwargs_factory = lambda: {}
                elif scenario == "by_individual":
                    up_indiv = inputs["up_indiv"]
                    if up_indiv is None:
                        skip_reason = "num_individuals == num_samples"
                        kwargs_factory = lambda: {}
                    else:
                        matrix = up_indiv if direction == "up" else inputs["down"]
                        kwargs_factory = lambda: {"by_individual": True}
                elif scenario == "init_xtx":
                    matrix = inputs["up_sample"] if direction == "up" else inputs["down"]
                    kwargs_factory = lambda: {"init": "xtx"}
                elif scenario == "init_vector":
                    init_vec = inputs["init_vec"]
                    matrix = inputs["up_sample"] if direction == "up" else inputs["down"]
                    kwargs_factory = lambda init_vec=init_vec: {"init": init_vec}
                elif scenario == "init_matrix":
                    init_mat = inputs["init_mat"]
                    matrix = inputs["up_sample"] if direction == "up" else inputs["down"]
                    kwargs_factory = lambda init_mat=init_mat: {"init": init_mat}
                elif scenario == "miss":
                    if op.sel_miss.nnz == 0:
                        skip_reason = "GRG has no missingness selector entries"
                        kwargs_factory = lambda: {}
                    elif direction == "up":
                        miss_up = inputs["miss_up"]

                        def kwargs_factory(miss_up=miss_up):
                            assert isinstance(miss_up, np.ndarray)
                            miss_up.fill(0.0)
                            return {"miss": miss_up}

                        matrix = inputs["up_sample"]
                    else:
                        miss_down = inputs["miss_down"]
                        matrix = inputs["down"]
                        kwargs_factory = lambda miss_down=miss_down: {"miss": miss_down}
                else:
                    raise ValueError(f"Unhandled matmul scenario: {scenario}")

                if skip_reason is not None:
                    summary_rows.append(
                        {
                            "config": label,
                            "scenario": scenario,
                            "direction": direction,
                            "k": int(k),
                            "skip": skip_reason,
                            "call_ms_mean": None,
                            "call_ms_std": None,
                            "host_gib": None,
                            "device_gib": None,
                            "note": f"skip: {skip_reason}",
                        }
                    )
                    continue

                if direction not in {"up", "down"}:
                    raise ValueError(f"Unknown direction {direction!r}")
                assert isinstance(matrix, np.ndarray)

                progress(f"{label}: scenario={scenario} direction={direction} k={k} start")

                calls = op._backend.mem_usage.calls
                start_idx = len(calls)
                ref_output: np.ndarray | None = None

                for warm_idx in range(n_warmup):
                    warm = np.asarray(op.matmul(matrix, direction, **kwargs_factory()))
                    if ref_output is None:
                        ref_output = warm
                    else:
                        _assert_outputs_allclose(
                            ref_output,
                            warm,
                            config=label,
                            scenario=scenario,
                            direction=direction,
                            k=int(k),
                            call_label=f"warmup[{warm_idx}]",
                            atol=OUTPUT_ATOL,
                            rtol=OUTPUT_RTOL,
                        )

                times: list[float] = []
                for trial_idx in range(n_trials):
                    t0 = perf_counter()
                    result = np.asarray(op.matmul(matrix, direction, **kwargs_factory()))
                    times.append(perf_counter() - t0)
                    if ref_output is None:
                        ref_output = result
                    else:
                        _assert_outputs_allclose(
                            ref_output,
                            result,
                            config=label,
                            scenario=scenario,
                            direction=direction,
                            k=int(k),
                            call_label=f"trial[{trial_idx}]",
                            atol=OUTPUT_ATOL,
                            rtol=OUTPUT_RTOL,
                        )

                end_idx = len(calls)
                host_bytes, device_bytes, mode = _validate_and_extract_runtime_memory(
                    call_slice=calls[start_idx:end_idx],
                    direction=direction,
                    k=int(k),
                    n_warmup=n_warmup,
                    n_trials=n_trials,
                )

                assert ref_output is not None
                ms = np.asarray(times, dtype=np.float64) * 1000.0
                mean_ms = float(np.mean(ms))
                std_ms = float(np.std(ms))

                config_token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_") or "value"
                scenario_token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in scenario).strip("_") or "value"
                direction_token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in direction).strip("_") or "value"
                output_path = output_dir / f"{len(output_rows):06d}_{config_token}_{scenario_token}_{direction_token}_k{int(k)}.npy"
                np.save(output_path, ref_output, allow_pickle=False)

                summary_rows.append(
                    {
                        "config": label,
                        "scenario": scenario,
                        "direction": direction,
                        "k": int(k),
                        "call_ms_mean": mean_ms,
                        "call_ms_std": std_ms,
                        "host_gib": _bytes_to_gib(host_bytes),
                        "device_gib": _bytes_to_gib(device_bytes),
                        "note": f"mode={mode}",
                    }
                )
                output_rows.append(
                    {
                        "config": label,
                        "scenario": scenario,
                        "direction": direction,
                        "k": int(k),
                        "path": str(output_path),
                    }
                )

    return summary_rows, output_rows


def print_summary_table(rows: list[dict[str, object]], *, skip_note: bool = False) -> None:
    if not rows:
        print("No benchmark results.")
        return

    title = "BENCHMARK SUMMARY (time + memory)"
    rendered: list[tuple[str, str, str, str, str, str, str, str]] = []
    for row in rows:
        config = str(row["config"])
        scenario = str(row["scenario"])
        direction = str(row["direction"])
        k_value = row.get("k")
        k_cell = "-" if k_value is None else str(int(k_value))
        if scenario in {"static", "static_est"}:
            call_cell = "-"
        elif "skip" in row:
            call_cell = f"SKIP ({row['skip']})"
        else:
            call_cell = f"{float(row['call_ms_mean']):.4f}+/-{float(row['call_ms_std']):.4f}"
        host = row.get("host_gib")
        device = row.get("device_gib")
        host_cell = "SKIP" if host is None else f"{float(host):.6f}"
        device_cell = "SKIP" if device is None else f"{float(device):.6f}"
        note = str(row.get("note", ""))
        rendered.append((config, scenario, direction, k_cell, call_cell, host_cell, device_cell, note))

    config_w = max(len("Config"), max(len(item[0]) for item in rendered))
    scenario_w = max(len("Scenario"), max(len(item[1]) for item in rendered))
    direction_w = max(len("Direction"), max(len(item[2]) for item in rendered))
    k_w = max(len("k"), max(len(item[3]) for item in rendered))
    call_w = max(len("Call ms"), max(len(item[4]) for item in rendered))
    host_w = max(len("Host GiB"), max(len(item[5]) for item in rendered))
    device_w = max(len("Device GiB"), max(len(item[6]) for item in rendered))

    header = f"{'Config':<{config_w}} {'Scenario':<{scenario_w}} "
    header += f"{'Direction':<{direction_w}} {'k':>{k_w}} {'Call ms':>{call_w}} "
    header += f"{'Host GiB':>{host_w}} {'Device GiB':>{device_w}}"
    if not skip_note:
        header += " Note"
    width = max(len(title), len(header))

    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(header)
    print("-" * width)

    for config, scenario, direction, k_cell, call_cell, host_cell, device_cell, note in rendered:
        row = (
            f"{config:<{config_w}} {scenario:<{scenario_w}} "
            f"{direction:<{direction_w}} {k_cell:>{k_w}} {call_cell:>{call_w}} "
            f"{host_cell:>{host_w}} {device_cell:>{device_w}}"
        )
        if not skip_note:
            row += f" {note}"
        print(row)


def assert_output_equivalence(
    rows: list[dict[str, object]],
    *,
    atol: float = OUTPUT_ATOL,
    rtol: float = OUTPUT_RTOL,
) -> None:
    classes: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        scenario = str(row["scenario"])
        direction = str(row["direction"])
        k = int(row["k"])
        key = ("baseline" if scenario == "miss" and direction == "up" else scenario, direction, k)
        classes.setdefault(key, []).append(row)

    compared = 0
    checked_classes = 0
    for key in sorted(classes):
        points = classes[key]
        if len(points) < 2:
            continue
        checked_classes += 1
        ref_row = points[0]
        ref_cfg = str(ref_row["config"])
        ref_scenario = str(ref_row["scenario"])
        ref_path = Path(str(ref_row["path"]))
        ref = np.load(ref_path, allow_pickle=False)
        for row in points[1:]:
            compared += 1
            cfg = str(row["config"])
            scenario = str(row["scenario"])
            arr_path = Path(str(row["path"]))
            arr = np.load(arr_path, allow_pickle=False)
            _assert_outputs_allclose(
                ref,
                arr,
                config=cfg,
                scenario=scenario,
                direction=key[1],
                k=int(key[2]),
                call_label=f"cross-config (ref={ref_cfg}:{ref_scenario} from {ref_path.name})",
                atol=atol,
                rtol=rtol,
            )

    print(
        "\nOutput equivalence passed: "
        f"classes={checked_classes}, comparisons={compared}, atol={atol}, rtol={rtol}"
    )


__all__ = [
    "DTYPE",
    "INDEX_DTYPE",
    "DEFAULT_MATMUL_OPTIONS",
    "OUTPUT_ATOL",
    "OUTPUT_RTOL",
    "configure_logging",
    "progress",
    "parse_csv_ints",
    "parse_k_hints",
    "parse_matmul_options",
    "parse_fmt_up_down",
    "expand_mkl_configs",
    "expand_cusparse_configs",
    "format_dry_run_line",
    "_validate_and_extract_runtime_memory",
    "benchmark_config",
    "print_summary_table",
    "assert_output_equivalence",
]
