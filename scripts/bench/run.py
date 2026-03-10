"""Benchmark execution and runtime diagnostics."""

from __future__ import annotations

import gc
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.bench.cases import build_case, build_inputs_by_k, configured_direction_names
from scripts.bench.cli import progress
from scripts.bench.configs import BenchConfig
from scripts.bench.report import (
    bytes_to_gib,
    compare_outputs,
    format_static_note,
    print_summary_table,
    skip_summary_row,
    summarize_intra_diagnostics,
    evaluate_output_equivalence,
)


@dataclass
class _IntraDiagnostics:
    ref_output: np.ndarray | None = None
    errors: int = 0
    trials: int = 0
    fail_indices: list[str] = field(default_factory=list)
    abs_sum: float = 0.0
    rel_sum: float = 0.0
    abs_max: float = 0.0
    rel_max: float = 0.0
    numeric_count: int = 0

    def observe(self, candidate: np.ndarray, *, tag: str, atol: float, rtol: float) -> None:
        if self.ref_output is None:
            self.ref_output = candidate
            return
        self.trials += 1
        ok, shape_ok, max_abs, max_rel = compare_outputs(
            self.ref_output,
            candidate,
            atol=atol,
            rtol=rtol,
        )
        if shape_ok:
            assert max_abs is not None and max_rel is not None
            self.numeric_count += 1
            self.abs_sum += max_abs
            self.rel_sum += max_rel
            self.abs_max = max(self.abs_max, max_abs)
            self.rel_max = max(self.rel_max, max_rel)
        if not ok:
            self.errors += 1
            self.fail_indices.append(tag)

    def summary(self) -> tuple[float | None, float | None, float | None, float | None]:
        if self.numeric_count == 0:
            return None, None, None, None
        return (
            float(self.abs_sum / self.numeric_count),
            float(self.abs_max),
            float(self.rel_sum / self.numeric_count),
            float(self.rel_max),
        )


@dataclass(frozen=True)
class _RunResult:
    output_path: Path
    mean_ms: float
    std_ms: float
    host_bytes: int
    device_bytes: int
    mode: str
    intra_errors: int
    intra_trials: int
    intra_fail_indices: list[str]
    abs_err_avg: float | None
    abs_err_max: float | None
    rel_err_avg: float | None
    rel_err_max: float | None


def _slug_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")
    return token or "value"


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


def _static_rows(*, op, label: str) -> list[dict[str, object]]:
    host_static = op._backend.mem_usage.host_static
    device_static = op._backend.mem_usage.device_static
    est_host, est_device = op._backend.estimate_static_bytes()
    return [
        {
            "config": label,
            "scenario": "static",
            "direction": "-",
            "k": None,
            "call_ms_mean": None,
            "call_ms_std": None,
            "host_gib": bytes_to_gib(host_static.total()),
            "device_gib": bytes_to_gib(device_static.total()),
            "note": format_static_note("actual_host", host_static)
            + " | "
            + format_static_note("actual_device", device_static),
        },
        {
            "config": label,
            "scenario": "static_est",
            "direction": "-",
            "k": None,
            "call_ms_mean": None,
            "call_ms_std": None,
            "host_gib": bytes_to_gib(est_host.total()),
            "device_gib": bytes_to_gib(est_device.total()),
            "note": format_static_note("est_host", est_host) + " | " + format_static_note("est_device", est_device),
        },
    ]


def _run_case(
    *,
    op,
    label: str,
    scenario: str,
    direction: str,
    k: int,
    matrix: np.ndarray,
    kwargs_factory,
    n_trials: int,
    n_warmup: int,
    output_dir: Path,
    output_index: int,
    output_atol: float,
    output_rtol: float,
) -> _RunResult:
    progress(f"{label}: scenario={scenario} direction={direction} k={k} start")

    calls = op._backend.mem_usage.calls
    start_idx = len(calls)
    diagnostics = _IntraDiagnostics()

    for warm_idx in range(n_warmup):
        warm = np.asarray(op.matmul(matrix, direction, **kwargs_factory()))
        diagnostics.observe(warm, tag=f"w{warm_idx + 1}", atol=output_atol, rtol=output_rtol)

    times: list[float] = []
    for trial_idx in range(n_trials):
        t0 = perf_counter()
        result = np.asarray(op.matmul(matrix, direction, **kwargs_factory()))
        times.append(perf_counter() - t0)
        diagnostics.observe(result, tag=f"b{trial_idx + 1}", atol=output_atol, rtol=output_rtol)

    end_idx = len(calls)
    host_bytes, device_bytes, mode = _validate_and_extract_runtime_memory(
        call_slice=calls[start_idx:end_idx],
        direction=direction,
        k=int(k),
        n_warmup=n_warmup,
        n_trials=n_trials,
    )

    assert diagnostics.ref_output is not None
    ms = np.asarray(times, dtype=np.float64) * 1000.0
    mean_ms = float(np.mean(ms))
    std_ms = float(np.std(ms))
    abs_err_avg, abs_err_max, rel_err_avg, rel_err_max = diagnostics.summary()

    config_token = _slug_token(label)
    scenario_token = _slug_token(scenario)
    direction_token = _slug_token(direction)
    output_path = output_dir / f"{output_index:06d}_{config_token}_{scenario_token}_{direction_token}_k{int(k)}.npy"
    np.save(output_path, diagnostics.ref_output, allow_pickle=False)

    return _RunResult(
        output_path=output_path,
        mean_ms=mean_ms,
        std_ms=std_ms,
        host_bytes=host_bytes,
        device_bytes=device_bytes,
        mode=mode,
        intra_errors=int(diagnostics.errors),
        intra_trials=int(diagnostics.trials),
        intra_fail_indices=list(diagnostics.fail_indices),
        abs_err_avg=abs_err_avg,
        abs_err_max=abs_err_max,
        rel_err_avg=rel_err_avg,
        rel_err_max=rel_err_max,
    )


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
    dtype: np.dtype,
    output_atol: float,
    output_rtol: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_by_k = build_inputs_by_k(op=op, ks=ks, seed_base=seed_base, dtype=dtype)
    summary_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []
    summary_rows.extend(_static_rows(op=op, label=label))
    directions = configured_direction_names(op._backend)

    for scenario in options:
        progress(f"{label}: scenario={scenario} start")
        for direction in directions:
            for k in ks:
                case = build_case(op=op, scenario=scenario, direction=direction, inputs=inputs_by_k[int(k)])
                if case.skip_reason is not None:
                    summary_rows.append(
                        skip_summary_row(
                            label=label,
                            scenario=scenario,
                            direction=direction,
                            k=int(k),
                            reason=case.skip_reason,
                        )
                    )
                    continue

                if direction not in {"up", "down"}:
                    raise ValueError(f"Unknown direction {direction!r}")
                assert isinstance(case.matrix, np.ndarray)
                result = _run_case(
                    op=op,
                    label=label,
                    scenario=scenario,
                    direction=direction,
                    k=int(k),
                    matrix=case.matrix,
                    kwargs_factory=case.kwargs_factory,
                    n_trials=n_trials,
                    n_warmup=n_warmup,
                    output_dir=output_dir,
                    output_index=len(output_rows),
                    output_atol=output_atol,
                    output_rtol=output_rtol,
                )
                summary_rows.append(
                    {
                        "config": label,
                        "scenario": scenario,
                        "direction": direction,
                        "k": int(k),
                        "path": str(result.output_path),
                        "call_ms_mean": result.mean_ms,
                        "call_ms_std": result.std_ms,
                        "host_gib": bytes_to_gib(result.host_bytes),
                        "device_gib": bytes_to_gib(result.device_bytes),
                        "note": f"mode={result.mode}",
                        "intra_errors": result.intra_errors,
                        "intra_trials": result.intra_trials,
                        "intra_fail_indices": result.intra_fail_indices,
                        "abs_err_avg": result.abs_err_avg,
                        "abs_err_max": result.abs_err_max,
                        "rel_err_avg": result.rel_err_avg,
                        "rel_err_max": result.rel_err_max,
                    }
                )
                output_rows.append(
                    {
                        "config": label,
                        "scenario": scenario,
                        "direction": direction,
                        "k": int(k),
                        "path": str(result.output_path),
                    }
                )

    return summary_rows, output_rows


def run_benchmark_suite(
    *,
    grg_path: str,
    configs: list[BenchConfig],
    ks: list[int],
    options: list[str],
    n_trials: int,
    n_warmup: int,
    dtype: np.dtype,
    index_dtype: np.dtype,
    output_atol: float,
    output_rtol: float,
    skip_note: bool,
    seed_base: int = 2026,
) -> None:
    from pygrgl_spmv import SpmvGRG

    progress(f"GRG file: {grg_path}")
    all_summary_rows: list[dict[str, object]] = []
    all_outputs: list[dict[str, object]] = []
    output_dir = Path(tempfile.mkdtemp(prefix="pygrgl_spmv_bench_refs_"))
    keep_output_dir = False
    try:
        for entry in configs:
            label = entry.label
            cfg = entry.config

            progress(f"{'=' * 72}")
            progress(f"loading operator {label}")
            t_load = perf_counter()
            op = SpmvGRG(grg_path, cfg, dtype, index_dtype)
            progress(f"{label}: operator ready in {(perf_counter() - t_load) * 1000.0:.2f} ms")

            cfg_rows, cfg_outputs = benchmark_config(
                op=op,
                label=label,
                ks=ks,
                options=options,
                n_trials=n_trials,
                n_warmup=n_warmup,
                seed_base=seed_base,
                output_dir=output_dir,
                dtype=dtype,
                output_atol=output_atol,
                output_rtol=output_rtol,
            )
            all_summary_rows.extend(cfg_rows)
            all_outputs.extend(cfg_outputs)
            del op
            gc.collect()

        cross_stats, per_config_cross = evaluate_output_equivalence(all_outputs, atol=output_atol, rtol=output_rtol)
        intra_errors, intra_trials = summarize_intra_diagnostics(all_summary_rows)
        print_summary_table(all_summary_rows, skip_note=skip_note)
        print(
            "\nCorrectness diagnostics: "
            f"intra_errors={intra_errors}/{intra_trials}, "
            f"cross_errors={cross_stats['errors']}/{cross_stats['comparisons']}"
        )
        config_order: dict[str, int] = {}
        for idx, entry in enumerate(configs):
            if entry.label not in config_order:
                config_order[entry.label] = idx
        print("Cross failures by config:")
        ordered_cfgs = sorted(
            config_order.keys(),
            key=lambda cfg: (-int(per_config_cross.get(cfg, {}).get("failures", 0)), config_order[cfg]),
        )
        for cfg in ordered_cfgs:
            stats = per_config_cross.get(cfg, {"failures": 0, "trials": 0})
            failures = int(stats.get("failures", 0))
            trials = int(stats.get("trials", 0))
            rate = 0.0 if trials == 0 else 100.0 * float(failures) / float(trials)
            print(f"  {cfg}: {failures}/{trials} ({rate:.1f}%)")
    except Exception:
        keep_output_dir = True
        print(f"\nSaved benchmark reference outputs to: {output_dir}")
        raise
    finally:
        if not keep_output_dir:
            shutil.rmtree(output_dir, ignore_errors=True)


__all__ = [
    "_validate_and_extract_runtime_memory",
    "benchmark_config",
    "run_benchmark_suite",
]
