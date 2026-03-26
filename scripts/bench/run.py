"""Benchmark execution and runtime diagnostics."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import pygrgl

from pygrgl_spmv.memory import MemoryAllocation, MemorySnapshot, live_snapshot, tree_rows
from scripts.bench.cases import build_case, build_inputs_by_k, configured_direction_names
from scripts.bench.cli import progress
from scripts.bench.configs import BenchConfig
from scripts.bench.report import (
    bytes_to_gib,
    compare_outputs,
    evaluate_output_equivalence,
    print_common_config,
    print_memory_table,
    print_runtime_table,
    summarize_config_display,
    summarize_intra_diagnostics,
)


@dataclass
class _IntraDiagnostics:
    steady_output: np.ndarray
    errors: int = 0
    trials: int = 0
    fail_indices: list[str] = field(default_factory=list)
    abs_sum: float = 0.0
    rel_sum: float = 0.0
    abs_max: float = 0.0
    rel_max: float = 0.0
    numeric_count: int = 0

    def observe(self, candidate: np.ndarray, *, tag: str, atol: float, rtol: float) -> None:
        self.trials += 1
        ok, shape_ok, max_abs, max_rel = compare_outputs(
            self.steady_output,
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
class _CaseResult:
    output: np.ndarray
    mean_ms: float
    std_ms: float
    memory_snapshot: MemorySnapshot
    intra_errors: int
    intra_trials: int
    intra_fail_indices: list[str]
    abs_err_avg: float | None
    abs_err_max: float | None
    rel_err_avg: float | None
    rel_err_max: float | None
    ref_error: int
    ref_abs_err_max: float | None
    ref_rel_err_max: float | None


def _row_signature(row: MemoryAllocation) -> tuple[object, ...]:
    # Timed benchmark validation checks stable memory shape, not allocator address reuse.
    return (
        int(row.nbytes),
        row.storage,
        tuple(sorted(row.owners)),
        tuple(sorted(row.activities)),
        tuple(sorted(row.retentions)),
        tuple(sorted(row.directions)),
        tuple(sorted(int(value) for value in row.slot_ks)),
        tuple(sorted(row.labels)),
        tuple(sorted(row.kinds)),
    )


def _snapshot_signature(snapshot: MemorySnapshot) -> tuple[object, ...]:
    return (
        str(snapshot.stage),
        None if snapshot.runtime_k is None else int(snapshot.runtime_k),
        snapshot.direction,
        tuple(_row_signature(row) for row in snapshot.allocations),
        tuple(sorted(snapshot.meta.items())),
    )


def _validate_and_extract_runtime_memory(
    *,
    call_slice: list[MemorySnapshot],
    retained_slice: list[MemorySnapshot | None],
    direction: str,
    k: int,
    n_warmup: int,
    n_trials: int,
) -> MemorySnapshot:
    expected = 1 + n_warmup + n_trials
    if len(call_slice) != expected:
        raise AssertionError(
            f"Expected {expected} memory records for direction={direction}, k={k}, got {len(call_slice)}"
        )
    if len(retained_slice) != expected:
        raise AssertionError(
            f"Expected {expected} retained memory records for direction={direction}, k={k}, got {len(retained_slice)}"
        )

    expected_stage = "run_up" if direction == "up" else "run_down"
    for idx, record in enumerate(call_slice):
        if str(record.stage) != expected_stage:
            raise AssertionError(
                f"Unexpected stage in memory record {idx}: expected {expected_stage}, got {record.stage}"
            )
        if int(record.runtime_k) != int(k):
            raise AssertionError(
                f"Unexpected runtime_k in memory record {idx}: expected {k}, got {record.runtime_k}"
            )
        if record.direction != direction:
            raise AssertionError(
                f"Unexpected direction in memory record {idx}: expected {direction}, got {record.direction}"
            )

    timed_records = call_slice[1 + n_warmup :]
    timed_retained = retained_slice[1 + n_warmup :]
    if len(timed_records) != n_trials:
        raise AssertionError(
            f"Expected {n_trials} timed memory records for direction={direction}, k={k}, got {len(timed_records)}"
        )
    if len(timed_retained) != n_trials:
        raise AssertionError(
            f"Expected {n_trials} timed retained memory records for direction={direction}, k={k}, got {len(timed_retained)}"
        )

    signatures = [_snapshot_signature(record) for record in timed_records]
    if len(set(signatures)) != 1:
        raise AssertionError(
            f"Call memory shape varies across timed trials for direction={direction}, k={k}"
        )
    retained_signatures = [
        None if record is None else _snapshot_signature(record)
        for record in timed_retained
    ]
    if len(set(retained_signatures)) != 1:
        raise AssertionError(
            f"Retained memory shape varies across timed trials for direction={direction}, k={k}"
        )

    return live_snapshot(timed_retained[0], timed_records[0])


def _clone_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    cloned: dict[str, object] = {}
    for key, value in kwargs.items():
        cloned[key] = value.copy() if isinstance(value, np.ndarray) else value
    return cloned


def _ground_truth_output(
    *,
    grg_ref,
    matrix: np.ndarray,
    direction: str,
    kwargs: dict[str, object],
) -> np.ndarray:
    traversal = pygrgl.TraversalDirection.UP if direction == "up" else pygrgl.TraversalDirection.DOWN
    return np.asarray(pygrgl.matmul(grg_ref, matrix, traversal, **kwargs))


def _summarize_reference_diagnostics(rows: list[dict[str, object]]) -> tuple[int, int, float | None, float | None]:
    errors = 0
    checked = 0
    abs_max: float | None = None
    rel_max: float | None = None
    for row in rows:
        if "skip" in row:
            continue
        checked += 1
        errors += int(row.get("ref_error", 0))
        row_abs = row.get("ref_abs_err_max")
        row_rel = row.get("ref_rel_err_max")
        if row_abs is not None:
            row_abs = float(row_abs)
            abs_max = row_abs if abs_max is None else max(abs_max, row_abs)
        if row_rel is not None:
            row_rel = float(row_rel)
            rel_max = row_rel if rel_max is None else max(rel_max, row_rel)
    return errors, checked, abs_max, rel_max


def _runtime_skip_row(*, label: str, scenario: str, direction: str, k: int, reason: str) -> dict[str, object]:
    return {
        "config": label,
        "scenario": scenario,
        "direction": direction,
        "k": int(k),
        "skip": reason,
        "note": f"skip: {reason}",
    }

def _memory_skip_rows(*, label: str, scenario: str, direction: str, k: int, reason: str) -> list[dict[str, object]]:
    return [
        {
            "config": label,
            "scenario": scenario,
            "case_direction": direction,
            "case_k": int(k),
            "level": 0,
            "node": node,
            "parent": "-",
            "skip": reason,
            "note": f"skip: {reason}",
        }
        for node in ("cuda_live", "cpu_live")
    ]


def _memory_rows(*, label: str, scenario: str, direction: str, k: int, snapshot: MemorySnapshot) -> list[dict[str, object]]:
    note = str(snapshot.meta.get("mode", ""))
    rows: list[dict[str, object]] = []
    last_leaf_key: tuple[object, ...] | None = None
    last_leaf_row: dict[str, object] | None = None
    for tree_row in tree_rows(snapshot):
        path = tree_row.path
        level = len(path) - 1
        parent = tree_row.parent()
        space = path[0].removesuffix("_live")
        base_row = {
            "config": label,
            "scenario": scenario,
            "case_direction": direction,
            "case_k": int(k),
            "level": level,
            "parent": parent,
            "gib": bytes_to_gib(tree_row.nbytes),
            "space": space,
            "note": note,
        }
        if not tree_row.is_leaf():
            rows.append(
                {
                    **base_row,
                    "node": tree_row.node(),
                    "owner": "-",
                    "active": "-",
                    "retention": "-",
                    "kinds": "",
                }
            )
            last_leaf_key = None
            last_leaf_row = None
            continue

        node = "|".join(sorted(tree_row.labels)) if tree_row.labels else "allocation"
        kinds = "|".join(sorted(tree_row.kinds))
        owner = "-" if tree_row.owner is None else str(tree_row.owner)
        activity = "-" if tree_row.activity is None else str(tree_row.activity)
        retention = "-" if tree_row.retention is None else str(tree_row.retention)
        leaf_key = (level, parent, node, space, owner, activity, retention, kinds)
        if leaf_key == last_leaf_key and last_leaf_row is not None:
            last_leaf_row["gib"] = float(last_leaf_row["gib"]) + bytes_to_gib(tree_row.nbytes)
            continue
        leaf_row = {
            **base_row,
            "node": node,
            "owner": owner,
            "active": activity,
            "retention": retention,
            "kinds": kinds,
        }
        rows.append(leaf_row)
        last_leaf_key = leaf_key
        last_leaf_row = leaf_row
    return rows


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
    output_atol: float,
    output_rtol: float,
    reference_output: np.ndarray,
) -> _CaseResult:
    progress(f"{label}: scenario={scenario} direction={direction} k={k} start")

    first = np.asarray(op.matmul(matrix, direction, **kwargs_factory()))
    first_call = op.memory.last_call
    if first_call is None:
        raise AssertionError(f"Expected last_call memory after direction={direction}, k={k} preflight")
    call_records = [first_call]
    retained_records = [op.memory.retained]
    diagnostics = _IntraDiagnostics(steady_output=np.array(first, copy=True))

    for warm_idx in range(n_warmup):
        warm = np.asarray(op.matmul(matrix, direction, **kwargs_factory()))
        if op.memory.last_call is None:
            raise AssertionError(f"Expected last_call memory after direction={direction}, k={k} warmup={warm_idx + 1}")
        call_records.append(op.memory.last_call)
        retained_records.append(op.memory.retained)
        diagnostics.observe(warm, tag=f"w{warm_idx + 1}", atol=output_atol, rtol=output_rtol)

    times: list[float] = []
    for trial_idx in range(n_trials):
        t0 = perf_counter()
        result = np.asarray(op.matmul(matrix, direction, **kwargs_factory()))
        times.append(perf_counter() - t0)
        if op.memory.last_call is None:
            raise AssertionError(f"Expected last_call memory after direction={direction}, k={k} trial={trial_idx + 1}")
        call_records.append(op.memory.last_call)
        retained_records.append(op.memory.retained)
        diagnostics.observe(result, tag=f"b{trial_idx + 1}", atol=output_atol, rtol=output_rtol)

    snapshot = _validate_and_extract_runtime_memory(
        call_slice=call_records,
        retained_slice=retained_records,
        direction=direction,
        k=int(k),
        n_warmup=n_warmup,
        n_trials=n_trials,
    )

    ms = np.asarray(times, dtype=np.float64) * 1000.0
    mean_ms = float(np.mean(ms))
    std_ms = float(np.std(ms))
    abs_err_avg, abs_err_max, rel_err_avg, rel_err_max = diagnostics.summary()
    ref_ok, ref_shape_ok, ref_abs_err_max, ref_rel_err_max = compare_outputs(
        reference_output,
        diagnostics.steady_output,
        atol=output_atol,
        rtol=output_rtol,
    )
    ref_error = 0 if ref_ok and ref_shape_ok else 1

    return _CaseResult(
        output=np.array(diagnostics.steady_output, copy=True),
        mean_ms=mean_ms,
        std_ms=std_ms,
        memory_snapshot=snapshot,
        intra_errors=int(diagnostics.errors),
        intra_trials=int(diagnostics.trials),
        intra_fail_indices=list(diagnostics.fail_indices),
        abs_err_avg=abs_err_avg,
        abs_err_max=abs_err_max,
        rel_err_avg=rel_err_avg,
        rel_err_max=rel_err_max,
        ref_error=ref_error,
        ref_abs_err_max=ref_abs_err_max,
        ref_rel_err_max=ref_rel_err_max,
    )


def benchmark_config(
    *,
    op,
    grg_ref,
    label: str,
    ks: list[int],
    options: list[str],
    n_trials: int,
    n_warmup: int,
    seed_base: int,
    dtype: np.dtype,
    output_atol: float,
    output_rtol: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    inputs_by_k = build_inputs_by_k(op=op, ks=ks, seed_base=seed_base, dtype=dtype)
    runtime_rows: list[dict[str, object]] = []
    memory_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []
    directions = configured_direction_names(op._backend)

    for scenario in options:
        progress(f"{label}: scenario={scenario} start")
        for direction in directions:
            for k in ks:
                case = build_case(op=op, scenario=scenario, direction=direction, inputs=inputs_by_k[int(k)])
                if case.skip_reason is not None:
                    runtime_rows.append(
                        _runtime_skip_row(
                            label=label,
                            scenario=scenario,
                            direction=direction,
                            k=int(k),
                            reason=case.skip_reason,
                        )
                    )
                    memory_rows.extend(
                        _memory_skip_rows(
                            label=label,
                            scenario=scenario,
                            direction=direction,
                            k=int(k),
                            reason=case.skip_reason,
                        )
                    )
                    continue

                assert isinstance(case.matrix, np.ndarray)
                kwargs = _clone_kwargs(case.kwargs_factory())
                reference_output = _ground_truth_output(
                    grg_ref=grg_ref,
                    matrix=case.matrix,
                    direction=direction,
                    kwargs=_clone_kwargs(kwargs),
                )
                result = _run_case(
                    op=op,
                    label=label,
                    scenario=scenario,
                    direction=direction,
                    k=int(k),
                    matrix=case.matrix,
                    kwargs_factory=lambda kwargs=kwargs: _clone_kwargs(kwargs),
                    n_trials=n_trials,
                    n_warmup=n_warmup,
                    output_atol=output_atol,
                    output_rtol=output_rtol,
                    reference_output=reference_output,
                )
                runtime_rows.append(
                    {
                        "config": label,
                        "scenario": scenario,
                        "direction": direction,
                        "k": int(k),
                        "call_ms_mean": result.mean_ms,
                        "call_ms_std": result.std_ms,
                        "note": f"mode={result.memory_snapshot.meta.get('mode', 'n/a')}",
                        "intra_errors": result.intra_errors,
                        "intra_trials": result.intra_trials,
                        "intra_fail_indices": result.intra_fail_indices,
                        "abs_err_avg": result.abs_err_avg,
                        "abs_err_max": result.abs_err_max,
                        "rel_err_avg": result.rel_err_avg,
                        "rel_err_max": result.rel_err_max,
                        "ref_error": result.ref_error,
                        "ref_abs_err_max": result.ref_abs_err_max,
                        "ref_rel_err_max": result.ref_rel_err_max,
                    }
                )
                memory_rows.extend(
                    _memory_rows(
                        label=label,
                        scenario=scenario,
                        direction=direction,
                        k=int(k),
                        snapshot=result.memory_snapshot,
                    )
                )
                output_rows.append(
                    {
                        "config": label,
                        "scenario": scenario,
                        "direction": direction,
                        "k": int(k),
                        "output": result.output,
                    }
                )

    return runtime_rows, memory_rows, output_rows


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
    common_config_lines, config_display = summarize_config_display(configs)
    all_runtime_rows: list[dict[str, object]] = []
    all_memory_rows: list[dict[str, object]] = []
    all_outputs: list[dict[str, object]] = []
    grg_ref = pygrgl.load_immutable_grg(grg_path)
    for entry in configs:
        label = entry.label

        progress(f"{'=' * 72}")
        progress(f"loading operator {label}")
        t_load = perf_counter()
        op = SpmvGRG(
            grg_path,
            entry.build_backend(),
            dtype,
            index_dtype,
            ordering=entry.ordering,
            intra_block_ordering=entry.intra_block_ordering,
        )
        progress(f"{label}: operator ready in {(perf_counter() - t_load) * 1000.0:.2f} ms")

        cfg_runtime_rows, cfg_memory_rows, cfg_outputs = benchmark_config(
            op=op,
            grg_ref=grg_ref,
            label=label,
            ks=ks,
            options=options,
            n_trials=n_trials,
            n_warmup=n_warmup,
            seed_base=seed_base,
            dtype=dtype,
            output_atol=output_atol,
            output_rtol=output_rtol,
        )
        all_runtime_rows.extend(cfg_runtime_rows)
        all_memory_rows.extend(cfg_memory_rows)
        all_outputs.extend(cfg_outputs)
        del op
        gc.collect()

    cross_stats, per_config_cross = evaluate_output_equivalence(all_outputs, atol=output_atol, rtol=output_rtol)
    intra_errors, intra_trials = summarize_intra_diagnostics(all_runtime_rows)
    ref_errors, ref_checked, ref_abs_max, ref_rel_max = _summarize_reference_diagnostics(all_runtime_rows)
    print_common_config(common_config_lines)
    print_runtime_table(all_runtime_rows, config_display=config_display, skip_note=skip_note)
    print_memory_table(all_memory_rows, config_display=config_display, skip_note=skip_note)
    print(
        "\nCorrectness diagnostics: "
        f"intra_errors={intra_errors}/{intra_trials}, "
        f"cross_errors={cross_stats['errors']}/{cross_stats['comparisons']}, "
        f"ref_errors={ref_errors}/{ref_checked}, "
        f"ref_abs_err_max={'n/a' if ref_abs_max is None else f'{ref_abs_max:.3e}'}, "
        f"ref_rel_err_max={'n/a' if ref_rel_max is None else f'{ref_rel_max:.3e}'}"
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
        print(f"  {config_display.get(cfg, cfg)}: {failures}/{trials} ({rate:.1f}%)")


__all__ = [
    "_ground_truth_output",
    "_summarize_reference_diagnostics",
    "_validate_and_extract_runtime_memory",
    "benchmark_config",
    "run_benchmark_suite",
]
