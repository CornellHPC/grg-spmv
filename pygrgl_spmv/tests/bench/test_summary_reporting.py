"""Unit tests for benchmark runtime/memory reporting helpers."""

from __future__ import annotations

import numpy as np
import pygrgl
import pytest
import scipy.sparse as sp

from pygrgl_spmv.backends.memory import MemoryRecord, MemoryUsage, ResidencyBytes, RuntimeBytes, StaticBytes
from pygrgl_spmv.backends.mkl import MklPlan
from scripts.bench.cli import parse_dtype, parse_index_dtype, tolerances_for_dtype
from scripts.bench.configs import BenchConfig, expand_triton_configs, format_dry_run_line, parse_plan_pair_literal
from scripts.bench.report import (
    evaluate_output_equivalence,
    print_memory_table,
    print_runtime_table,
    summarize_intra_diagnostics,
)
from scripts.bench.run import (
    _ground_truth_output,
    _summarize_reference_diagnostics,
    _validate_and_extract_runtime_memory,
    benchmark_config,
)


def _equiv_row(*, config, scenario, direction, k, output):
    return {
        "config": config,
        "scenario": scenario,
        "direction": direction,
        "k": k,
        "output": np.asarray(output),
    }


def test_dtype_parsers_and_tolerances():
    assert parse_dtype("float32") == np.dtype(np.float32)
    assert parse_dtype("float64") == np.dtype(np.float64)
    assert parse_index_dtype("int32") == np.dtype(np.int32)
    assert parse_index_dtype("int64") == np.dtype(np.int64)
    assert tolerances_for_dtype(np.float32) == (1e-2, 1e-1)
    assert tolerances_for_dtype(np.float64) == (1e-8, 1e-5)
    with pytest.raises(ValueError, match="--dtype"):
        parse_dtype("float16")
    with pytest.raises(ValueError, match="--index-dtype"):
        parse_index_dtype("uint32")


def test_format_dry_run_line_includes_dtype_and_index_dtype():
    line = format_dry_run_line(
        BenchConfig(
            label="x",
            backend_name="mkl",
            plan_up_text=str(MklPlan.from_dict({"k_hint": None, "store": "N", "fmt": "CSR", "n_threads": 1})),
            plan_down_text=None,
            instrumentation=False,
            build_backend=lambda: None,  # type: ignore[return-value]
        ),
        [1, 4],
        ["baseline"],
        dtype=np.float32,
        index_dtype=np.int32,
    )
    assert "dtype=float32" in line
    assert "index_dtype=int32" in line
    assert "instrumentation=off" in line


def test_expand_triton_configs_one_sided_exhaustive():
    up_configs = expand_triton_configs(
        [parse_plan_pair_literal("[k_hint=1,store=*,fmt=*][]")],
        log_level="WARNING",
    )
    down_configs = expand_triton_configs(
        [parse_plan_pair_literal("[][k_hint=1,store=*,fmt=*]")],
        log_level="WARNING",
    )
    assert [str(cfg.plan_up_text) for cfg in up_configs] == [
        "[k_hint=1,store=N,fmt=CSC,scratch=none]",
        "[k_hint=1,store=N,fmt=CSR,scratch=none]",
    ]
    assert [str(cfg.plan_down_text) for cfg in down_configs] == [
        "[k_hint=1,store=T,fmt=CSC,scratch=none]",
        "[k_hint=1,store=T,fmt=CSR,scratch=none]",
    ]


def test_expand_triton_configs_accept_none_hint():
    configs = expand_triton_configs(
        [parse_plan_pair_literal("[k_hint=none,store=N,fmt=CSR][]")],
        log_level="WARNING",
    )
    assert len(configs) == 1
    assert configs[0].plan_up_text == "[k_hint=none,store=N,fmt=CSR,scratch=none]"


def test_expand_triton_configs_with_instrumentation():
    configs = expand_triton_configs(
        [parse_plan_pair_literal("[k_hint=1,store=N,fmt=CSR][]")],
        log_level="WARNING",
        instrumentation=True,
    )
    assert len(configs) == 1
    assert configs[0].label.endswith("-instr")
    assert configs[0].instrumentation is True


@pytest.mark.smoke
def test_output_equivalence_passes_for_same_class():
    rows = [
        _equiv_row(config="cfg-a", scenario="baseline", direction="up", k=4, output=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)),
        _equiv_row(
            config="cfg-b",
            scenario="baseline",
            direction="up",
            k=4,
            output=np.array([[1.0 + 1e-10, 2.0], [3.0, 4.0 - 1e-10]], dtype=np.float64),
        ),
    ]
    summary, per_config = evaluate_output_equivalence(rows, atol=1e-8, rtol=1e-5)
    assert summary == {"classes": 1, "comparisons": 1, "errors": 0}
    assert per_config["cfg-a"] == {"failures": 0, "trials": 1}
    assert per_config["cfg-b"] == {"failures": 0, "trials": 1}


def test_runtime_table_keeps_input_order(capsys):
    rows = [
        {
            "config": "cfg-z",
            "scenario": "baseline",
            "direction": "up",
            "k": 4,
            "call_ms_mean": 1.0,
            "call_ms_std": 0.1,
            "note": "x",
            "intra_errors": 0,
            "intra_trials": 0,
            "intra_fail_indices": [],
            "abs_err_avg": None,
            "abs_err_max": None,
            "rel_err_avg": None,
            "rel_err_max": None,
        },
        {
            "config": "cfg-a",
            "scenario": "init_vector",
            "direction": "down",
            "k": 1,
            "skip": "reason",
            "note": "skip: reason",
        },
    ]
    print_runtime_table(rows)
    out = capsys.readouterr().out
    assert out.find("cfg-z") < out.find("cfg-a")


def test_memory_table_has_no_config_delimiters(capsys):
    rows = [
        {"config": "cfg-a", "scenario": "baseline", "direction": "up", "k": 1, "kind": "Total GiB", "host_gib": 0.1, "device_gib": 0.2, "note": "a"},
        {"config": "cfg-b", "scenario": "baseline", "direction": "down", "k": 1, "kind": "Dynamic WS GiB", "host_gib": 0.3, "device_gib": 0.4, "note": "b"},
    ]
    print_memory_table(rows)
    out = capsys.readouterr().out
    dash_lines = [line for line in out.splitlines() if line and set(line) == {"-"}]
    assert len(dash_lines) == 1


def test_runtime_table_skip_note_hides_note_column(capsys):
    rows = [
        {
            "config": "cfg-a",
            "scenario": "baseline",
            "direction": "up",
            "k": 1,
            "call_ms_mean": 1.0,
            "call_ms_std": 0.0,
            "note": "mode=graph",
            "intra_errors": 0,
            "intra_trials": 0,
            "intra_fail_indices": [],
            "abs_err_avg": None,
            "abs_err_max": None,
            "rel_err_avg": None,
            "rel_err_max": None,
        }
    ]
    print_runtime_table(rows, skip_note=True)
    out = capsys.readouterr().out
    assert " Note" not in out
    assert "mode=graph" not in out


def test_analyze_cusparse_summary_accepts_skip_note_runtime_rows(tmp_path, capsys):
    from scripts.analyze_cusparse_summary import parse_full_log

    rows = [
        {
            "config": (
                "cusparse-up=[k_hint=none,store=N,fmt=CSR,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT,scratch=none]"
                "-down=<unspecified>"
            ),
            "scenario": "baseline",
            "direction": "up",
            "k": 1,
            "call_ms_mean": 1.0,
            "call_ms_std": 0.0,
            "note": "mode=graph",
            "intra_errors": 0,
            "intra_trials": 0,
            "intra_fail_indices": [],
            "abs_err_avg": None,
            "abs_err_max": None,
            "rel_err_avg": None,
            "rel_err_max": None,
        }
    ]
    print_runtime_table(rows, skip_note=True)
    summary = capsys.readouterr().out
    path = tmp_path / "cusparse.log"
    path.write_text(f"prefix\n{summary}\nCorrectness diagnostics:\n", encoding="utf-8")

    frame = parse_full_log(path, expected_direction="up")

    assert list(frame["note"]) == [""]
    assert list(frame["mode"]) == ["unknown"]
    assert list(frame["config"]) == [rows[0]["config"]]


def test_runtime_table_renders_err_trials_and_fail_tags(capsys):
    rows = [
        {
            "config": "cfg-a",
            "scenario": "baseline",
            "direction": "up",
            "k": 2,
            "call_ms_mean": 1.0,
            "call_ms_std": 0.0,
            "note": "mode=dynamic",
            "intra_errors": 2,
            "intra_trials": 5,
            "intra_fail_indices": ["w2", "b1"],
            "abs_err_avg": 1.25e-3,
            "abs_err_max": 3.5e-2,
            "rel_err_avg": 2.0e-4,
            "rel_err_max": 7.5e-3,
        }
    ]
    print_runtime_table(rows)
    out = capsys.readouterr().out
    assert "Err/Trials" in out
    assert "Abs Err (avg/max)" in out
    assert "Rel Err (avg/max)" in out
    assert "2/5" in out
    assert "1.250e-03/3.500e-02" in out
    assert "2.000e-04/7.500e-03" in out
    assert "intra_fail=w2,b1" in out


def test_validate_runtime_memory_fails_when_timed_values_vary():
    calls = [
        MemoryRecord(
            stage="run_up",
            runtime_k=4,
            host=RuntimeBytes(level_buffers=1, inputs=2, outputs=3, aux=4),
            device=RuntimeBytes(level_buffers=10, inputs=20, outputs=30, aux=40),
            static_ws=ResidencyBytes(device_bytes=1),
            dynamic_ws=ResidencyBytes(device_bytes=2),
            staging=ResidencyBytes(device_bytes=3),
            meta={"direction": "up", "mode": "graph"},
        ),
        MemoryRecord(
            stage="run_up",
            runtime_k=4,
            host=RuntimeBytes(level_buffers=1, inputs=2, outputs=3, aux=4),
            device=RuntimeBytes(level_buffers=10, inputs=20, outputs=30, aux=40),
            static_ws=ResidencyBytes(device_bytes=1),
            dynamic_ws=ResidencyBytes(device_bytes=2),
            staging=ResidencyBytes(device_bytes=3),
            meta={"direction": "up", "mode": "graph"},
        ),
        MemoryRecord(
            stage="run_up",
            runtime_k=4,
            host=RuntimeBytes(level_buffers=2, inputs=2, outputs=3, aux=4),
            device=RuntimeBytes(level_buffers=10, inputs=20, outputs=30, aux=40),
            static_ws=ResidencyBytes(device_bytes=1),
            dynamic_ws=ResidencyBytes(device_bytes=2),
            staging=ResidencyBytes(device_bytes=3),
            meta={"direction": "up", "mode": "graph"},
        ),
    ]
    with pytest.raises(AssertionError, match="vary across timed trials"):
        _validate_and_extract_runtime_memory(call_slice=calls, direction="up", k=4, n_warmup=0, n_trials=2)


class _FakeBackend:
    def __init__(self, *, directions=("up", "down")):
        self.mem_usage = MemoryUsage()
        self.mem_usage.host_static.level_offsets = 8
        self.mem_usage.device_static.blocks_up = 16
        self._plan_up = object() if "up" in directions else None
        self._plan_down = object() if "down" in directions else None

    def estimate_static_bytes(self):
        host = StaticBytes(blocks_up=4)
        device = StaticBytes(blocks_up=12)
        return host, device


class _FakeOp:
    def __init__(self, *, directions=("up", "down")):
        self.num_samples = 3
        self.num_mutations = 2
        self.num_nodes = 4
        self.num_individuals = 3
        self.sel_miss = sp.csr_matrix((self.num_mutations, self.num_nodes), dtype=np.float64)
        self._backend = _FakeBackend(directions=directions)

    def matmul(self, matrix, direction, **_kwargs):
        k = int(matrix.shape[0])
        if direction == "up":
            self._backend.mem_usage.record(
                stage="run_up",
                runtime_k=k,
                host_runtime=RuntimeBytes(level_buffers=100),
                device_runtime=RuntimeBytes(level_buffers=200),
                static_ws=ResidencyBytes(device_bytes=10, note="graph_up(active)"),
                dynamic_ws=ResidencyBytes(device_bytes=20, note="dynamic_up"),
                staging=ResidencyBytes(device_bytes=30, note="up:buffers"),
                meta={"direction": "up", "mode": "n/a"},
            )
            return np.zeros((k, self.num_mutations), dtype=np.float64)
        if direction == "down":
            self._backend.mem_usage.record(
                stage="run_down",
                runtime_k=k,
                host_runtime=RuntimeBytes(level_buffers=300),
                device_runtime=RuntimeBytes(level_buffers=400),
                static_ws=ResidencyBytes(device_bytes=11, note="graph_down(active)"),
                dynamic_ws=ResidencyBytes(device_bytes=21, note="dynamic_down"),
                staging=ResidencyBytes(device_bytes=31, note="down:buffers"),
                meta={"direction": "down", "mode": "n/a"},
            )
            return np.zeros((k, self.num_samples), dtype=np.float64)
        raise ValueError(direction)


def test_benchmark_config_orders_rows_by_execution():
    import scripts.bench.run as bench_run

    op = _FakeOp()
    grg_ref = object()
    monkey_ref = lambda **kwargs: np.zeros(
        (
            kwargs["matrix"].shape[0],
            op.num_mutations if kwargs["direction"] == "up" else op.num_samples,
        ),
        dtype=np.float64,
    )
    original_reference = bench_run._ground_truth_output
    bench_run._ground_truth_output = monkey_ref
    try:
        runtime_rows, memory_rows, output_rows = benchmark_config(
            op=op,
            grg_ref=grg_ref,
            label="cfg-x",
            ks=[1, 2],
            options=["baseline"],
            n_trials=1,
            n_warmup=0,
            seed_base=123,
            dtype=np.float64,
            output_atol=1e-8,
            output_rtol=1e-5,
        )
    finally:
        bench_run._ground_truth_output = original_reference

    assert [(row["direction"], row["k"]) for row in runtime_rows] == [
        ("up", 1),
        ("up", 2),
        ("down", 1),
        ("down", 2),
    ]
    assert len(memory_rows) == 16
    assert [row["kind"] for row in memory_rows[:4]] == [
        "Total GiB",
        "Static WS GiB",
        "Dynamic WS GiB",
        "Staging GiB",
    ]
    assert len(output_rows) == 4


def test_benchmark_config_checks_warmup_outputs():
    import scripts.bench.run as bench_run

    class _FakeWarmupMismatchOp(_FakeOp):
        def __init__(self):
            super().__init__()
            self._call_counter = {"up": 0, "down": 0}

        def matmul(self, matrix, direction, **kwargs):
            out = super().matmul(matrix, direction, **kwargs)
            idx = self._call_counter[direction]
            self._call_counter[direction] += 1
            if direction == "up" and idx == 1:
                out = out.copy()
                out[0, 0] = 1.0
            return out

    op = _FakeWarmupMismatchOp()
    grg_ref = object()
    original_reference = bench_run._ground_truth_output
    bench_run._ground_truth_output = lambda **kwargs: np.zeros(
        (
            kwargs["matrix"].shape[0],
            op.num_mutations if kwargs["direction"] == "up" else op.num_samples,
        ),
        dtype=np.float64,
    )
    try:
        runtime_rows, memory_rows, output_rows = benchmark_config(
            op=op,
            grg_ref=grg_ref,
            label="cfg-x",
            ks=[1],
            options=["baseline"],
            n_trials=1,
            n_warmup=2,
            seed_base=123,
            dtype=np.float64,
            output_atol=1e-8,
            output_rtol=1e-5,
        )
    finally:
        bench_run._ground_truth_output = original_reference

    up_row = next(row for row in runtime_rows if row["direction"] == "up")
    assert up_row["intra_errors"] == 1
    assert up_row["intra_trials"] == 3
    assert up_row["intra_fail_indices"] == ["w1"]
    assert len(memory_rows) == 8
    assert len(output_rows) == 2


def test_run_benchmark_suite_prints_both_tables(monkeypatch, capsys):
    import pygrgl_spmv
    import scripts.bench.run as bench_run

    config = BenchConfig(
        label="cfg-x",
        backend_name="fake",
        plan_up_text=None,
        plan_down_text=None,
        instrumentation=False,
        build_backend=lambda: object(),
    )

    monkeypatch.setattr(pygrgl_spmv, "SpmvGRG", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(bench_run.pygrgl, "load_immutable_grg", lambda _path: object())
    monkeypatch.setattr(
        bench_run,
        "benchmark_config",
        lambda **_kwargs: (
            [
                {
                    "config": "cfg-x",
                    "scenario": "baseline",
                    "direction": "up",
                    "k": 1,
                    "call_ms_mean": 1.0,
                    "call_ms_std": 0.0,
                    "note": "mode=n/a",
                    "intra_errors": 0,
                    "intra_trials": 0,
                    "intra_fail_indices": [],
                    "abs_err_avg": None,
                    "abs_err_max": None,
                    "rel_err_avg": None,
                    "rel_err_max": None,
                    "ref_error": 0,
                    "ref_abs_err_max": 0.0,
                    "ref_rel_err_max": 0.0,
                }
            ],
            [
                {"config": "cfg-x", "scenario": "baseline", "direction": "up", "k": 1, "kind": "Total GiB", "host_gib": 0.0, "device_gib": 0.0, "note": "mode=n/a"},
                {"config": "cfg-x", "scenario": "baseline", "direction": "up", "k": 1, "kind": "Static WS GiB", "host_gib": 0.0, "device_gib": 0.0, "note": "none"},
                {"config": "cfg-x", "scenario": "baseline", "direction": "up", "k": 1, "kind": "Dynamic WS GiB", "host_gib": 0.0, "device_gib": 0.0, "note": "none"},
                {"config": "cfg-x", "scenario": "baseline", "direction": "up", "k": 1, "kind": "Staging GiB", "host_gib": 0.0, "device_gib": 0.0, "note": "none"},
            ],
            [_equiv_row(config="cfg-x", scenario="baseline", direction="up", k=1, output=np.array([[1.0]], dtype=np.float64))],
        ),
    )

    bench_run.run_benchmark_suite(
        grg_path="fake.grg",
        configs=[config],
        ks=[1],
        options=["baseline"],
        n_trials=1,
        n_warmup=0,
        dtype=np.float64,
        index_dtype=np.int32,
        output_atol=1e-8,
        output_rtol=1e-5,
        skip_note=False,
    )

    out = capsys.readouterr().out
    assert "BENCHMARK RUNTIME SUMMARY" in out
    assert "BENCHMARK MEMORY SUMMARY" in out


def test_summarize_intra_diagnostics_counts_runtime_rows_only():
    rows = [
        {"scenario": "baseline", "intra_errors": 2, "intra_trials": 5},
        {"scenario": "init_vector", "intra_errors": 1, "intra_trials": 3},
        {"scenario": "miss", "skip": "reason", "intra_errors": 99, "intra_trials": 99},
    ]
    assert summarize_intra_diagnostics(rows) == (3, 8)


def test_ground_truth_output_matches_pygrgl():
    grg = pygrgl.load_immutable_grg("pygrgl_spmv/tests/data/msprime.example.igd.final.grg")
    rng = np.random.default_rng(123)
    matrix = rng.standard_normal((1, grg.num_samples), dtype=np.float64)
    expected = np.asarray(pygrgl.matmul(grg, matrix, pygrgl.TraversalDirection.UP))
    actual = _ground_truth_output(grg_ref=grg, matrix=matrix, direction="up", kwargs={})
    np.testing.assert_allclose(actual, expected)


def test_summarize_reference_diagnostics():
    errors, checked, abs_max, rel_max = _summarize_reference_diagnostics(
        [
            {"scenario": "baseline", "ref_error": 1, "ref_abs_err_max": 1e-6, "ref_rel_err_max": 2e-6},
            {"scenario": "baseline", "ref_error": 0, "ref_abs_err_max": 3e-6, "ref_rel_err_max": 1e-6},
        ]
    )
    assert (errors, checked) == (1, 2)
    assert abs_max == pytest.approx(3e-6)
    assert rel_max == pytest.approx(2e-6)
