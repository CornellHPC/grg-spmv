"""Unit tests for benchmark summary and equivalence helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from pygrgl_spmv.backends.memory import MemoryRecord, MemoryUsage, RuntimeBytes, StaticBytes
from pygrgl_spmv.backends.mkl import MklPlan
from scripts.bench.cli import parse_dtype, parse_index_dtype, tolerances_for_dtype
from scripts.bench.configs import BenchConfig, format_dry_run_line
from scripts.bench.report import evaluate_output_equivalence, print_summary_table, summarize_intra_diagnostics
from scripts.bench.run import _validate_and_extract_runtime_memory, benchmark_config


def _save_arr(path, arr):
    np.save(path, arr, allow_pickle=False)
    return str(path)


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
            config={
                "type": "mkl",
                "plan_up": MklPlan.from_any({"k_hint": None, "store": "N", "fmt": "CSR", "n_threads": 1}),
                "plan_down": None,
                "log_level": "WARNING",
            },
        ),
        [1, 4],
        ["baseline"],
        dtype=np.float32,
        index_dtype=np.int32,
    )
    assert "dtype=float32" in line
    assert "index_dtype=int32" in line


@pytest.mark.smoke
def test_output_equivalence_passes_for_same_class(tmp_path):
    p0 = _save_arr(tmp_path / "a.npy", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    p1 = _save_arr(tmp_path / "b.npy", np.array([[1.0 + 1e-10, 2.0], [3.0, 4.0 - 1e-10]], dtype=np.float64))
    rows = [
        {"config": "cfg-a", "scenario": "baseline", "direction": "up", "k": 4, "path": p0},
        {"config": "cfg-b", "scenario": "baseline", "direction": "up", "k": 4, "path": p1},
    ]
    summary, per_config = evaluate_output_equivalence(rows, atol=1e-8, rtol=1e-5)
    assert summary == {"classes": 1, "comparisons": 1, "errors": 0}
    assert per_config["cfg-a"] == {"failures": 0, "trials": 1}
    assert per_config["cfg-b"] == {"failures": 0, "trials": 1}


def test_output_equivalence_collects_failures(tmp_path):
    p0 = _save_arr(tmp_path / "a.npy", np.array([[1.0, 2.0]], dtype=np.float64))
    p1 = _save_arr(tmp_path / "b.npy", np.array([[1.0, 9.0]], dtype=np.float64))
    rows = [
        {"config": "cfg-a", "scenario": "baseline", "direction": "up", "k": 4, "path": p0},
        {"config": "cfg-b", "scenario": "baseline", "direction": "up", "k": 4, "path": p1},
    ]
    summary, per_config = evaluate_output_equivalence(rows, atol=1e-8, rtol=1e-5)
    assert summary == {"classes": 1, "comparisons": 1, "errors": 1}
    assert per_config["cfg-a"] == {"failures": 1, "trials": 1}
    assert per_config["cfg-b"] == {"failures": 1, "trials": 1}


def test_output_equivalence_maps_up_miss_to_baseline(tmp_path):
    p0 = _save_arr(tmp_path / "a.npy", np.array([[11.0, 12.0]], dtype=np.float64))
    p1 = _save_arr(tmp_path / "b.npy", np.array([[11.0, 12.0]], dtype=np.float64))
    rows = [
        {"config": "cfg-a", "scenario": "baseline", "direction": "up", "k": 2, "path": p0},
        {"config": "cfg-b", "scenario": "miss", "direction": "up", "k": 2, "path": p1},
    ]
    summary, per_config = evaluate_output_equivalence(rows, atol=1e-8, rtol=1e-5)
    assert summary == {"classes": 1, "comparisons": 1, "errors": 0}
    assert per_config["cfg-a"] == {"failures": 0, "trials": 1}
    assert per_config["cfg-b"] == {"failures": 0, "trials": 1}


def test_output_equivalence_pairwise_counts(tmp_path):
    p0 = _save_arr(tmp_path / "a.npy", np.array([[1.0]], dtype=np.float64))
    p1 = _save_arr(tmp_path / "b.npy", np.array([[2.0]], dtype=np.float64))
    p2 = _save_arr(tmp_path / "c.npy", np.array([[1.0]], dtype=np.float64))
    rows = [
        {"config": "cfg-a", "scenario": "baseline", "direction": "up", "k": 1, "path": p0},
        {"config": "cfg-b", "scenario": "baseline", "direction": "up", "k": 1, "path": p1},
        {"config": "cfg-c", "scenario": "baseline", "direction": "up", "k": 1, "path": p2},
    ]
    summary, per_config = evaluate_output_equivalence(rows, atol=1e-8, rtol=1e-5)
    assert summary == {"classes": 1, "comparisons": 3, "errors": 2}
    assert per_config["cfg-a"] == {"failures": 1, "trials": 2}
    assert per_config["cfg-b"] == {"failures": 2, "trials": 2}
    assert per_config["cfg-c"] == {"failures": 1, "trials": 2}


def test_output_equivalence_is_order_invariant_for_asymmetric_allclose_case(tmp_path):
    p0 = _save_arr(tmp_path / "a.npy", np.array([[1000.0]], dtype=np.float64))
    p1 = _save_arr(tmp_path / "b.npy", np.array([[1105.0]], dtype=np.float64))
    rows_ab = [
        {"config": "cfg-a", "scenario": "baseline", "direction": "up", "k": 1, "path": p0},
        {"config": "cfg-b", "scenario": "baseline", "direction": "up", "k": 1, "path": p1},
    ]
    rows_ba = [
        {"config": "cfg-b", "scenario": "baseline", "direction": "up", "k": 1, "path": p1},
        {"config": "cfg-a", "scenario": "baseline", "direction": "up", "k": 1, "path": p0},
    ]
    summary_ab, per_cfg_ab = evaluate_output_equivalence(rows_ab, atol=1e-8, rtol=1e-1)
    summary_ba, per_cfg_ba = evaluate_output_equivalence(rows_ba, atol=1e-8, rtol=1e-1)
    assert summary_ab == summary_ba == {"classes": 1, "comparisons": 1, "errors": 1}
    assert per_cfg_ab == per_cfg_ba == {
        "cfg-a": {"failures": 1, "trials": 1},
        "cfg-b": {"failures": 1, "trials": 1},
    }


@pytest.mark.smoke
def test_print_summary_table_keeps_input_order(capsys):
    rows = [
        {
            "config": "cfg-z",
            "scenario": "baseline",
            "direction": "up",
            "k": 4,
            "call_ms_mean": 1.0,
            "call_ms_std": 0.1,
            "host_gib": 0.1,
            "device_gib": 0.2,
            "note": "x",
        },
        {
            "config": "cfg-a",
            "scenario": "init_vector",
            "direction": "down",
            "k": 1,
            "skip": "reason",
            "call_ms_mean": None,
            "call_ms_std": None,
            "host_gib": None,
            "device_gib": None,
            "note": "skip: reason",
        },
    ]
    print_summary_table(rows)
    out = capsys.readouterr().out
    assert out.find("cfg-z") < out.find("cfg-a")


def test_print_summary_table_has_no_config_delimiters(capsys):
    rows = [
        {
            "config": "cfg-a",
            "scenario": "baseline",
            "direction": "up",
            "k": 1,
            "call_ms_mean": 1.0,
            "call_ms_std": 0.0,
            "host_gib": 0.1,
            "device_gib": 0.2,
            "note": "a",
        },
        {
            "config": "cfg-b",
            "scenario": "baseline",
            "direction": "down",
            "k": 1,
            "call_ms_mean": 2.0,
            "call_ms_std": 0.0,
            "host_gib": 0.3,
            "device_gib": 0.4,
            "note": "b",
        },
    ]
    print_summary_table(rows)
    out = capsys.readouterr().out
    dash_lines = [line for line in out.splitlines() if line and set(line) == {"-"}]
    assert len(dash_lines) == 1


def test_print_summary_table_skip_note_hides_note_column(capsys):
    rows = [
        {
            "config": "cfg-a",
            "scenario": "baseline",
            "direction": "up",
            "k": 1,
            "call_ms_mean": 1.0,
            "call_ms_std": 0.0,
            "host_gib": 0.1,
            "device_gib": 0.2,
            "note": "mode=graph",
        }
    ]
    print_summary_table(rows, skip_note=True)
    out = capsys.readouterr().out
    assert " Note" not in out
    assert "mode=graph" not in out


def test_analyze_cusparse_summary_accepts_skip_note_runtime_rows(tmp_path, capsys):
    from scripts.analyze_cusparse_summary import parse_full_log

    rows = [
        {
            "config": (
                "cusparse-up=[k_hint=none,store=N,fmt=CSR,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT]"
                "-down=<unspecified>"
            ),
            "scenario": "baseline",
            "direction": "up",
            "k": 1,
            "call_ms_mean": 1.0,
            "call_ms_std": 0.0,
            "host_gib": 0.1,
            "device_gib": 0.2,
            "note": "mode=graph",
            "intra_errors": 0,
            "intra_trials": 1,
            "abs_err_avg": 0.0,
            "abs_err_max": 0.0,
            "rel_err_avg": 0.0,
            "rel_err_max": 0.0,
        }
    ]
    print_summary_table(rows, skip_note=True)
    summary = capsys.readouterr().out
    path = tmp_path / "cusparse.log"
    path.write_text(f"prefix\n{summary}\nCorrectness diagnostics:\n", encoding="utf-8")

    frame = parse_full_log(path, expected_direction="up")

    assert list(frame["note"]) == [""]
    assert list(frame["mode"]) == ["unknown"]
    assert list(frame["config"]) == [rows[0]["config"]]
    assert list(frame["device_gib"]) == [pytest.approx(0.2)]


def test_print_summary_table_renders_err_trials_and_fail_tags(capsys):
    rows = [
        {
            "config": "cfg-a",
            "scenario": "baseline",
            "direction": "up",
            "k": 2,
            "call_ms_mean": 1.0,
            "call_ms_std": 0.0,
            "host_gib": 0.1,
            "device_gib": 0.2,
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
    print_summary_table(rows)
    out = capsys.readouterr().out
    assert "Err/Trials" in out
    assert "Abs Err (avg/max)" in out
    assert "Rel Err (avg/max)" in out
    assert "2/5" in out
    assert "1.250e-03/3.500e-02" in out
    assert "2.000e-04/7.500e-03" in out
    assert "intra_fail=w2,b1" in out
    assert "cross_fail" not in out


def test_validate_runtime_memory_fails_when_timed_values_vary():
    calls = [
        MemoryRecord(
            stage="run_up",
            runtime_k=4,
            host=RuntimeBytes(level_buffers=1, inputs=2, outputs=3, aux=4),
            device=RuntimeBytes(level_buffers=10, inputs=20, outputs=30, aux=40),
            meta={"direction": "up", "mode": "graph"},
        ),
        MemoryRecord(
            stage="run_up",
            runtime_k=4,
            host=RuntimeBytes(level_buffers=2, inputs=2, outputs=3, aux=4),
            device=RuntimeBytes(level_buffers=10, inputs=20, outputs=30, aux=40),
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
        self.n = 3
        self.m = 2
        self.K = 4
        self.num_individuals = 3
        self.sel_miss = sp.csr_matrix((self.m, self.K), dtype=np.float64)
        self._backend = _FakeBackend(directions=directions)

    def matmul(self, matrix, direction, **_kwargs):
        k = int(matrix.shape[0])
        if direction == "up":
            self._backend.mem_usage.record(
                stage="run_up",
                runtime_k=k,
                host_runtime=RuntimeBytes(level_buffers=100),
                device_runtime=RuntimeBytes(level_buffers=200),
                meta={"direction": "up", "mode": "n/a"},
            )
            return np.zeros((k, self.m), dtype=np.float64)
        if direction == "down":
            self._backend.mem_usage.record(
                stage="run_down",
                runtime_k=k,
                host_runtime=RuntimeBytes(level_buffers=300),
                device_runtime=RuntimeBytes(level_buffers=400),
                meta={"direction": "down", "mode": "n/a"},
            )
            return np.zeros((k, self.n), dtype=np.float64)
        raise ValueError(direction)


def test_benchmark_config_orders_rows_by_execution(tmp_path):
    op = _FakeOp()
    summary_rows, output_rows = benchmark_config(
        op=op,
        label="cfg-x",
        ks=[1, 2],
        options=["baseline"],
        n_trials=1,
        n_warmup=0,
        seed_base=123,
        output_dir=tmp_path,
        dtype=np.float64,
        output_atol=1e-8,
        output_rtol=1e-5,
    )

    assert [row["scenario"] for row in summary_rows[:2]] == ["static", "static_est"]
    assert [(row["direction"], row["k"]) for row in summary_rows[2:]] == [
        ("up", 1),
        ("up", 2),
        ("down", 1),
        ("down", 2),
    ]
    assert len(output_rows) == 4
    for row in output_rows:
        assert Path(str(row["path"])).exists()


def test_benchmark_config_omits_unspecified_direction_rows(tmp_path):
    op = _FakeOp(directions=("up",))
    summary_rows, output_rows = benchmark_config(
        op=op,
        label="cfg-x",
        ks=[1],
        options=["baseline"],
        n_trials=1,
        n_warmup=0,
        seed_base=123,
        output_dir=tmp_path,
        dtype=np.float64,
        output_atol=1e-8,
        output_rtol=1e-5,
    )

    assert [row["direction"] for row in summary_rows] == ["-", "-", "up"]
    assert "skip" not in summary_rows[-1]
    assert len(output_rows) == 1
    assert output_rows[0]["direction"] == "up"
    assert output_rows[0]["scenario"] == "baseline"


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


def test_benchmark_config_checks_warmup_outputs(tmp_path):
    op = _FakeWarmupMismatchOp()
    summary_rows, output_rows = benchmark_config(
        op=op,
        label="cfg-x",
        ks=[1],
        options=["baseline"],
        n_trials=1,
        n_warmup=2,
        seed_base=123,
        output_dir=tmp_path,
        dtype=np.float64,
        output_atol=1e-8,
        output_rtol=1e-5,
    )
    assert len(output_rows) == 2
    up_row = next(row for row in summary_rows if row.get("scenario") == "baseline" and row.get("direction") == "up")
    assert up_row["intra_errors"] == 1
    assert up_row["intra_trials"] == 2
    assert up_row["intra_fail_indices"] == ["w2"]
    assert up_row["abs_err_avg"] == pytest.approx(0.5)
    assert up_row["abs_err_max"] == pytest.approx(1.0)
    assert up_row["rel_err_avg"] == pytest.approx(5e29)
    assert up_row["rel_err_max"] == pytest.approx(1e30)


def test_summarize_intra_diagnostics_counts_runtime_rows_only():
    rows = [
        {"scenario": "static"},
        {"scenario": "static_est"},
        {"scenario": "baseline", "intra_errors": 2, "intra_trials": 5},
        {"scenario": "init_vector", "intra_errors": 1, "intra_trials": 3},
        {"scenario": "miss", "skip": "reason", "intra_errors": 99, "intra_trials": 99},
    ]
    assert summarize_intra_diagnostics(rows) == (3, 8)
