"""Unit tests for benchmark summary and equivalence helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from pygrgl_spmv.backends.memory import MemoryRecord, MemoryUsage, RuntimeBytes, StaticBytes
from scripts.bench import (
    _validate_and_extract_runtime_memory,
    assert_output_equivalence,
    benchmark_config,
    print_summary_table,
)


def _save_arr(path, arr):
    np.save(path, arr, allow_pickle=False)
    return str(path)


@pytest.mark.smoke
def test_output_equivalence_passes_for_same_class(tmp_path):
    p0 = _save_arr(tmp_path / "a.npy", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    p1 = _save_arr(tmp_path / "b.npy", np.array([[1.0 + 1e-10, 2.0], [3.0, 4.0 - 1e-10]], dtype=np.float64))
    rows = [
        {"config": "cfg-a", "scenario": "baseline", "direction": "up", "k": 4, "path": p0},
        {"config": "cfg-b", "scenario": "baseline", "direction": "up", "k": 4, "path": p1},
    ]
    assert_output_equivalence(rows, atol=1e-8, rtol=1e-5)


def test_output_equivalence_fails_with_detailed_message(tmp_path):
    p0 = _save_arr(tmp_path / "a.npy", np.array([[1.0, 2.0]], dtype=np.float64))
    p1 = _save_arr(tmp_path / "b.npy", np.array([[1.0, 9.0]], dtype=np.float64))
    rows = [
        {"config": "cfg-a", "scenario": "baseline", "direction": "up", "k": 4, "path": p0},
        {"config": "cfg-b", "scenario": "baseline", "direction": "up", "k": 4, "path": p1},
    ]
    with pytest.raises(AssertionError, match="Output mismatch"):
        assert_output_equivalence(rows, atol=1e-8, rtol=1e-5)


def test_output_equivalence_maps_up_miss_to_baseline(tmp_path):
    p0 = _save_arr(tmp_path / "a.npy", np.array([[11.0, 12.0]], dtype=np.float64))
    p1 = _save_arr(tmp_path / "b.npy", np.array([[11.0, 12.0]], dtype=np.float64))
    rows = [
        {"config": "cfg-a", "scenario": "baseline", "direction": "up", "k": 2, "path": p0},
        {"config": "cfg-b", "scenario": "miss", "direction": "up", "k": 2, "path": p1},
    ]
    assert_output_equivalence(rows, atol=1e-8, rtol=1e-5)


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
    def __init__(self):
        self.mem_usage = MemoryUsage()
        self.mem_usage.host_static.level_offsets = 8
        self.mem_usage.device_static.blocks_up = 16

    def estimate_static_bytes(self):
        host = StaticBytes(blocks_up=4)
        device = StaticBytes(blocks_up=12)
        return host, device


class _FakeOp:
    def __init__(self):
        self.n = 3
        self.m = 2
        self.K = 4
        self.num_individuals = 3
        self.sel_miss = sp.csr_matrix((self.m, self.K), dtype=np.float64)
        self._backend = _FakeBackend()

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
    with pytest.raises(AssertionError, match="warmup"):
        benchmark_config(
            op=op,
            label="cfg-x",
            ks=[1],
            options=["baseline"],
            n_trials=1,
            n_warmup=2,
            seed_base=123,
            output_dir=tmp_path,
        )
