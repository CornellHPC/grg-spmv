"""Unit tests for benchmark reporting helpers with the flat memory ledger."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import scipy.sparse as sp

from pygrgl_spmv.memory import (
    MemoryAllocation,
    MemoryBinding,
    MemoryLedger,
    MemoryRole,
    MemorySnapshot,
    alloc_field,
    capture_snapshot,
    child_field,
    live_snapshot,
)
from pygrgl_spmv.backends.mkl import MklPlan
from scripts.bench.cli import parse_dtype, parse_index_dtype, tolerances_for_dtype
from scripts.bench.configs import BenchConfig, format_dry_run_line
from scripts.bench.report import (
    bytes_to_gib,
    evaluate_output_equivalence,
    print_common_config,
    print_memory_table,
    print_runtime_table,
    summarize_config_display,
)
from scripts.bench.run import _memory_rows, _validate_and_extract_runtime_memory, benchmark_config


def _alloc(
    *,
    nbytes: int,
    storage: str,
    owner: str,
    retention: str,
    activity: str,
    label: str,
    kind: str,
    direction: str | None = None,
    slot_k: int | None = None,
) -> MemoryAllocation:
    return MemoryAllocation(
        nbytes=nbytes,
        storage=storage,
        bindings=frozenset(
            (
                MemoryBinding(
                    owner=owner,
                    retention=retention,
                    activity=activity,
                    direction=direction,
                    slot_k=slot_k,
                    roles=frozenset((MemoryRole(label=label, kind=kind),)),
                ),
            )
        ),
    )


def _snapshot(*, stage: str, k: int, direction: str, node_bytes: int) -> MemorySnapshot:
    return MemorySnapshot(
        stage=stage,
        runtime_k=k,
        allocations=[
            _alloc(
                nbytes=node_bytes,
                storage="torch",
                owner="backend",
                retention="call",
                activity="yes",
                label="node_state",
                kind="state",
            ),
        ],
        direction=direction,
        meta={"mode": "graph"},
    )


@dataclass
class _CallMem:
    node_state: np.ndarray | None = alloc_field(label="node_state", kind="state", owner="backend", retention="call", activity="yes", default=None)


@dataclass
class _RootMem:
    call: _CallMem = child_field(owner="backend", retention="call", activity="yes")


def test_dtype_parsers_and_tolerances():
    assert parse_dtype("float32") == np.dtype(np.float32)
    assert parse_dtype("float64") == np.dtype(np.float64)
    assert parse_index_dtype("int32") == np.dtype(np.int32)
    assert parse_index_dtype("int64") == np.dtype(np.int64)
    assert tolerances_for_dtype(np.float32) == (1e-2, 1e-1)
    assert tolerances_for_dtype(np.float64) == (1e-8, 1e-5)


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


def test_summarize_config_display_single_config_moves_everything_to_common():
    configs = [
        BenchConfig(
            label="cfg-a",
            backend_name="cusparse",
            plan_up_text="[k_hint=2,store=T,fmt=CSC,opA=T]",
            plan_down_text="[k_hint=2,store=T,fmt=CSC,opA=N]",
            instrumentation=False,
            build_backend=lambda: None,  # type: ignore[return-value]
        )
    ]
    common_lines, display_by_label = summarize_config_display(configs)
    assert "backend=cusparse" in common_lines
    assert "instrumentation=off" in common_lines
    assert "up[k_hint=2,store=T,fmt=CSC,opA=T]" in common_lines
    assert "down[k_hint=2,store=T,fmt=CSC,opA=N]" in common_lines
    assert display_by_label["cfg-a"] == "-"


def test_summarize_config_display_shows_only_differences():
    configs = [
        BenchConfig(
            label="cfg-a",
            backend_name="cusparse",
            plan_up_text="[k_hint=2,store=T,fmt=CSC,opA=T,algo=CSR_ALG1]",
            plan_down_text="[k_hint=2,store=T,fmt=CSC,opA=N,algo=CSR_ALG1]",
            instrumentation=False,
            build_backend=lambda: None,  # type: ignore[return-value]
        ),
        BenchConfig(
            label="cfg-b",
            backend_name="cusparse",
            plan_up_text="[k_hint=2,store=T,fmt=CSC,opA=N,algo=CSR_ALG1]",
            plan_down_text="[k_hint=2,store=T,fmt=CSC,opA=N,algo=CSR_ALG2]",
            instrumentation=False,
            build_backend=lambda: None,  # type: ignore[return-value]
        ),
    ]
    common_lines, display_by_label = summarize_config_display(configs)
    assert "backend=cusparse" in common_lines
    assert "instrumentation=off" in common_lines
    assert "up[k_hint=2,store=T,fmt=CSC,algo=CSR_ALG1]" in common_lines
    assert "down[k_hint=2,store=T,fmt=CSC,opA=N]" in common_lines
    assert display_by_label["cfg-a"] == "up[opA=T] down[algo=CSR_ALG1]"
    assert display_by_label["cfg-b"] == "up[opA=N] down[algo=CSR_ALG2]"


def test_summarize_config_display_falls_back_to_labels_when_reduced_text_collides():
    configs = [
        BenchConfig(
            label="baseline",
            backend_name="cusparse",
            plan_up_text="[k_hint=2,store=T,fmt=CSC,opA=T]",
            plan_down_text="[k_hint=2,store=T,fmt=CSC,opA=N]",
            instrumentation=False,
            build_backend=lambda: None,  # type: ignore[return-value]
        ),
        BenchConfig(
            label="candidate",
            backend_name="cusparse",
            plan_up_text="[k_hint=2,store=T,fmt=CSC,opA=T]",
            plan_down_text="[k_hint=2,store=T,fmt=CSC,opA=N]",
            instrumentation=False,
            build_backend=lambda: None,  # type: ignore[return-value]
        ),
    ]
    common_lines, display_by_label = summarize_config_display(configs)
    assert "backend=cusparse" in common_lines
    assert display_by_label["baseline"] == "baseline"
    assert display_by_label["candidate"] == "candidate"


def test_summarize_config_display_rejects_duplicate_labels():
    configs = [
        BenchConfig(
            label="dup",
            backend_name="cusparse",
            plan_up_text="[k_hint=2,store=T,fmt=CSC,opA=T]",
            plan_down_text=None,
            instrumentation=False,
            build_backend=lambda: None,  # type: ignore[return-value]
        ),
        BenchConfig(
            label="dup",
            backend_name="mkl",
            plan_up_text="[k_hint=none,store=N,fmt=CSR,n_threads=1]",
            plan_down_text=None,
            instrumentation=False,
            build_backend=lambda: None,  # type: ignore[return-value]
        ),
    ]
    with pytest.raises(ValueError, match="labels must be unique"):
        summarize_config_display(configs)


def test_output_equivalence_passes_for_same_class():
    rows = [
        {"config": "cfg-a", "scenario": "baseline", "direction": "up", "k": 4, "output": np.array([[1.0, 2.0]])},
        {"config": "cfg-b", "scenario": "baseline", "direction": "up", "k": 4, "output": np.array([[1.0, 2.0]])},
    ]
    summary, per_config = evaluate_output_equivalence(rows, atol=1e-8, rtol=1e-5)
    assert summary == {"classes": 1, "comparisons": 1, "errors": 0}
    assert per_config["cfg-a"] == {"failures": 0, "trials": 1}
    assert per_config["cfg-b"] == {"failures": 0, "trials": 1}


def test_validate_runtime_memory_fails_when_timed_values_vary():
    calls = [
        _snapshot(stage="run_up", k=4, direction="up", node_bytes=10),
        _snapshot(stage="run_up", k=4, direction="up", node_bytes=10),
        _snapshot(stage="run_up", k=4, direction="up", node_bytes=11),
    ]
    retained = [None, None, None]
    with pytest.raises(AssertionError, match="Call memory shape varies across timed trials"):
        _validate_and_extract_runtime_memory(call_slice=calls, retained_slice=retained, direction="up", k=4, n_warmup=0, n_trials=2)


def test_validate_runtime_memory_ignores_physical_allocation_identity():
    calls = [
        capture_snapshot(_RootMem(call=_CallMem(node_state=np.ones((4,), dtype=np.float64))), stage="run_up", runtime_k=4, direction="up", meta={"mode": "graph"}),
        capture_snapshot(_RootMem(call=_CallMem(node_state=np.ones((4,), dtype=np.float64))), stage="run_up", runtime_k=4, direction="up", meta={"mode": "graph"}),
        capture_snapshot(_RootMem(call=_CallMem(node_state=np.ones((4,), dtype=np.float64))), stage="run_up", runtime_k=4, direction="up", meta={"mode": "graph"}),
    ]
    retained = [None, None, None]
    snapshot = _validate_and_extract_runtime_memory(call_slice=calls, retained_slice=retained, direction="up", k=4, n_warmup=0, n_trials=2)
    assert int(snapshot.runtime_k) == 4


def test_validate_runtime_memory_ignores_active_retained_pointer_churn():
    retained_a = MemorySnapshot(
        stage="retained",
        runtime_k=None,
        allocations=[
            _alloc(
                nbytes=8,
                storage="torch",
                owner="backend",
                retention="captured",
                activity="no",
                direction="up",
                slot_k=4,
                label="node_state",
                kind="state",
            ),
        ],
        meta={},
        _alloc_keys=(("retained_alloc", 1),),
    )
    retained_b = MemorySnapshot(
        stage="retained",
        runtime_k=None,
        allocations=[
            _alloc(
                nbytes=8,
                storage="torch",
                owner="backend",
                retention="captured",
                activity="no",
                direction="up",
                slot_k=4,
                label="node_state",
                kind="state",
            ),
        ],
        meta={},
        _alloc_keys=(("retained_alloc", 2),),
    )
    calls = [
        MemorySnapshot(
            stage="run_up",
            runtime_k=4,
            allocations=[
                _alloc(
                    nbytes=10,
                    storage="torch",
                    owner="caller",
                    retention="call",
                    activity="yes",
                    label="node_state",
                    kind="state",
                ),
            ],
            direction="up",
            active_alloc_keys={retained_a._alloc_keys[0]},
            meta={"mode": "graph"},
        ),
        MemorySnapshot(
            stage="run_up",
            runtime_k=4,
            allocations=[
                _alloc(
                    nbytes=10,
                    storage="torch",
                    owner="caller",
                    retention="call",
                    activity="yes",
                    label="node_state",
                    kind="state",
                ),
            ],
            direction="up",
            active_alloc_keys={retained_a._alloc_keys[0]},
            meta={"mode": "graph"},
        ),
        MemorySnapshot(
            stage="run_up",
            runtime_k=4,
            allocations=[
                _alloc(
                    nbytes=10,
                    storage="torch",
                    owner="caller",
                    retention="call",
                    activity="yes",
                    label="node_state",
                    kind="state",
                ),
            ],
            direction="up",
            active_alloc_keys={retained_b._alloc_keys[0]},
            meta={"mode": "graph"},
        ),
    ]

    snapshot = _validate_and_extract_runtime_memory(
        call_slice=calls,
        retained_slice=[retained_a, retained_a, retained_b],
        direction="up",
        k=4,
        n_warmup=0,
        n_trials=2,
    )

    assert int(snapshot.runtime_k) == 4


def test_validate_runtime_memory_still_fails_when_shape_metadata_changes():
    calls = [
        _snapshot(stage="run_up", k=4, direction="up", node_bytes=10),
        _snapshot(stage="run_up", k=4, direction="up", node_bytes=10),
        MemorySnapshot(
            stage="run_up",
            runtime_k=4,
            allocations=[
                _alloc(
                    nbytes=10,
                    storage="torch",
                    owner="caller",
                    retention="call",
                    activity="yes",
                    label="node_state",
                    kind="state",
                ),
            ],
            direction="up",
            meta={"mode": "graph"},
        ),
    ]
    retained = [None, None, None]
    with pytest.raises(AssertionError, match="Call memory shape varies across timed trials"):
        _validate_and_extract_runtime_memory(call_slice=calls, retained_slice=retained, direction="up", k=4, n_warmup=0, n_trials=2)


def test_live_snapshot_promotes_active_allocation_rows_without_duplication():
    retained = MemorySnapshot(
        stage="retained",
        runtime_k=None,
        allocations=[
            _alloc(
                nbytes=8,
                storage="torch",
                owner="backend",
                retention="captured",
                activity="no",
                direction="up",
                slot_k=4,
                label="node_state",
                kind="state",
            ),
        ],
        meta={},
    )
    last_call = MemorySnapshot(
        stage="run_up",
        runtime_k=4,
        allocations=[
            _alloc(
                nbytes=4,
                storage="numpy",
                owner="caller",
                retention="call",
                activity="yes",
                label="output",
                kind="output",
            ),
        ],
        direction="up",
        active_alloc_keys={retained._alloc_keys[0]},
        meta={"mode": "graph"},
    )
    merged = live_snapshot(retained, last_call)
    assert merged is not None
    assert len(merged.allocations) == 2
    assert merged.allocations[0].activity == "yes"


def test_memory_rows_preserve_retained_binding_paths_and_case_columns():
    retained = MemorySnapshot(
        stage="retained",
        runtime_k=None,
        allocations=[
            _alloc(
                nbytes=8,
                storage="torch",
                owner="backend",
                retention="captured",
                activity="no",
                direction="down",
                slot_k=1,
                label="cache_hit",
                kind="state",
            ),
            _alloc(
                nbytes=6,
                storage="torch",
                owner="backend",
                retention="staging",
                activity="no",
                direction="up",
                slot_k=3,
                label="idle_stage",
                kind="temporary",
            ),
        ],
        meta={},
    )
    last_call = MemorySnapshot(
        stage="run_up",
        runtime_k=2,
        allocations=[
            _alloc(
                nbytes=4,
                storage="numpy",
                owner="caller",
                retention="call",
                activity="yes",
                label="output",
                kind="output",
            ),
        ],
        direction="up",
        active_alloc_keys={retained._alloc_keys[0]},
        meta={"mode": "graph"},
    )
    merged = live_snapshot(retained, last_call)
    assert merged is not None
    rows = _memory_rows(label="cfg-a", scenario="baseline", direction="up", k=2, snapshot=merged)

    promoted = next(row for row in rows if row["node"] == "cache_hit")
    inactive = next(row for row in rows if row["node"] == "idle_stage")
    assert promoted["case_direction"] == "up"
    assert promoted["case_k"] == 2
    assert promoted["parent"] == "cuda_live/retained/down/k=1"
    assert promoted["active"] == "yes"
    assert inactive["case_direction"] == "up"
    assert inactive["case_k"] == 2
    assert inactive["parent"] == "cuda_live/retained/up/k=3"
    assert inactive["active"] == "no"


def test_memory_snapshot_rejects_reserved_semantic_meta_keys():
    with pytest.raises(ValueError, match="reserved semantic key"):
        MemorySnapshot(
            stage="run_up",
            runtime_k=4,
            allocations=[
                _alloc(
                    nbytes=10,
                    storage="torch",
                    owner="backend",
                    retention="call",
                    activity="yes",
                    label="node_state",
                    kind="state",
                )
            ],
            meta={"active_alloc_keys": frozenset()},
        )


class _FakeBackend:
    def __init__(self, *, directions=("up", "down")):
        self._plan_up = object() if "up" in directions else None
        self._plan_down = object() if "down" in directions else None


class _FakeOp:
    def __init__(self, *, directions=("up", "down")):
        self.num_samples = 3
        self.num_mutations = 2
        self.num_nodes = 4
        self.num_individuals = 3
        self.sel_miss = sp.csr_matrix((self.num_mutations, self.num_nodes), dtype=np.float64)
        self._backend = _FakeBackend(directions=directions)
        self.memory = MemoryLedger()
        self.memory.retained = None

    def matmul(self, matrix, direction, **_kwargs):
        k = int(matrix.shape[0])
        if direction == "up":
            self.memory.last_call = _snapshot(stage="run_up", k=k, direction="up", node_bytes=200)
            return np.zeros((k, self.num_mutations), dtype=np.float64)
        if direction == "down":
            self.memory.last_call = _snapshot(stage="run_down", k=k, direction="down", node_bytes=400)
            return np.zeros((k, self.num_samples), dtype=np.float64)
        raise ValueError(direction)


def test_benchmark_config_orders_rows_by_execution():
    import scripts.bench.run as bench_run

    op = _FakeOp()
    original_reference = bench_run._ground_truth_output
    bench_run._ground_truth_output = lambda **kwargs: np.zeros(
        (
            kwargs["matrix"].shape[0],
            op.num_mutations if kwargs["direction"] == "up" else op.num_samples,
        ),
        dtype=np.float64,
    )
    try:
        runtime_rows, memory_rows, _ = benchmark_config(
            op=op,
            grg_ref=object(),
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

    assert [row["k"] for row in runtime_rows] == [1, 2, 1, 2]
    assert memory_rows[0]["node"] == "cuda_live"
    assert memory_rows[1]["node"] == "call"


def test_compact_memory_rows_aggregate_repeated_leaves():
    snapshot = MemorySnapshot(
        stage="run_up",
        runtime_k=2,
        allocations=[
            MemoryAllocation(
                nbytes=4,
                storage="torch",
                bindings=frozenset(
                    (
                        MemoryBinding(
                            owner="backend",
                            retention="persistent",
                            activity="always",
                            roles=frozenset(
                                (
                                    MemoryRole(label="blocks_up", kind="sparse"),
                                    MemoryRole(label="blocks_down", kind="sparse"),
                                )
                            ),
                        ),
                    )
                ),
            ),
            MemoryAllocation(
                nbytes=6,
                storage="torch",
                bindings=frozenset(
                    (
                        MemoryBinding(
                            owner="backend",
                            retention="persistent",
                            activity="always",
                            roles=frozenset(
                                (
                                    MemoryRole(label="blocks_up", kind="sparse"),
                                    MemoryRole(label="blocks_down", kind="sparse"),
                                )
                            ),
                        ),
                    )
                ),
            ),
            _alloc(
                nbytes=8,
                storage="numpy",
                owner="caller",
                retention="call",
                activity="yes",
                label="output",
                kind="output",
            ),
        ],
        direction="up",
        meta={"mode": "graph"},
    )
    rows = _memory_rows(label="cfg-a", scenario="baseline", direction="up", k=2, snapshot=snapshot)
    block_rows = [row for row in rows if row["node"] == "blocks_down|blocks_up"]
    assert len(block_rows) == 1
    assert float(block_rows[0]["gib"]) == bytes_to_gib(10)
    assert block_rows[0]["parent"] == "cuda_live/retained"
    assert all("backend/always" not in str(row["parent"]) for row in rows)


def test_compact_memory_rows_accept_multi_slot_retained_paths():
    snapshot = MemorySnapshot(
        stage="run_up",
        runtime_k=2,
        allocations=[
            MemoryAllocation(
                nbytes=8,
                storage="torch",
                bindings=frozenset(
                    (
                        MemoryBinding(
                            owner="backend",
                            retention="staging",
                            activity="no",
                            direction="up",
                            slot_k=1,
                            roles=frozenset((MemoryRole(label="shared_stage", kind="temporary"),)),
                        ),
                        MemoryBinding(
                            owner="backend",
                            retention="staging",
                            activity="no",
                            direction="up",
                            slot_k=2,
                            roles=frozenset((MemoryRole(label="shared_stage", kind="temporary"),)),
                        ),
                    )
                ),
            ),
        ],
        direction="up",
        meta={"mode": "graph"},
    )
    rows = _memory_rows(label="cfg-a", scenario="baseline", direction="up", k=2, snapshot=snapshot)
    shared = next(row for row in rows if row["node"] == "shared_stage")
    assert shared["parent"] == "cuda_live/retained/up/k=1|2"
    assert shared["active"] == "no"


def test_memory_table_renders_compact_columns_and_level_separators(capsys):
    rows = [
        {
            "config": "cfg-a",
            "scenario": "baseline",
            "case_direction": "up",
            "case_k": 1,
            "level": 0,
            "node": "cuda_live",
            "parent": "-",
            "gib": 1.0,
            "space": "cuda",
            "owner": "-",
            "active": "-",
            "retention": "-",
            "kinds": "",
            "note": "mode=graph",
        },
        {
            "config": "cfg-a",
            "scenario": "baseline",
            "case_direction": "up",
            "case_k": 1,
            "level": 1,
            "node": "node_state",
            "parent": "cuda_live/call/up/k=1",
            "gib": 0.5,
            "space": "cuda",
            "owner": "backend",
            "active": "yes",
            "retention": "call",
            "kinds": "state",
            "note": "",
        },
    ]
    print_memory_table(rows)
    out = capsys.readouterr().out
    assert "CaseDir" in out
    assert "CaseK" in out
    assert "Parent" in out
    assert "Space" in out
    assert "Owner" in out
    assert "Active" in out
    assert "Retention" in out
    assert "Kinds" in out
    assert "Labels" not in out
    assert "LEVEL 1" in out
    assert "cuda_live/call/up/k=1" in out


def test_runtime_table_uses_compact_config_display_and_common_banner(capsys):
    configs = [
        BenchConfig(
            label="cfg-a",
            backend_name="mkl",
            plan_up_text="[k_hint=none,store=N,fmt=CSR,n_threads=1]",
            plan_down_text=None,
            instrumentation=False,
            build_backend=lambda: None,  # type: ignore[return-value]
        )
    ]
    common_lines, display_by_label = summarize_config_display(configs)
    print_common_config(common_lines)
    print_runtime_table(
        [
            {
                "config": "cfg-a",
                "scenario": "baseline",
                "direction": "up",
                "k": 2,
                "call_ms_mean": 1.0,
                "call_ms_std": 0.0,
                "note": "mode=dynamic",
                "intra_errors": 0,
                "intra_trials": 1,
                "intra_fail_indices": [],
                "abs_err_avg": 0.0,
                "abs_err_max": 0.0,
                "rel_err_avg": 0.0,
                "rel_err_max": 0.0,
            }
        ],
        config_display=display_by_label,
    )
    out = capsys.readouterr().out
    assert "Common config:" in out
    assert "backend=mkl" in out
    assert "up[k_hint=none,store=N,fmt=CSR,n_threads=1]" in out
    assert "cfg-a" not in out


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
    assert "2/5" in out
    assert "intra_fail=w2,b1" in out
