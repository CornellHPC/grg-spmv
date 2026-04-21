from __future__ import annotations

import argparse
import numpy as np
import pytest

from pygrgl_spmv.backends.cusparse import SpMMAlgorithm
from scripts.bench.cusparse import (
    DEFAULT_CUSPARSE_PLAN_NAME,
    add_cusparse_plan_arg,
    parse_cusparse_plan,
)
from scripts.bench.cli import add_common_args, parse_args
from scripts.bench.run import benchmark_runtime


class _FakeGrg:
    num_samples = 5
    num_mutations = 7

    def __init__(self, calls: list[tuple[str, tuple[int, int]]]) -> None:
        self._calls = calls

    def matmul(self, matrix, direction):
        arr = np.asarray(matrix)
        self._calls.append((str(direction), tuple(int(v) for v in arr.shape)))
        width = self.num_mutations if str(direction) == "up" else self.num_samples
        return np.zeros((arr.shape[0], width), dtype=arr.dtype)


class _FakeRuntime:
    def __init__(self, layout) -> None:
        self._layout = layout
        self.grgs = (_FakeGrg(self._layout["calls"]),)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_benchmark_runtime_honors_warmup_and_trials(capsys):
    calls: list[tuple[str, tuple[int, int]]] = []
    benchmark_runtime(
        runtime_cls=_FakeRuntime,
        layout={"calls": calls},
        direction="up",
        k=3,
        dtype=np.float64,
        warmup=2,
        trials=5,
    )
    assert calls == [("up", (3, 5))] * 7
    assert "mean_ms=" in capsys.readouterr().out


def _parse_gpu_bench_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    add_common_args(parser, gpu=True)
    return parse_args(parser.parse_args(argv), gpu=True)


def test_gpu_bench_parser_defaults_allow_residency():
    args = _parse_gpu_bench_args(["--artifact", "/tmp/a.grg_spmv", "--vram-budget-bytes", "123"])
    assert args.allow_residency is True


def test_gpu_bench_parser_accepts_no_allow_residency():
    args = _parse_gpu_bench_args(
        ["--artifact", "/tmp/a.grg_spmv", "--vram-budget-bytes", "123", "--no-allow-residency"]
    )
    assert args.allow_residency is False


def _parse_cusparse_plan_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    add_cusparse_plan_arg(parser)
    return parser.parse_args(argv)


def test_cusparse_plan_arg_defaults_to_exhaustive_best_plan():
    args = _parse_cusparse_plan_args([])
    pair = args.plan
    assert pair == parse_cusparse_plan(DEFAULT_CUSPARSE_PLAN_NAME)
    assert pair.plan_up.store == "N"
    assert pair.plan_up.fmt == "CSR"
    assert pair.plan_up.order_b.name == "COL"
    assert pair.plan_up.order_c.name == "COL"
    assert pair.plan_up.algo == SpMMAlgorithm.DEFAULT
    assert pair.plan_down.store == "T"
    assert pair.plan_down.fmt == "CSC"
    assert pair.plan_down.order_b.name == "COL"
    assert pair.plan_down.order_c.name == "COL"
    assert pair.plan_down.algo == SpMMAlgorithm.DEFAULT


def test_cusparse_plan_arg_accepts_named_preset():
    args = _parse_cusparse_plan_args(["--plan", "shared-csr-reinterpret"])
    assert args.plan.plan_up.store == "N"
    assert args.plan.plan_up.fmt == "CSR"
    assert args.plan.plan_up.op_b.name == "T"
    assert args.plan.plan_up.order_b.name == "COL"
    assert args.plan.plan_down.store == "T"
    assert args.plan.plan_down.fmt == "CSC"
    assert args.plan.plan_down.op_b.name == "T"
    assert args.plan.plan_down.order_b.name == "COL"


def test_cusparse_plan_arg_accepts_json_plan():
    args = _parse_cusparse_plan_args(
        [
            "--plan",
            (
                '{"plan_up": {"store": "N", "fmt": "CSR", "opA": "N", '
                '"opB": "N", "orderB": "ROW", "orderC": "ROW", '
                '"algo": "CSR_ALG1"}, "plan_down": null}'
            ),
        ]
    )
    assert args.plan.plan_up.algo == SpMMAlgorithm.CSR_ALG1
    assert args.plan.plan_down is None


def test_cusparse_plan_arg_rejects_invalid_value():
    with pytest.raises(SystemExit):
        _parse_cusparse_plan_args(["--plan", "not-a-plan"])
