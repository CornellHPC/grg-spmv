from __future__ import annotations

import numpy as np

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
