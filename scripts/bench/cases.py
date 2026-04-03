"""Benchmark input generation and scenario construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _BenchInputs:
    up_sample: np.ndarray
    down: np.ndarray
    init_vec: np.ndarray
    init_mat: np.ndarray
    miss_down: np.ndarray
    miss_up: np.ndarray
    up_indiv: np.ndarray | None


@dataclass(frozen=True)
class _ScenarioCase:
    matrix: np.ndarray | None
    kwargs_factory: Callable[[], dict[str, object]]
    skip_reason: str | None = None


def configured_direction_names(backend) -> list[str]:
    if hasattr(backend, "_configured_directions"):
        return [str(direction.value) for direction in backend._configured_directions()]

    directions = []
    _missing = object()
    backend_plan_up = getattr(backend, "_plan_up", _missing)
    backend_plan_down = getattr(backend, "_plan_down", _missing)
    if backend_plan_up is _missing and backend_plan_down is _missing:
        return ["up", "down"]
    if backend_plan_up is not None and backend_plan_up is not _missing:
        directions.append("up")
    if backend_plan_down is not None and backend_plan_down is not _missing:
        directions.append("down")
    return directions


def build_inputs_by_k(*, op, ks: list[int], seed_base: int, dtype: np.dtype) -> dict[int, _BenchInputs]:
    inputs_by_k: dict[int, _BenchInputs] = {}
    for k in ks:
        rng = np.random.default_rng(seed_base + int(k))
        up_sample = rng.standard_normal((k, op.num_samples), dtype=dtype)
        down = rng.standard_normal((k, op.num_mutations), dtype=dtype)
        init_vec = rng.standard_normal(k, dtype=dtype)
        init_mat = rng.standard_normal((k, op.num_nodes), dtype=dtype)
        miss_down = rng.standard_normal((k, op.num_mutations), dtype=dtype)
        miss_up = np.zeros((k, op.num_mutations), dtype=dtype)
        up_indiv = None
        if op.num_individuals != op.num_samples:
            up_indiv = rng.standard_normal((k, op.num_individuals), dtype=dtype)
        inputs_by_k[int(k)] = _BenchInputs(
            up_sample=up_sample,
            down=down,
            init_vec=init_vec,
            init_mat=init_mat,
            miss_down=miss_down,
            miss_up=miss_up,
            up_indiv=up_indiv,
        )
    return inputs_by_k


def build_case(*, op, scenario: str, direction: str, inputs: _BenchInputs) -> _ScenarioCase:
    backend_module = str(getattr(op._backend, "__class__", type(op._backend)).__module__).lower()
    runtime_k = int(inputs.up_sample.shape[0] if direction == "up" else inputs.down.shape[0])
    if "triton" in backend_module and runtime_k != 1:
        return _ScenarioCase(
            matrix=None,
            kwargs_factory=lambda: {},
            skip_reason="streamed Triton backend supports runtime k == 1 only",
        )

    if scenario == "baseline":
        return _ScenarioCase(
            matrix=inputs.up_sample if direction == "up" else inputs.down,
            kwargs_factory=lambda: {},
        )

    if scenario == "by_individual":
        if inputs.up_indiv is None:
            return _ScenarioCase(
                matrix=None,
                kwargs_factory=lambda: {},
                skip_reason="num_individuals == num_samples",
            )
        return _ScenarioCase(
            matrix=inputs.up_indiv if direction == "up" else inputs.down,
            kwargs_factory=lambda: {"by_individual": True},
        )

    if scenario == "init_xtx":
        return _ScenarioCase(
            matrix=inputs.up_sample if direction == "up" else inputs.down,
            kwargs_factory=lambda: {"init": "xtx"},
        )

    if scenario == "init_vector":
        return _ScenarioCase(
            matrix=inputs.up_sample if direction == "up" else inputs.down,
            kwargs_factory=lambda init_vec=inputs.init_vec: {"init": init_vec},
        )

    if scenario == "init_matrix":
        return _ScenarioCase(
            matrix=inputs.up_sample if direction == "up" else inputs.down,
            kwargs_factory=lambda init_mat=inputs.init_mat: {"init": init_mat},
        )

    if scenario == "miss":
        if op.sel_miss.nnz == 0:
            return _ScenarioCase(
                matrix=None,
                kwargs_factory=lambda: {},
                skip_reason="GRG has no missingness selector entries",
            )
        if direction == "up":

            def kwargs_factory(miss_up=inputs.miss_up):
                miss_up.fill(0.0)
                return {"miss": miss_up}

            return _ScenarioCase(matrix=inputs.up_sample, kwargs_factory=kwargs_factory)

        return _ScenarioCase(
            matrix=inputs.down,
            kwargs_factory=lambda miss_down=inputs.miss_down: {"miss": miss_down},
        )

    raise ValueError(f"Unhandled matmul scenario: {scenario}")


__all__ = [
    "_BenchInputs",
    "_ScenarioCase",
    "build_case",
    "build_inputs_by_k",
    "configured_direction_names",
]
