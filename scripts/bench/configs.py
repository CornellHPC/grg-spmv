"""Plan-pair parsing and backend config expansion for benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

PlanSpec = dict[str, str]
PlanPairSpec = tuple[PlanSpec | None, PlanSpec | None]

_PAIR_LITERAL_RE = re.compile(r"^\[(.*?)\]\[(.*?)\]$")


@dataclass(frozen=True)
class BenchConfig:
    label: str
    config: dict[str, object]


def _parse_plan_group(body: str, *, raw: str) -> PlanSpec | None:
    if body == "":
        return None
    mapping: PlanSpec = {}
    for part in (chunk.strip() for chunk in body.split(",") if chunk.strip()):
        if "=" not in part:
            raise ValueError(f"Invalid plan field {part!r} in {raw!r}")
        key, value = (token.strip() for token in part.split("=", 1))
        if not key or not value:
            raise ValueError(f"Invalid plan field {part!r} in {raw!r}")
        if key in mapping:
            raise ValueError(f"Duplicate plan field {key!r} in {raw!r}")
        mapping[key] = value
    return mapping


def parse_plan_pair_literal(raw: str) -> PlanPairSpec:
    value = "".join(str(raw).split())
    match = _PAIR_LITERAL_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid --plan-up-down literal {raw!r}; expected [..][..]")
    pair = (_parse_plan_group(match.group(1), raw=raw), _parse_plan_group(match.group(2), raw=raw))
    if pair == (None, None):
        raise ValueError("Invalid --plan-up-down literal; both sides cannot be empty")
    return pair


def spec_to_literal(spec: PlanSpec) -> str:
    return "[" + ",".join(f"{key}={spec[key]}" for key in spec) + "]"


def spec_has_pattern(spec: PlanSpec | None) -> bool:
    return spec is not None and any(value == "*" or str(value).startswith("!") for value in spec.values())


def render_plan(plan: object | None) -> str:
    if plan is None:
        return "<unspecified>"
    return str(plan)


def make_config_label(backend: str, plan_up: object | None, plan_down: object | None) -> str:
    return f"{backend}-up={render_plan(plan_up)}-down={render_plan(plan_down)}"


def _bench_config(backend: str, plan_up: object | None, plan_down: object | None, *, log_level: str) -> BenchConfig:
    return BenchConfig(
        label=make_config_label(backend, plan_up, plan_down),
        config={
            "type": backend,
            "plan_up": plan_up,
            "plan_down": plan_down,
            "log_level": str(log_level).upper(),
        },
    )


def expand_mkl_configs(plan_pair_specs: list[PlanPairSpec], log_level: str) -> list[BenchConfig]:
    from pygrgl_spmv.backends.mkl import MklPlan

    configs: list[BenchConfig] = []
    for up_spec, down_spec in plan_pair_specs:
        if spec_has_pattern(up_spec) or spec_has_pattern(down_spec):
            raise ValueError("MKL benchmark plans must be fully concrete; wildcard/negation expansion is cuSPARSE-only for now")
        configs.append(
            _bench_config(
                "mkl",
                None if up_spec is None else MklPlan.from_any(up_spec),
                None if down_spec is None else MklPlan.from_any(down_spec),
                log_level=log_level,
            )
        )
    return configs


def _cusparse_runtime_supported(plan) -> bool:
    return plan.supported and not (plan.fmt == plan.fmt.CSC and plan.algo == plan.algo.CSR_ALG3)


def _expand_cusparse_side(spec: PlanSpec | None, *, want_up: bool):
    import pygrgl
    from pygrgl_spmv.backends.cusparse import CusparsePlan

    if spec is None:
        return [None]
    direction = pygrgl.TraversalDirection.UP if want_up else pygrgl.TraversalDirection.DOWN
    return [
        plan
        for plan in CusparsePlan.expand_literal(spec_to_literal(spec))
        if _cusparse_runtime_supported(plan) and plan.direction == direction
    ]


def expand_cusparse_configs(plan_pair_specs: list[PlanPairSpec], log_level: str) -> list[BenchConfig]:
    configs: list[BenchConfig] = []
    for up_spec, down_spec in plan_pair_specs:
        for plan_up in _expand_cusparse_side(up_spec, want_up=True):
            for plan_down in _expand_cusparse_side(down_spec, want_up=False):
                configs.append(_bench_config("cusparse", plan_up, plan_down, log_level=log_level))
    return configs


def format_dry_run_line(entry: BenchConfig, ks, options, *, dtype, index_dtype) -> str:
    cfg = entry.config
    parts = [
        entry.label,
        f"ks={','.join(str(k) for k in ks)}",
        f"options={','.join(options)}",
        f"dtype={np.dtype(dtype).name}",
        f"index_dtype={np.dtype(index_dtype).name}",
    ]
    if "plan_up" in cfg:
        parts.append(f"plan_up={render_plan(cfg['plan_up'])}")
    if "plan_down" in cfg:
        parts.append(f"plan_down={render_plan(cfg['plan_down'])}")
    return " ".join(parts)


__all__ = [
    "BenchConfig",
    "PlanPairSpec",
    "PlanSpec",
    "expand_cusparse_configs",
    "expand_mkl_configs",
    "format_dry_run_line",
    "make_config_label",
    "parse_plan_pair_literal",
    "render_plan",
    "spec_has_pattern",
    "spec_to_literal",
]
