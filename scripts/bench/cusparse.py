"""Benchmark cuSPARSE runtime configurations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping

from pygrgl_spmv.backends.cusparse import (
    CusparsePlanPair,
    CusparseRuntime,
    plan_cusparse_layout,
)

from .cli import add_common_args, parse_args
from .run import baseline_requirements, benchmark_runtime


DEFAULT_CUSPARSE_PLAN_NAME = "exhaustive-best"


def _plan(
    *,
    store: str,
    fmt: str,
    op_a: str = "N",
    op_b: str = "N",
    order_b: str = "ROW",
    order_c: str = "ROW",
    algo: str = "DEFAULT",
    scratch: str = "none",
) -> dict[str, str]:
    return {
        "store": store,
        "fmt": fmt,
        "opA": op_a,
        "opB": op_b,
        "orderB": order_b,
        "orderC": order_c,
        "algo": algo,
        "scratch": scratch,
    }


CUSPARSE_PLAN_PRESETS: dict[str, tuple[dict[str, str] | None, dict[str, str] | None]] = {
    DEFAULT_CUSPARSE_PLAN_NAME: (
        _plan(store="N", fmt="CSR", order_b="COL", order_c="COL"),
        _plan(store="T", fmt="CSC", order_b="COL", order_c="COL"),
    ),
    "shared-csr": (
        _plan(store="N", fmt="CSR"),
        _plan(store="T", fmt="CSC"),
    ),
    "shared-csr-reinterpret": (
        _plan(store="N", fmt="CSR", op_b="T", order_b="COL"),
        _plan(store="T", fmt="CSC", op_b="T", order_b="COL"),
    ),
}


def parse_cusparse_plan(value: str) -> CusparsePlanPair:
    token = str(value).strip()
    if token in CUSPARSE_PLAN_PRESETS:
        plan_up, plan_down = CUSPARSE_PLAN_PRESETS[token]
        return CusparsePlanPair.from_dicts(plan_up, plan_down)
    try:
        raw = json.loads(token)
    except json.JSONDecodeError as exc:
        known = ", ".join(sorted(CUSPARSE_PLAN_PRESETS))
        raise argparse.ArgumentTypeError(f"unknown cuSPARSE plan preset {token!r}; known presets: {known}") from exc
    if not isinstance(raw, Mapping):
        raise argparse.ArgumentTypeError("cuSPARSE JSON plan must be an object with plan_up and plan_down")
    allowed = {"plan_up", "plan_down"}
    extra = sorted(set(raw) - allowed)
    if extra:
        raise argparse.ArgumentTypeError(f"unknown cuSPARSE JSON plan field(s): {extra}")
    plan_up = raw.get("plan_up")
    plan_down = raw.get("plan_down")
    if plan_up is not None and not isinstance(plan_up, Mapping):
        raise argparse.ArgumentTypeError("cuSPARSE JSON plan_up must be an object or null")
    if plan_down is not None and not isinstance(plan_down, Mapping):
        raise argparse.ArgumentTypeError("cuSPARSE JSON plan_down must be an object or null")
    try:
        return CusparsePlanPair.from_dicts(plan_up, plan_down)
    except (KeyError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def add_cusparse_plan_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--plan",
        type=parse_cusparse_plan,
        default=parse_cusparse_plan(DEFAULT_CUSPARSE_PLAN_NAME),
        metavar="PLAN",
        help=(
            "cuSPARSE plan preset or JSON object with plan_up/plan_down. "
            f"Default: {DEFAULT_CUSPARSE_PLAN_NAME}. "
            f"Presets: {', '.join(sorted(CUSPARSE_PLAN_PRESETS))}"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CusparseRuntime")
    add_common_args(parser, gpu=True)
    add_cusparse_plan_arg(parser)
    raw_args = parser.parse_args()
    args = parse_args(raw_args, gpu=True)
    layout = plan_cusparse_layout(
        artifacts=[args.artifact],
        pair=raw_args.plan,
        dtype=args.dtype,
        requirements=baseline_requirements(direction=args.direction, k=args.k),
        vram_budget_bytes=args.vram_budget_bytes,
        ring_buffer_size=args.ring_buffer_size,
        allow_residency=args.allow_residency,
        device=args.device,
        stream=args.stream,
    )
    benchmark_runtime(
        runtime_cls=CusparseRuntime,
        layout=layout,
        direction=args.direction,
        k=args.k,
        dtype=args.dtype,
        warmup=args.warmup,
        trials=args.trials,
    )


if __name__ == "__main__":
    main()
