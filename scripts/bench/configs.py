"""Plan-pair parsing and backend-builder expansion for benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import re

import numpy as np

from pygrgl_spmv.backends import BackendBase

PlanSpec = dict[str, str]
PlanPairSpec = tuple[PlanSpec | None, PlanSpec | None]
_PAIR_LITERAL_RE = re.compile(r"^\[(.*?)\]\[(.*?)\]$")


@dataclass(frozen=True)
class BenchConfig:
    label: str
    backend_name: str
    ordering: str
    intra_block_ordering: str
    plan_up_text: str | None
    plan_down_text: str | None
    instrumentation: bool
    build_backend: Callable[[], BackendBase]


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


def spec_to_literal(spec: Mapping[str, str]) -> str:
    return "[" + ",".join(f"{key}={spec[key]}" for key in spec) + "]"


def spec_has_pattern(spec: PlanSpec | None) -> bool:
    return spec is not None and any(value == "*" or str(value).startswith("!") for value in spec.values())


def _render_plan_text(plan: str | None) -> str:
    return "<unspecified>" if plan is None else plan


def make_config_label(
    backend_name: str,
    ordering: str,
    intra_block_ordering: str,
    plan_up_text: str | None,
    plan_down_text: str | None,
) -> str:
    return (
        f"{backend_name}-order={ordering}-intra={intra_block_ordering}-"
        f"up={_render_plan_text(plan_up_text)}-down={_render_plan_text(plan_down_text)}"
    )


def _bench_config(
    *,
    backend_name: str,
    ordering: str,
    intra_block_ordering: str,
    plan_up_text: str | None,
    plan_down_text: str | None,
    log_level: str,
    instrumentation: bool,
    build_backend: Callable[[], BackendBase],
) -> BenchConfig:
    label = make_config_label(backend_name, ordering, intra_block_ordering, plan_up_text, plan_down_text)
    if instrumentation:
        label = f"{label}-instr"
    return BenchConfig(
        label=label,
        backend_name=backend_name,
        ordering=str(ordering),
        intra_block_ordering=str(intra_block_ordering),
        plan_up_text=plan_up_text,
        plan_down_text=plan_down_text,
        instrumentation=bool(instrumentation),
        build_backend=build_backend,
    )


def expand_mkl_configs(
    plan_pair_specs: list[PlanPairSpec],
    orderings: list[str],
    intra_block_orderings: list[str],
    log_level: str,
    instrumentation: bool = False,
) -> list[BenchConfig]:
    from pygrgl_spmv.backends.mkl import MklBackend, MklPlan, MklPlanPair

    configs: list[BenchConfig] = []
    for ordering in orderings:
        for intra_block_ordering in intra_block_orderings:
            for up_spec, down_spec in plan_pair_specs:
                if spec_has_pattern(up_spec) or spec_has_pattern(down_spec):
                    raise ValueError("MKL benchmark plans must be fully concrete; wildcard/negation expansion is cuSPARSE-only for now")
                pair = MklPlanPair.from_dicts(up_spec, down_spec)
                configs.append(
                    _bench_config(
                        backend_name="mkl",
                        ordering=ordering,
                        intra_block_ordering=intra_block_ordering,
                        plan_up_text=None if pair.plan_up is None else str(pair.plan_up),
                        plan_down_text=None if pair.plan_down is None else str(pair.plan_down),
                        log_level=log_level,
                        instrumentation=instrumentation,
                        build_backend=lambda pair=pair, log_level=log_level, instrumentation=instrumentation: MklBackend(
                            pair=pair,
                            log_level=log_level,
                            instrumentation=instrumentation,
                        ),
                    )
                )
    return configs


def _expand_cusparse_side(spec: PlanSpec | None, *, want_up: bool):
    import pygrgl
    from pygrgl_spmv.backends.cusparse import CusparsePlan

    if spec is None:
        return [None]
    direction = pygrgl.TraversalDirection.UP if want_up else pygrgl.TraversalDirection.DOWN
    return [
        plan
        for plan in CusparsePlan.expand_literal(spec_to_literal(spec))
        if plan.supported and plan.direction == direction
    ]


def expand_cusparse_configs(
    plan_pair_specs: list[PlanPairSpec],
    orderings: list[str],
    intra_block_orderings: list[str],
    log_level: str,
    instrumentation: bool = False,
) -> list[BenchConfig]:
    from pygrgl_spmv.backends.cusparse import CusparseBackend, CusparsePlanPair

    configs: list[BenchConfig] = []
    for ordering in orderings:
        for intra_block_ordering in intra_block_orderings:
            for up_spec, down_spec in plan_pair_specs:
                for plan_up in _expand_cusparse_side(up_spec, want_up=True):
                    for plan_down in _expand_cusparse_side(down_spec, want_up=False):
                        pair = CusparsePlanPair(plan_up=plan_up, plan_down=plan_down)
                        configs.append(
                            _bench_config(
                                backend_name="cusparse",
                                ordering=ordering,
                                intra_block_ordering=intra_block_ordering,
                                plan_up_text=None if pair.plan_up is None else str(pair.plan_up),
                                plan_down_text=None if pair.plan_down is None else str(pair.plan_down),
                                log_level=log_level,
                                instrumentation=instrumentation,
                                build_backend=lambda pair=pair, log_level=log_level, instrumentation=instrumentation: CusparseBackend(
                                    pair=pair,
                                    log_level=log_level,
                                    instrumentation=instrumentation,
                                ),
                            )
                        )
    return configs


def _expand_triton_candidates(raw: str, all_values, parser, *, field_name: str):
    token = str(raw).strip()
    if token == "*":
        return list(all_values)
    if token.startswith("!"):
        excluded = [piece for piece in token.split("!") if piece]
        if not excluded:
            raise ValueError(f"Invalid negation token for {field_name}: {raw!r}")
        excluded_values = {parser(piece) for piece in excluded}
        return [value for value in all_values if value not in excluded_values]
    return [parser(token)]


def _expand_triton_side(spec: PlanSpec | None, *, want_up: bool):
    from pygrgl_spmv.backends.triton import TritonPlan
    from pygrgl_spmv.backends.types import SparseFormat, StoredMatrix, parse_sparse_format, parse_store

    if spec is None:
        return [None]
    allowed_keys = {"k_hint", "store", "fmt", "scratch"}
    extra = sorted(set(spec) - allowed_keys)
    if extra:
        raise ValueError(f"Unknown Triton plan field(s): {extra}")
    required_keys = {"k_hint", "store", "fmt"}
    missing = sorted(required_keys - set(spec))
    if missing:
        raise ValueError(f"Missing Triton plan field(s): {missing}")
    k_hint_token = str(spec["k_hint"]).strip().lower()
    if k_hint_token not in {"1", "none"}:
        raise ValueError(f"Triton benchmark plans require k_hint=none or 1, got {spec['k_hint']!r}")
    scratch = str(spec.get("scratch", "none")).strip()
    if scratch == "*" or scratch.startswith("!"):
        raise ValueError(f"Triton benchmark scratch does not support wildcard/negation, got {scratch!r}")

    required_store = StoredMatrix.N if want_up else StoredMatrix.T
    stores = _expand_triton_candidates(spec["store"], list(StoredMatrix), parse_store, field_name="store")
    fmts = _expand_triton_candidates(spec["fmt"], [SparseFormat.CSR, SparseFormat.CSC], parse_sparse_format, field_name="fmt")
    plans = [
        TritonPlan.from_dict({"k_hint": None if k_hint_token == "none" else 1, "store": store.value, "fmt": fmt.value, "scratch": scratch})
        for store in stores
        if store == required_store
        for fmt in fmts
    ]
    plans.sort(key=str)
    return plans


def expand_triton_configs(
    plan_pair_specs: list[PlanPairSpec],
    orderings: list[str],
    intra_block_orderings: list[str],
    log_level: str,
    instrumentation: bool = False,
) -> list[BenchConfig]:
    from pygrgl_spmv.backends.triton import TritonBackend, TritonPlanPair

    configs: list[BenchConfig] = []
    for ordering in orderings:
        for intra_block_ordering in intra_block_orderings:
            for up_spec, down_spec in plan_pair_specs:
                for plan_up in _expand_triton_side(up_spec, want_up=True):
                    for plan_down in _expand_triton_side(down_spec, want_up=False):
                        pair = TritonPlanPair(plan_up=plan_up, plan_down=plan_down)
                        configs.append(
                            _bench_config(
                                backend_name="triton",
                                ordering=ordering,
                                intra_block_ordering=intra_block_ordering,
                                plan_up_text=None if pair.plan_up is None else str(pair.plan_up),
                                plan_down_text=None if pair.plan_down is None else str(pair.plan_down),
                                log_level=log_level,
                                instrumentation=instrumentation,
                                build_backend=lambda pair=pair, log_level=log_level, instrumentation=instrumentation: TritonBackend(
                                    pair=pair,
                                    log_level=log_level,
                                    instrumentation=instrumentation,
                                ),
                            )
                        )
    return configs


def format_dry_run_line(entry: BenchConfig, ks: list[int], options: list[str], *, dtype: np.dtype, index_dtype: np.dtype) -> str:
    parts = [
        entry.label,
        f"ks={','.join(str(k) for k in ks)}",
        f"options={','.join(options)}",
        f"ordering={entry.ordering}",
        f"intra_block_ordering={entry.intra_block_ordering}",
        f"dtype={np.dtype(dtype).name}",
        f"index_dtype={np.dtype(index_dtype).name}",
        f"instrumentation={'on' if entry.instrumentation else 'off'}",
        f"plan_up={_render_plan_text(entry.plan_up_text)}",
        f"plan_down={_render_plan_text(entry.plan_down_text)}",
    ]
    return " ".join(parts)


__all__ = [
    "BenchConfig",
    "PlanPairSpec",
    "PlanSpec",
    "expand_cusparse_configs",
    "expand_mkl_configs",
    "expand_triton_configs",
    "format_dry_run_line",
    "make_config_label",
    "parse_plan_pair_literal",
    "spec_has_pattern",
    "spec_to_literal",
]
