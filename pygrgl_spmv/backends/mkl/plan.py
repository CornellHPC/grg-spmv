"""MKL plan type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from pygrgl_spmv.backends import _parse_optional_k_hint
from pygrgl_spmv.backends.types import (
    Direction,
    SparseFormat,
    StoredMatrix,
    parse_sparse_format,
    parse_store,
    transpose_compatible_format,
)


def _parse_cpu_affinity(value: object) -> "tuple[int, ...] | None":
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(int(c) for c in value)
    s = str(value).strip()
    result: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))
    return tuple(result)


def _parse_bool_field(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    raise ValueError(f"Expected boolean (true/false), got {value!r}")


@dataclass(frozen=True)
class MklPlan:
    """Explicit MKL traversal/storage plan."""

    store: StoredMatrix
    fmt: SparseFormat
    n_threads: int = 0
    k_hint: int | None = None
    optimize: bool = True
    cpu_affinity: "tuple[int, ...] | None" = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_hint", _parse_optional_k_hint(self.k_hint))
        object.__setattr__(self, "cpu_affinity", _parse_cpu_affinity(self.cpu_affinity))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "MklPlan":
        allowed_keys = {"store", "fmt", "n_threads", "k_hint", "optimize", "cpu_affinity"}
        extra = sorted(set(value) - allowed_keys)
        if extra:
            raise ValueError(f"Unknown MklPlan field(s): {extra}")
        return cls(
            store=parse_store(value["store"]),
            fmt=parse_sparse_format(value["fmt"]),
            n_threads=int(value.get("n_threads", 0)),
            k_hint=_parse_optional_k_hint(value.get("k_hint")),
            optimize=_parse_bool_field(value.get("optimize"), True),
            cpu_affinity=_parse_cpu_affinity(value.get("cpu_affinity")),
        )

    def can_share_storage_with(self, other: "MklPlan") -> bool:
        if self.store == other.store:
            return self.fmt == other.fmt
        return transpose_compatible_format(self.fmt) == other.fmt

    def __str__(self) -> str:
        return (
            "["
            f"k_hint={'none' if self.k_hint is None else self.k_hint},"
            f"store={self.store.value},fmt={self.fmt.value},"
            f"n_threads={self.n_threads},"
            f"optimize={self.optimize}"
            "]"
        )

@dataclass(frozen=True)
class MklPlanPair:
    plan_up: MklPlan | None
    plan_down: MklPlan | None

    def __post_init__(self) -> None:
        if self.plan_up is None and self.plan_down is None:
            raise ValueError("At least one of plan_up/plan_down must be provided")

    @classmethod
    def from_dicts(
        cls,
        plan_up: Mapping[str, object] | None,
        plan_down: Mapping[str, object] | None,
    ) -> "MklPlanPair":
        return cls(
            plan_up=None if plan_up is None else MklPlan.from_dict(plan_up),
            plan_down=None if plan_down is None else MklPlan.from_dict(plan_down),
        )

    def direction_enabled(self, direction: Direction) -> bool:
        return (self.plan_up is not None) if direction == Direction.UP else (self.plan_down is not None)


__all__ = ["MklPlan", "MklPlanPair"]
