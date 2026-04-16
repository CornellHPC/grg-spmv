"""Concrete MKL plan types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from pygrgl_spmv.backends.types import (
    SparseFormat,
    StoredMatrix,
    parse_sparse_format,
    parse_store,
    transpose_compatible_format,
)


@dataclass(frozen=True)
class MklPlan:
    store: StoredMatrix
    fmt: SparseFormat
    n_threads: int = 0
    optimize: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "store", parse_store(self.store))
        object.__setattr__(self, "fmt", parse_sparse_format(self.fmt))
        if int(self.n_threads) < 0:
            raise ValueError(f"n_threads must be >= 0, got {self.n_threads}")
        object.__setattr__(self, "optimize", bool(self.optimize))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "MklPlan":
        allowed = {"store", "fmt", "n_threads", "optimize"}
        extra = sorted(set(value) - allowed)
        if extra:
            raise ValueError(f"unknown MklPlan field(s): {extra}")
        return cls(
            store=parse_store(value["store"]),
            fmt=parse_sparse_format(value["fmt"]),
            n_threads=int(value.get("n_threads", 0)),
            optimize=bool(value.get("optimize", True)),
        )

    def can_share_storage_with(self, other: "MklPlan") -> bool:
        if self.store == other.store:
            return self.fmt == other.fmt
        return transpose_compatible_format(self.fmt) == other.fmt


@dataclass(frozen=True)
class MklPlanPair:
    plan_up: MklPlan | None
    plan_down: MklPlan | None

    def __post_init__(self) -> None:
        if self.plan_up is None and self.plan_down is None:
            raise ValueError("at least one of plan_up/plan_down must be provided")

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


__all__ = ["MklPlan", "MklPlanPair"]
