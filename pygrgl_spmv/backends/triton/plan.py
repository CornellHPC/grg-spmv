"""Plan type for the Triton backend."""

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

_ALLOWED_FORMATS = {SparseFormat.CSR, SparseFormat.CSC}


def _normalize_scratch(value: object) -> str:
    token = "none" if value is None else str(value).strip().lower()
    if token in {"", "none"}:
        return "none"
    if token == "all":
        return "all"
    parts = token.split("|")
    if any(part == "" for part in parts):
        raise ValueError(f"Invalid Triton scratch specification: {value!r}")
    levels: list[int] = []
    seen: set[int] = set()
    for part in parts:
        level = int(part)
        if level < 0:
            raise ValueError(f"Triton scratch levels must be non-negative, got {value!r}")
        if level in seen:
            raise ValueError(f"Duplicate Triton scratch level {level} in {value!r}")
        seen.add(level)
        levels.append(level)
    return "|".join(str(level) for level in sorted(levels))


@dataclass(frozen=True)
class TritonPlan:
    """Explicit Triton traversal/storage plan for singleton-vector execution."""

    store: StoredMatrix
    fmt: SparseFormat
    k_hint: int
    scratch: str = "none"

    def __post_init__(self) -> None:
        parsed_k = _parse_optional_k_hint(self.k_hint)
        if parsed_k != 1:
            raise ValueError(f"Triton backend requires k_hint=1, got {self.k_hint!r}")
        if self.fmt not in _ALLOWED_FORMATS:
            raise ValueError(f"Triton backend supports only CSR/CSC formats, got {self.fmt.value}")
        object.__setattr__(self, "k_hint", int(parsed_k))
        object.__setattr__(self, "scratch", _normalize_scratch(self.scratch))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TritonPlan":
        allowed_keys = {"store", "fmt", "k_hint", "scratch"}
        extra = sorted(set(value) - allowed_keys)
        if extra:
            raise ValueError(f"Unknown TritonPlan field(s): {extra}")
        return cls(
            store=parse_store(value["store"]),
            fmt=parse_sparse_format(value["fmt"]),
            k_hint=value["k_hint"],
            scratch=value.get("scratch", "none"),
        )

    def can_share_storage_with(self, other: "TritonPlan") -> bool:
        if self.store == other.store:
            return self.fmt == other.fmt
        return transpose_compatible_format(self.fmt) == other.fmt

    def __str__(self) -> str:
        return f"[k_hint={self.k_hint},store={self.store.value},fmt={self.fmt.value},scratch={self.scratch}]"

@dataclass(frozen=True)
class TritonPlanPair:
    plan_up: TritonPlan | None
    plan_down: TritonPlan | None

    def __post_init__(self) -> None:
        if self.plan_up is None and self.plan_down is None:
            raise ValueError("At least one of plan_up/plan_down must be provided")
        if self.plan_up is not None and self.plan_up.store != StoredMatrix.N:
            raise ValueError(f"Triton plan_up expects store=N, got {self.plan_up.store.value}")
        if self.plan_down is not None and self.plan_down.store != StoredMatrix.T:
            raise ValueError(f"Triton plan_down expects store=T, got {self.plan_down.store.value}")

    @classmethod
    def from_dicts(
        cls,
        plan_up: Mapping[str, object] | None,
        plan_down: Mapping[str, object] | None,
    ) -> "TritonPlanPair":
        return cls(
            plan_up=None if plan_up is None else TritonPlan.from_dict(plan_up),
            plan_down=None if plan_down is None else TritonPlan.from_dict(plan_down),
        )

    def direction_enabled(self, direction: Direction) -> bool:
        return (self.plan_up is not None) if direction == Direction.UP else (self.plan_down is not None)


__all__ = ["TritonPlan", "TritonPlanPair"]
