"""Concrete cuSPARSE plan types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping

import pygrgl

from pygrgl_spmv.backends.cusparse.ffi import (
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_TRANSPOSE,
    CUSPARSE_ORDER_COL,
    CUSPARSE_ORDER_ROW,
    CUSPARSE_SPMM_ALG_DEFAULT,
    CUSPARSE_SPMM_COO_ALG1,
    CUSPARSE_SPMM_COO_ALG2,
    CUSPARSE_SPMM_COO_ALG3,
    CUSPARSE_SPMM_COO_ALG4,
    CUSPARSE_SPMM_CSR_ALG1,
    CUSPARSE_SPMM_CSR_ALG2,
    CUSPARSE_SPMM_CSR_ALG3,
)
from pygrgl_spmv.backends.types import SparseFormat, StoredMatrix, parse_sparse_format, parse_store, transpose_compatible_format


class Operation(IntEnum):
    N = CUSPARSE_OPERATION_NON_TRANSPOSE
    T = CUSPARSE_OPERATION_TRANSPOSE


class DenseOrder(IntEnum):
    COL = CUSPARSE_ORDER_COL
    ROW = CUSPARSE_ORDER_ROW


class SpMMAlgorithm(IntEnum):
    DEFAULT = CUSPARSE_SPMM_ALG_DEFAULT
    COO_ALG1 = CUSPARSE_SPMM_COO_ALG1
    COO_ALG2 = CUSPARSE_SPMM_COO_ALG2
    COO_ALG3 = CUSPARSE_SPMM_COO_ALG3
    COO_ALG4 = CUSPARSE_SPMM_COO_ALG4
    CSR_ALG1 = CUSPARSE_SPMM_CSR_ALG1
    CSR_ALG2 = CUSPARSE_SPMM_CSR_ALG2
    CSR_ALG3 = CUSPARSE_SPMM_CSR_ALG3


def _parse_operation(value) -> Operation:
    if isinstance(value, Operation):
        return value
    key = str(value).strip().upper()
    if key == "N":
        return Operation.N
    if key == "T":
        return Operation.T
    return Operation(int(key))


def _parse_dense_order(value) -> DenseOrder:
    if isinstance(value, DenseOrder):
        return value
    key = str(value).strip().upper()
    if key == "ROW":
        return DenseOrder.ROW
    if key == "COL":
        return DenseOrder.COL
    return DenseOrder(int(key))


def _parse_algo(value) -> SpMMAlgorithm:
    if isinstance(value, SpMMAlgorithm):
        return value
    key = str(value).strip().upper()
    return SpMMAlgorithm[key]


def _normalize_scratch(value: object) -> str:
    token = "none" if value is None else str(value).strip().lower()
    if token in {"", "none"}:
        return "none"
    if token == "all":
        return "all"
    parts = token.split("|")
    if any(part == "" for part in parts):
        raise ValueError(f"invalid cuSPARSE scratch specification: {value!r}")
    levels: list[int] = []
    seen: set[int] = set()
    for part in parts:
        level = int(part)
        if level < 0:
            raise ValueError(f"cuSPARSE scratch levels must be non-negative, got {value!r}")
        if level in seen:
            raise ValueError(f"duplicate cuSPARSE scratch level {level} in {value!r}")
        seen.add(level)
        levels.append(level)
    return "|".join(str(level) for level in sorted(levels))


@dataclass(frozen=True)
class CusparsePlan:
    store: StoredMatrix
    fmt: SparseFormat
    op_a: Operation
    op_b: Operation
    order_b: DenseOrder
    order_c: DenseOrder
    algo: SpMMAlgorithm
    scratch: str = "none"

    def __post_init__(self) -> None:
        object.__setattr__(self, "store", parse_store(self.store))
        object.__setattr__(self, "fmt", parse_sparse_format(self.fmt))
        object.__setattr__(self, "op_a", _parse_operation(self.op_a))
        object.__setattr__(self, "op_b", _parse_operation(self.op_b))
        object.__setattr__(self, "order_b", _parse_dense_order(self.order_b))
        object.__setattr__(self, "order_c", _parse_dense_order(self.order_c))
        object.__setattr__(self, "algo", _parse_algo(self.algo))
        object.__setattr__(self, "scratch", _normalize_scratch(self.scratch))
        if not self.supported:
            raise ValueError(f"unsupported cuSPARSE plan: {self}")

    @property
    def direction(self):
        return (
            pygrgl.TraversalDirection.UP
            if (self.store == StoredMatrix.N) == (self.op_a == Operation.N)
            else pygrgl.TraversalDirection.DOWN
        )

    @property
    def supported(self) -> bool:
        if self.algo == SpMMAlgorithm.DEFAULT:
            return self.fmt in {SparseFormat.CSR, SparseFormat.CSC, SparseFormat.COO}
        if self.algo in {SpMMAlgorithm.COO_ALG1, SpMMAlgorithm.COO_ALG2, SpMMAlgorithm.COO_ALG3, SpMMAlgorithm.COO_ALG4}:
            return self.fmt == SparseFormat.COO
        if self.algo in {SpMMAlgorithm.CSR_ALG1, SpMMAlgorithm.CSR_ALG2}:
            return self.fmt in {SparseFormat.CSR, SparseFormat.CSC}
        if self.algo == SpMMAlgorithm.CSR_ALG3:
            return self.fmt == SparseFormat.CSR and self.op_a == Operation.N
        return False

    @classmethod
    def from_dict(cls, mapping: Mapping[str, object]) -> "CusparsePlan":
        allowed = {"store", "fmt", "opA", "opB", "orderB", "orderC", "algo", "scratch"}
        extra = sorted(set(mapping) - allowed)
        if extra:
            raise ValueError(f"unknown CusparsePlan field(s): {extra}")
        return cls(
            store=parse_store(mapping["store"]),
            fmt=parse_sparse_format(mapping["fmt"]),
            op_a=_parse_operation(mapping["opA"]),
            op_b=_parse_operation(mapping["opB"]),
            order_b=_parse_dense_order(mapping["orderB"]),
            order_c=_parse_dense_order(mapping["orderC"]),
            algo=_parse_algo(mapping["algo"]),
            scratch=mapping.get("scratch", "none"),
        )

    def can_share_storage_with(self, other: "CusparsePlan") -> bool:
        if self.store == other.store:
            return self.fmt == other.fmt
        if self.fmt == SparseFormat.COO and other.fmt == SparseFormat.COO:
            return False
        return transpose_compatible_format(self.fmt) == other.fmt


@dataclass(frozen=True)
class CusparsePlanPair:
    plan_up: CusparsePlan | None
    plan_down: CusparsePlan | None

    def __post_init__(self) -> None:
        if self.plan_up is None and self.plan_down is None:
            raise ValueError("at least one of plan_up/plan_down must be provided")
        if self.plan_up is not None and str(getattr(self.plan_up.direction, "name", self.plan_up.direction)).lower() != "up":
            raise ValueError(f"cuSPARSE plan_up expects a UP plan, got {self.plan_up}")
        if self.plan_down is not None and str(getattr(self.plan_down.direction, "name", self.plan_down.direction)).lower() != "down":
            raise ValueError(f"cuSPARSE plan_down expects a DOWN plan, got {self.plan_down}")

    @classmethod
    def from_dicts(
        cls,
        plan_up: Mapping[str, object] | None,
        plan_down: Mapping[str, object] | None,
    ) -> "CusparsePlanPair":
        return cls(
            plan_up=None if plan_up is None else CusparsePlan.from_dict(plan_up),
            plan_down=None if plan_down is None else CusparsePlan.from_dict(plan_down),
        )


__all__ = [
    "CusparsePlan",
    "CusparsePlanPair",
    "DenseOrder",
    "Operation",
    "SpMMAlgorithm",
]
