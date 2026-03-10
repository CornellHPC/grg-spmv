"""Public cuSPARSE SpMM planning types grounded in CUDA 12.9.0 docs.

The planner is intentionally pure and small: it parses explicit plan literals,
normalizes enum values, and exposes doc-driven properties such as support,
determinism, and storage sharing. Runtime execution still allocates buffers and
runs preprocess unconditionally for now; ``need_buffer`` and
``need_preprocess`` are future-optimization metadata.

The rules in this module are grounded on the CUDA 12.9.0 cuSPARSE SpMM docs:
https://docs.nvidia.com/cuda/archive/12.9.0/cusparse/index.html#cusparsespmm
"""

from __future__ import annotations

from functools import cache
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from itertools import product
from typing import Any, Mapping
import re

import pygrgl

from pygrgl_spmv.backends.cuda_utils import (
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
from pygrgl_spmv.backends import _parse_optional_k_hint
from pygrgl_spmv.backends.types import SparseFormat, StoredMatrix, parse_sparse_format, parse_store, transpose_compatible_format

_LITERAL_RE = re.compile(r"^\[(.*)\]$")
_REQUIRED_KEYS = ("k_hint", "store", "fmt", "opA", "opB", "orderB", "orderC", "algo")
_DOC_CUDA_VERSION = (12, 9, 0)
_SUPPORTED_CUDA_MAJOR = 12


class Operation(IntEnum):
    """cuSPARSE transpose flags."""

    N = CUSPARSE_OPERATION_NON_TRANSPOSE
    T = CUSPARSE_OPERATION_TRANSPOSE


class DenseOrder(IntEnum):
    """cuSPARSE dense matrix orders."""

    COL = CUSPARSE_ORDER_COL
    ROW = CUSPARSE_ORDER_ROW


class SpMMAlgorithm(IntEnum):
    """cuSPARSE SpMM algorithms."""

    DEFAULT = CUSPARSE_SPMM_ALG_DEFAULT
    COO_ALG1 = CUSPARSE_SPMM_COO_ALG1
    COO_ALG2 = CUSPARSE_SPMM_COO_ALG2
    COO_ALG3 = CUSPARSE_SPMM_COO_ALG3
    COO_ALG4 = CUSPARSE_SPMM_COO_ALG4
    CSR_ALG1 = CUSPARSE_SPMM_CSR_ALG1
    CSR_ALG2 = CUSPARSE_SPMM_CSR_ALG2
    CSR_ALG3 = CUSPARSE_SPMM_CSR_ALG3


def _probe_cuda_version() -> tuple[int, int, int]:
    import cupy as cp

    value = int(cp.cuda.runtime.runtimeGetVersion())
    return _validate_cuda_version((value // 1000, (value % 1000) // 10, value % 10))


def _validate_cuda_version(version: Any) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in tuple(version))
    if len(parts) != 3:
        raise ValueError(f"cuda_version must be a 3-tuple, got {version!r}")
    if parts[0] != _SUPPORTED_CUDA_MAJOR:
        raise ValueError(f"cuSPARSE backend requires a CUDA 12.x runtime, got {parts!r}")
    return parts


@cache
def _runtime_cuda_version() -> tuple[int, int, int]:
    return _validate_cuda_version(_probe_cuda_version())


def _parse_k_hint(value: Any) -> int | None:
    return _parse_optional_k_hint(value)


def _parse_operation(value: Any) -> Operation:
    if isinstance(value, Operation):
        return value
    if isinstance(value, IntEnum):
        return Operation(int(value))
    key = str(value).strip().upper()
    match key:
        case "N":
            return Operation.N
        case "T":
            return Operation.T
        case _:
            return Operation(int(key))


def _parse_dense_order(value: Any) -> DenseOrder:
    if isinstance(value, DenseOrder):
        return value
    if isinstance(value, IntEnum):
        return DenseOrder(int(value))
    key = str(value).strip().upper()
    match key:
        case "ROW":
            return DenseOrder.ROW
        case "COL":
            return DenseOrder.COL
        case _:
            return DenseOrder(int(key))


def _parse_algo(value: Any) -> SpMMAlgorithm:
    if isinstance(value, SpMMAlgorithm):
        return value
    if isinstance(value, IntEnum):
        return SpMMAlgorithm(int(value))
    key = str(value).strip().upper()
    match key:
        case "DEFAULT":
            return SpMMAlgorithm.DEFAULT
        case "COO_ALG1":
            return SpMMAlgorithm.COO_ALG1
        case "COO_ALG2":
            return SpMMAlgorithm.COO_ALG2
        case "COO_ALG3":
            return SpMMAlgorithm.COO_ALG3
        case "COO_ALG4":
            return SpMMAlgorithm.COO_ALG4
        case "CSR_ALG1":
            return SpMMAlgorithm.CSR_ALG1
        case "CSR_ALG2":
            return SpMMAlgorithm.CSR_ALG2
        case "CSR_ALG3":
            return SpMMAlgorithm.CSR_ALG3
        case _:
            return SpMMAlgorithm(int(key))


def _expand_candidates(raw: str, all_values, parser, *, field_name: str):
    token = str(raw).strip()
    if token == "*":
        return list(all_values)
    if field_name == "k_hint" and token.startswith("!"):
        raise ValueError("k_hint does not support negation")
    if token.startswith("!"):
        excluded = [piece for piece in token.split("!") if piece]
        if not excluded:
            raise ValueError(f"Invalid negation token for {field_name}: {raw!r}")
        excluded_values = {parser(piece) for piece in excluded}
        return [value for value in all_values if value not in excluded_values]
    return [parser(token)]




@dataclass(frozen=True)
class CusparsePlan:
    """Explicit cuSPARSE SpMM plan for one traversal direction and runtime-k policy."""

    k_hint: int | None
    store: StoredMatrix
    fmt: SparseFormat
    op_a: Operation
    op_b: Operation
    order_b: DenseOrder
    order_c: DenseOrder
    algo: SpMMAlgorithm

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_hint", _parse_k_hint(self.k_hint))

    @property
    def cuda_version(self) -> tuple[int, int, int]:
        return _runtime_cuda_version()

    @classmethod
    def from_any(cls, value: Any) -> CusparsePlan | None:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.from_literal(value)
        if isinstance(value, Mapping):
            return cls.from_mapping(value)
        raise TypeError(f"Cannot construct CusparsePlan from {type(value).__name__}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> CusparsePlan:
        extra = sorted(set(mapping) - set(_REQUIRED_KEYS))
        if extra:
            raise ValueError(f"Unknown CusparsePlan field(s): {extra}")
        missing = [key for key in _REQUIRED_KEYS if key not in mapping]
        if missing:
            raise ValueError(f"Missing CusparsePlan field(s): {missing}")
        return cls(
            k_hint=_parse_k_hint(mapping["k_hint"]),
            store=parse_store(mapping["store"]),
            fmt=parse_sparse_format(mapping["fmt"]),
            op_a=_parse_operation(mapping["opA"]),
            op_b=_parse_operation(mapping["opB"]),
            order_b=_parse_dense_order(mapping["orderB"]),
            order_c=_parse_dense_order(mapping["orderC"]),
            algo=_parse_algo(mapping["algo"]),
        )

    @classmethod
    def from_literal(cls, raw: str) -> CusparsePlan:
        mapping = parse_plan_literal(raw)
        if any(value == "*" or str(value).startswith("!") for value in mapping.values()):
            raise ValueError(f"Wildcard or negation is not allowed in concrete plan literal {raw!r}")
        return cls.from_mapping(mapping)

    @classmethod
    def expand_literal(cls, raw: str) -> list[CusparsePlan]:
        mapping = parse_plan_literal(raw)
        if mapping["k_hint"] == "*":
            raise ValueError("k_hint must not be '*' in plan literals")
        candidates = {
            "k_hint": [_parse_k_hint(mapping["k_hint"])],
            "store": _expand_candidates(mapping["store"], list(StoredMatrix), parse_store, field_name="store"),
            "fmt": _expand_candidates(mapping["fmt"], list(SparseFormat), parse_sparse_format, field_name="fmt"),
            "opA": _expand_candidates(mapping["opA"], list(Operation), _parse_operation, field_name="opA"),
            "opB": _expand_candidates(mapping["opB"], list(Operation), _parse_operation, field_name="opB"),
            "orderB": _expand_candidates(mapping["orderB"], list(DenseOrder), _parse_dense_order, field_name="orderB"),
            "orderC": _expand_candidates(mapping["orderC"], list(DenseOrder), _parse_dense_order, field_name="orderC"),
            "algo": _expand_candidates(mapping["algo"], list(SpMMAlgorithm), _parse_algo, field_name="algo"),
        }
        plans: list[CusparsePlan] = []
        for store, fmt, op_a, op_b, order_b, order_c, algo in product(
            candidates["store"],
            candidates["fmt"],
            candidates["opA"],
            candidates["opB"],
            candidates["orderB"],
            candidates["orderC"],
            candidates["algo"],
        ):
            plan = cls(
                k_hint=candidates["k_hint"][0],
                store=store,
                fmt=fmt,
                op_a=op_a,
                op_b=op_b,
                order_b=order_b,
                order_c=order_c,
                algo=algo,
            )
            if plan.supported:
                plans.append(plan)
        plans.sort(key=lambda plan: plan.sort_key())
        return plans

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
            return self.fmt in {SparseFormat.CSR, SparseFormat.CSC} and self.op_a == Operation.N
        return False

    @property
    def deterministic(self) -> bool:
        if self.algo == SpMMAlgorithm.CSR_ALG3:
            return True
        if self.algo == SpMMAlgorithm.COO_ALG2:
            return self.op_a == Operation.N
        return False

    @property
    def need_buffer(self) -> bool:
        return self.algo not in {SpMMAlgorithm.COO_ALG1, SpMMAlgorithm.COO_ALG3, SpMMAlgorithm.COO_ALG4}

    @property
    def need_preprocess(self) -> bool:
        return self.algo in {SpMMAlgorithm.DEFAULT, SpMMAlgorithm.CSR_ALG1, SpMMAlgorithm.CSR_ALG3}

    def can_share_storage_with(self, other: CusparsePlan) -> bool:
        if self.store == other.store:
            return self.fmt == other.fmt
        # The CUDA 12.9.0 cuSPARSE docs assume COO is sorted by row, so an opposite-
        # orientation transpose cannot safely reuse raw row/col storage by
        # simply swapping the pointers.
        if self.fmt == SparseFormat.COO and other.fmt == SparseFormat.COO:
            return False
        return transpose_compatible_format(self.fmt) == other.fmt

    def __str__(self) -> str:
        return (
            "["
            f"k_hint={'none' if self.k_hint is None else self.k_hint},"
            f"store={self.store.value},"
            f"fmt={self.fmt.value},"
            f"opA={self.op_a.name},"
            f"opB={self.op_b.name},"
            f"orderB={self.order_b.name},"
            f"orderC={self.order_c.name},"
            f"algo={self.algo.name}"
            "]"
        )

    def sort_key(self) -> tuple[Any, ...]:
        return (
            -1 if self.k_hint is None else int(self.k_hint),
            self.store.value,
            self.fmt.value,
            int(self.op_a),
            int(self.op_b),
            int(self.order_b),
            int(self.order_c),
            int(self.algo),
        )


def parse_plan_literal(raw: str) -> dict[str, str]:
    match = _LITERAL_RE.fullmatch(str(raw).strip())
    if match is None:
        raise ValueError(f"Invalid plan literal {raw!r}; expected [k=v,...]")
    body = match.group(1)
    pairs = [chunk.strip() for chunk in body.split(",") if chunk.strip()]
    mapping: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid plan field {pair!r} in {raw!r}")
        key, value = pair.split("=", 1)
        k = key.strip()
        v = value.strip()
        if not k or not v:
            raise ValueError(f"Invalid plan field {pair!r} in {raw!r}")
        if k in mapping:
            raise ValueError(f"Duplicate plan field {k!r} in {raw!r}")
        mapping[k] = v
    missing = [key for key in _REQUIRED_KEYS if key not in mapping]
    if missing:
        raise ValueError(f"Missing plan field(s) {missing} in {raw!r}")
    extra = sorted(set(mapping) - set(_REQUIRED_KEYS))
    if extra:
        raise ValueError(f"Unknown plan field(s) {extra} in {raw!r}")
    return mapping


__all__ = [
    "CusparsePlan",
    "DenseOrder",
    "Operation",
    "SpMMAlgorithm",
    "StoredMatrix",
    "SparseFormat",
    "parse_plan_literal",
]
