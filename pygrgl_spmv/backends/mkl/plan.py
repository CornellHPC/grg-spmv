"""MKL plan type."""

from __future__ import annotations

from dataclasses import dataclass

from pygrgl_spmv.backends import _parse_optional_k_hint
from pygrgl_spmv.backends.types import (
    SparseFormat,
    StoredMatrix,
    parse_sparse_format,
    parse_store,
    transpose_compatible_format,
)


@dataclass(frozen=True)
class MklPlan:
    """Explicit MKL traversal/storage plan."""

    store: StoredMatrix
    fmt: SparseFormat
    n_threads: int = 0
    k_hint: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_hint", _parse_optional_k_hint(self.k_hint))

    @classmethod
    def from_any(cls, value):
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            allowed_keys = {"store", "fmt", "n_threads", "k_hint"}
            extra = sorted(set(value) - allowed_keys)
            if extra:
                raise ValueError(f"Unknown MklPlan field(s): {extra}")
            n_threads = int(value.get("n_threads", 0))
            return cls(
                store=parse_store(value["store"]),
                fmt=parse_sparse_format(value["fmt"]),
                n_threads=n_threads,
                k_hint=_parse_optional_k_hint(value.get("k_hint")),
            )
        if hasattr(value, "store") and hasattr(value, "fmt"):
            n_threads = int(getattr(value, "n_threads", 0))
            return cls(
                store=parse_store(getattr(value, "store")),
                fmt=parse_sparse_format(getattr(value, "fmt")),
                n_threads=n_threads,
                k_hint=_parse_optional_k_hint(getattr(value, "k_hint", None)),
            )
        raise TypeError(f"Cannot construct MklPlan from {type(value).__name__}")

    def can_share_storage_with(self, other: "MklPlan") -> bool:
        if self.store == other.store:
            return self.fmt == other.fmt
        return transpose_compatible_format(self.fmt) == other.fmt

    def __str__(self) -> str:
        return (
            "["
            f"k_hint={'none' if self.k_hint is None else self.k_hint},"
            f"store={self.store.value},fmt={self.fmt.value},"
            f"n_threads={self.n_threads}"
            "]"
        )


__all__ = ["MklPlan"]
