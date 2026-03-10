"""Shared backend enums and parsers."""

from __future__ import annotations

from enum import StrEnum


class Direction(StrEnum):
    """Traversal direction used by backend internals."""

    UP = "up"
    DOWN = "down"


class InitMode(StrEnum):
    """Node initialization modes for a single traversal."""

    NONE = "none"
    XTX = "xtx"
    VECTOR = "vector"
    MATRIX = "matrix"


class StoredMatrix(StrEnum):
    """Whether a backend stores ``A`` or ``A.T``."""

    N = "N"
    T = "T"


class SparseFormat(StrEnum):
    """Sparse storage formats shared by backends."""

    CSR = "CSR"
    CSC = "CSC"
    COO = "COO"


_TRANSPOSE_FMT_MAP = {
    SparseFormat.CSR: SparseFormat.CSC,
    SparseFormat.CSC: SparseFormat.CSR,
    SparseFormat.COO: SparseFormat.COO,
}


def parse_direction(direction: str | Direction) -> Direction:
    """Parse a traversal direction string/enum into Direction."""
    if isinstance(direction, Direction):
        return direction
    key = str(direction).lower()
    match key:
        case "up":
            return Direction.UP
        case "down":
            return Direction.DOWN
        case _:
            raise ValueError(f"Unknown direction {direction!r}; expected up or down")


def parse_init_mode(init_mode: str | InitMode) -> InitMode:
    """Parse an init mode string/enum into InitMode."""
    if isinstance(init_mode, InitMode):
        return init_mode
    key = str(init_mode).lower()
    match key:
        case "none":
            return InitMode.NONE
        case "xtx":
            return InitMode.XTX
        case "vector":
            return InitMode.VECTOR
        case "matrix":
            return InitMode.MATRIX
        case _:
            raise ValueError(
                "Unknown init_mode "
                f"{init_mode!r}; expected one of (none, xtx, vector, matrix)"
            )


def parse_store(value: str | StoredMatrix) -> StoredMatrix:
    """Parse whether a backend stores ``A`` or ``A.T``."""
    if isinstance(value, StoredMatrix):
        return value
    key = str(value).strip().upper()
    match key:
        case "N":
            return StoredMatrix.N
        case "T":
            return StoredMatrix.T
        case _:
            raise ValueError(f"Unknown store {value!r}; expected N or T")


def parse_sparse_format(value: str | SparseFormat) -> SparseFormat:
    """Parse sparse storage format."""
    if isinstance(value, SparseFormat):
        return value
    key = str(value).strip().upper()
    match key:
        case "CSR":
            return SparseFormat.CSR
        case "CSC":
            return SparseFormat.CSC
        case "COO":
            return SparseFormat.COO
        case _:
            raise ValueError(f"Unknown sparse format {value!r}; expected CSR, CSC, or COO")


def transpose_compatible_format(fmt: SparseFormat) -> SparseFormat:
    """Return the transpose-compatible sparse format."""
    return _TRANSPOSE_FMT_MAP[parse_sparse_format(fmt)]


__all__ = [
    "Direction",
    "InitMode",
    "StoredMatrix",
    "SparseFormat",
    "parse_direction",
    "parse_init_mode",
    "parse_sparse_format",
    "parse_store",
    "transpose_compatible_format",
]
