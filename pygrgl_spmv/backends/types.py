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
