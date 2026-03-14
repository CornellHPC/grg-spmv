"""Minimal NVTX helpers for backend instrumentation paths."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator


def _format_field(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


@dataclass(frozen=True)
class NvtxTracer:
    prefix: str
    mark_fn: Callable[[str], None]
    push_fn: Callable[[str], Any]
    pop_fn: Callable[[], Any]

    def label(self, name: str, /, **fields: object) -> str:
        tokens = [f"{key}={_format_field(value)}" for key, value in fields.items() if value is not None]
        suffix = "" if not tokens else "[" + ",".join(tokens) + "]"
        return f"{self.prefix}.{name}{suffix}"

    def mark(self, name: str, /, **fields: object) -> None:
        self.mark_fn(self.label(name, **fields))

    @contextmanager
    def range(self, name: str, /, **fields: object) -> Iterator[None]:
        self.push_fn(self.label(name, **fields))
        try:
            yield
        finally:
            self.pop_fn()


def make_cupy_tracer(prefix: str, cp: Any) -> NvtxTracer:
    return NvtxTracer(
        prefix=prefix,
        mark_fn=cp.cuda.nvtx.Mark,
        push_fn=cp.cuda.nvtx.RangePush,
        pop_fn=cp.cuda.nvtx.RangePop,
    )


def make_torch_tracer(prefix: str, torch_mod: Any) -> NvtxTracer:
    return NvtxTracer(
        prefix=prefix,
        mark_fn=torch_mod.cuda.nvtx.mark,
        push_fn=torch_mod.cuda.nvtx.range_push,
        pop_fn=torch_mod.cuda.nvtx.range_pop,
    )


__all__ = ["NvtxTracer", "make_cupy_tracer", "make_torch_tracer"]
