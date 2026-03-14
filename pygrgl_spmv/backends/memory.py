"""Memory usage tracking for backend setup and matmul calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass
class StaticBytes:
    """Persistent memory components allocated during backend setup."""

    level_offsets: int = 0
    sample_perm: int = 0
    inv_sample_perm: int = 0
    coalescence_counts: int = 0
    xtx_init: int = 0
    blocks_up: int = 0
    blocks_down: int = 0
    selector_mut: int = 0
    selector_miss: int = 0
    workspace: int = 0

    def total(self) -> int:
        return (
            int(self.level_offsets)
            + int(self.sample_perm)
            + int(self.inv_sample_perm)
            + int(self.coalescence_counts)
            + int(self.xtx_init)
            + int(self.blocks_up)
            + int(self.blocks_down)
            + int(self.selector_mut)
            + int(self.selector_miss)
            + int(self.workspace)
        )


@dataclass
class RuntimeBytes:
    """Per-call memory components for one matmul invocation."""

    level_buffers: int = 0
    inputs: int = 0
    outputs: int = 0
    aux: int = 0

    def total(self) -> int:
        return int(self.level_buffers) + int(self.inputs) + int(self.outputs) + int(self.aux)


@dataclass
class MemoryRecord:
    stage: str
    runtime_k: int | None
    host: RuntimeBytes = field(default_factory=RuntimeBytes)
    device: RuntimeBytes = field(default_factory=RuntimeBytes)
    meta: dict[str, object] = field(default_factory=dict)


@dataclass
class MemoryUsage:
    """Lifecycle memory tracker for one backend instance."""

    host_static: StaticBytes = field(default_factory=StaticBytes)
    device_static: StaticBytes = field(default_factory=StaticBytes)
    calls: list[MemoryRecord] = field(default_factory=list)

    def reset(self) -> None:
        self.host_static = StaticBytes()
        self.device_static = StaticBytes()
        self.calls.clear()

    def record(
        self,
        *,
        stage: str,
        runtime_k: int | None,
        host_runtime: RuntimeBytes | Mapping[str, int] | None = None,
        device_runtime: RuntimeBytes | Mapping[str, int] | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> None:
        host = _normalize_runtime(host_runtime)
        device = _normalize_runtime(device_runtime)
        self.calls.append(
            MemoryRecord(
                stage=str(stage),
                runtime_k=None if runtime_k is None else int(runtime_k),
                host=host,
                device=device,
                meta=dict(meta) if meta else {},
            )
        )


def _normalize_runtime(values: RuntimeBytes | Mapping[str, int] | None) -> RuntimeBytes:
    if values is None:
        return RuntimeBytes()
    if isinstance(values, RuntimeBytes):
        return RuntimeBytes(
            level_buffers=int(values.level_buffers),
            inputs=int(values.inputs),
            outputs=int(values.outputs),
            aux=int(values.aux),
        )
    runtime = RuntimeBytes()
    for key, value in values.items():
        if not hasattr(runtime, key):
            raise KeyError(f"Unknown runtime memory attribute: {key!r}")
        setattr(runtime, key, int(value))
    return runtime


__all__ = [
    "MemoryRecord",
    "MemoryUsage",
    "RuntimeBytes",
    "StaticBytes",
]
