"""Memory usage tracking for backend setup and matmul calls."""

from __future__ import annotations

from dataclasses import dataclass, field


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
class ResidencyBytes:
    """Retained residency for one reporting bucket."""

    host_bytes: int = 0
    device_bytes: int = 0
    note: str = ""


@dataclass
class MemoryRecord:
    stage: str
    runtime_k: int | None
    host: RuntimeBytes = field(default_factory=RuntimeBytes)
    device: RuntimeBytes = field(default_factory=RuntimeBytes)
    static_ws: ResidencyBytes = field(default_factory=ResidencyBytes)
    dynamic_ws: ResidencyBytes = field(default_factory=ResidencyBytes)
    staging: ResidencyBytes = field(default_factory=ResidencyBytes)
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
        host_runtime: RuntimeBytes | None = None,
        device_runtime: RuntimeBytes | None = None,
        static_ws: ResidencyBytes | None = None,
        dynamic_ws: ResidencyBytes | None = None,
        staging: ResidencyBytes | None = None,
        meta: dict[str, object] | None = None,
    ) -> None:
        self.calls.append(
            MemoryRecord(
                stage=str(stage),
                runtime_k=None if runtime_k is None else int(runtime_k),
                host=RuntimeBytes() if host_runtime is None else RuntimeBytes(
                    level_buffers=int(host_runtime.level_buffers),
                    inputs=int(host_runtime.inputs),
                    outputs=int(host_runtime.outputs),
                    aux=int(host_runtime.aux),
                ),
                device=RuntimeBytes() if device_runtime is None else RuntimeBytes(
                    level_buffers=int(device_runtime.level_buffers),
                    inputs=int(device_runtime.inputs),
                    outputs=int(device_runtime.outputs),
                    aux=int(device_runtime.aux),
                ),
                static_ws=ResidencyBytes() if static_ws is None else ResidencyBytes(
                    host_bytes=int(static_ws.host_bytes),
                    device_bytes=int(static_ws.device_bytes),
                    note=str(static_ws.note),
                ),
                dynamic_ws=ResidencyBytes() if dynamic_ws is None else ResidencyBytes(
                    host_bytes=int(dynamic_ws.host_bytes),
                    device_bytes=int(dynamic_ws.device_bytes),
                    note=str(dynamic_ws.note),
                ),
                staging=ResidencyBytes() if staging is None else ResidencyBytes(
                    host_bytes=int(staging.host_bytes),
                    device_bytes=int(staging.device_bytes),
                    note=str(staging.note),
                ),
                meta=dict(meta) if meta else {},
            )
        )


__all__ = [
    "MemoryRecord",
    "MemoryUsage",
    "ResidencyBytes",
    "RuntimeBytes",
    "StaticBytes",
]
