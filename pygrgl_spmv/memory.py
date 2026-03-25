"""Physical-allocation memory snapshots for ``SpmvGRG``."""

from __future__ import annotations

from collections import deque
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp

OWNER_VALUES = {"operator", "backend", "caller"}
ACTIVITY_VALUES = {"always", "yes", "no"}
RETENTION_VALUES = {"call", "persistent", "captured", "on_demand", "staging"}
KIND_VALUES = {
    "mapping",
    "tables",
    "coalescence",
    "init",
    "sparse",
    "selector",
    "state",
    "input",
    "output",
    "scratch",
    "auxiliary",
    "temporary",
}
STORAGE_VALUES = {"numpy", "torch", "cupy", "cuda_vmm"}
SPACE_VALUES = {"cpu", "cuda"}
FIELD_KIND_ALLOC = "alloc"
FIELD_KIND_CHILD = "child"
FIELD_KIND_IGNORE = "ignore"
DEFAULT_TREE_LEVELS = ("space", "scope", "direction", "slot_k")
AllocKey = tuple[object, ...]
_BindingKey = tuple[str, str, str, str | None, int | None]
_RESERVED_SNAPSHOT_META_KEYS = frozenset({"direction", "active_alloc_keys"})
_OWNER_ORDER = {"caller": 0, "operator": 1, "backend": 2}
_RETENTION_ORDER = {"call": 0, "persistent": 1, "captured": 2, "on_demand": 3, "staging": 4}
_ACTIVITY_ORDER = {"yes": 0, "always": 1, "no": 2}


def _is_dataclass_instance(value: Any) -> bool:
    return is_dataclass(value) and not isinstance(value, type)


def _require_member(*, label: str, value: str | None, allowed: set[str]) -> str:
    if value is None:
        raise ValueError(f"{label} is required")
    token = str(value)
    if token not in allowed:
        raise ValueError(f"Unknown {label} {token!r}; expected one of {sorted(allowed)}")
    return token


def _path_text(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _normalize_direction(direction: str | None, *, label: str) -> str | None:
    if direction is None:
        return None
    token = str(direction)
    if token not in {"up", "down"}:
        raise ValueError(f"Unknown {label} {direction!r}; expected 'up' or 'down'")
    return token


def _normalize_alloc_key(key: AllocKey) -> AllocKey:
    if isinstance(key, tuple):
        return tuple(key)
    if isinstance(key, list):
        return tuple(key)
    raise TypeError(f"Allocation key must be a tuple or list, got {type(key).__name__}")


def _normalize_alloc_keys(keys: Iterable[AllocKey]) -> frozenset[AllocKey]:
    return frozenset(_normalize_alloc_key(key) for key in keys)


def _sorted_owners(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted((str(value) for value in values), key=lambda value: (_OWNER_ORDER[value], value)))


def _sorted_retentions(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted((str(value) for value in values), key=lambda value: (_RETENTION_ORDER[value], value)))


def _sorted_activities(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted((str(value) for value in values), key=lambda value: (_ACTIVITY_ORDER[value], value)))


def _sorted_directions(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted((str(value) for value in values), key=lambda value: (0 if value == "up" else 1, value)))


def _join_tokens(values: Iterable[str]) -> str | None:
    tokens = tuple(str(value) for value in values)
    if not tokens:
        return None
    return "|".join(tokens)


def _join_slot_ks(values: Iterable[int]) -> str | None:
    tokens = tuple(str(int(value)) for value in sorted(int(value) for value in values))
    if not tokens:
        return None
    return "|".join(tokens)


@dataclass(frozen=True)
class MemoryRole:
    label: str
    kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "kind", _require_member(label="kind", value=self.kind, allowed=KIND_VALUES))


@dataclass(frozen=True)
class MemoryBinding:
    owner: str
    retention: str
    activity: str
    direction: str | None = None
    slot_k: int | None = None
    roles: frozenset[MemoryRole] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner", _require_member(label="owner", value=self.owner, allowed=OWNER_VALUES))
        object.__setattr__(self, "retention", _require_member(label="retention", value=self.retention, allowed=RETENTION_VALUES))
        object.__setattr__(self, "activity", _require_member(label="activity", value=self.activity, allowed=ACTIVITY_VALUES))
        object.__setattr__(self, "direction", _normalize_direction(self.direction, label="direction"))
        object.__setattr__(self, "slot_k", None if self.slot_k is None else int(self.slot_k))
        object.__setattr__(self, "roles", frozenset(self.roles))
        if not self.roles:
            raise ValueError("MemoryBinding.roles must be non-empty")

    @property
    def labels(self) -> frozenset[str]:
        return frozenset(role.label for role in self.roles)

    @property
    def kinds(self) -> frozenset[str]:
        return frozenset(role.kind for role in self.roles)


@dataclass(frozen=True)
class VmmAliasedAlloc:
    ptr: int
    physical_nbytes: int
    logical_nbytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "ptr", int(self.ptr))
        object.__setattr__(self, "physical_nbytes", int(self.physical_nbytes))
        object.__setattr__(self, "logical_nbytes", int(self.logical_nbytes))
        if self.ptr <= 0:
            raise ValueError(f"VMM pointer must be positive, got {self.ptr}")
        if self.physical_nbytes <= 0:
            raise ValueError(f"VMM physical_nbytes must be positive, got {self.physical_nbytes}")
        if self.logical_nbytes <= 0:
            raise ValueError(f"VMM logical_nbytes must be positive, got {self.logical_nbytes}")


@dataclass(frozen=True)
class MemoryAllocation:
    nbytes: int
    storage: str
    bindings: frozenset[MemoryBinding] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "nbytes", int(self.nbytes))
        object.__setattr__(self, "storage", _require_member(label="storage", value=self.storage, allowed=STORAGE_VALUES))
        object.__setattr__(self, "bindings", frozenset(self.bindings))
        if self.nbytes <= 0:
            raise ValueError(f"Allocation nbytes must be positive, got {self.nbytes}")
        if not self.bindings:
            raise ValueError("MemoryAllocation.bindings must be non-empty")

    @property
    def space(self) -> str:
        return "cpu" if self.storage == "numpy" else "cuda"

    @property
    def roles(self) -> frozenset[MemoryRole]:
        return frozenset(role for binding in self.bindings for role in binding.roles)

    @property
    def labels(self) -> frozenset[str]:
        return frozenset(role.label for role in self.roles)

    @property
    def kinds(self) -> frozenset[str]:
        return frozenset(role.kind for role in self.roles)

    @property
    def owners(self) -> frozenset[str]:
        return frozenset(binding.owner for binding in self.bindings)

    @property
    def retentions(self) -> frozenset[str]:
        return frozenset(binding.retention for binding in self.bindings)

    @property
    def activities(self) -> frozenset[str]:
        return frozenset(binding.activity for binding in self.bindings)

    @property
    def directions(self) -> frozenset[str]:
        return frozenset(
            binding.direction
            for binding in self.bindings
            if binding.direction is not None
        )

    @property
    def slot_ks(self) -> frozenset[int]:
        return frozenset(
            int(binding.slot_k)
            for binding in self.bindings
            if binding.slot_k is not None
        )

    @property
    def has_call_binding(self) -> bool:
        return "call" in self.retentions

    @property
    def owner(self) -> str | None:
        values = self.owners
        return next(iter(values)) if len(values) == 1 else None

    @property
    def retention(self) -> str | None:
        values = self.retentions
        return next(iter(values)) if len(values) == 1 else None

    @property
    def activity(self) -> str | None:
        values = self.activities
        return next(iter(values)) if len(values) == 1 else None

    @property
    def direction(self) -> str | None:
        values = self.directions
        return next(iter(values)) if len(values) == 1 else None

    @property
    def slot_k(self) -> int | None:
        values = self.slot_ks
        return next(iter(values)) if len(values) == 1 else None

    @property
    def owner_text(self) -> str | None:
        return _join_tokens(_sorted_owners(self.owners))

    @property
    def retention_text(self) -> str | None:
        return _join_tokens(_sorted_retentions(self.retentions))

    @property
    def activity_text(self) -> str | None:
        return _join_tokens(_sorted_activities(self.activities))


@dataclass(frozen=True)
class MemorySnapshot:
    stage: str
    runtime_k: int | None
    allocations: tuple[MemoryAllocation, ...]
    direction: str | None = None
    active_alloc_keys: frozenset[AllocKey] = frozenset()
    meta: dict[str, object] = field(default_factory=dict)
    _alloc_keys: tuple[AllocKey, ...] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", str(self.stage))
        object.__setattr__(self, "runtime_k", None if self.runtime_k is None else int(self.runtime_k))
        allocations = tuple(self.allocations)
        object.__setattr__(self, "allocations", allocations)
        object.__setattr__(self, "direction", _normalize_direction(self.direction, label="direction"))
        object.__setattr__(self, "active_alloc_keys", _normalize_alloc_keys(self.active_alloc_keys))
        meta = dict(self.meta)
        reserved = sorted(_RESERVED_SNAPSHOT_META_KEYS.intersection(meta))
        if reserved:
            raise ValueError(f"MemorySnapshot.meta contains reserved semantic key(s): {reserved}")
        object.__setattr__(self, "meta", meta)
        alloc_keys = tuple(_normalize_alloc_key(key) for key in self._alloc_keys)
        if not alloc_keys:
            alloc_keys = tuple(("manual_alloc", id(self), idx) for idx in range(len(allocations)))
        if len(alloc_keys) != len(allocations):
            raise ValueError(
                f"MemorySnapshot._alloc_keys length mismatch: {len(alloc_keys)} keys for {len(allocations)} allocations"
            )
        if len(set(alloc_keys)) != len(alloc_keys):
            raise ValueError("MemorySnapshot._alloc_keys must be unique")
        object.__setattr__(self, "_alloc_keys", alloc_keys)


@dataclass
class MemoryLedger:
    retained: MemorySnapshot | None = None
    last_call: MemorySnapshot | None = None

    def reset(self) -> None:
        self.retained = None
        self.last_call = None


@dataclass(frozen=True)
class MemoryTreeRow:
    path: tuple[str, ...]
    nbytes: int
    storage: str | None = None
    owner: str | None = None
    retention: str | None = None
    activity: str | None = None
    labels: frozenset[str] = frozenset()
    kinds: frozenset[str] = frozenset()

    def node(self) -> str:
        return self.path[-1]

    def parent(self) -> str:
        return "-" if len(self.path) == 1 else "/".join(self.path[:-1])

    def is_leaf(self) -> bool:
        return self.owner is not None


def _field_with_mem(
    *,
    mem_kind: str,
    metadata: dict[str, object],
    default: object = MISSING,
    default_factory: object = MISSING,
):
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("Cannot specify both default and default_factory")
    payload = {"mem": mem_kind, **metadata}
    if default_factory is not MISSING:
        return field(default_factory=default_factory, metadata=payload)
    if default is not MISSING:
        return field(default=default, metadata=payload)
    return field(metadata=payload)


def alloc_field(
    *,
    label: str,
    kind: str,
    owner: str | None = None,
    retention: str | None = None,
    activity: str | None = None,
    direction: str | None = None,
    slot_k: int | None = None,
    default: object = MISSING,
    default_factory: object = MISSING,
):
    if owner is not None:
        _require_member(label="owner", value=owner, allowed=OWNER_VALUES)
    _require_member(label="kind", value=kind, allowed=KIND_VALUES)
    if retention is not None:
        _require_member(label="retention", value=retention, allowed=RETENTION_VALUES)
    if activity is not None:
        _require_member(label="activity", value=activity, allowed=ACTIVITY_VALUES)
    if direction is not None and str(direction) not in {"up", "down"}:
        raise ValueError(f"Unknown direction {direction!r}; expected 'up' or 'down'")
    return _field_with_mem(
        mem_kind=FIELD_KIND_ALLOC,
        metadata={
            "label": str(label),
            "kind": str(kind),
            "owner": owner,
            "retention": retention,
            "activity": activity,
            "direction": direction,
            "slot_k": slot_k,
        },
        default=default,
        default_factory=default_factory,
    )


def child_field(
    *,
    owner: str | None = None,
    retention: str | None = None,
    activity: str | None = None,
    direction: str | None = None,
    slot_k_from_attr: str | None = None,
    slot_k_from_dict_key: bool = False,
    default: object = MISSING,
    default_factory: object = MISSING,
):
    if owner is not None:
        _require_member(label="owner", value=owner, allowed=OWNER_VALUES)
    if retention is not None:
        _require_member(label="retention", value=retention, allowed=RETENTION_VALUES)
    if activity is not None:
        _require_member(label="activity", value=activity, allowed=ACTIVITY_VALUES)
    if direction is not None and str(direction) not in {"up", "down"}:
        raise ValueError(f"Unknown direction {direction!r}; expected 'up' or 'down'")
    return _field_with_mem(
        mem_kind=FIELD_KIND_CHILD,
        metadata={
            "owner": owner,
            "retention": retention,
            "activity": activity,
            "direction": direction,
            "slot_k_from_attr": slot_k_from_attr,
            "slot_k_from_dict_key": bool(slot_k_from_dict_key),
        },
        default=default,
        default_factory=default_factory,
    )


def ignore_field(*, default: object = MISSING, default_factory: object = MISSING):
    return _field_with_mem(
        mem_kind=FIELD_KIND_IGNORE,
        metadata={},
        default=default,
        default_factory=default_factory,
    )


@dataclass(frozen=True)
class _Scope:
    owner: str | None = None
    retention: str | None = None
    activity: str | None = None
    direction: str | None = None
    slot_k: int | None = None


@dataclass(frozen=True)
class _FieldSpec:
    name: str
    mem_kind: str
    label: str | None = None
    kind: str | None = None
    owner: str | None = None
    retention: str | None = None
    activity: str | None = None
    direction: str | None = None
    slot_k: int | None = None
    slot_k_from_attr: str | None = None
    slot_k_from_dict_key: bool = False


@lru_cache(maxsize=None)
def _field_specs(cls: type[object]) -> tuple[_FieldSpec, ...]:
    if not hasattr(cls, "__dataclass_fields__"):
        raise RuntimeError(f"{cls.__name__} is not a dataclass type")
    specs: list[_FieldSpec] = []
    for dc_field in fields(cls):
        meta = dict(dc_field.metadata)
        mem_kind = meta.get("mem")
        if mem_kind not in {FIELD_KIND_ALLOC, FIELD_KIND_CHILD, FIELD_KIND_IGNORE}:
            raise RuntimeError(
                f"Field {cls.__name__}.{dc_field.name} is missing memory metadata; use alloc_field(), child_field(), or ignore_field()"
            )
        if mem_kind == FIELD_KIND_ALLOC:
            specs.append(
                _FieldSpec(
                    name=dc_field.name,
                    mem_kind=FIELD_KIND_ALLOC,
                    label=str(meta["label"]),
                    kind=str(meta["kind"]),
                    owner=None if meta.get("owner") is None else str(meta["owner"]),
                    retention=None if meta.get("retention") is None else str(meta["retention"]),
                    activity=None if meta.get("activity") is None else str(meta["activity"]),
                    direction=None if meta.get("direction") is None else str(meta["direction"]),
                    slot_k=None if meta.get("slot_k") is None else int(meta["slot_k"]),
                )
            )
            continue
        if mem_kind == FIELD_KIND_CHILD:
            specs.append(
                _FieldSpec(
                    name=dc_field.name,
                    mem_kind=FIELD_KIND_CHILD,
                    owner=None if meta.get("owner") is None else str(meta["owner"]),
                    retention=None if meta.get("retention") is None else str(meta["retention"]),
                    activity=None if meta.get("activity") is None else str(meta["activity"]),
                    direction=None if meta.get("direction") is None else str(meta["direction"]),
                    slot_k_from_attr=None if meta.get("slot_k_from_attr") is None else str(meta["slot_k_from_attr"]),
                    slot_k_from_dict_key=bool(meta.get("slot_k_from_dict_key", False)),
                )
            )
            continue
        specs.append(_FieldSpec(name=dc_field.name, mem_kind=FIELD_KIND_IGNORE))
    return tuple(specs)


def _follow_numpy_base(array: np.ndarray) -> np.ndarray:
    root = np.asarray(array)
    while isinstance(root.base, np.ndarray):
        root = root.base
    return root


# Ordinary array carriers report logical payload bytes. Only explicit physical
# carriers such as VmmAliasedAlloc report physical reservation bytes.
def _numpy_key(value: np.ndarray) -> tuple[AllocKey, int, str]:
    root = _follow_numpy_base(np.asarray(value))
    ptr = int(root.__array_interface__["data"][0])
    nbytes = int(root.nbytes)
    return ("numpy", ptr, nbytes), nbytes, "numpy"


def _torch_key(value: Any) -> tuple[AllocKey, int, str]:
    # Project invariant: torch-backed allocations only come from the Triton
    # path, so they must already be CUDA-resident when captured by the ledger.
    assert getattr(getattr(value, "device", None), "type", None) == "cuda", (
        "Memory ledger expects torch tensors to live on CUDA"
    )
    storage = value.untyped_storage()
    ptr = int(storage.data_ptr())
    nbytes = int(storage.nbytes())
    return ("torch", ptr, nbytes), nbytes, "torch"


def _follow_cupy_base(value: Any) -> Any:
    root = value
    while True:
        base = getattr(root, "base", None)
        if base is None or not hasattr(base, "data"):
            return root
        root = base


def _cupy_key(value: Any) -> tuple[AllocKey, int, str]:
    root = _follow_cupy_base(value)
    mem = getattr(getattr(root, "data", None), "mem", None)
    ptr = getattr(mem, "ptr", None)
    nbytes = getattr(root, "nbytes", None)
    if ptr is None or nbytes is None:
        raise TypeError(
            f"Unsupported CuPy allocation carrier {type(value)!r}; expected data.mem.ptr and nbytes"
        )
    size = int(nbytes)
    return ("cupy", int(ptr), size), size, "cupy"


def _vmm_key(value: VmmAliasedAlloc) -> tuple[AllocKey, int, str]:
    return ("cuda_vmm", int(value.ptr), int(value.physical_nbytes)), int(value.physical_nbytes), "cuda_vmm"


def _sparse_payloads(value: Any) -> Iterable[np.ndarray]:
    obj = getattr(value, "_mat", value)
    for attr in ("data", "indices", "indptr", "row", "col"):
        payload = getattr(obj, attr, None)
        if payload is not None:
            yield np.asarray(payload)


def _supported_scalar_alloc(value: Any) -> bool:
    return (
        isinstance(value, VmmAliasedAlloc)
        or sp.issparse(value)
        or hasattr(value, "__array_interface__")
        or hasattr(value, "untyped_storage")
        or hasattr(getattr(value, "data", None), "mem")
    )


def _key_nbytes_storage(value: Any) -> tuple[AllocKey, int, str]:
    if isinstance(value, VmmAliasedAlloc):
        return _vmm_key(value)
    if sp.issparse(value):
        raise TypeError("Sparse matrices must be expanded before keying")
    if hasattr(value, "__array_interface__"):
        return _numpy_key(np.asarray(value))
    if hasattr(value, "untyped_storage"):
        return _torch_key(value)
    if hasattr(getattr(value, "data", None), "mem"):
        return _cupy_key(value)
    raise TypeError(f"Unsupported allocation carrier {type(value)!r}")


@dataclass
class _CollectedAlloc:
    nbytes: int
    storage: str
    bindings: dict[_BindingKey, set[MemoryRole]] = field(default_factory=dict)

    def add_binding(self, *, binding_key: _BindingKey, role: MemoryRole) -> None:
        self.bindings.setdefault(binding_key, set()).add(role)

    def materialize(self) -> MemoryAllocation:
        return MemoryAllocation(
            nbytes=self.nbytes,
            storage=self.storage,
            bindings=frozenset(
                MemoryBinding(
                    owner=owner,
                    retention=retention,
                    activity=activity,
                    direction=direction,
                    slot_k=slot_k,
                    roles=frozenset(roles),
                )
                for (owner, retention, activity, direction, slot_k), roles in self.bindings.items()
            ),
        )


def _binding_sort_key(binding: MemoryBinding) -> tuple[object, ...]:
    return (
        _RETENTION_ORDER[binding.retention],
        _OWNER_ORDER[binding.owner],
        _ACTIVITY_ORDER[binding.activity],
        0 if binding.direction == "up" else 1 if binding.direction == "down" else 2,
        -1 if binding.slot_k is None else int(binding.slot_k),
        tuple(sorted(binding.labels)),
        tuple(sorted(binding.kinds)),
    )


def _merge_bindings(*binding_sets: Iterable[MemoryBinding]) -> frozenset[MemoryBinding]:
    merged: dict[_BindingKey, set[MemoryRole]] = {}
    for bindings in binding_sets:
        for binding in bindings:
            key: _BindingKey = (
                binding.owner,
                binding.retention,
                binding.activity,
                binding.direction,
                binding.slot_k,
            )
            merged.setdefault(key, set()).update(binding.roles)
    return frozenset(
        MemoryBinding(
            owner=owner,
            retention=retention,
            activity=activity,
            direction=direction,
            slot_k=slot_k,
            roles=frozenset(roles),
        )
        for (owner, retention, activity, direction, slot_k), roles in merged.items()
    )


def _promote_active_bindings(bindings: Iterable[MemoryBinding]) -> frozenset[MemoryBinding]:
    promoted: list[MemoryBinding] = []
    for binding in bindings:
        if binding.activity == "no":
            promoted.append(
                MemoryBinding(
                    owner=binding.owner,
                    retention=binding.retention,
                    activity="yes",
                    direction=binding.direction,
                    slot_k=binding.slot_k,
                    roles=binding.roles,
                )
            )
            continue
        promoted.append(binding)
    return _merge_bindings(promoted)


def _merge_allocations(left: MemoryAllocation, right: MemoryAllocation) -> MemoryAllocation:
    if left.nbytes != right.nbytes or left.storage != right.storage:
        raise RuntimeError(
            f"Conflicting allocation merge facts: left={(left.nbytes, left.storage)!r}, right={(right.nbytes, right.storage)!r}"
        )
    return MemoryAllocation(
        nbytes=left.nbytes,
        storage=left.storage,
        bindings=_merge_bindings(left.bindings, right.bindings),
    )


class _AllocCollector:
    def __init__(
        self,
        *,
        stage: str,
        runtime_k: int | None,
        direction: str | None = None,
        active_alloc_keys: Iterable[AllocKey] = (),
        meta: dict[str, object] | None = None,
    ) -> None:
        self._stage = str(stage)
        self._runtime_k = None if runtime_k is None else int(runtime_k)
        self._direction = _normalize_direction(direction, label="direction")
        self._active_alloc_keys = _normalize_alloc_keys(active_alloc_keys)
        self._meta = {} if meta is None else dict(meta)
        self._rows: dict[AllocKey, _CollectedAlloc] = {}
        self._stack: list[tuple[Any, _Scope, tuple[str, ...]]] = []

    def add_root(self, root: Any, *, label: str) -> None:
        if root is None:
            raise RuntimeError(f"Memory root {label} is None")
        if not _is_dataclass_instance(root):
            raise RuntimeError(f"Memory root {label} must be a dataclass instance, got {type(root).__name__}")
        self._stack.append((root, _Scope(), (str(label),)))

    def finish(self) -> MemorySnapshot:
        while self._stack:
            value, scope, path = self._stack.pop()
            self._visit_dataclass(value, scope=scope, path=path)
        rows = [(key, collected.materialize()) for key, collected in self._rows.items()]
        rows.sort(key=lambda item: self._sort_key(item[1]))
        return MemorySnapshot(
            stage=self._stage,
            runtime_k=self._runtime_k,
            allocations=tuple(row for _, row in rows),
            direction=self._direction,
            active_alloc_keys=self._active_alloc_keys,
            meta=dict(self._meta),
            _alloc_keys=tuple(key for key, _ in rows),
        )

    def _visit_dataclass(self, value: Any, *, scope: _Scope, path: tuple[str, ...]) -> None:
        if not _is_dataclass_instance(value):
            raise RuntimeError(f"Expected dataclass instance at {_path_text(path)!r}, got {type(value).__name__}")
        specs = _field_specs(type(value))
        for spec in specs:
            field_path = (*path, spec.name)
            item = getattr(value, spec.name)
            if spec.mem_kind == FIELD_KIND_IGNORE:
                self._check_ignored(item, path=field_path)
                continue
            if spec.mem_kind == FIELD_KIND_CHILD:
                self._push_child(item, spec=spec, scope=scope, path=field_path)
                continue
            self._add_alloc(item, spec=spec, scope=scope, path=field_path)

    def _check_ignored(self, value: Any, *, path: tuple[str, ...]) -> None:
        if value is None:
            return
        if _supported_scalar_alloc(value):
            raise RuntimeError(
                f"ignore field {_path_text(path)!r} holds a supported allocation carrier {type(value).__name__}"
            )
        if _is_dataclass_instance(value):
            raise RuntimeError(
                f"ignore field {_path_text(path)!r} holds a dataclass instance {type(value).__name__}"
            )
        if isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                self._check_ignored(item, path=(*path, str(idx)))
            return
        if isinstance(value, dict):
            for key, item in value.items():
                self._check_ignored(item, path=(*path, f"[{key}]"))

    def _push_child(self, value: Any, *, spec: _FieldSpec, scope: _Scope, path: tuple[str, ...]) -> None:
        if value is None:
            return
        self._push_child_value(value, spec=spec, scope=scope, path=path, dict_key=None)

    def _push_child_value(
        self,
        value: Any,
        *,
        spec: _FieldSpec,
        scope: _Scope,
        path: tuple[str, ...],
        dict_key: int | None,
    ) -> None:
        if value is None:
            return
        if _is_dataclass_instance(value):
            child_scope = self._child_scope(scope=scope, spec=spec, value=value, dict_key=dict_key)
            self._stack.append((value, child_scope, path))
            return
        if isinstance(value, dict):
            for key, item in reversed(list(value.items())):
                if item is None:
                    continue
                if not isinstance(key, int):
                    raise RuntimeError(
                        f"child field {_path_text(path)!r} expected int dict keys, got {type(key).__name__}"
                    )
                self._push_child_value(item, spec=spec, scope=scope, path=(*path, f"k={int(key)}"), dict_key=key)
            return
        if isinstance(value, (list, tuple)):
            for idx in range(len(value) - 1, -1, -1):
                item = value[idx]
                if item is None:
                    continue
                self._push_child_value(item, spec=spec, scope=scope, path=(*path, str(idx)), dict_key=dict_key)
            return
        raise RuntimeError(
            f"child field {_path_text(path)!r} must hold dataclass instances or containers of dataclass instances"
        )

    def _child_scope(self, *, scope: _Scope, spec: _FieldSpec, value: Any, dict_key: int | None) -> _Scope:
        if not _is_dataclass_instance(value):
            raise RuntimeError(f"child field {type(value).__name__} must hold dataclass instances")
        slot_k = scope.slot_k
        if spec.slot_k_from_attr is not None:
            attr_value = getattr(value, spec.slot_k_from_attr)
            if attr_value is None:
                slot_k = None
            else:
                slot_k = int(attr_value)
        if spec.slot_k_from_dict_key:
            if dict_key is None:
                raise RuntimeError("slot_k_from_dict_key requires an int dict key")
            slot_k = int(dict_key)
        return _Scope(
            owner=scope.owner if spec.owner is None else spec.owner,
            retention=scope.retention if spec.retention is None else spec.retention,
            activity=scope.activity if spec.activity is None else spec.activity,
            direction=scope.direction if spec.direction is None else spec.direction,
            slot_k=slot_k,
        )

    def _add_alloc(self, value: Any, *, spec: _FieldSpec, scope: _Scope, path: tuple[str, ...]) -> None:
        if value is None:
            return
        role = MemoryRole(label=str(spec.label), kind=str(spec.kind))
        final_owner = scope.owner if spec.owner is None else spec.owner
        final_retention = scope.retention if spec.retention is None else spec.retention
        final_activity = scope.activity if spec.activity is None else spec.activity
        final_direction = scope.direction if spec.direction is None else spec.direction
        final_slot_k = scope.slot_k if spec.slot_k is None else int(spec.slot_k)
        if final_owner is None:
            raise RuntimeError(f"Allocation field {_path_text(path)!r} has no resolved owner")
        if final_retention is None:
            raise RuntimeError(f"Allocation field {_path_text(path)!r} has no resolved retention")
        if final_activity is None:
            raise RuntimeError(f"Allocation field {_path_text(path)!r} has no resolved activity")
        self._add_alloc_value(
            value,
            role=role,
            owner=final_owner,
            retention=final_retention,
            activity=final_activity,
            direction=final_direction,
            slot_k=final_slot_k,
            path=path,
        )

    def _add_alloc_value(
        self,
        value: Any,
        *,
        role: MemoryRole,
        owner: str,
        retention: str,
        activity: str,
        direction: str | None,
        slot_k: int | None,
        path: tuple[str, ...],
    ) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                self._add_alloc_value(
                    item,
                    role=role,
                    owner=owner,
                    retention=retention,
                    activity=activity,
                    direction=direction,
                    slot_k=slot_k,
                    path=(*path, str(idx)),
                )
            return
        if sp.issparse(value):
            for idx, payload in enumerate(_sparse_payloads(value)):
                self._add_alloc_value(
                    payload,
                    role=role,
                    owner=owner,
                    retention=retention,
                    activity=activity,
                    direction=direction,
                    slot_k=slot_k,
                    path=(*path, f"payload{idx}"),
                )
            return
        key, nbytes, storage = _key_nbytes_storage(value)
        if nbytes == 0:
            return
        if nbytes < 0:
            raise RuntimeError(f"Negative allocation size at {_path_text(path)!r}: {nbytes}")
        binding_key: _BindingKey = (
            str(owner),
            str(retention),
            str(activity),
            None if direction is None else str(direction),
            None if slot_k is None else int(slot_k),
        )
        collected = self._rows.get(key)
        if collected is None:
            collected = _CollectedAlloc(nbytes=int(nbytes), storage=str(storage))
            self._rows[key] = collected
        elif collected.nbytes != int(nbytes) or collected.storage != str(storage):
            raise RuntimeError(
                f"Conflicting allocation facts for {_path_text(path)!r}: "
                f"existing={(collected.nbytes, collected.storage)!r}, new={(nbytes, storage)!r}"
            )
        collected.add_binding(binding_key=binding_key, role=role)

    @staticmethod
    def _sort_key(row: MemoryAllocation) -> tuple[object, ...]:
        return (
            0 if row.space == "cuda" else 1,
            _sorted_retentions(row.retentions),
            _sorted_directions(row.directions),
            tuple(sorted(int(value) for value in row.slot_ks)),
            _sorted_owners(row.owners),
            _sorted_activities(row.activities),
            tuple(sorted(row.labels)),
            tuple(sorted(row.kinds)),
            int(row.nbytes),
            row.storage,
        )


def capture_snapshot(
    *roots: Any,
    stage: str,
    runtime_k: int | None,
    direction: str | None = None,
    active_alloc_keys: Iterable[AllocKey] = (),
    meta: dict[str, object] | None = None,
) -> MemorySnapshot:
    collector = _AllocCollector(
        stage=stage,
        runtime_k=runtime_k,
        direction=direction,
        active_alloc_keys=active_alloc_keys,
        meta=meta,
    )
    for idx, root in enumerate(roots):
        collector.add_root(root, label=f"root{idx}")
    return collector.finish()


def live_snapshot(retained: MemorySnapshot | None, last_call: MemorySnapshot | None) -> MemorySnapshot | None:
    if retained is None:
        return last_call
    if last_call is None:
        return retained
    retained_rows = dict(zip(retained._alloc_keys, retained.allocations, strict=True))
    call_rows = dict(zip(last_call._alloc_keys, last_call.allocations, strict=True))
    merged_keys = list(retained._alloc_keys)
    for key in last_call._alloc_keys:
        if key not in retained_rows:
            merged_keys.append(key)

    merged_pairs: list[tuple[AllocKey, MemoryAllocation]] = []
    for key in merged_keys:
        retained_row = retained_rows.get(key)
        call_row = call_rows.get(key)
        if retained_row is not None and key in last_call.active_alloc_keys:
            retained_row = MemoryAllocation(
                nbytes=retained_row.nbytes,
                storage=retained_row.storage,
                bindings=_promote_active_bindings(retained_row.bindings),
            )
        if retained_row is None:
            assert call_row is not None
            merged_pairs.append((key, call_row))
            continue
        if call_row is None:
            merged_pairs.append((key, retained_row))
            continue
        merged_pairs.append((key, _merge_allocations(retained_row, call_row)))

    merged_pairs.sort(key=lambda item: _AllocCollector._sort_key(item[1]))
    meta = dict(retained.meta)
    meta.update(last_call.meta)
    return MemorySnapshot(
        stage=str(last_call.stage),
        runtime_k=last_call.runtime_k,
        allocations=tuple(row for _, row in merged_pairs),
        direction=last_call.direction,
        active_alloc_keys=last_call.active_alloc_keys,
        meta=meta,
        _alloc_keys=tuple(key for key, _ in merged_pairs),
    )


def _alloc_keys_for_value(value: Any) -> set[AllocKey]:
    keys: set[AllocKey] = set()
    _collect_alloc_keys(value, keys)
    return keys


def _collect_alloc_keys(value: Any, keys: set[AllocKey]) -> None:
    if value is None:
        return
    if isinstance(value, VmmAliasedAlloc):
        key, nbytes, _storage = _key_nbytes_storage(value)
        if nbytes > 0:
            keys.add(key)
        return
    if _is_dataclass_instance(value):
        for dc_field in fields(type(value)):
            mem_kind = dc_field.metadata.get("mem")
            field_value = getattr(value, dc_field.name)
            if mem_kind == FIELD_KIND_ALLOC or mem_kind == FIELD_KIND_CHILD:
                _collect_alloc_keys(field_value, keys)
            elif mem_kind != FIELD_KIND_IGNORE:
                raise RuntimeError(
                    f"Field {type(value).__name__}.{dc_field.name} is missing memory metadata; "
                    "use alloc_field(), child_field(), or ignore_field()"
                )
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_alloc_keys(item, keys)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_alloc_keys(item, keys)
        return
    if sp.issparse(value):
        for payload in _sparse_payloads(value):
            _collect_alloc_keys(payload, keys)
        return
    key, nbytes, _storage = _key_nbytes_storage(value)
    if nbytes > 0:
        keys.add(key)


def _leaf_name(labels: frozenset[str]) -> str:
    return "|".join(sorted(labels)) if labels else "allocation"


def _primary_scope(row: MemoryAllocation) -> str:
    return "call" if row.has_call_binding else "retained"


def _primary_direction(snapshot: MemorySnapshot, row: MemoryAllocation) -> str | None:
    if row.has_call_binding:
        return snapshot.direction
    directions = _sorted_directions(row.directions)
    if len(directions) == 1:
        return directions[0]
    return _join_tokens(directions)


def _primary_slot_k(snapshot: MemorySnapshot, row: MemoryAllocation) -> int | None:
    if row.has_call_binding:
        return None if snapshot.runtime_k is None else int(snapshot.runtime_k)
    slot_ks = sorted(int(value) for value in row.slot_ks)
    if len(slot_ks) == 1:
        return slot_ks[0]
    return None


def _tree_segments(snapshot: MemorySnapshot, row: MemoryAllocation, *, levels: tuple[str, ...]) -> tuple[str, ...]:
    segments: list[str] = []
    for level in levels:
        if level == "space":
            segments.append("cuda_live" if row.space == "cuda" else "cpu_live")
            continue
        if level == "scope":
            segments.append(_primary_scope(row))
            continue
        if level == "retention":
            token = row.retention_text
            if token is not None:
                segments.append(token)
            continue
        if level == "direction":
            token = _primary_direction(snapshot, row)
            if token:
                segments.append(token)
            continue
        if level == "slot_k":
            slot_k = _primary_slot_k(snapshot, row)
            if slot_k is not None:
                segments.append(f"k={int(slot_k)}")
                continue
            slot_ks = row.slot_ks
            if slot_ks and not row.has_call_binding:
                token = _join_slot_ks(slot_ks)
                if token is not None:
                    segments.append(f"k={token}")
            continue
        if level == "owner":
            token = row.owner_text
            if token is not None:
                segments.append(token)
            continue
        if level == "activity":
            token = row.activity_text
            if token is not None:
                segments.append(token)
            continue
        raise ValueError(f"Unknown tree level {level!r}")
    return tuple(segments)


def tree_rows(snapshot: MemorySnapshot, *, levels: tuple[str, ...] = DEFAULT_TREE_LEVELS) -> list[MemoryTreeRow]:
    counts: dict[tuple[tuple[str, ...], str], int] = {}
    base_paths: list[tuple[MemoryAllocation, tuple[str, ...], str]] = []
    for row in snapshot.allocations:
        path = _tree_segments(snapshot, row, levels=tuple(levels))
        leaf = _leaf_name(row.labels)
        counts[(path, leaf)] = counts.get((path, leaf), 0) + 1
        base_paths.append((row, path, leaf))

    leaf_paths: list[tuple[MemoryAllocation, tuple[str, ...]]] = []
    seen: dict[tuple[tuple[str, ...], str], int] = {}
    for row, path, leaf in base_paths:
        token = (path, leaf)
        seen[token] = seen.get(token, 0) + 1
        suffix = f"#{seen[token]}" if counts[token] > 1 else ""
        leaf_paths.append((row, (*path, f"{leaf}{suffix}")))

    all_paths: set[tuple[str, ...]] = set()
    leaves: dict[tuple[str, ...], MemoryTreeRow] = {}
    for row, path in leaf_paths:
        for size in range(1, len(path) + 1):
            all_paths.add(path[:size])
        leaves[path] = MemoryTreeRow(
            path=path,
            nbytes=int(row.nbytes),
            storage=row.storage,
            owner=row.owner_text,
            retention=row.retention_text,
            activity=row.activity_text,
            labels=row.labels,
            kinds=row.kinds,
        )

    children: dict[tuple[str, ...], list[tuple[str, ...]]] = {path: [] for path in all_paths}
    for path in all_paths:
        if len(path) == 1:
            continue
        children.setdefault(path[:-1], []).append(path)

    rolled: dict[tuple[str, ...], MemoryTreeRow] = {}
    for path in sorted(all_paths, key=len, reverse=True):
        leaf = leaves.get(path)
        if leaf is not None:
            rolled[path] = leaf
            continue
        rolled[path] = MemoryTreeRow(
            path=path,
            nbytes=int(sum(rolled[child].nbytes for child in children.get(path, ()))),
        )

    rows: list[MemoryTreeRow] = []
    queue: deque[tuple[str, ...]] = deque(sorted((path for path in all_paths if len(path) == 1), key=_root_sort_key))
    seen_paths: set[tuple[str, ...]] = set()
    current_direction = snapshot.direction
    current_k = None if snapshot.runtime_k is None else int(snapshot.runtime_k)
    while queue:
        path = queue.popleft()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        rows.append(rolled[path])
        ordered_children = sorted(
            children.get(path, ()),
            key=lambda child: _tree_child_sort_key(
                parent=path,
                child=child,
                current_direction=current_direction,
                current_k=current_k,
                leaf=rolled[child].is_leaf(),
            ),
        )
        queue.extend(ordered_children)
    return rows


def _root_sort_key(path: tuple[str, ...]) -> tuple[int, str]:
    return (0 if path[-1] == "cuda_live" else 1, path[-1])


def _tree_child_sort_key(
    *,
    parent: tuple[str, ...],
    child: tuple[str, ...],
    current_direction: str | None,
    current_k: int | None,
    leaf: bool,
) -> tuple[object, ...]:
    segment = child[-1]
    if len(parent) == 1:
        return (0 if segment == "call" else 1 if segment == "retained" else 2, segment)
    if segment in {"up", "down"}:
        return (0 if segment == current_direction else 1, segment)
    if segment.startswith("k="):
        token = segment.split("=", 1)[1]
        if "|" in token:
            return (2, token)
        value = int(token)
        return (0 if current_k is not None and value == current_k else 1, value)
    return (1 if leaf else 0, segment)


__all__ = [
    "ACTIVITY_VALUES",
    "AllocKey",
    "KIND_VALUES",
    "MemoryAllocation",
    "MemoryBinding",
    "MemoryLedger",
    "MemoryRole",
    "MemorySnapshot",
    "MemoryTreeRow",
    "OWNER_VALUES",
    "RETENTION_VALUES",
    "SPACE_VALUES",
    "STORAGE_VALUES",
    "VmmAliasedAlloc",
    "alloc_field",
    "capture_snapshot",
    "child_field",
    "ignore_field",
    "live_snapshot",
    "tree_rows",
]
