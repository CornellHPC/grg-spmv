Parent docs: [Project README](../../README.md)

# Backends

This package contains the backend implementations used by `SpmvGRG`.

## Layout

- `base.py`: shared backend scaffolding, validation, and fail-fast memory-capture hooks
- `reference.py`: reference CPU backend used for correctness and init-bias generation
- `mkl/`: MKL backend
- `triton/`: Triton backend
- `cusparse/`: cuSPARSE backend

## Child docs

- [MKL Backend](mkl/README.md)
- [Triton Backend](triton/README.md)
- [cuSPARSE Backend](cusparse/README.md)

## Shared backend concepts

Backends are constructed from explicit backend-specific plan pairs.

- MKL uses `MklPlanPair`
- Triton uses `TritonPlanPair`
- cuSPARSE uses `CusparsePlanPair`

Either side of a plan pair may be omitted.

- `plan_up=None` builds a DOWN-only backend
- `plan_down=None` builds an UP-only backend

`log_level` controls verbosity only.

`instrumentation` enables slower observability behavior. Each backend decides how that affects execution, but the flag is runtime configuration, not plan state.

GPU backends also require mandatory `device` and `stream` constructor arguments:

- CUDA device ordinal such as `0`
- raw `cudaStream_t` handle, including `0`
- any object implementing `__cuda_stream__()`

Typical null-stream call sites use `device=0, stream=0` in internal
tests/benchmarks or `device=0, stream=cupy.cuda.Stream.null` in user code.

## GPU stream model

GPU backends use four stream roles:

- caller stream: the supplied `stream=` handle on the declared `device=`
- root stream: one backend-owned stream per backend instance
- level streams: backend-owned worker streams
- scratch streams: backend-owned helper streams for scratch-enabled levels

`stream=0` means the null stream on the declared device. Non-null external
streams must resolve to that same device at backend construction time.

Setup-time GPU allocation/upload, staging, graph warmup/capture/replay, and
root-side gathers run on the root stream.

Level and scratch work run on the backend-owned worker streams.

Each wavefront joins per-level completion back onto root before return, so any
root-stream wait or synchronize is also the boundary for the worker work from
that launch.

## Memory ledger

`SpmvGRG.memory` is the public memory-accounting surface.

The ledger stores:

- one retained snapshot captured after operator construction and refreshed only when retained state changes
- one `last_call` snapshot overwritten on each successful `SpmvGRG.matmul()` call

Each snapshot stores flat allocations, not a prebuilt tree.

Each backend exposes:

- one persistent retained-memory root stored on the backend instance
- one lexical call-memory root that exists only while a capture scope is active

Each allocation row records:

- counted bytes
  - logical payload bytes for ordinary `numpy`, `torch`, and `cupy` arrays
  - physical reservation bytes only for explicit physical carriers such as `cuda_vmm`
- storage carrier: `numpy`, `torch`, `cupy`, or `cuda_vmm`
- one or more logical bindings
  - owner: `caller`, `operator`, or `backend`
  - retention: `call`, `persistent`, `captured`, `on_demand`, `staging`
  - activity:
    - `always`: retained and always live
    - `yes`: live because the current call is using it
    - `no`: retained but inactive for the current call
  - optional direction / retained-slot `k`
  - merged logical roles (`label`, `kind`)

The ledger counts live data buffers visible to this library at the snapshot checkpoint.

The ledger does not count:

- Python object headers
- opaque library metadata such as CUDA streams, CUDA events, cuSPARSE descriptors, or MKL internal handle overhead
- allocator reserve, pool slack, or unrelated process RSS

## Derived tree

Human-facing reports call `tree_rows(snapshot, ...)` to derive an additive tree from flat allocations.

The default grouping is:

- space root: `cuda_live` or `cpu_live`
- `call` vs `retained`
- direction when present
- retained `k=<value>` when present

When one retained physical allocation is bound to multiple observed directions or slot bindings in the same snapshot, the tree keeps that exact merged binding such as `up|down` or `k=1|2`.

Leaf rows are named from merged logical labels.
Leaf annotations summarize merged bindings. When a physical allocation has multiple bindings, owner/retention/activity are rendered as joined values.

## Table columns

The benchmark memory table renders the derived tree with these columns:

- `CaseDir`
- `CaseK`
- `Node`
- `Parent`
- `GiB`
- `Space`
- `Owner`
- `Active`
- `Retention`
- `Kinds`
- `Note`

## Fail-fast capture contract

`SpmvGRG.matmul()` opens a backend call-capture scope only after all input validation and internal conversions succeed.

Every successful backend run method must publish exactly one `CallCapture` before return.

Direct backend API calls do not participate in memory capture unless a capture scope is already active.

`SpmvGRG.matmul()` consumes that capture immediately after the backend call and raises if:

- no capture was published
- the nonce is stale
- the published direction is wrong
- the published runtime `k` is wrong

Snapshot construction also raises if:

- a producer memory root is missing
- a field in a walked memory dataclass is unclassified
- an ignored field hides a supported allocation carrier
- the same physical allocation is seen with conflicting physical facts

`BackendBase` also enforces two lifetime rules:

- backend call memory exists only inside an active capture scope and is dropped structurally when that scope exits
- generic setup payload kept on `BackendBase` after `setup()` must be explicitly classified as `retained`, `borrowed`, or `dropped`
