Parent docs: [Backend Docs](../README.md)

# Triton Backend

The Triton backend provides GPU sparse traversal with Triton CSR/CSC kernels.

## Public API

Exports from [__init__.py](__init__.py):

- `TritonBackend`
- `TritonPlan`
- `TritonPlanPair`

`TritonBackend` requires mandatory `device=` and `stream=` constructor
arguments.

- `device`: visible CUDA ordinal such as `0`
- `stream`: accepted forms are:
  - raw `cudaStream_t` integer handle such as `0`
  - any CUDA Stream Protocol object such as `cupy.cuda.Stream.null`

Protocol-backed stream objects are retained by the backend for its lifetime.
Raw integer handles are treated as non-owning.
`stream=0` means the null stream on the declared device. Non-null external
streams must belong to that same device.

## Plans

`TritonPlan` is defined in [plan.py](plan.py).

Fields:

- `store`
- `fmt`
- `k_hint`
- `scratch`

Validation:

- supported formats are `CSR` and `CSC`
- `k_hint` must be `none` or `1`
- `scratch` accepts:
  - `none`
  - `all`
  - an explicit `|`-separated level list

`TritonPlanPair` enforces directional store conventions:

- UP expects `store=N`
- DOWN expects `store=T`

## Runtime behavior

`TritonBackend` is implemented in [backend.py](backend.py).

Setup:

- uploads sparse structure to device tensors
- uploads selector tensors
- autotunes the Triton kernel family for each configured direction
- creates captured workspaces when the effective `k_hint` is present

Execution:

- uses a single dense `node_state` tensor per workspace
- exposes per-level `level_views` as slices of `node_state`
- launches level-wise sparse kernels
- uses helper streams and scratch buffers on levels enabled by the `scratch` plan field
- gathers endpoint outputs through retained staging buffers

The supplied `stream=` is the caller stream on the declared `device=`.

The backend owns:

- one root stream per backend instance
- backend-owned level streams
- backend-owned scratch streams

Setup-time GPU allocation/upload, staging, autotune uploads, graph
warmup/capture/replay, and root-side gathers run on root.

Triton wavefront work runs on the level and scratch streams.

Each wavefront joins per-level completion back onto root before return.

`level_views` share storage with `node_state`; they collapse onto the same physical allocation row and only add an extra logical role.

## Captured, on-demand, and staging state

Retained execution state is split into:

- `captured`: captured graph workspaces built during setup
- `on_demand`: lazily created workspaces for uncaptured execution
- `staging`: retained input/output/init helper buffers

The backend keeps at most one captured workspace per direction and at most one on-demand workspace per direction.

## Instrumentation

`instrumentation=True` enables NVTX tracing and uses the uncaptured execution path.

When instrumentation is enabled and a plan specifies `k_hint`, the backend warns and uses the effective uncaptured path.

## Memory ledger mapping

Retained device leaves include:

- sparse block structure
- selector tensors
- captured workspaces
- on-demand workspaces
- staging buffers

Call snapshots mark the workspace and staging objects used by the current case as `Active=yes`.
