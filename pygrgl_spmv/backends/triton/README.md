Parent docs: [Backend Docs](../README.md)

# Triton Backend

The Triton backend provides GPU sparse traversal with Triton CSR/CSC kernels.

## Public API

Exports from [__init__.py](__init__.py):

- `TritonBackend`
- `TritonPlan`
- `TritonPlanPair`

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
- uploads `sample_rows` and selector tensors
- autotunes the Triton kernel family for each configured direction
- creates captured workspaces when the effective `k_hint` is present

Execution:

- uses a single dense `node_state` tensor per workspace
- exposes per-level `level_views` as slices of `node_state`
- launches level-wise sparse kernels
- uses helper streams and scratch buffers on levels enabled by the `scratch` plan field
- gathers endpoint outputs through retained staging buffers

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

- uploaded sample-row mapping tensor
- sparse block structure
- selector tensors
- captured workspaces
- on-demand workspaces
- staging buffers

Call snapshots mark the workspace and staging objects used by the current case as `Active=yes`.
