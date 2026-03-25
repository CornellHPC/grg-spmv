Parent docs: [Backend Docs](../README.md)

# cuSPARSE Backend

The cuSPARSE backend provides GPU sparse traversal with explicit cuSPARSE SpMM plans.

## Public API

Exports from [__init__.py](__init__.py):

- `CusparseBackend`
- `CusparsePlan`
- `CusparsePlanPair`
- `DenseOrder`
- `Operation`
- `SparseFormat`
- `SpMMAlgorithm`
- `is_valid_combo`

## Plans

`CusparsePlan` is defined in [plan.py](plan.py).

Fields:

- `store`
- `fmt`
- `opA`
- `opB`
- `orderB`
- `orderC`
- `algo`
- `scratch`
- `k_hint`

Useful plan properties:

- `direction`
- `supported`
- `deterministic`
- `need_buffer`
- `need_preprocess`
- `can_share_storage_with(...)`

The plan object is the execution contract for sparse descriptors, dense layout, buffer requirements, and scratch scheduling.

## Shared values

Sparse blocks store binary ones as a shared values source instead of one value buffer per block.

`backend.py` builds one `_SharedOnes` object with:

- `logical_nbytes`
- `physical_nbytes`
- `vmm`

Possible paths:

- VMM-backed shared physical storage when CUDA VMM is available and profitable
- one materialized all-ones array otherwise

The memory ledger counts:

- logical bytes for the materialized all-ones array path
- physical bytes for the VMM path through `VmmAliasedAlloc`

## Dense-state layout

The backend maintains one `_DenseState` per workspace.

Each level owns canonical dense buffers in `orderC`.

The source-side dense view is chosen from the plan:

- direct alias when `orderB == orderC` and `opB == N`
- descriptor reinterpretation when `orderB != orderC` and `opB == T`
- explicit source buffers otherwise

This makes dense-layout behavior explicit and keeps source-buffer ownership visible in the memory ledger.

## Scratch scheduling

The `scratch` field controls which destination levels use helper-stream scratch scheduling.

Accepted values are:

- `none`
- `all`
- explicit `|`-separated destination levels

For scratch-enabled levels:

- each op gets its own scratch buffer
- helper streams launch SpMM into scratch buffers
- the destination stream reduces scratch buffers into the canonical level buffer in a deterministic order

## Retained execution state

Retained execution memory is organized into:

- `captured`: captured workspaces built during setup for the effective `k_hint`
- `on_demand`: lazily created workspaces for uncaptured execution
- `staging`: retained input/output/init helper buffers, keyed by direction and runtime `k`

The backend keeps at most one captured workspace per direction and at most one on-demand workspace per direction. Staging is keyed by direction and runtime `k`.

## Instrumentation

`instrumentation=True` enables NVTX tracing and uses the uncaptured execution path.

When instrumentation is enabled and a plan specifies `k_hint`, the backend warns and does not build captured workspaces for that hint.

## Memory ledger mapping

Retained device leaves include:

- sparse block index payload
- shared values
- selector payload
- sample-routing arrays
- captured workspaces
- on-demand workspaces
- staging buffers

The ledger does not count:

- cuSPARSE dense/sparse descriptors
- CUDA streams
- CUDA events

The `vmm` annotation is attached to the shared-values row when that source uses CUDA VMM.
