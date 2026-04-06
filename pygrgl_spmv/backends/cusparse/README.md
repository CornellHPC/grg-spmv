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

`CusparseBackend` requires mandatory `device=`, `stream=`, and
`ring_buffer_size=` constructor
arguments.

- `device`: visible CUDA ordinal such as `0`
- `stream`: accepted forms are:
  - raw `cudaStream_t` integer handle such as `0`
  - any CUDA Stream Protocol object such as `cupy.cuda.Stream.null`
- `ring_buffer_size`: number of streamed sparse-structure slots shared across
  UP and DOWN

`stream=0` means the null stream on the declared device. Non-null external
streams must belong to that same device.

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
`need_preprocess` is retained as doc-grounded metadata; the current runtime path
does not call `cusparseSpMM_preprocess()`.

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

## Streamed sparse structure

Uncaptured cuSPARSE execution no longer uploads every sparse block to device
during `setup()`.

Instead it keeps pinned host CSR/CSC/COO structure for retained blocks and
copies each block into one of `ring_buffer_size` shared device slots just
before the corresponding `cusparseSpMM()` launch. Slot assignment is static, so
the dynamic streamed path keeps fixed device addresses.

Selectors and the shared all-ones values source remain device-resident.

The backend resolves slot dtypes from each stored block's final
CSR/CSC/COO shape and `nnz`, then streams setup one block at a time:
materialize one stored block, pin its sparse structure, drop the temporary
SciPy block, and continue. It uses cheap bounds to choose `int32` when that is
provably safe, otherwise keeps conservative `int64` structure, and widens into
shared slot families only when needed. CSR/CSC keep separate offset/index slot
families; COO still uses one common coordinate dtype. Pinned copies are
range-checked against those chosen slot dtypes.
Transposed COO blocks are canonicalized to row-sorted COO before upload because
cuSPARSE assumes COO coordinates are sorted by row.

## Captured cuSPARSE workspaces

Captured and dynamic cuSPARSE execution use the same slot-backed sparse path.

- pinned host CSR/CSC/COO structure remains retained on the host
- each logical op keeps a static slot assignment
- one sparse descriptor per slot is reused for both dynamic execution and graph
  capture
- one SpMM external buffer is pre-allocated per slot from the maximum queried
  `cusparseSpMM_bufferSize()` seen on that slot
- graph capture includes the slot H2D copies plus `cusparseSpMM()` launches

Graph-build failures are treated as hard setup errors. The backend does not
silently downgrade a failed captured workspace to dynamic execution.

## Dense-state layout

The backend maintains one `_DenseState` per workspace. Each level owns
canonical dense buffers in `orderC`.

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

The supplied `stream=` is the caller stream on the declared `device=`. The
backend also owns one root stream, one copy stream per slot, level streams, and
scratch streams.

Setup-time GPU allocation, graph warmup/capture/replay, and root-side gathers
run on root. Sparse H2D copies run on the slot copy streams. SpMM wavefront
work runs on the level and scratch streams.

Each wavefront joins per-level completion back onto root before return.

## Retained execution state

Retained execution memory is organized into:

- `captured`: captured workspaces built during setup for the effective `k_hint`
- `on_demand`: lazily created workspaces for uncaptured execution
- `staging`: retained input/output/init helper buffers, keyed by direction and runtime `k`

The backend keeps at most one captured workspace per direction and at most one
on-demand workspace per direction. Staging is keyed by direction and runtime
`k`.

Captured cuSPARSE workspaces retain the same sparse structure and SpMM ext shape
as the dynamic streamed path, so both stay bounded by `ring_buffer_size`.

## Instrumentation

`instrumentation=True` enables NVTX tracing and uses the uncaptured execution path.

When instrumentation is enabled and a plan specifies `k_hint`, the backend warns and does not build captured workspaces for that hint.

## Memory ledger mapping

Retained device leaves include:

- shared values
- slot buffers
- selector payload
- captured workspaces
- on-demand workspaces
- staging buffers

Retained CPU leaves include:

- pinned host block structure

The ledger does not count:

- cuSPARSE dense/sparse descriptors
- CUDA streams
- CUDA events

The `vmm` annotation is attached to the shared-values row when that source uses CUDA VMM.
