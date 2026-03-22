# Backend Memory Model

This document describes how memory usage is tracked for all backends through
`Backend.mem_usage` (`MemoryUsage` dataclass).

## Shared Memory Tracking

`mem_usage` has three parts:

1. `host_static: StaticBytes`
2. `device_static: StaticBytes`
3. `calls: list[MemoryRecord]`

Each runtime `MemoryRecord` now also carries retained residency buckets:

- `static_ws`
- `dynamic_ws`
- `staging`

### Static memory (`StaticBytes`) keys

- `level_offsets`
- `sample_perm`
- `inv_sample_perm`
- `coalescence_counts`
- `xtx_init`
- `blocks_up`
- `blocks_down`
- `selector_mut`
- `selector_miss`
- `workspace`

For the GPU backends, `workspace` means setup-retained execution workspaces
only. GPU XTX bias is no longer backend-owned static state, so `xtx_init`
remains zero for GPU static accounting.

### Runtime memory (`RuntimeBytes`) keys

- `level_buffers`
- `inputs`
- `outputs`
- `aux`

GPU runtime accounting reports the active runtime footprint in `RuntimeBytes`.
Retained workspace/staging bytes are reported separately through the residency
buckets. `aux` includes retained inactive bytes that are not part of the
current `level_buffers` / `inputs` / `outputs` slices.

### Residency buckets

- `static_ws`: setup-retained static workspaces
- `dynamic_ws`: lazily created dynamic workspaces
- `staging`: miss/output/init/XTX state

For the GPU backends, the intended ownership split is:

- workspaces hold execution state only
- staging holds optional miss/output/init/XTX state

### Actual vs Estimated static memory

- actual static memory comes from `backend.setup()` accounting
- estimated static memory comes from `backend.estimate_static_bytes()`
- tests assert alignment between measured and estimated static fields

## Layout

- `pygrgl_spmv/backends/base.py`: shared backend scaffolding, setup payloads, and validation
- `pygrgl_spmv/backends/reference.py`: `ReferenceBackend` and `ReferencePlan`
- `pygrgl_spmv/backends/mkl/`: MKL plan, backend, and FFI modules
- `pygrgl_spmv/backends/cusparse/`: cuSPARSE plan, backend, and FFI modules

## Plan-driven backends

Backends are constructed from explicit backend-specific plan-pair objects.

- `MklPlan` carries MKL-specific storage/runtime hints (`store`, `fmt`,
  `n_threads`, `k_hint`)
- `CusparsePlan` carries explicit cuSPARSE SpMM choices (`store`, `fmt`,
  `opA`, `opB`, `orderB`, `orderC`, `algo`, `scratch`, and `k_hint`); the
  current CUDA runtime version remains available as an environment-derived
  property
- each backend exposes a backend-specific `*PlanPair` type that validates
  its `plan_up` / `plan_down` pair before backend construction

The plan object is the single source of truth for backend execution semantics.
Runtime helper objects may still cache raw storage buffers or workspaces, but
should not duplicate plan metadata such as algorithm, dense order, or transpose
mode.

Either side of a `*PlanPair` may be omitted: `plan_up=None` builds a DOWN-only
backend and `plan_down=None` builds an UP-only backend.

## Backend config instrumentation

All public backends accept a common backend-config flag:

- `instrumentation=False` (default): keep the normal fast path
- `instrumentation=True`: enable slower observability behavior

This flag is runtime/config state, not plan state. `log_level` only controls
verbosity and must not change execution mode by itself.

## MKL backend details

- sparse blocks are MKL handle payloads backed by host sparse arrays
- selector buffers are stored on host
- `device_static` remains zero for MKL by design
- common host arrays are included in measured and estimated static host usage
- `n_threads` and `k_hint` are applied per direction; one side does not override
  the other unless both traversals truly share the same handle and transpose
  mode

## Triton backend details

- sparse blocks are structure-only device CSR/CSC payloads
- graph and dynamic workspaces are kept as separate concepts even though the
  current kernels only support singleton-vector execution
- public `k_hint` accepts only `none` or `1`
- `instrumentation=True` ignores configured `k_hint` and uses the effective
  `k_hint=none` path
- optional miss/output/init/XTX buffers live in per-direction staging
- `device_static.workspace` counts only retained graph workspaces

## cuSPARSE backend details

- sparse blocks are device payloads with separate sparse descriptors for graph
  and dynamic preprocess state
- CSR/CSC/COO block values are binary ones; the backend uses one shared ones
  source across all sparse blocks instead of one `data` allocation per block
- when CUDA VMM is supported and one minimum-granularity VMM tile is smaller
  than a materialized ones array, that shared ones source is virtually aliased
  from one physical allocation tile using the CUDA Driver API
- the VMM tile size, reservation alignment, and physical byte accounting all
  use the allocation minimum granularity; the recommended granularity is logged
  as a performance hint only
- when VMM is unsupported, unavailable in the current context, or offers no
  memory savings, the backend uses one shared materialized all-ones array
- warnings are reserved for VMM query/build failures; normal materialized-path
  selection is INFO-only
- selector row/col index buffers are stored on device
- `instrumentation=True` ignores configured `k_hint` and retains no static
  workspace for that hint
- setup-retained workspaces contain only the buffers needed for the minimal
  node-output matvec for that execution path
- optional miss/output/init/XTX buffers live in per-direction-per-`k` staging
- runtime memory tracks dense level buffers, active staging buffers, eager
  scratch/ext buffers, and retained inactive staging/dynamic state in `aux`
- static block memory accounting reflects the physical shared ones allocation,
  not the reserved virtual alias range

### cuSPARSE package layout

- `pygrgl_spmv/backends/cusparse/backend.py`: `CusparseBackend`, sparse-block descriptors, selector routing, dense-view logic, workspace helpers, and execution
- `pygrgl_spmv/backends/cusparse/plan.py`: `CusparsePlan` and its enums/parsers
- `pygrgl_spmv/backends/cusparse/ffi.py`: ctypes bindings and CUDA/cuSPARSE constants

### `CusparsePlan`

`CusparsePlan` is grounded on CUDA 12.9.0 SpMM semantics and exposes the
current CUDA 12.x runtime version as an environment-derived property backed by
`cupy.cuda.runtime.runtimeGetVersion()`.

It exposes doc-driven properties such as:

- `direction`
- `supported`
- `deterministic`
- `need_buffer`
- `need_preprocess`
- `can_share_storage_with(...)`

The executor now consumes `need_buffer` and `need_preprocess` directly, while
still auditing `need_buffer` against the runtime `cusparseSpMM_bufferSize`
result and warning when the plan expectation disagrees with the queried buffer
size.

The `scratch` field controls which destination levels use the multi-stream
scratch scheduler. Accepted values are:

- `none`
- `all`
- a `|`-separated list of destination levels such as `0|2|3`

### Dense layout execution

The cuSPARSE backend now treats the dense side of each direction plan
(`order_b`, `order_c`, `op_b`) explicitly.

For one direction and one runtime `k`, each level owns a canonical dense state
buffer stored in `order_c`. The backend then derives the source-side `matB`
view in one of three ways:

- direct alias: `order_b == order_c` and `op_b == N`
- descriptor reinterpretation: `order_b != order_c` and `op_b == T`
- one-per-level repack into a separate source buffer for all remaining cases

This keeps the wavefront level-oriented: if repacking is needed, it happens once
per completed destination level, not once per block SpMM call.

### Scratch scheduling and workspace lifecycle

For scratch-enabled levels, cuSPARSE now mirrors Triton's helper-stream
scheduler shape:

- each op for the destination level gets its own scratch buffer
- helper streams launch `SpMM` into scratch buffers with `beta=0`
- the destination level stream reduces scratch buffers back into the canonical
  level buffer in a deterministic order
- source repack/publication happens only after the reduction finishes

Workspace allocation follows the backend memory model:

- setup-retained workspaces are whichever hinted graph workspaces are actually
  built in `setup()` for the effective mode
- other dynamic workspaces are allocated lazily on first use
- output staging, miss staging, init staging, and XTX bias live in staging
- `device_static.workspace` counts only the setup-retained workspaces
