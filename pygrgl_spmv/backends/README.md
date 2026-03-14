# Backend Memory Model

This document describes how memory usage is tracked for all backends through
`Backend.mem_usage` (`MemoryUsage` dataclass).

## Shared Memory Tracking

`mem_usage` has three parts:

1. `host_static: StaticBytes`
2. `device_static: StaticBytes`
3. `calls: list[MemoryRecord]`

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

### Runtime memory (`RuntimeBytes`) keys

- `level_buffers`
- `inputs`
- `outputs`
- `aux`

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
  `opA`, `opB`, `orderB`, `orderC`, `algo`, and `k_hint`); the current CUDA
  runtime version remains available as an environment-derived property
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

## cuSPARSE backend details

- sparse blocks are device payloads with separate sparse descriptors for graph
  and dynamic preprocess state
- selector row/col index buffers are stored on device
- optional `xtx_init` vectors are stored on device when coalescence counts are
  available
- runtime memory tracks dense level buffers, input/output staging buffers, and
  external cuSPARSE work buffers

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

`need_buffer` and `need_preprocess` are currently metadata only: the executor
still allocates work buffers and runs preprocess unconditionally. This keeps
runtime control flow simple while preserving the information needed for later
optimization work.

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
