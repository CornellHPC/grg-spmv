# Backend Memory Model

This document describes how memory usage is tracked for all backends through
`Backend.mem_usage` (`MemoryUsage` dataclass).

## Shared Memory Tracking (all backends)

`mem_usage` has three parts:

1. `host_static: StaticBytes`
2. `device_static: StaticBytes`
3. `calls: list[MemoryRecord]` (one runtime record per matmul call)

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

`host_static` and `device_static` use the same key schema so estimates and
measured values are directly comparable field-by-field.

### Block-format resolution (`fmt_up`, `fmt_down`)

All backends resolve block storage policy once in `Backend.__init__`:

- at least one of `fmt_up` / `fmt_down` must be set
- supported formats are `csr`, `csc`, `coo`
- if one side is `None`, the transpose-compatible format is inferred
- if both sides are transpose-compatible (for example `csr/csc`, `csc/csr`,
  `coo/coo`), one physical block set is stored and the other traversal aliases
  it via transpose semantics
- if both sides are explicitly non-compatible, both block sets are materialized

This policy is backend-agnostic and directly drives the `blocks_up` /
`blocks_down` static-memory counters.

### Runtime memory (`RuntimeBytes`) keys

- `level_buffers`
- `inputs`
- `outputs`
- `aux`

Each `MemoryRecord` stores host/device runtime bytes plus metadata
(`direction`, mode flags, etc.).

### Actual vs Estimated static memory

- **Actual** static memory comes from `backend.setup()` accounting.
- **Estimated** static memory comes from `backend.estimate_static_bytes()`.
- Tests assert alignment between estimated and actual static fields for both
  host and device schemas.

---

## MKL backend details

- Sparse blocks (`blocks_up`, `blocks_down`) are MKL handle payloads backed by
  host-side sparse arrays.
- Selector buffers are stored on host (`selector_mut`, `selector_miss`).
- `device_static` remains zero for MKL by design.
- Common host arrays (`level_offsets`, permutations, optional coalescence/xtx)
  are included in both measured and estimated static host usage.

Runtime calls track host-side level buffers and temporary arrays in
`mem_usage.calls`.

---

## cuSPARSE backend details

- Sparse block payloads are device arrays (`blocks_up`, `blocks_down`).
- Selector row/col index buffers are device arrays (`selector_mut`,
  `selector_miss`).
- Optional XTX init buffers (`xtx_init`) are tracked on device when
  coalescence counts are available.
- Host static usage still includes common metadata arrays
  (`level_offsets`, permutations, optional coalescence counts).

Runtime calls track both host and device transient usage, including workspace
buffers, staging buffers, and per-call auxiliaries.
