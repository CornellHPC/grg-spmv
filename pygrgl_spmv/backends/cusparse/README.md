Parent docs: [Backend Docs](../README.md)

# cuSPARSE

`CusparseRuntime` owns one shared GPU execution arena across all bound GRGs in a layout.

`CusparseLayout` plans:

- resident vs streamed sparse blocks
- heterogeneous ring slots
- exact `cusparseSpMM_bufferSize` ext buffers
- shared dense workspaces and source buffers
- selector payloads
- optional XTX bias arrays
- shared-ones mode and bytes

VMM shared-ones support is kept and planned explicitly. Budgeting counts physical bytes, not logical bytes.
Planner `vram_budget_bytes=0` means use `required_budget_for_full_residency`; planners do not query current free VRAM.

## Planning Warning

cuSPARSE planning queries exact `cusparseSpMM_bufferSize` values using sparse
descriptors whose `indices`, `indptr`, and values pointers are null. This is
observed but unsupported cuSPARSE behavior: the buffer-size query appears to use
descriptor metadata rather than sparse contents. The behavior is isolated to
layout planning because loading real sparse structures there would require
repeated full passes over large artifacts. Runtime execution still uses real
`indices` and `indptr` arrays.
