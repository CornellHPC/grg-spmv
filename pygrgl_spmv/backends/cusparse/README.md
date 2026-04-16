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
