Parent docs: [Backend Docs](../README.md)

# MKL

`MklRuntime` owns one shared host execution arena across all bound GRGs in a layout.

`MklPlan` keeps only:

- `store`
- `fmt`
- `n_threads`
- `optimize`

`MklRuntime` supports `float32` and `float64`.

`plan_mkl_layout(...)` sizes retained sparse structure, selectors, dense workspaces, and one shared sparse-value buffer.
The shared-value budget is physical retained memory, not the logical aliased span.

`optimize=True` enables MKL handle hints plus `mkl_sparse_optimize()` during runtime entry.
