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

MKL thread counts use `MKL_Set_Num_Threads_Local`.
Runtime entry scopes the local count only around setup and handle optimization on the entering thread.
Each matmul scopes the local count again on the calling thread before invoking MKL.

One `MklRuntime` still rejects concurrent matmul calls because its workspaces are shared.
Use one runtime per worker thread for parallel MKL execution.
