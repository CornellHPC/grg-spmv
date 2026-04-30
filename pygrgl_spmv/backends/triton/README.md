Parent docs: [Backend Docs](../README.md)

# Triton

`TritonRuntime` keeps the streamed slot/level/scratch execution model, but all owned device buffers are allocated in `__enter__()`.

`TritonLayout` plans:

- resident vs streamed sparse blocks
- heterogeneous ring slots
- selector tensors
- shared dense workspaces
- shared scratch buffers

Planner `vram_budget_bytes=0` means use `required_budget_for_full_residency`; planners do not query current free VRAM.
