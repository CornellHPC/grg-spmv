# pygrgl-spmv

`pygrgl-spmv` is a runtime-owned sparse matmul prototype for GRG genotype matrices.

## Core workflow

```python
import numpy as np

from pygrgl_spmv import RuntimeRequirements, convert
from pygrgl_spmv.backends.cusparse import CusparsePlanPair, CusparseRuntime, plan_cusparse_layout

artifact = convert("A.grg", "artifacts")
req = RuntimeRequirements(
    max_k_up=8,
    max_k_down=8,
    need_down_miss_input=True,
    need_up_miss_output=False,
    need_init_vector=True,
    need_init_matrix=False,
    need_init_xtx=True,
)
pair = CusparsePlanPair.from_dicts(
    {"store": "N", "fmt": "CSR", "opA": "N", "opB": "N", "orderB": "ROW", "orderC": "ROW", "algo": "DEFAULT", "scratch": "none"},
    {"store": "T", "fmt": "CSC", "opA": "N", "opB": "N", "orderB": "ROW", "orderC": "ROW", "algo": "DEFAULT", "scratch": "none"},
)
layout = plan_cusparse_layout(
    artifacts=[artifact],
    pair=pair,
    dtype=np.float64,
    requirements=req,
    vram_budget_bytes=8_000_000_000,
    ring_buffer_size=4,
    device=0,
    stream=0,
)

with CusparseRuntime(layout) as runtime:
    (A,) = runtime.grgs
    with A.prepare_matmul_cuda(direction="up", k=1) as op:
        op.input.copy_(op.input.new_tensor(np.ones((1, A.num_samples), dtype=np.float64)))
        op()
        y = op.output.cpu().numpy().copy()
```

## Public surface

- Package root exports:
  - `convert(...) -> Path`
  - `RuntimeRequirements`
  - `plan_reference_layout(...)`, `ReferenceRuntime`
  - `plan_mkl_layout(...)`, `MklRuntime`
- GPU backends are imported from their subpackages:
  - `pygrgl_spmv.backends.triton`
  - `pygrgl_spmv.backends.cusparse`
- GPU execution is centered on `grg.prepare_matmul_cuda(...)`; eager `grg.matmul(...)` is a NumPy convenience wrapper over that prepared path

## Notes

- planners and runtimes consume `.grg_spmv` artifacts only
- runtime-owned buffers are allocated in `__enter__()`
- one runtime owns one shared execution arena across all `runtime.grgs`
- concurrent calls on one runtime fail fast
- Triton and cuSPARSE support declared `max_k >= 1`
- GPU layouts can mix resident and streamed sparse blocks under a VRAM budget
- `ring_buffer_size=0` is valid for GPU layouts only when the budget keeps every sparse block resident
- the package root intentionally stays CPU-safe and does not re-export GPU runtime symbols

## Benchmark scripts

- `uv run python -m scripts.bench.mkl`
- `uv run python -m scripts.bench.triton`
- `uv run python -m scripts.bench.cusparse`

See [scripts/README.md](scripts/README.md) for the minimal benchmark CLI.
