# pygrgl-spmv

`pygrgl-spmv` provides sparse matmul backends for GRG-based genotype matrix traversal.

## Installation

```bash
pip install pygrgl-spmv
```

Optional extras:

```bash
pip install "pygrgl-spmv[gpu]"
pip install "pygrgl-spmv[dev]"
```

## Quick usage

```python
import numpy as np
from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.backends.mkl import MklPlan

op = SpmvGRG(
    "/path/to/file.grg",
    {
        "type": "mkl",
        "plan_up": MklPlan.from_any({"k_hint": None, "store": "N", "fmt": "CSR", "n_threads": 1}),
        "plan_down": MklPlan.from_any({"k_hint": None, "store": "T", "fmt": "CSC", "n_threads": 1}),
    },
    np.float64,
    np.uintp,
    cache_dir="pygrgl_spmv_cache",
)
```

Set `plan_up` or `plan_down` to `None` to build a one-sided operator.

## Cache behavior

- `SpmvGRG` stores/loads NPZ caches under `cache_dir`
- default cache root: `./pygrgl_spmv_cache`
- cache file paths encode the full GRG path to avoid collisions between GRGs with the same filename in different directories

## Benchmarks

Benchmark scripts:

- `python -m scripts.bench.mkl`
- `python -m scripts.bench.cusparse`

See `scripts/README.md` for the explicit `--plan-up-down` syntax, wildcard expansion, and search commands.

Internally, the GRG implementation now lives under `pygrgl_spmv/grg/`, and backend implementations live under `pygrgl_spmv/backends/mkl/` and `pygrgl_spmv/backends/cusparse/`.

## Tests

See `pygrgl_spmv/tests/README.md` for test layout, marker policy, CLI options, and recommended commands.

## Backend memory tracking

See `pygrgl_spmv/backends/README.md` for static/runtime memory tracking, plan objects, and backend-specific accounting details.
