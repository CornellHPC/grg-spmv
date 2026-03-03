# pygrgl-spmv

`pygrgl-spmv` provides sparse matmul backends for GRG-based genotype matrix traversal.

## Installation

```bash
pip install pygrgl-spmv
```

Optional extras:

```bash
pip install "pygrgl-spmv[gpu]"   # cuSPARSE / CuPy support
pip install "pygrgl-spmv[dev]"   # tests + plotting + tooling
```

## Quick usage

```python
import numpy as np
from pygrgl_spmv import SpmvGRG

op = SpmvGRG(
    "/path/to/file.grg",
    {"type": "mkl", "n_threads": 1},
    np.float64,
    np.uintp,
    cache_dir="pygrgl_spmv_cache",  # optional, defaults to ./pygrgl_spmv_cache
)
```

## Cache behavior

- `SpmvGRG` stores/load NPZ caches under `cache_dir`.
- default cache root: `./pygrgl_spmv_cache`
- cache file paths encode the full GRG path to avoid collisions between GRGs
  with the same filename in different directories.

## Benchmarks

Benchmark scripts:

- `python -m scripts.bench.mkl`
- `python -m scripts.bench.cusparse`

See `scripts/README.md` for flags, output schema, and examples.

## Tests

See `pygrgl_spmv/tests/README.md` for test layout, marker policy, CLI options,
and recommended commands.

## Backend memory tracking

See `pygrgl_spmv/backends/README.md` for static/runtime memory tracking,
backend-specific accounting details, and estimate-vs-measured semantics.
