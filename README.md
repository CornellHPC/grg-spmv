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

## Quick start with `load()`

`load()` is the explicit-config convenience API. It does not probe for a default
backend. You must point `PYGRGL_SPMV_CONFIG` at a JSON file.

```python
import os
from importlib.resources import as_file, files

import numpy as np

from pygrgl_spmv import load

cfg = files("pygrgl_spmv.configs").joinpath("reference-default.json")
with as_file(cfg) as cfg_path:
    os.environ["PYGRGL_SPMV_CONFIG"] = str(cfg_path)
    op = load(
        "/path/to/file.grg",
        np.float64,
        artifact_dir="pygrgl_spmv_artifacts",
    )
```

Bundled sample configs:

- `reference-default.json`: base-install CPU reference backend
- `mkl-default.json`: MKL backend, requires `libmkl_rt.so`
- `cusparse-default.json`: cuSPARSE backend, requires CuPy + CUDA
- `triton-default.json`: Triton backend, requires Torch + Triton + CUDA

The JSON schema is strict:

- `backend` must be one of `reference`, `mkl`, `cusparse`, or `triton`
- the selected backend section must be present
- each `up` / `down` entry must be either `null` or a full backend plan dict
- all fields must be written explicitly; `load()` does not fill in omitted plan fields

One-sided example:

```json
{
  "backend": "reference",
  "reference": {
    "log_level": "WARNING",
    "up": {
      "store": "N",
      "fmt": "CSR",
      "k_hint": null
    },
    "down": null
  }
}
```

Successful backend selection is logged at INFO level on the `pygrgl_spmv` logger.

## Manual backend construction

If you want to construct backends directly, `SpmvGRG` still accepts an explicit
backend instance.

```python
import numpy as np

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.backends.mkl import MklBackend, MklPlanPair

backend = MklBackend(
    pair=MklPlanPair.from_dicts(
        {"k_hint": None, "store": "N", "fmt": "CSR", "n_threads": 1},
        {"k_hint": None, "store": "T", "fmt": "CSC", "n_threads": 1},
    ),
    log_level="WARNING",
)

op = SpmvGRG("/path/to/file.grg", backend, np.float64, artifact_dir="pygrgl_spmv_artifacts")
```

Set either side of the plan pair to `None` to build a one-sided operator.

GPU backends (`CusparseBackend`, `TritonBackend`) require explicit
`device=`, `stream=`, and `ring_buffer_size=` constructor arguments.

## `convert()`

`convert()` compiles a `.grg` or loaded `pygrgl.ImmutableGRG` into a
`CompiledOperatorState`, optionally saving a `.grg_spmv` artifact.

```python
from pygrgl_spmv import convert

state = convert("/path/to/file.grg")
saved = convert("/path/to/file.grg", output_dir="artifacts")
```

## Artifact behavior

- `.grg` input builds or reuses a cached `.grg_spmv` artifact under `artifact_dir`
- `.grg_spmv` input loads the artifact directly
- loaded `ImmutableGRG` input compiles in memory and leaves `artifact_path` unset

## Documentation

- [GRG Operator Docs](pygrgl_spmv/grg/README.md)
- [Backend Docs](pygrgl_spmv/backends/README.md)
- [Benchmark Script Docs](scripts/README.md)
- [Test Suite Docs](pygrgl_spmv/tests/README.md)

## Benchmarks

Benchmark entrypoints:

- `python -m scripts.bench.mkl`
- `python -m scripts.bench.cusparse`
- `python -m scripts.bench.triton`

See `scripts/README.md` for explicit plan syntax, wildcard expansion, and benchmark-only flags.

## Tests

Recommended command:

```bash
uv run pytest -q pygrgl_spmv/tests --backend all
```
