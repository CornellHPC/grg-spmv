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
from pygrgl_spmv.backends.mkl import MklBackend, MklPlanPair

pair = MklPlanPair.from_dicts(
    {"k_hint": None, "store": "N", "fmt": "CSR", "n_threads": 1},
    {"k_hint": None, "store": "T", "fmt": "CSC", "n_threads": 1},
)
backend = MklBackend(pair=pair, instrumentation=False, log_level="WARNING")

op = SpmvGRG(
    "/path/to/file.grg",
    backend,
    np.float64,
    np.int32,
    artifact_dir="pygrgl_spmv_artifacts",
)
```

Set either side of the plan pair to `None` to build a one-sided operator.

GPU backends (`CusparseBackend`, `TritonBackend`) additionally require
mandatory `device=` and `stream=` constructor arguments.

- `device` is a visible CUDA ordinal such as `0`
- `stream` is either a raw `cudaStream_t` handle such as `0`, or a CUDA Stream
  Protocol object such as `cupy.cuda.Stream.null`

`stream=0` means the null stream on the declared `device`. Non-null external
streams must belong to that same device.

`log_level` controls verbosity only. Set `instrumentation=True` when you want
profiling/observability behavior that may reduce absolute performance.

## Artifact behavior

- `SpmvGRG` stores/loads standalone `.grg_spmv` artifacts under `artifact_dir` when you construct from a `.grg`
- default artifact root: `./pygrgl_spmv_artifacts`
- derived artifact paths encode the full GRG path to avoid collisions between GRGs with the same filename in different directories
- you can also construct `SpmvGRG` directly from a `.grg_spmv` file without the original `.grg`

## Documentation

- [GRG Operator Docs](pygrgl_spmv/grg/README.md)
- [Backend Docs](pygrgl_spmv/backends/README.md)
- [Benchmark Script Docs](scripts/README.md)
- [Test Suite Docs](pygrgl_spmv/tests/README.md)

## Package layout

- `pygrgl_spmv/grg/`: operator construction, artifact I/O, and GRG compilation
- `pygrgl_spmv/backends/`: backend implementations and backend-specific plan types
- `pygrgl_spmv/memory.py`: additive live-memory ledger
- `scripts/bench/`: benchmark runner, reporting, and CLI helpers

## Benchmarks

Benchmark scripts:

- `python -m scripts.bench.mkl`
- `python -m scripts.bench.cusparse`
- `python -m scripts.bench.triton`

See `scripts/README.md` for the explicit `--plan-up-down` syntax, wildcard
expansion, search commands, and the shared `--instrumentation` flag.

## Tests

See `pygrgl_spmv/tests/README.md` for test layout, marker policy, CLI options, and recommended commands.

## Memory ledger

See `pygrgl_spmv/backends/README.md` for the additive live-memory ledger, snapshot timing, and benchmark memory-table taxonomy.
