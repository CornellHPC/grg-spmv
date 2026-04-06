Parent docs: [Project README](../../README.md)

# Test Suite Guide

## Layout

- `test_memory_ledger.py`
  - flat-allocation collector behavior and lifecycle invariants
- `backends/`
  - backend-specific behavior for MKL and cuSPARSE
  - backend base/reference behavior
  - backend memory-ledger integration behavior
- `operator/`
  - traversal correctness and numerical invariants
  - public matmul option semantics
  - artifact lifecycle and observability behaviors
  - operator-owned memory-ledger snapshots
- `bench/`
  - benchmark helper behavior for summary tables, ledger rendering, and equivalence checks
- `endtoend/`
  - cross-method integration semantics for full GRG workflows
- `data/`
  - immutable fixture datasets used by end-to-end and missingness tests

Implementation paths exercised by these tests:

- GRG code lives under `pygrgl_spmv/grg/`
- backend code lives under `pygrgl_spmv/backends/base.py`, `reference.py`, `mkl/`, `triton/`, and `cusparse/`
- benchmark helper tests target the split modules in `scripts/bench/`

Observability rule covered by the suite:

- `log_level` controls logger verbosity
- operator build logs follow normal Python/root logger inheritance
- compile RSS checkpoints are emitted on the `pygrgl_spmv.grg.compile` logger
- GPU setup RSS checkpoints are emitted on the backend logger
- `instrumentation` is the opt-in flag for slower profiling/observability behavior

Memory-ledger rule covered by the suite:

- `SpmvGRG.memory` owns one retained snapshot plus the latest call snapshot
- backend call captures are fail-fast
- backend call memory is lexical and exists only inside an active capture scope
- validation failures before backend execution must not leave capture active
- backend generic setup payload must be explicitly retained, borrowed, or dropped
- benchmark memory tables are rendered from canonical `tree_rows()` output plus benchmark-local case metadata
- GPU retained sparse memory is now split between CPU pinned host block
  structure and shared CUDA slot buffers
- GPU setup streams stored sparse blocks one at a time during host pinning
- Triton runtime execution is singleton-only (`k == 1`)

## Markers

- `smoke`: core fast checks selected by `--smoke`
- `stress`: literal machine-scale stress tests selected by `--stress`
- `gpu`: requires CuPy/cuSPARSE
- `mkl`: requires MKL runtime

`--smoke` runs only tests marked `smoke`.
Stress tests are skipped unless `--stress` is supplied.

The giant streamed exact-output and ring-3 OOM contract tests live under `stress`.
The many-small-block streamed-overlap tests are ordinary GPU tests.
cuSPARSE large-stream stress keeps `nnz` below the CUDA 12.9 near-`2^31`
cuSPARSE SpMM bug boundary and forces int64 slot families locally in the test.

## CLI Options

- `--backend {all,mkl,cusparse,triton}`
- `--smoke`
- `--stress`
- `--grg <path>`: primary GRG used by traversal/backend tests
- `--missing-grg <path>`: missingness GRG used by missingness tests

## Recommended Commands

Full suite on a small GRG:

```bash
uv run pytest -q pygrgl_spmv/tests --backend all
```

Smoke/core suite on a larger GRG:

```bash
uv run pytest -q pygrgl_spmv/tests --backend all --smoke \
  --grg /path/to/large.grg
```

## Fixture Data

- `msprime.example.igd.final.grg` (small primary fixture)
- `test-200-samples.miss.igd` (source IGD for missingness fixture)
- `test-200-samples.miss.final.grg` (missingness fixture)

Regenerate missingness fixture:

```bash
uv run grg construct --force -p 10 -j 4 \
  pygrgl_spmv/tests/data/test-200-samples.miss.igd \
  -o pygrgl_spmv/tests/data/test-200-samples.miss.final.grg
```
