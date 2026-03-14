# Test Suite Guide

## Layout

- `backends/`
  - backend-specific behavior for MKL and cuSPARSE
  - backend base/reference behavior
  - static memory accounting alignment checks
- `operator/`
  - traversal correctness and numerical invariants
  - public matmul option semantics
  - artifact lifecycle + observability behaviors
- `bench/`
  - benchmark helper behavior (summary table + equivalence checks)
- `endtoend/`
  - cross-method integration semantics for full GRG workflows
- `data/`
  - immutable fixture datasets used by end-to-end and missingness tests

Implementation paths exercised by these tests:

- GRG code lives under `pygrgl_spmv/grg/`
- backend code lives under `pygrgl_spmv/backends/base.py`, `reference.py`, `mkl/`, and `cusparse/`
- benchmark helper tests target the split modules in `scripts/bench/`

Observability rule covered by the suite:

- `log_level` changes verbosity only
- `instrumentation` is the opt-in flag for slower profiling/observability behavior

## Markers

- `smoke`: core fast checks selected by `--smoke`
- `gpu`: requires CuPy/cuSPARSE
- `mkl`: requires MKL runtime

`--smoke` runs only tests marked `smoke`.

## CLI Options

- `--backend {all,mkl,cusparse}`
- `--smoke`
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
