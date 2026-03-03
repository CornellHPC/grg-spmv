# Benchmark scripts

Backend-specific entrypoints:

- `python -m scripts.bench.mkl`
- `python -m scripts.bench.cusparse`

Each script expands backend configurations, runs all requested scenarios, then
prints one unified summary table.

## Summary table

Columns:

- `Config`
- `Scenario`
- `Direction`
- `k`
- `Call ms`
- `Host GiB`
- `Device GiB`
- `Note`

Row semantics:

- `static`: measured static memory (`grg._backend.mem_usage.host_static/device_static`)
- `static_est`: estimated static memory (`backend.estimate_static_bytes()`)
- runtime rows: per-scenario timed calls and runtime memory
- skip rows: `Call ms=SKIP (...)`, memory columns `SKIP`

Rows are printed in execution order. There are no config block delimiters.

## Output equivalence checks

Benchmark correctness checks use full-output arrays:

- warmup/trial outputs for each case must match (`np.allclose`)
- cross-config outputs are compared within equivalence classes:
  - same `(scenario, direction, k)`
  - plus `up/miss` mapped to `up/baseline`

Reference outputs are written to temporary `.npy` files during the run and
cleaned up on success.

## Common flags

- `--grg`
- `--ks`
- `--trials`
- `--warmup`
- `--k-hints`
- `--fmt-up-down`
- `--matmul-options`
- `--log-level`
- `--dry-run`
- `--skip-note` (hide the `Note` column in summary output)

MKL-only flag:

- `--threads`

## Examples

MKL:

```bash
uv run python -m scripts.bench.mkl \
  --grg /pscratch/sd/q/qys/grg/msprime.example.igd.final.grg \
  --ks 4,16 \
  --warmup 1 --trials 3 \
  --threads 1,4 \
  --k-hints none,4 \
  --fmt-up-down csr,none/coo,coo \
  --matmul-options baseline,init_vector
```

cuSPARSE:

```bash
uv run python -m scripts.bench.cusparse \
  --grg /pscratch/sd/q/qys/grg/msprime.example.igd.final.grg \
  --ks 4,16 \
  --warmup 1 --trials 3 \
  --k-hints none,4 \
  --fmt-up-down csr,none/none,csc \
  --matmul-options baseline,init_vector
```

Dry run:

```bash
uv run python -m scripts.bench.cusparse \
  --dry-run \
  --k-hints none,4 \
  --ks 1,4 \
  --fmt-up-down csr,none/none,csc
```
