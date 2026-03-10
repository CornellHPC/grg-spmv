# Benchmark scripts

Backend-specific entrypoints:

- `python -m scripts.bench.mkl`
- `python -m scripts.bench.cusparse`

Each script expands explicit backend plan pairs, runs the requested scenarios,
then prints one unified summary table.

Both entrypoints call the same shared benchmark runner and therefore execute
the same stress/correctness workflow (intra-case + cross-config diagnostics).

## Summary table

Columns:

- `Config`
- `Scenario`
- `Direction`
- `k`
- `Call ms`
- `Err/Trials`
- `Abs Err (avg/max)`
- `Rel Err (avg/max)`
- `Host GiB`
- `Device GiB`
- `Note`

Row semantics:

- `static`: measured static memory (`grg._backend.mem_usage.host_static/device_static`)
- `static_est`: estimated static memory (`backend.estimate_static_bytes()`)
- runtime rows: per-scenario timed calls and runtime memory
- runtime rows also include correctness diagnostics:
  - `Err/Trials = intra_errors/intra_trials` (intra-case only)
  - `Abs Err (avg/max)`: average/max absolute error over intra comparisons
  - `Rel Err (avg/max)`: average/max relative error over intra comparisons
- directions with no configured plan are omitted entirely
- skip rows: `Call ms=SKIP (...)`, memory columns `SKIP`

## Output equivalence checks

Benchmark correctness checks use full-output arrays and do not stop the run:

- intra-case warmup/trial outputs are checked with `np.allclose`
- cross-config outputs are checked with full pairwise comparisons inside each equivalence class:
  - same `(scenario, direction, k)`
  - plus `up/miss` mapped to `up/baseline`
  - each pair must pass `np.allclose(a, b)` and `np.allclose(b, a)`
- mismatch indices are appended to `Note` for intra-case only
- cross diagnostics are reported after the table

## Common flags

- `--grg`
- `--ks`
- `--trials`
- `--warmup`
- `--plan-up-down`
- `--matmul-options`
- `--log-level`
- `--dtype` (`float32` or `float64`)
- `--index-dtype` (`int32` or `int64`)
- `--dry-run`
- `--skip-note`

`--plan-up-down` uses two adjacent bracketed plan literals:

```text
[k_hint=none,store=N,fmt=CSR,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT]
[k_hint=none,store=T,fmt=CSC,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT]
```

Pass them as one argument with no delimiter between the two bracket groups:

```bash
--plan-up-down="[...][...]"
```

For cuSPARSE, `*` is allowed for every field except `k_hint`. Negation is also supported for enum-like fields, for example `fmt=!COO` or `fmt=!CSR!CSC`. Empty sides are allowed: `[][PLAN]` means benchmark DOWN only, `[PLAN][]` means benchmark UP only. The current executor now supports the broader docs-valid dense-side plan space, including `opB=T` and column-major `orderB/orderC`, except for separately-guarded runtime quirks such as `CSC + CSR_ALG3`.

## Examples

MKL:

```bash
uv run python -m scripts.bench.mkl \
  --grg /pscratch/sd/q/qys/grg/msprime.example.igd.final.grg \
  --ks 4,16 \
  --warmup 1 --trials 3 \
  --dtype float64 \
  --index-dtype int64 \
  --plan-up-down="[k_hint=none,store=N,fmt=CSR,n_threads=1][k_hint=none,store=T,fmt=CSC,n_threads=1]" \
  --matmul-options baseline,init_vector
```

cuSPARSE dry run of the full executable search space on `k=1`:

```bash
module load cudatoolkit/12.9

uv run python -m scripts.bench.cusparse \
  --dry-run \
  --ks 1 \
  --plan-up-down="[k_hint=none,store=*,fmt=*,opA=*,opB=*,orderB=*,orderC=*,algo=*][k_hint=none,store=*,fmt=*,opA=*,opB=*,orderB=*,orderC=*,algo=*]" \
  --plan-up-down="[k_hint=1,store=*,fmt=*,opA=*,opB=*,orderB=*,orderC=*,algo=*][k_hint=1,store=*,fmt=*,opA=*,opB=*,orderB=*,orderC=*,algo=*]"
```

Negation and empty-side examples:

```bash
uv run python -m scripts.bench.cusparse \
  --dry-run \
  --ks 1 \
  --plan-up-down="[][k_hint=1,store=*,fmt=!COO,opA=*,opB=*,orderB=*,orderC=*,algo=*]"
```

Concrete cuSPARSE search command to find the fastest executable plans for `k=1`:

```bash
module load cudatoolkit/12.9

uv run python -m scripts.bench.cusparse \
  --matmul-options baseline \
  --ks 1 \
  --trials 30 \
  --warmup 5 \
  --skip-note \
  --plan-up-down="[k_hint=none,store=*,fmt=*,opA=*,opB=*,orderB=*,orderC=*,algo=*][k_hint=none,store=*,fmt=*,opA=*,opB=*,orderB=*,orderC=*,algo=*]" \
  --plan-up-down="[k_hint=1,store=*,fmt=*,opA=*,opB=*,orderB=*,orderC=*,algo=*][k_hint=1,store=*,fmt=*,opA=*,opB=*,orderB=*,orderC=*,algo=*]"
```
