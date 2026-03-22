# Benchmark scripts

Backend-specific entrypoints:

- `python -m scripts.bench.mkl`
- `python -m scripts.bench.cusparse`
- `python -m scripts.bench.triton`

Each script expands explicit backend plan pairs, runs the requested scenarios,
then prints separate runtime and memory summary tables.

Both entrypoints call the same shared benchmark runner and therefore execute
the same stress/correctness workflow (intra-case + cross-config diagnostics).

Internal layout:

- `scripts/bench/cli.py`: CLI parsing, dtype/index parsing, logging
- `scripts/bench/configs.py`: plan-pair parsing, config expansion, dry-run formatting
- `scripts/bench/cases.py`: generated inputs and scenario construction
- `scripts/bench/run.py`: benchmark execution and runtime diagnostics
- `scripts/bench/report.py`: summary rendering and output-equivalence checks

## Runtime table

Columns:

- `Config`
- `Scenario`
- `Direction`
- `k`
- `Call ms`
- `Err/Trials`
- `Abs Err (avg/max)`
- `Rel Err (avg/max)`
- `Note`

Row semantics:

- one row per timed runtime case
- the runner performs one untimed preflight call first, so timed rows reflect steady-state behavior even when `--warmup=0`
- `Err/Trials = intra_errors/intra_trials` over warmup+timed outputs relative to the preflight output
- directions with no configured plan are omitted entirely
- skip rows show `Call ms=SKIP (...)`

## Memory table

Columns:

- `Config`
- `Scenario`
- `Direction`
- `k`
- `Kind`
- `Host GiB`
- `Device GiB`
- `Note`

Row semantics:

- each executed runtime case expands into four rows:
  - `Total GiB`
  - `Static WS GiB`
  - `Dynamic WS GiB`
  - `Staging GiB`
- `Total GiB` comes from the recorded runtime totals
- the three `* GiB` residency rows report retained workspace/staging bytes
- skip cases emit four memory rows with `Host GiB=SKIP` / `Device GiB=SKIP`

## Output equivalence checks

Benchmark correctness checks use full-output arrays in memory and do not stop the run:

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
- `--instrumentation`
- `--dtype` (`float32` or `float64`)
- `--index-dtype` (`int32` or `int64`)
- `--dry-run`
- `--skip-note`

`--log-level` changes verbosity only. `--instrumentation` is the opt-in switch
for slower observability/profiling behavior:

- MKL: per-level CPU-side wavefront debug logging
- cuSPARSE: NVTX ranges/marks and dynamic scheduler execution
- Triton: NVTX ranges/marks and dynamic scheduler execution

`--plan-up-down` uses two adjacent bracketed plan literals:

```text
[k_hint=none,store=N,fmt=CSR,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT,scratch=none]
[k_hint=none,store=T,fmt=CSC,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT,scratch=none]
```

Pass them as one argument with no delimiter between the two bracket groups:

```bash
--plan-up-down="[...][...]"
```

For cuSPARSE, `*` is allowed for every field except `k_hint`. Negation is also supported for enum-like fields, for example `fmt=!COO` or `fmt=!CSR!CSC`. `scratch` is a concrete execution-policy field and accepts `none`, `all`, or an explicit level list such as `1|2`; wildcard and negation are not supported for `scratch`. Empty sides are allowed: `[][PLAN]` means benchmark DOWN only, `[PLAN][]` means benchmark UP only. The current executor now supports the broader docs-valid dense-side plan space, including `opB=T` and column-major `orderB/orderC`. Unsupported combinations such as `CSC + CSR_ALG3` are rejected directly by `CusparsePlan.supported`.

For Triton, benchmark plans currently expose only `k_hint`, `store`, and `fmt`.
`k_hint` must be either `none` or `1`. Wildcards and negation are supported
for `store` and `fmt`, but not for `k_hint`.
One-sided dry-run examples:

```bash
uv run python -m scripts.bench.triton \
  --dry-run \
  --ks 1 \
  --matmul-options baseline \
  --plan-up-down="[k_hint=1,store=*,fmt=*][]"

uv run python -m scripts.bench.triton \
  --dry-run \
  --ks 1 \
  --matmul-options baseline \
  --plan-up-down="[][k_hint=1,store=*,fmt=*]"
```

## Examples

MKL:

```bash
uv run python -m scripts.bench.mkl \
  --grg /pscratch/sd/q/qys/grg/msprime.example.igd.final.grg \
  --ks 4,16 \
  --warmup 1 --trials 3 \
  --dtype float64 \
  --index-dtype int32 \
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

Triton logged benchmark examples:

```bash
mkdir -p bench_logs/triton/run1

uv run python -m scripts.bench.triton \
  --ks 1 \
  --warmup 3 \
  --trials 10 \
  --dtype float64 \
  --matmul-options baseline \
  --plan-up-down="[k_hint=1,store=*,fmt=*][]" \
  2>&1 | tee bench_logs/triton/run1/fp64_up.log

uv run python -m scripts.bench.triton \
  --ks 1 \
  --warmup 3 \
  --trials 10 \
  --dtype float64 \
  --matmul-options baseline \
  --plan-up-down="[][k_hint=1,store=*,fmt=*]" \
  2>&1 | tee bench_logs/triton/run1/fp64_down.log
```
