Parent docs: [Project README](../README.md)

# Benchmark Scripts

The benchmark surface is intentionally minimal and runtime-centric.

- `python -m scripts.bench.mkl`
- `python -m scripts.bench.triton`
- `python -m scripts.bench.cusparse`

Use them as:

- `uv run python -m scripts.bench.mkl`
- `uv run python -m scripts.bench.triton`
- `uv run python -m scripts.bench.cusparse`

Each script benchmarks one `.grg_spmv` artifact with a backend plan and prints mean/std call time.
GPU benchmarks drive `grg.prepare_matmul_cuda(...)` directly; CPU benchmarks still call eager `grg.matmul(...)`.

Common flags:

- `--artifact /path/to/file.grg_spmv`
- `--direction {up,down}`
- `--k <rows>`
- `--trials <count>`
- `--warmup <count>`
- `--dtype {float32,float64}`

GPU-only flags:

- `--device`
- `--stream`
- `--ring-buffer-size`
- `--vram-budget-bytes`
- `--allow-residency` / `--no-allow-residency`

`--ring-buffer-size 0` is valid only for fully resident GPU layouts, so it cannot be combined with `--no-allow-residency`.

cuSPARSE-only flags:

- `--plan PLAN`

`PLAN` may be a named preset or a JSON object with `plan_up` and `plan_down` fields accepted by `CusparsePlanPair.from_dicts(...)`.
The default is `exhaustive-best`, selected from an exhaustive CSR/CSC cuSPARSE sweep with `float64`, `scratch=none`, and residency allowed.
