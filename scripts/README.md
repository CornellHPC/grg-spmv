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

Each script benchmarks one `.grg_spmv` artifact with one canonical backend plan and prints mean/std call time.

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

`--ring-buffer-size 0` is valid only for fully resident GPU layouts.
