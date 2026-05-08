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

`--vram-budget-bytes` defaults to `0`, which means the planner uses the required full-residency budget rather than probing GPU memory.
`--ring-buffer-size 0` means no streaming slots; it is valid only for fully resident GPU layouts, so it cannot be combined with `--no-allow-residency`.

cuSPARSE-only flags:

- `--plan PLAN`

`PLAN` may be a named preset or a JSON object with `plan_up` and `plan_down` fields accepted by `CusparsePlanPair.from_dicts(...)`.
The default is `exhaustive-best`, selected from an exhaustive CSR/CSC cuSPARSE sweep with `float64`, `scratch=none`, and residency allowed.

## BOLT-LMM-inf Reference Comparison

`python -m scripts.bolt_lmm_inf` is now a reference-comparison driver for the GRG BOLT-LMM-inf prototype. It prepares matched GRG/PLINK inputs, builds or preflights official BOLT-LMM v2.5, simulates a phenotype, runs official `--lmmInfOnly`, runs the local GRG implementation, and writes per-SNP comparison output.

The GRG path is intentionally BOLT-parity oriented: solver control flow and stats formatting are kept close to official BOLT-LMM rather than adding local numerical guards.

The recommended 1000 Genomes smoke uses chr19-22 GRG/PLINK files, with 32 sampled singleton GRG/BIM matches per chromosome:

```bash
uv run python -u -m scripts.bolt_lmm_inf \
  --workDir "$SCRATCH/bolt_lmm_inf_1000genome_chr19_22" \
  --artifactCache "$SCRATCH/grg/pygrgl_spmv_artifacts" \
  --grgDir /global/cfs/projectdirs/m4341/grg/1000genome \
  --plinkDir /global/cfs/projectdirs/m4341/grg/1000genome/plink \
  --chromosomes 19,20,21,22 \
  --snpsPerChrom 32 \
  --seed 12345 \
  --simH2 0.3 \
  --numThreads "$(nproc)" \
  --device 0 \
  --vramBudgetBytes 0 \
  --ringBufferSize 0 \
  --logLevel INFO \
  --covarMaxLevels 10
```

The old key/value summary benchmark contract has been removed. `--grgDir`, `--plinkDir`, and `--chromosomes` select the source data, `--snpsPerChrom` controls random singleton-match sampling per chromosome (`0` means all singleton matches), and `--simH2` controls the simulated phenotype heritability. Without `--covarFile`, the driver caches `PC1`-`PC20` and `SEX` under `inputs/covariates.tsv` and forwards those covariates to both official BOLT and the GRG path. Outputs are written under `--workDir`, including filtered PLINK inputs, `inputs/run_manifest.tsv`, `inputs/covariates.tsv`, `inputs/pheno.tsv`, `bolt.log`, `bolt.stats`, `grg.stats`, and `summary.json`; the script prints one JSON summary to stdout and reports comparison errors without threshold-gating CLI exit status. See [bolt_lmm_inf/README.md](bolt_lmm_inf/README.md) for details.

## PCA Benchmark

`python -m scripts.pca_bench` benchmarks PCA eigensolvers on one `.grg_spmv` artifact with the cuSPARSE backend and the default `exhaustive-best` plan.

Full benchmark:

```bash
uv run python -u -m scripts.pca_bench --artifact "$SCRATCH/grg/pygrgl_spmv_artifacts/example.grg_spmv" --pcs 20
```

The randomized Rayleigh-Ritz solver defaults to `--rr-oversample 40 --rr-power-iters 10`. LOBPCG methods are reported as `status=skipped` when local SciPy/CuPy would switch to their dense fallback (`num_mutations < 5 * pcs`).
