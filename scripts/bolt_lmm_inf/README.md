# BOLT-LMM-inf GRG Benchmark

`python -m scripts.bolt_lmm_inf` runs a lean BOLT-LMM-inf style workload over chromosome `.grg_spmv` artifacts with the cuSPARSE backend. It is a benchmark and validation driver for GRG sparse matrix-vector operations inside a realistic computational genetics loop.

## Purpose And Non-Goals

The script exercises these operations repeatedly:

```text
X @ a
X.T @ v
X_S @ (X_S.T @ v)
```

`X` is the centered, standardized genotype matrix, with individuals by SNPs. The benchmark estimates infinitesimal variance components, solves leave-one-chromosome-out residuals, calibrates score statistics, and scans calibrated chi-square values.

This is not a full BOLT-LMM replacement. It does not implement covariates, phenotype files, sample or SNP QC, PLINK/BGEN/dosage readers, the non-infinitesimal BOLT mixture model, per-SNP text output, or biological interpretation. The output is a key/value summary TSV.

## Data Path

Raw GRG mode discovers raw chromosome files:

```bash
uv run python -u -m scripts.bolt_lmm_inf \
  --grg-dir /global/cfs/projectdirs/m4341/grg/sim/500k/grg \
  --artifact-cache /path/to/cache \
  --summary-file bolt_summary.tsv
```

`--grg-dir` is the root containing `chr*.grg`. `--artifact-cache` is required in this mode. If the corresponding `.grg_spmv` artifact is missing, the script converts the raw GRG lazily into the cache before running. Existing cached artifacts are validated before use; unsupported artifacts fail fast and are not modified.

Direct artifact mode bypasses discovery and conversion:

```bash
uv run python -u -m scripts.bolt_lmm_inf \
  --artifacts /path/to/chr21.grg_spmv /path/to/chr22.grg_spmv \
  --chromosomes 21,22 \
  --summary-file bolt_summary.tsv
```

`--artifacts` is mutually exclusive with `--grg-dir`. Direct artifacts must be either all `chr<num>`-labeled, or all unlabeled with one explicit `--chromosomes` label per path. The default `--chromosomes all` selects every discovered raw GRG or every labeled direct artifact. Use `--chromosomes 21,22` to narrow a run.

Before execution, artifact metadata is validated across selected chromosomes. All selected artifacts must have the same individual and sample counts, ploidy must be 2, and missing data is rejected.

## Memory Controls

When `--vram-budget-bytes` is omitted, the script probes current free VRAM and subtracts a phenotype-mode-specific BOLT-side reservation before planning. The reservation covers metadata, phenotype, REML, LOCO, calibration/effect-check, and scan arrays. An explicit budget is passed through unchanged; `--vram-budget-bytes 0` asks the planner for full residency without probing. The default `--ring-buffer-size` is `0`, meaning no streaming slots.

## Genotype Math

For each chromosome, allele counts are initialized with an UP traversal from an all-ones sample vector. A SNP is used only when its count is strictly between 0 and the sample count. Reference-monomorphic and alternate-monomorphic SNPs are skipped and reported.

For used SNP `j`, the script applies the centered standardized genotype column:

```text
freq_j = count_j / num_samples
mu_j = ploidy * freq_j
sigma_j = sqrt(ploidy * freq_j * (1 - freq_j))
x_j = (g_j - mu_j) / sigma_j
```

DOWN operations form `X_chr @ weights` by dividing weights by `sigma` before traversal and subtracting the mean term. UP operations form `X_chr.T @ center(v)` and scale by `1 / sigma`. Whole-genome or LOCO GRM products use:

```text
K_S v = center(X_S @ (X_S.T @ center(v))) / |S|
```

where `S` is either all used SNPs or all used SNPs outside one chromosome.

## Phenotype Simulation

Null mode draws standard normal noise on the CUDA device, centers it, and rescales it to empirical variance 1:

```bash
uv run python -u -m scripts.bolt_lmm_inf \
  --grg-dir /path/to/grg \
  --artifact-cache /path/to/cache \
  --phenotype-mode null \
  --summary-file bolt_null_summary.tsv
```

Null metrics include `phenotype.mode = null`, `phenotype.true_h2 = 0`, and `phenotype.var` near 1.

Infinitesimal mode draws one Gaussian effect per used SNP with base scale `1 / sqrt(M)`, applies `X beta` to form `g`, and draws noise `e`. The noise is centered and orthogonalized against `g`. The two components are then scaled as:

```text
g <- g * sqrt(sim_h2 / var(g))
e <- e * sqrt((1 - sim_h2) / var(e))
y = center(g + e)
```

Because `g` and `e` are orthogonal after construction, the requested, true, and empirical heritability match up to floating-point error, and `phenotype.genetic_var + phenotype.noise_var` matches `phenotype.var`.

Use `--n-effect-check N` in infinitesimal mode to sample true effects before simulation. The summary reports aggregate recovery diagnostics after LOCO residuals are solved. `--n-effect-check 0` disables those fields.

## REML

The REML step estimates variance components for the infinitesimal model. It builds Monte Carlo genetic and environmental components, then uses CG solves inside a small secant search over:

```text
delta = sigma_e2 / sigma_g2
H(delta) = K + delta I
h2 = 1 / (1 + delta)
```

The search is constrained by `_REML_MIN_H2 = 1e-8` and `_REML_MAX_H2 = 0.99`. The final summary reports:

```text
reml.sigma_g2
reml.sigma_e2
reml.h2
reml.delta
reml.mc_trials
```

`sigma_g2` and `sigma_e2` must be finite and positive. REML `h2` is a model estimate; in null runs it should be small but is not expected to be exactly zero because of the lower bound.

## LOCO Residuals

For each selected chromosome `c`, the script excludes that chromosome from the random effect and solves:

```text
V_c r_c = y
V_c = sigma_g2 K_-c + sigma_e2 I
```

The residual vector `r_c = V_c^-1 y` is used for every SNP on chromosome `c`. This is the LOCO path used by both calibration and scan.

## Calibration And Scan

Calibration samples `min(--n-calib, num_mutations.used)` used SNPs. For a calibration SNP on chromosome `c`, the script solves:

```text
V_c q = x_j
score_j = x_j.T r_c
prospective_j = score_j^2 / (x_j.T q)
```

It then estimates the BOLT-LMM-inf scalar:

```text
c_inf = sum(score_j^2) / sum(prospective_j)
```

The genome scan computes all chromosome scores, skips monomorphic SNPs, and reports:

```text
chi2_j = score_j^2 / c_inf
p_j = Pr[ChiSquare(df=1) >= chi2_j]
```

The summary includes per-chromosome and genome chi-square totals, a 20-bin p-value histogram, an approximate histogram KS distance, approximate lambda GC from the median p-value, and optional top-hit fields. `--scan-top-k 0` disables top-hit fields.

## Validation

For null runs, inspect:

```text
phenotype.true_h2
phenotype.var
reml.h2
scan.p_hist.*
scan.p_hist.ks_approx
scan.lambda_gc_approx
```

Expected null behavior is `phenotype.true_h2 = 0`, empirical phenotype variance near 1, a small but nonzero REML `h2`, a roughly flat p-value histogram, and lambda GC close to 1. Small runs can be noisy; strict stochastic uniformity checks require larger sample sizes.

For infinitesimal runs, inspect:

```text
phenotype.requested_h2
phenotype.true_h2
phenotype.empirical_h2
phenotype.genetic_var
phenotype.noise_var
phenotype.genetic_noise_dot
reml.h2
effect_check.*
```

The requested, true, and empirical `h2` should match by construction. Genetic and noise variances should add to phenotype variance, and the genetic/noise dot product should be near zero. `effect_check.beta_slope` near 1 is aggregate evidence that sampled true effects are recoverable, while individual SNP estimates can be much noisier than the true effects.

## Summary Metric Reference

General run metadata:

- `backend`: backend name, currently `cusparse`.
- `chromosomes`: comma-separated selected chromosome labels.
- `num_individuals`, `num_samples`: shared artifact sample dimensions.
- `artifact_count`: selected artifact count.
- `seed`: base seed split into phenotype, analysis/calibration, and validation RNGs.
- `cg_tol`, `cg_max_iter`: configured CG controls.
- `n_effect_check`, `scan_top_k`: configured diagnostic and scan output sizes.

Mutation counts:

- `num_mutations.raw`, `num_mutations.used`, `num_mutations.monomorphic_skipped`: genome totals.
- `chr<label>.num_mutations.raw`, `.used`, `.monomorphic_skipped`, `.monomorphic_ref`, `.monomorphic_alt`: per-chromosome counts.

Count contract: `used + monomorphic_skipped = raw`, and per-chromosome totals sum to genome totals.

Phenotype metrics:

- Null mode emits `phenotype.mode`, `phenotype.var`, and `phenotype.true_h2`.
- Infinitesimal mode also emits `phenotype.requested_h2`, `phenotype.empirical_h2`, `phenotype.genetic_var`, `phenotype.noise_var`, and `phenotype.genetic_noise_dot`.

Null contract: `phenotype.true_h2 = 0` and `phenotype.var` is approximately 1.

Infinitesimal contract: requested, true, and empirical `h2` match by construction; genetic and noise variances sum to phenotype variance; `phenotype.genetic_noise_dot` is near zero.

REML and calibration:

- `reml.sigma_g2`, `reml.sigma_e2`, `reml.h2`, `reml.delta`, `reml.mc_trials`.
- `calibration.c_inf`, `calibration.num_snps_requested`, `calibration.num_snps_effective`.

REML contract: variance components are finite and positive, and `reml.h2 = 1 / (1 + reml.delta)` within `_REML_MIN_H2` and `_REML_MAX_H2`.

Calibration contract: `calibration.c_inf > 0`, and `calibration.num_snps_effective = min(--n-calib, num_mutations.used)`.

CG metrics:

- `cg.reml.solves`, `.iterations`, `.max_iterations`, `.max_rel_resid`.
- `cg.loco.solves`, `.iterations`, `.max_iterations`, `.max_rel_resid`.
- `cg.calibration.solves`, `.iterations`, `.max_iterations`, `.max_rel_resid`.
- `cg.effect_check.solves`, `.iterations`, `.max_iterations`, `.max_rel_resid`.

CG contract: solve counts and iteration counts are nonnegative. Successful max residuals should be at or below `--cg-tol` up to floating-point margin.

Timing and layout metrics:

- `setup.<stage>.seconds`: setup timings, including artifact resolution/conversion, optional free-memory probe, layout planning, runtime initialization, frequency setup, and phenotype simulation.
- `setup.total_elapsed.seconds`: full wall-clock elapsed time.
- `layout.bytes.total`, `layout.bytes.<category>`: planned device memory by category.
- `matmul.up.k1.calls`, `.total_ms`, `.avg_ms`: timed UP traversals.
- `matmul.down.k1.calls`, `.total_ms`, `.avg_ms`: timed DOWN traversals.

Scan metrics:

- `scan.chr<label>.num_snps`, `.sum_chi2`, `.mean_chi2`, `.max_chi2`, `.min_p`.
- `scan.genome.num_snps`, `.sum_chi2`, `.mean_chi2`, `.max_chi2`, `.min_p`.
- `scan.p_hist.bin_count`, `scan.p_hist.bin<idx>.count`, `scan.p_hist.ks_approx`.
- `scan.lambda_gc_approx`.
- `scan.top<rank>.chr`, `.local_idx`, `.chi2`, `.p` when `--scan-top-k > 0`.

Scan contract: histogram bin counts sum to `scan.genome.num_snps`. Null p-values should be roughly uniform, and `scan.lambda_gc_approx` should be close to 1 under null. Top hits are sorted by descending chi-square, and top-hit p-values correspond to their chi-square values.

Effect-check metrics:

- `effect_check.count`.
- `effect_check.beta_corr`, `effect_check.beta_slope`, `effect_check.beta_rmse`, `effect_check.beta_mae`, `effect_check.sign_concordance`.

Effect-check fields are emitted only for infinitesimal mode when sampled effects exist. A slope near 1 is an aggregate diagnostic; individual effects can be noisy.
