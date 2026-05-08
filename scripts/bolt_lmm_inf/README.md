# BOLT-LMM-inf Reference Comparison

`uv run python -m scripts.bolt_lmm_inf` compares the GRG/cuSPARSE BOLT-LMM-inf prototype against the official BOLT-LMM v2.5 `--lmmInfOnly` path.

The driver is intentionally narrow. It prepares matched GRG/PLINK inputs, simulates one phenotype, runs official BOLT, runs the local GRG implementation, and compares per-SNP association output.

The local path intentionally follows official BOLT-LMM v2.5 control flow over defensive numerical behavior. In particular, its conjugate-gradient solve mirrors `Bolt::conjGradSolve`: all right-hand sides advance as one batch, denominator pathologies are not masked, and reaching the iteration limit returns the last iterate rather than raising.

## Project-Standard Dataset

The recommended 1000 Genomes smoke uses chr19-22 data:

```text
GRG:   /global/cfs/projectdirs/m4341/grg/1000genome
PLINK: /global/cfs/projectdirs/m4341/grg/1000genome/plink
CHR:   19,20,21,22
SNPs:  32 sampled BIM rows per chromosome
```

Use `--grgDir`, `--plinkDir`, and `--chromosomes` to point the harness at another matched GRG/PLINK dataset. Both nested `plink/chr19/chr19.merged.bim` and flat `*.chr19.*.bim` layouts are supported.

## Quick Start

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

This recommended smoke samples 32 random singleton GRG/BIM matches per chromosome from chr19-22, for 128 total SNPs. Use `--snpsPerChrom 0` to use the full singleton GRG/BIM intersection on the selected chromosomes. `--simH2` controls the target heritability of the simulated phenotype.

If `--covarFile` is not supplied, the driver writes cached `inputs/covariates.tsv` containing `PC1`-`PC20` and `SEX`, then passes `--qCovarCol PC1` ... `PC20` and `--covarCol SEX` to both official BOLT and the local GRG path. A user covariate file can be supplied with repeated `--covarCol` and `--qCovarCol`; every FAM sample must be present exactly once with no `-9` or `NA` values in requested columns.

All matched singleton SNPs that survive BOLT covariate projection are used as model/GRM SNPs; there is no separate model-SNP subset.

The script writes all generated files under `--workDir`, including:

- `inputs/run_manifest.tsv`
- `inputs/covariates.tsv`
- `inputs/covariates.meta.json`
- `inputs/pheno.tsv`
- `inputs/pheno.meta.json`
- `inputs/pheno_sim.log`
- `inputs/chr*.sample.bim` or `inputs/chr*.full.bim`
- `inputs/chr*.sample.bed` or `inputs/chr*.full.bed`
- `bolt.log`
- `bolt.stats`
- `grg.stats`
- `summary.json`

It prints one compact JSON summary to stdout and writes the same object to `summary.json`. Timing and a `pprint` comparison report go to stderr. The comparison report contains numeric error summaries; the CLI exits nonzero for operational/runtime failures, not for threshold-sized numeric deltas.

Both sampled and full mode materialize filtered BED/BIM files under `inputs/`. Cached covariates and phenotypes are keyed by seed, selected variants, FAM identity, input file fingerprints, and relevant simulator settings.

The current integration smoke uses `--snpsPerChrom 32` and `--simH2 0.3`; that gives 128 matched/model SNPs across chr19-22.

## BOLT Build

The checked-in `bolt/BOLT-LMM_v2.5/bolt` binary is not assumed usable. The driver caches the official BOLT build under the resolved artifact cache, at `bolt_lmm_cache/BOLT-LMM_v2.5/src/bolt`.

Build or preflight it directly with:

```bash
uv run python -m scripts.bolt_lmm_inf.build_bolt \
  --cacheDir "$SCRATCH/grg/pygrgl_spmv_artifacts/bolt_lmm_cache" \
  --jobs "$(nproc)" \
  --log-level INFO
```

The cached build downloads the official BOLT-LMM v2.5 tarball if needed, patches only cached `src/NonlinearOptMulti.cpp` with the `--lmmInfOnly`-safe NLopt stub, builds `src/bolt`, and preflights it with `bolt -h`.

That cached patch replaces the NLopt-dependent REML-AI implementation with a stub that throws if called. The `--lmmInfOnly` reference path does not use that code, so this keeps the official source buildable without local NLopt while still failing loudly if the wrong path is reached.

## Matching Rules

For each chromosome, the driver:

1. loads GRG mutation metadata;
2. counts mutation positions;
3. discards all GRG mutations whose position occurs more than once;
4. matches BIM rows by `(bp, allele1, allele0)` to GRG `(position, allele, ref_allele)`;
5. rewrites BIM SNP IDs to `chr:bp:allele1:allele0` to avoid official BOLT duplicate-ID masking;
6. decodes PLINK BED rows to get exact allele1 dosage sums and squared sums;
7. builds the BOLT covariate basis;
8. recomputes raw projected SNP norms and BOLT `Xnorm2` after covariate projection.

The driver only analyzes singleton rows. A base-pair position (`BP`) is only the physical chromosome coordinate, not a complete variant identity. If more than one BIM row or more than one GRG mutation uses the same `BP`, every row at that position is filtered out. This includes different alleles at the same coordinate, indel/symbolic representations, and exact duplicate `(BP, allele, ref)` keys. Sampled mode keeps scanning until it finds the requested number of singleton matches. Full mode materializes all matched singleton BIM/GRG intersection rows after repeated-BP filtering and skips unmatched BIM singletons.

The standard 1000 Genomes chr19-22 PLINK hard calls currently scan with no missing calls, and every PLINK singleton key is present in GRG. The GRG-only singleton keys at chr19:21078649, chr21:34957154, and chr22:39187252 fall at repeated BIM positions, so they are intentionally excluded by the BIM singleton filter.

## Statistics

The local output uses official BOLT's verbose column order:

```text
SNP CHR BP GENPOS ALLELE1 ALLELE0 A1FREQ F_MISS CHISQ_LINREG P_LINREG BETA SE CHISQ_BOLT_LMM_INF P_BOLT_LMM_INF
```

`GENPOS`, `F_MISS`, and statistic columns are serialized to match BOLT's `getSnpStats()` row formatting. SNPs killed by covariate projection are still written, using BOLT's bad-SNP chi-square sentinel and p-value `1`.

The CLI comparison reports identity alignment and numeric error summaries for `A1FREQ`, `CHISQ_LINREG`, `BETA`, `SE`, `CHISQ_BOLT_LMM_INF`, and the p-value columns. Threshold pass/fail is enforced by tests or external wrappers, not by the CLI.

## Scope

This is a reference-comparison tool, not a general BOLT replacement. It does not implement sample QC, BGEN/dosage input, the non-infinitesimal mixture model, or full REML-AI.
