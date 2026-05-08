from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.bolt_lmm_inf.core import (
    STANDARD_CHROMOSOMES,
    STANDARD_GRG_DIR,
    STANDARD_PLINK_DIR,
    BimRecord,
    BoostMt19937,
    BoostNormalDistribution,
    CalibrationResult,
    ChromosomeFiles,
    ChromosomeManifest,
    FamSample,
    StatComparisonThresholds,
    Variant,
    VarianceFit,
    compare_stat_files,
    discover_chromosome_files,
    match_grg_to_bim,
    read_stats_file,
    simulate_cached_phenotype,
    write_grg_stats,
    _require_projected_model_snps,
)


DEFAULT_ARTIFACT_CACHE = Path(os.environ.get("SCRATCH", ".")).expanduser() / "grg" / "pygrgl_spmv_artifacts"
TINY_PHENO_GRG = Path(__file__).resolve().parent / "data" / "test-200-samples.miss.final.grg"


def _has_cusparse_runtime() -> bool:
    try:
        import cupy as cp
        import torch

        cp.cuda.runtime.getDeviceCount()
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _artifact_cache() -> Path:
    return Path(os.environ.get("BOLT_LMM_INF_TEST_ARTIFACT_CACHE", str(DEFAULT_ARTIFACT_CACHE))).expanduser()


class _NumpyCompat:
    @staticmethod
    def asarray(values, dtype=None):
        return np.asarray(values, dtype=dtype)

    @staticmethod
    def asnumpy(values):
        return np.asarray(values)


class _FakeStatsOps:
    cp = _NumpyCompat
    device = contextlib.nullcontext()
    dim = 4

    def __init__(self, manifest: ChromosomeManifest):
        self.manifests = (manifest,)
        self.states = {int(manifest.chrom): SimpleNamespace(local_indices=np.asarray([], dtype=np.int64))}

    def project(self, values):
        return np.asarray(values, dtype=np.float64)

    def scores(self, _chrom: int, _residual):
        return np.zeros(1, dtype=np.float64)


def _variant(chrom: int, idx: int, *, proj_norm2: float = 1.0) -> Variant:
    return Variant(
        global_idx=idx,
        chrom=chrom,
        local_idx=idx,
        bed_row=idx,
        snp_id=f"{chrom}:{1000 + idx}:A:G",
        bp=1000 + idx,
        genetic_pos="0",
        allele1="A",
        allele0="G",
        missing=0,
        mean=1.0,
        mean_center_norm2=1.0,
        proj_norm2=proj_norm2,
        norm_scale=1.0,
        x_norm2=proj_norm2,
        a1freq=0.5,
    )


def _manifest(chrom: int, variants: tuple[Variant, ...]) -> ChromosomeManifest:
    prefix = Path(f"chr{chrom}")
    return ChromosomeManifest(
        chrom=chrom,
        grg_path=prefix.with_suffix(".grg"),
        bed_path=prefix.with_suffix(".bed"),
        bim_path=prefix.with_suffix(".bim"),
        fam_path=prefix.with_suffix(".fam"),
        variants=variants,
    )


def test_boost_normal_distribution_matches_bolt_sequence():
    rng = BoostMt19937(12346)
    randn = BoostNormalDistribution()
    values = [randn(rng) for _ in range(5)]
    assert values == pytest.approx(
        [
            1.3215278348242439,
            -0.05406356468865104,
            0.25961218757166832,
            -1.8261137440309931,
            -0.11777653168380715,
        ],
        rel=0.0,
        abs=1e-15,
    )


def test_grg_pheno_sim_rng_wrapper_is_deterministic_for_uncached_calls(tmp_path):
    if not TINY_PHENO_GRG.exists():
        pytest.skip(f"tiny GRG fixture unavailable: {TINY_PHENO_GRG}")
    samples = tuple(FamSample(str(i), str(i), (str(i), str(i))) for i in range(200))
    outputs = []
    metrics = []
    for suffix in ("a", "b"):
        y, pheno_metrics = simulate_cached_phenotype(
            grg_paths=(TINY_PHENO_GRG,),
            samples=samples,
            pheno_path=tmp_path / f"pheno_{suffix}.tsv",
            meta_path=tmp_path / f"pheno_{suffix}.meta.json",
            log_path=tmp_path / f"pheno_{suffix}.log",
            seed=2026,
            sim_h2=0.3,
            num_causal_per_file=2,
        )
        outputs.append(y)
        metrics.append(pheno_metrics)
    np.testing.assert_array_equal(outputs[0], outputs[1])
    cached, cached_metrics = simulate_cached_phenotype(
        grg_paths=(TINY_PHENO_GRG,),
        samples=samples,
        pheno_path=tmp_path / "pheno_a.tsv",
        meta_path=tmp_path / "pheno_a.meta.json",
        log_path=tmp_path / "pheno_a.log",
        seed=2026,
        sim_h2=0.3,
        num_causal_per_file=2,
    )
    np.testing.assert_array_equal(outputs[0], cached)
    assert cached_metrics.keys() == metrics[0].keys()
    assert cached_metrics["phenotype.cache_hit"] == 1.0
    assert cached_metrics["phenotype.h2"] == metrics[0]["phenotype.h2"]


def test_match_grg_to_bim_keeps_singleton_intersection(monkeypatch):
    class _Mutation:
        def __init__(self, position: int, allele: str, ref_allele: str):
            self.position = position
            self.allele = allele
            self.ref_allele = ref_allele

    class _FakeGrg:
        def __init__(self):
            self.mutations = (
                _Mutation(100, "A", "G"),
                _Mutation(200, "A", "T"),
                _Mutation(300, "C", "CA"),
            )
            self.num_mutations = len(self.mutations)

        def get_mutation_by_id(self, local_idx: int):
            return self.mutations[int(local_idx)]

    monkeypatch.setitem(
        sys.modules,
        "pygrgl",
        SimpleNamespace(load_immutable_grg=lambda _path, load_up_edges=False: _FakeGrg()),
    )
    files = ChromosomeFiles(
        chrom=19,
        grg=Path("chr19.grg"),
        bed=Path("chr19.bed"),
        bim=Path("chr19.bim"),
        fam=Path("chr19.fam"),
    )
    records = (
        BimRecord(19, "match", "0", 100, "A", "G", 0),
        BimRecord(19, "missing", "0", 200, "C", "T", 1),
        BimRecord(19, "repeat-a", "0", 300, "C", "CA", 2),
        BimRecord(19, "repeat-b", "0", 300, "CA", "C", 3),
    )

    manifest = match_grg_to_bim(files, records, bim_repeated_bps={300})

    assert [variant.snp_id for variant in manifest.variants] == ["19:100:A:G"]
    assert [variant.local_idx for variant in manifest.variants] == [0]
    assert [variant.bed_row for variant in manifest.variants] == [0]


def test_write_grg_stats_matches_bolt_identity_and_bad_snp_text(tmp_path):
    variant = Variant(
        global_idx=0,
        chrom=19,
        local_idx=0,
        bed_row=0,
        snp_id="19:123456:A:G",
        bp=123456,
        genetic_pos="0.123456789",
        allele1="A",
        allele0="G",
        missing=0,
        mean=1.0,
        mean_center_norm2=1.0,
        proj_norm2=0.05,
        norm_scale=1.0,
        x_norm2=0.05,
        a1freq=0.5,
    )
    manifest = _manifest(19, (variant,))
    stats_path = tmp_path / "grg.stats"

    write_grg_stats(
        ops=_FakeStatsOps(manifest),
        y=np.asarray([1.0, 2.0, 3.0, 4.0]),
        residuals={19: np.zeros(4, dtype=np.float64)},
        fit=VarianceFit(log_delta=0.0, sigma_g2=1.0, sigma_e2=1.0, h2=0.5, delta=1.0, all_hinv_y=None),
        calibration=CalibrationResult(
            factor=1.0,
            std=0.0,
            ratio_of_medians=1.0,
            median_of_ratios=1.0,
            selected_snps=(),
            tried_snps=0,
            vinv_scale_by_chrom={19: 1.0},
        ),
        path=stats_path,
    )

    row = read_stats_file(stats_path)[0]
    assert row["GENPOS"] == "0.123457"
    assert row["F_MISS"] == "0"
    assert row["CHISQ_LINREG"] == "-1e+09"
    assert row["P_LINREG"] == "1.0E+00"
    assert row["BETA"] == "0"
    assert row["SE"] == "-nan"
    assert row["CHISQ_BOLT_LMM_INF"] == "-1e+09"
    assert row["P_BOLT_LMM_INF"] == "1.0E+00"


def test_require_projected_model_snps_rejects_empty_projected_chromosome():
    chr19 = _manifest(19, (_variant(19, 0), _variant(19, 1)))
    chr20 = _manifest(20, (_variant(20, 0, proj_norm2=0.05),))
    with pytest.raises(ValueError, match="chr20"):
        _require_projected_model_snps((chr19, chr20))

    assert _require_projected_model_snps((chr19,)) == 2

    chr21 = _manifest(21, (_variant(21, 0),))
    with pytest.raises(ValueError, match="at least two eligible projected SNPs"):
        _require_projected_model_snps((chr21,))


@pytest.mark.gpu
@pytest.mark.cusparse
def test_bolt_lmm_inf_32_snp_smoke(tmp_path):
    if not _has_cusparse_runtime():
        pytest.skip("cuSPARSE runtime unavailable (CuPy + CUDA not found)")
    try:
        discover_chromosome_files(STANDARD_GRG_DIR, STANDARD_PLINK_DIR, STANDARD_CHROMOSOMES)
    except FileNotFoundError as exc:
        pytest.skip(f"external 1000 Genomes chr19-22 dataset unavailable: {exc}")

    artifact_cache = _artifact_cache()
    if not artifact_cache.is_dir() or not os.access(artifact_cache, os.R_OK | os.W_OK | os.X_OK):
        pytest.skip(f"BOLT-LMM-inf artifact cache unavailable: {artifact_cache}")

    work_dir = tmp_path / "bolt_lmm_inf_32"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "scripts.bolt_lmm_inf",
        "--workDir",
        str(work_dir),
        "--artifactCache",
        str(artifact_cache),
        "--grgDir",
        str(STANDARD_GRG_DIR),
        "--plinkDir",
        str(STANDARD_PLINK_DIR),
        "--chromosomes",
        ",".join(map(str, STANDARD_CHROMOSOMES)),
        "--snpsPerChrom",
        "32",
        "--seed",
        "12345",
        "--simH2",
        "0.3",
        "--numThreads",
        "1",
        "--device",
        "0",
        "--vramBudgetBytes",
        "0",
        "--ringBufferSize",
        "0",
        "--logLevel",
        "INFO",
        "--covarMaxLevels",
        "10",
    ]
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=1800,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    summary = json.loads(result.stdout)
    assert summary["snpsPerChrom"] == 32
    assert summary["matched_snps"] == 128
    assert summary["model_snps_count"] == 128
    assert Path(summary["covar_file"]).exists()
    assert Path(summary["pheno_file"]).exists()
    assert summary["q_covar_cols"] == [f"PC{i}" for i in range(1, 21)]
    assert summary["covar_cols"] == ["SEX"]
    assert summary["Cindep"] == 22
    assert "LOCO solve" not in summary["timing"]
    assert "grg.cg.loco.solves" not in summary["local"]
    summary_file = Path(summary["summary_json"])
    assert summary_file.exists()
    summary_from_file = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary_from_file["summary_json"] == summary["summary_json"]
    assert summary_from_file["bolt_stats"] == summary["bolt_stats"]
    assert summary_from_file["grg_stats"] == summary["grg_stats"]
    assert summary_from_file["snpsPerChrom"] == summary["snpsPerChrom"]
    assert summary_from_file["matched_snps"] == summary["matched_snps"]
    strict = compare_stat_files(
        Path(summary["bolt_stats"]),
        Path(summary["grg_stats"]),
        thresholds=StatComparisonThresholds(),
    )
    assert strict["passed"] is True
