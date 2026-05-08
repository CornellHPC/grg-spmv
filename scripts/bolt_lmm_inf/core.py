"""Reference-comparison driver for the GRG BOLT-LMM-inf prototype."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Iterable, Sequence
import contextlib
from dataclasses import dataclass, replace
import hashlib
from importlib import metadata as importlib_metadata
import json
import logging
import math
import os
from pathlib import Path
import pprint
import re
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np

from pygrgl_spmv.grg import RuntimeRequirements, convert
from pygrgl_spmv.grg.artifact import artifact_path_for_grg, scan_grg_spmv
from scripts.bench.cusparse import DEFAULT_CUSPARSE_PLAN_NAME, parse_cusparse_plan


DTYPE = np.dtype(np.float64)
STANDARD_GRG_DIR = Path("/global/cfs/projectdirs/m4341/grg/1000genome")
STANDARD_PLINK_DIR = Path("/global/cfs/projectdirs/m4341/grg/1000genome/plink")
STANDARD_CHROMOSOMES = (19, 20, 21, 22)
STANDARD_SNPS_PER_CHROM = 32
DEFAULT_NUM_CALIB_SNPS = 30
DEFAULT_H2_EST_MC_TRIALS = 3
DEFAULT_CG_TOL = 5e-4
DEFAULT_MAX_ITERS = 10_000
BOLT_RANDOM_SEED = 12345
BOLT_BAD_SNP_STAT = -1e9
COVARIATE_CACHE_SCHEMA_VERSION = 1
PHENOTYPE_CACHE_SCHEMA_VERSION = 2
BOLT_LMM_INF_CODE_VERSION = "lean-covariate-aware-v1"
GENERATED_Q_COVARS = tuple(f"PC{i}" for i in range(1, 21))
GENERATED_COVARS = ("SEX",)
BOOST_NORMAL_HEADER = Path("/usr/include/boost/random/normal_distribution.hpp")
BOOST_EXPONENTIAL_HEADER = Path("/usr/include/boost/random/exponential_distribution.hpp")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BimRecord:
    chrom: int
    snp_id: str
    genetic_pos: str
    bp: int
    allele1: str
    allele0: str
    row: int

    @property
    def unique_id(self) -> str:
        return f"{self.chrom}:{self.bp}:{self.allele1}:{self.allele0}"

    @property
    def key(self) -> tuple[int, str, str]:
        return (int(self.bp), self.allele1, self.allele0)


@dataclass(frozen=True)
class FamSample:
    fid: str
    iid: str
    raw_fields: tuple[str, ...]


@dataclass(frozen=True)
class Variant:
    global_idx: int
    chrom: int
    local_idx: int
    bed_row: int
    snp_id: str
    bp: int
    genetic_pos: str
    allele1: str
    allele0: str
    missing: int = 0
    mean: float = float("nan")
    mean_center_norm2: float = float("nan")
    proj_norm2: float = float("nan")
    norm_scale: float = float("nan")
    x_norm2: float = float("nan")
    a1freq: float = float("nan")

    @property
    def plink_line(self) -> str:
        return f"{self.chrom}\t{self.snp_id}\t{self.genetic_pos}\t{self.bp}\t{self.allele1}\t{self.allele0}"


@dataclass(frozen=True)
class ChromosomeFiles:
    chrom: int
    grg: Path
    bed: Path
    bim: Path
    fam: Path


@dataclass(frozen=True)
class ChromosomeManifest:
    chrom: int
    grg_path: Path
    bed_path: Path
    bim_path: Path
    fam_path: Path
    variants: tuple[Variant, ...]

    @property
    def snp_ids(self) -> tuple[str, ...]:
        return tuple(variant.snp_id for variant in self.variants)

    @property
    def local_indices(self) -> np.ndarray:
        return np.asarray([variant.local_idx for variant in self.variants], dtype=np.int64)

    @property
    def model_count(self) -> int:
        return sum(1 for variant in self.variants if is_model_variant(variant))


@dataclass(frozen=True)
class BedDosageStats:
    sum_g: np.ndarray
    sumsq_g: np.ndarray
    missing: np.ndarray


@dataclass
class CgStats:
    solves: int = 0
    iterations: int = 0
    max_iterations: int = 0
    max_rel_resid: float = 0.0

    def add(self, iterations: int, rel_resid: float) -> None:
        iters = int(iterations)
        self.solves += 1
        self.iterations += iters
        self.max_iterations = max(self.max_iterations, iters)
        self.max_rel_resid = max(self.max_rel_resid, float(rel_resid))


@dataclass(frozen=True)
class StatComparisonThresholds:
    freq_atol: float = 5e-7
    beta_rtol: float = 5e-4
    beta_atol: float = 5e-6
    se_rtol: float = 5e-4
    se_atol: float = 5e-6
    chisq_rtol: float = 5e-4
    chisq_atol: float = 5e-6


@dataclass(frozen=True)
class VarianceFit:
    log_delta: float
    sigma_g2: float
    sigma_e2: float
    h2: float
    delta: float
    all_hinv_y: Any


@dataclass(frozen=True)
class McScalingResult:
    log_delta: float
    f_jacks: tuple[float, ...]
    f_rands_as_data: tuple[float, ...]
    sigma2_k: float
    all_hinv_y: Any

    @property
    def f_reml(self) -> float:
        return float(self.f_jacks[-1])


@dataclass(frozen=True)
class CalibrationResult:
    factor: float
    std: float
    ratio_of_medians: float
    median_of_ratios: float
    selected_snps: tuple[str, ...]
    tried_snps: int
    vinv_scale_by_chrom: dict[int, float]


@dataclass
class ChromOpsState:
    manifest: ChromosomeManifest
    grg: Any
    up_op: Any
    down_op: Any
    views: dict[str, Any]
    mean: Any
    scale: Any
    mean_scale: Any
    local_indices: Any


def _as_float(value: Any) -> float:
    if hasattr(value, "get"):
        return float(value.get())
    return float(value)


def _array_module(value):
    if type(value).__module__.split(".", 1)[0] == "cupy":
        import cupy as cp

        return cp
    return np


def _dot(left, right) -> float:
    xp = _array_module(left)
    return _as_float(xp.sum(left * right))


# Python equivalent of BOLT::initMarker()'s final projMaskSnps eligibility.
def is_model_variant(variant: Variant) -> bool:
    return (
        float(variant.mean_center_norm2) > 0.0
        and float(variant.proj_norm2) >= 0.1
        and float(variant.norm_scale) > 0.0
        and float(variant.x_norm2) > 0.0
    )


@dataclass(frozen=True)
class CovariateBasis:
    """BOLT-style orthonormal covariate basis, including the all-ones vector."""

    basis: np.ndarray
    covar_cols: tuple[str, ...]
    q_covar_cols: tuple[str, ...]
    covar_max_levels: int
    source_path: Path | None = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.basis, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("covariate basis must be two-dimensional")
        if arr.shape[0] < 1:
            raise ValueError("covariate basis must include at least one sample")
        object.__setattr__(self, "basis", np.ascontiguousarray(arr))

    @property
    def nused(self) -> int:
        return int(self.basis.shape[0])

    @property
    def cindep(self) -> int:
        return int(self.basis.shape[1])

    @property
    def dim(self) -> int:
        return int(self.nused - self.cindep)

    @classmethod
    def intercept_only(cls, n: int) -> "CovariateBasis":
        n_int = int(n)
        if n_int < 1:
            raise ValueError("sample count must be positive")
        basis = np.full((n_int, 1), 1.0 / math.sqrt(float(n_int)), dtype=np.float64)
        return cls(basis=basis, covar_cols=(), q_covar_cols=(), covar_max_levels=10)

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        *,
        covar_cols: Sequence[str],
        q_covar_cols: Sequence[str],
        covar_max_levels: int,
        source_path: Path | None,
    ) -> "CovariateBasis":
        covars = np.asarray(matrix, dtype=np.float64)
        if covars.ndim != 2:
            raise ValueError("covariate matrix must be two-dimensional")
        if covars.shape[0] < 1 or covars.shape[1] < 1:
            raise ValueError("covariate matrix must be non-empty")
        if covars.shape[1] > covars.shape[0]:
            raise ValueError("number of covariate columns cannot exceed sample count")
        u, s, _vt = np.linalg.svd(np.asfortranarray(covars), full_matrices=False)
        if s.size == 0 or s[0] <= 0.0:
            raise ValueError("covariate matrix is rank-deficient with no independent columns")
        rank = int(np.count_nonzero(s >= (s[0] * 1e-8)))
        return cls(
            basis=u[:, :rank],
            covar_cols=tuple(str(value) for value in covar_cols),
            q_covar_cols=tuple(str(value) for value in q_covar_cols),
            covar_max_levels=int(covar_max_levels),
            source_path=None if source_path is None else Path(source_path),
        )

    def project_host(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        was_vector = arr.ndim == 1
        mat = arr.reshape((self.nused, 1)) if was_vector else arr
        if mat.shape[0] != self.nused:
            raise ValueError(f"vector has {mat.shape[0]} rows; expected {self.nused}")
        projected = mat - self.basis @ (self.basis.T @ mat)
        return projected[:, 0] if was_vector else projected

    def project_host_inplace(self, values: np.ndarray) -> np.ndarray:
        values[...] = self.project_host(values)
        return values

    def project_device(self, values):
        xp = _array_module(values)
        if xp is np:
            return self.project_host(values)
        q = xp.asarray(self.basis, dtype=DTYPE)
        arr = xp.asarray(values, dtype=DTYPE)
        was_vector = arr.ndim == 1
        mat = arr.reshape((self.nused, 1)) if was_vector else arr
        projected = mat - q @ (q.T @ mat)
        return projected[:, 0] if was_vector else projected

    def project_device_inplace(self, values):
        values[...] = self.project_device(values)
        return values


class BoostMt19937:
    """Small Boost.Random mt19937 clone for BOLT's calibration SNP sampler."""

    _n = 624
    _m = 397
    _r = 31
    _a = 0x9908B0DF
    _u = 11
    _d = 0xFFFFFFFF
    _s = 7
    _b = 0x9D2C5680
    _t = 15
    _c = 0xEFC60000
    _l = 18
    _f = 1812433253
    _mask = 0xFFFFFFFF
    _upper_mask = 0x80000000
    _lower_mask = 0x7FFFFFFF

    def __init__(self, seed: int):
        self.x = [0] * self._n
        self.i = self._n
        self.seed(seed)

    def seed(self, value: int) -> None:
        self.x[0] = int(value) & self._mask
        for idx in range(1, self._n):
            prev = self.x[idx - 1]
            self.x[idx] = (self._f * (prev ^ (prev >> 30)) + idx) & self._mask
        self.i = self._n
        self._normalize_state()

    def _normalize_state(self) -> None:
        y0 = self.x[self._m - 1] ^ self.x[self._n - 1]
        if y0 & (1 << 31):
            y0 = ((y0 ^ self._a) << 1) | 1
        else:
            y0 <<= 1
        self.x[0] = (self.x[0] & self._upper_mask) | (y0 & self._lower_mask)
        if not any(self.x):
            self.x[0] = 1 << 31

    def _twist(self) -> None:
        for idx in range(0, self._n - self._m):
            y = (self.x[idx] & self._upper_mask) | (self.x[idx + 1] & self._lower_mask)
            self.x[idx] = (self.x[idx + self._m] ^ (y >> 1) ^ ((self.x[idx + 1] & 1) * self._a)) & self._mask
        for idx in range(self._n - self._m, self._n - 1):
            y = (self.x[idx] & self._upper_mask) | (self.x[idx + 1] & self._lower_mask)
            self.x[idx] = (self.x[idx - (self._n - self._m)] ^ (y >> 1) ^ ((self.x[idx + 1] & 1) * self._a)) & self._mask
        y = (self.x[self._n - 1] & self._upper_mask) | (self.x[0] & self._lower_mask)
        self.x[self._n - 1] = (self.x[self._m - 1] ^ (y >> 1) ^ ((self.x[0] & 1) * self._a)) & self._mask
        self.i = 0

    def __call__(self) -> int:
        if self.i == self._n:
            self._twist()
        z = self.x[self.i]
        self.i += 1
        z ^= (z >> self._u) & self._d
        z ^= (z << self._s) & self._b
        z ^= (z << self._t) & self._c
        z ^= z >> self._l
        return z & self._mask


def boost_uniform_int_0_2pow30(rng: BoostMt19937) -> int:
    """Match boost::uniform_int<>(0, 1<<30) for a 32-bit mt19937 engine."""
    max_value = 1 << 30
    bucket_size = 3
    while True:
        result = rng() // bucket_size
        if result <= max_value:
            return int(result)


_BOOST_TABLES: dict[tuple[Path, str], tuple[float, ...]] = {}


def _boost_table(header: Path, table_name: str) -> tuple[float, ...]:
    key = (Path(header), str(table_name))
    cached = _BOOST_TABLES.get(key)
    if cached is not None:
        return cached
    text = Path(header).read_text(encoding="utf-8")
    match = re.search(rf"{re.escape(table_name)}\[\d+\]\s*=\s*\{{(?P<body>.*?)\}};", text, flags=re.S)
    if match is None:
        raise RuntimeError(f"could not find Boost.Random {table_name} in {header}")
    values = tuple(float(item.strip()) for item in match.group("body").replace("\n", " ").split(",") if item.strip())
    _BOOST_TABLES[key] = values
    return values


def boost_uniform_01(rng: BoostMt19937) -> float:
    """Match boost::random::uniform_01<double> for boost::mt19937."""
    return float(rng()) / 4294967296.0


def boost_generate_int_float_pair_8(rng: BoostMt19937) -> tuple[float, int]:
    """Match boost::random::detail::generate_int_float_pair<double, 8>."""
    first = int(rng())
    bucket = first & 0xFF
    r = float(first >> 8) / 16777216.0
    second = int(rng())
    r += float(second & ((1 << 29) - 1))
    r /= float(1 << 29)
    return r, bucket


class BoostExponentialDistribution:
    """Boost.Random exponential_distribution<> clone for the normal tail path."""

    def __init__(self, lambda_arg: float = 1.0):
        self.lambda_arg = float(lambda_arg)
        if self.lambda_arg <= 0.0:
            raise ValueError("lambda_arg must be positive")
        self._table_x = _boost_table(BOOST_EXPONENTIAL_HEADER, "table_x")
        self._table_y = _boost_table(BOOST_EXPONENTIAL_HEADER, "table_y")

    def __call__(self, rng: BoostMt19937) -> float:
        table_x = self._table_x
        table_y = self._table_y
        shift = 0.0
        while True:
            r, i = boost_generate_int_float_pair_8(rng)
            x = r * table_x[i]
            if x < table_x[i + 1]:
                return (shift + x) / self.lambda_arg
            if i == 0:
                shift += table_x[1]
                continue
            y01 = boost_uniform_01(rng)
            y = table_y[i] + y01 * (table_y[i + 1] - table_y[i])
            y_above_ubound = (table_x[i] - table_x[i + 1]) * y01 - (table_x[i] - x)
            y_above_lbound = y - (table_y[i + 1] + (table_x[i + 1] - x) * table_y[i + 1])
            if y_above_ubound < 0.0 and (y_above_lbound < 0.0 or y < math.exp(-x)):
                return (x + shift) / self.lambda_arg


class BoostNormalDistribution:
    """Boost.Random normal_distribution<> clone used by BOLT's MC scaling."""

    def __init__(self, mean: float = 0.0, sigma: float = 1.0):
        self.mean = float(mean)
        self.sigma = float(sigma)
        if self.sigma < 0.0:
            raise ValueError("sigma must be nonnegative")
        self._table_x = _boost_table(BOOST_NORMAL_HEADER, "table_x")
        self._table_y = _boost_table(BOOST_NORMAL_HEADER, "table_y")

    def __call__(self, rng: BoostMt19937) -> float:
        unit = self._unit(rng)
        return unit * self.sigma + self.mean

    def _unit(self, rng: BoostMt19937) -> float:
        table_x = self._table_x
        table_y = self._table_y
        while True:
            r, bucket = boost_generate_int_float_pair_8(rng)
            sign = (bucket & 1) * 2 - 1
            i = bucket >> 1
            x = r * table_x[i]
            if x < table_x[i + 1]:
                return x * sign
            if i == 0:
                return self._tail(rng) * sign

            y01 = boost_uniform_01(rng)
            y = table_y[i] + y01 * (table_y[i + 1] - table_y[i])
            if table_x[i] >= 1.0:
                y_above_ubound = (table_x[i] - table_x[i + 1]) * y01 - (table_x[i] - x)
                y_above_lbound = y - (table_y[i] + (table_x[i] - x) * table_y[i] * table_x[i])
            else:
                y_above_lbound = (table_x[i] - table_x[i + 1]) * y01 - (table_x[i] - x)
                y_above_ubound = y - (table_y[i] + (table_x[i] - x) * table_y[i] * table_x[i])
            if y_above_ubound < 0.0 and (y_above_lbound < 0.0 or y < math.exp(-(x * x / 2.0))):
                return x * sign

    def _tail(self, rng: BoostMt19937) -> float:
        tail_start = self._table_x[1]
        exp_x = BoostExponentialDistribution(tail_start)
        exp_y = BoostExponentialDistribution()
        while True:
            x = exp_x(rng)
            y = exp_y(rng)
            if 2.0 * y > x * x:
                return x + tail_start


def _find_one(patterns: Sequence[Path], *, label: str) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(pattern.parent.glob(pattern.name)))
    unique = tuple(dict.fromkeys(path.resolve() for path in matches if path.exists()))
    if len(unique) != 1:
        raise FileNotFoundError(f"expected exactly one {label}, found {len(unique)}: {[str(path) for path in unique[:5]]}")
    return unique[0]


def _chrom_file_patterns(parent: Path, chrom: int, suffix: str) -> tuple[Path, ...]:
    token = f"chr{int(chrom)}"
    return (
        parent / f"{token}.{suffix}",
        parent / f"{token}.*.{suffix}",
        parent / f"{token}_*.{suffix}",
        parent / f"{token}-*.{suffix}",
        parent / f"*.{token}.*.{suffix}",
        parent / f"*.{token}_*.{suffix}",
        parent / f"*.{token}-*.{suffix}",
    )


def discover_chromosome_files(grg_dir: Path, plink_dir: Path, chromosomes: Iterable[int]) -> tuple[ChromosomeFiles, ...]:
    grg_root = Path(grg_dir).expanduser()
    plink_root = Path(plink_dir).expanduser()
    files: list[ChromosomeFiles] = []
    for chrom in chromosomes:
        grg = _find_one(_chrom_file_patterns(grg_root, int(chrom), "grg"), label=f"chr{chrom} GRG")
        plink_chr_dir = plink_root / f"chr{chrom}"
        prefix_patterns = (*_chrom_file_patterns(plink_chr_dir, int(chrom), "bim"), *_chrom_file_patterns(plink_root, int(chrom), "bim"))
        bim = _find_one(prefix_patterns, label=f"chr{chrom} BIM")
        bed = bim.with_suffix(".bed")
        fam = bim.with_suffix(".fam")
        if not bed.exists():
            raise FileNotFoundError(bed)
        if not fam.exists():
            raise FileNotFoundError(fam)
        files.append(ChromosomeFiles(int(chrom), grg, bed, bim, fam))
    return tuple(files)


def read_bim(path: Path) -> tuple[BimRecord, ...]:
    records: list[BimRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            fields = line.rstrip("\n").split()
            if len(fields) < 6:
                raise ValueError(f"invalid BIM line {row + 1} in {path}: expected at least 6 fields")
            records.append(
                BimRecord(
                    chrom=int(fields[0]),
                    snp_id=fields[1],
                    genetic_pos=fields[2],
                    bp=int(fields[3]),
                    allele1=fields[4],
                    allele0=fields[5],
                    row=row,
                )
            )
    if not records:
        raise ValueError(f"BIM file is empty: {path}")
    return tuple(records)


def scan_bim_repeated_bps(path: Path) -> tuple[int, set[int]]:
    counts: dict[int, int] = {}
    rows = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            fields = line.rstrip("\n").split()
            if len(fields) < 6:
                raise ValueError(f"invalid BIM line {row + 1} in {path}: expected at least 6 fields")
            bp = int(fields[3])
            counts[bp] = counts.get(bp, 0) + 1
            rows += 1
    if rows == 0:
        raise ValueError(f"BIM file is empty: {path}")
    return rows, {bp for bp, count in counts.items() if count > 1}


def select_matched_bim_records(
    files: ChromosomeFiles,
    *,
    row_count: int,
    bim_repeated_bps: set[int],
    snps_per_chrom: int,
    seed: int,
    batch_size: int = 256,
) -> tuple[tuple[BimRecord, int], ...]:
    rows = int(row_count)
    requested = int(snps_per_chrom)
    if requested < 1:
        raise ValueError("snps_per_chrom must be positive for sampled matched BIM selection")
    if requested > rows:
        raise ValueError(f"--snpsPerChrom={requested} exceeds chr{files.chrom} BIM row count {rows}")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(files.chrom)]))
    row_order = rng.permutation(rows)
    selected_by_row: dict[int, tuple[BimRecord, int]] = {}
    chunk_size = max(int(batch_size), requested * 8)
    for start in range(0, rows, chunk_size):
        chunk_rows = row_order[start : start + chunk_size]
        records = read_selected_bim(files.bim, sorted(int(row) for row in chunk_rows))
        records_by_row = {int(record.row): record for record in records}
        candidate_records = tuple(records_by_row[int(row)] for row in chunk_rows)
        manifest = match_grg_to_bim(files, candidate_records, bim_repeated_bps=bim_repeated_bps)
        variant_by_row = {int(variant.bed_row): variant for variant in manifest.variants}
        for row in chunk_rows:
            row_int = int(row)
            variant = variant_by_row.get(row_int)
            if variant is not None:
                selected_by_row[row_int] = (records_by_row[row_int], int(variant.local_idx))
                if len(selected_by_row) == requested:
                    return tuple(selected_by_row[row] for row in sorted(selected_by_row))
    raise ValueError(
        f"--snpsPerChrom={requested} requested {requested} singleton GRG/BIM matches on chr{files.chrom}, "
        f"but only found {len(selected_by_row)} among {rows} BIM rows"
    )


def read_selected_bim(path: Path, selected_rows: Sequence[int]) -> tuple[BimRecord, ...]:
    targets = tuple(int(row) for row in selected_rows)
    if any(row < 0 for row in targets):
        raise ValueError("selected BIM rows must be nonnegative")
    if tuple(sorted(targets)) != targets:
        raise ValueError("selected BIM rows must be sorted")
    if len(set(targets)) != len(targets):
        raise ValueError("selected BIM rows must be unique")
    records: list[BimRecord] = []
    target_index = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            if target_index >= len(targets):
                break
            if row != targets[target_index]:
                continue
            fields = line.rstrip("\n").split()
            if len(fields) < 6:
                raise ValueError(f"invalid BIM line {row + 1} in {path}: expected at least 6 fields")
            records.append(
                BimRecord(
                    chrom=int(fields[0]),
                    snp_id=fields[1],
                    genetic_pos=fields[2],
                    bp=int(fields[3]),
                    allele1=fields[4],
                    allele0=fields[5],
                    row=row,
                )
            )
            target_index += 1
    if len(records) != len(targets):
        missing = targets[len(records) :]
        preview = ", ".join(str(row) for row in missing[:5])
        raise ValueError(f"{path} ended before selected BIM rows were found: {preview}")
    return tuple(records)


def _bed_row_bytes(n_individuals: int) -> int:
    n = int(n_individuals)
    if n < 1:
        raise ValueError("n_individuals must be positive")
    return (n + 3) // 4


def _validate_bed_size(path: Path, *, row_bytes: int, n_variants: int) -> None:
    expected_size = 3 + int(row_bytes) * int(n_variants)
    actual_size = Path(path).stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{path} has {actual_size} bytes; expected {expected_size} for {n_variants} SNP-major rows")


def _contiguous_runs(sorted_rows: Sequence[int]) -> Iterable[tuple[int, int]]:
    if not sorted_rows:
        return
    start = int(sorted_rows[0])
    prev = start
    length = 1
    for row_value in sorted_rows[1:]:
        row = int(row_value)
        if row == prev + 1:
            length += 1
        else:
            yield start, length
            start = row
            length = 1
        prev = row
    yield start, length


def write_selected_bed(
    source_bed: Path,
    output_bed: Path,
    selected_rows: Sequence[int],
    *,
    n_individuals: int,
    source_variant_count: int,
) -> int:
    rows = tuple(int(row) for row in selected_rows)
    if tuple(sorted(rows)) != rows:
        raise ValueError("selected BED rows must be sorted")
    if len(set(rows)) != len(rows):
        raise ValueError("selected BED rows must be unique")
    if rows and (rows[0] < 0 or rows[-1] >= int(source_variant_count)):
        raise ValueError("selected BED row is outside the source variant range")

    row_bytes = _bed_row_bytes(int(n_individuals))
    _validate_bed_size(Path(source_bed), row_bytes=row_bytes, n_variants=int(source_variant_count))
    out = Path(output_bed)
    out.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    with Path(source_bed).open("rb") as src, out.open("wb") as dst:
        magic = src.read(3)
        if magic != b"\x6c\x1b\x01":
            raise ValueError(f"{source_bed} is not a SNP-major PLINK BED file")
        dst.write(magic)
        for start_row, run_length in _contiguous_runs(rows):
            run_bytes = int(run_length) * row_bytes
            src.seek(3 + int(start_row) * row_bytes)
            remaining = run_bytes
            while remaining:
                chunk = src.read(min(remaining, 8 * 1024 * 1024))
                if not chunk:
                    raise ValueError(f"short read from {source_bed} at SNP row {start_row}")
                dst.write(chunk)
                copied += len(chunk)
                remaining -= len(chunk)
    return copied


def read_fam(path: Path) -> tuple[FamSample, ...]:
    samples: list[FamSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = tuple(line.rstrip("\n").split())
            if len(fields) < 2:
                raise ValueError(f"invalid FAM line {line_number} in {path}: expected at least FID IID")
            samples.append(FamSample(fid=fields[0], iid=fields[1], raw_fields=fields))
    if not samples:
        raise ValueError(f"FAM file is empty: {path}")
    return tuple(samples)


def assert_same_fam(fam_paths: Iterable[Path]) -> tuple[FamSample, ...]:
    paths = tuple(Path(path) for path in fam_paths)
    if not paths:
        raise ValueError("at least one FAM path is required")
    first = read_fam(paths[0])
    first_pairs = tuple((sample.fid, sample.iid) for sample in first)
    for path in paths[1:]:
        current = read_fam(path)
        current_pairs = tuple((sample.fid, sample.iid) for sample in current)
        if current_pairs != first_pairs:
            raise ValueError(f"FAM sample order differs between {paths[0]} and {path}")
    return first


def _variant_from_bim_record(files: ChromosomeFiles, record: BimRecord, *, local_idx: int, bed_row: int | None = None) -> Variant:
    return Variant(
        global_idx=-1,
        chrom=int(files.chrom),
        local_idx=int(local_idx),
        bed_row=int(record.row if bed_row is None else bed_row),
        snp_id=record.unique_id,
        bp=int(record.bp),
        genetic_pos=record.genetic_pos,
        allele1=record.allele1,
        allele0=record.allele0,
    )


def match_grg_to_bim(
    files: ChromosomeFiles,
    bim_records: Sequence[BimRecord],
    *,
    bim_repeated_bps: set[int],
) -> ChromosomeManifest:
    import pygrgl

    records = tuple(bim_records)
    local_bp_counts: dict[int, int] = {}
    for record in records:
        if int(record.chrom) != int(files.chrom):
            raise ValueError(f"BIM row {record.row + 1} in {files.bim} has chromosome {record.chrom}; expected {files.chrom}")
        local_bp_counts[int(record.bp)] = local_bp_counts.get(int(record.bp), 0) + 1
    excluded_bps = set(int(bp) for bp in bim_repeated_bps)
    excluded_bps.update(bp for bp, count in local_bp_counts.items() if count > 1)
    candidate_records = tuple(record for record in records if int(record.bp) not in excluded_bps)
    candidate_bps = {int(record.bp) for record in candidate_records}
    candidate_keys = {record.key for record in candidate_records}

    # GRG-only mutations at repeated BIM positions are outside the comparison
    # set because this harness is BIM-driven and analyzes singleton BIM rows.
    LOGGER.info("Matching chr%s selected BIM keys against GRG mutation metadata", files.chrom)
    grg = pygrgl.load_immutable_grg(str(files.grg), load_up_edges=False)
    position_counts: dict[int, int] = {}
    for local_idx in range(int(grg.num_mutations)):
        mutation = grg.get_mutation_by_id(int(local_idx))
        bp = int(round(float(mutation.position)))
        if bp not in candidate_bps:
            continue
        position_counts[bp] = position_counts.get(bp, 0) + 1

    singleton_by_key: dict[tuple[int, str, str], int] = {}
    for local_idx in range(int(grg.num_mutations)):
        mutation = grg.get_mutation_by_id(int(local_idx))
        bp = int(round(float(mutation.position)))
        if position_counts.get(bp) != 1:
            continue
        key = (bp, str(mutation.allele), str(mutation.ref_allele))
        if key not in candidate_keys:
            continue
        singleton_by_key[key] = int(local_idx)

    variants: list[Variant] = []
    for record in candidate_records:
        if position_counts.get(int(record.bp), 0) > 1:
            continue
        local_idx = singleton_by_key.get(record.key)
        if local_idx is None:
            continue
        variants.append(_variant_from_bim_record(files, record, local_idx=int(local_idx)))
    return ChromosomeManifest(
        chrom=int(files.chrom),
        grg_path=files.grg,
        bed_path=files.bed,
        bim_path=files.bim,
        fam_path=files.fam,
        variants=tuple(variants),
    )


def _bed_lookup_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    codes = ((np.arange(256, dtype=np.uint16)[:, None] >> np.array([0, 2, 4, 6], dtype=np.uint16)) & 0b11).astype(np.uint8)
    geno = np.take(np.asarray([2, -1, 1, 0], dtype=np.int16), codes)
    valid = geno >= 0
    sums = np.where(valid, geno, 0).sum(axis=1, dtype=np.int16)
    sumsqs = np.where(valid, geno * geno, 0).sum(axis=1, dtype=np.int16)
    missing = np.count_nonzero(~valid, axis=1).astype(np.int16)
    return geno, sums, sumsqs, missing


def read_bed_snp_stats(bed_path: Path, *, n_individuals: int, n_variants: int, chunk_rows: int = 1024) -> BedDosageStats:
    path = Path(bed_path)
    row_bytes = (int(n_individuals) + 3) // 4
    expected_size = 3 + int(row_bytes) * int(n_variants)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{path} has {actual_size} bytes; expected {expected_size} for {n_variants} SNP-major rows and {n_individuals} samples")
    geno_lut, sum_lut, sumsq_lut, miss_lut = _bed_lookup_tables()
    sum_g = np.empty(int(n_variants), dtype=np.float64)
    sumsq_g = np.empty(int(n_variants), dtype=np.float64)
    missing = np.empty(int(n_variants), dtype=np.int32)
    valid_last = int(n_individuals) % 4
    if valid_last == 0:
        valid_last = 4

    with path.open("rb") as handle:
        magic = handle.read(3)
        if magic != b"\x6c\x1b\x01":
            raise ValueError(f"{path} is not a SNP-major PLINK BED file")
        for start in range(0, int(n_variants), int(chunk_rows)):
            rows = min(int(chunk_rows), int(n_variants) - start)
            raw = handle.read(rows * row_bytes)
            if len(raw) != rows * row_bytes:
                raise ValueError(f"short read from {path} at SNP row {start}")
            data = np.frombuffer(raw, dtype=np.uint8).reshape(rows, row_bytes)
            if valid_last == 4:
                sum_g[start : start + rows] = sum_lut[data].sum(axis=1, dtype=np.int64)
                sumsq_g[start : start + rows] = sumsq_lut[data].sum(axis=1, dtype=np.int64)
                missing[start : start + rows] = miss_lut[data].sum(axis=1, dtype=np.int64)
            else:
                body = data[:, :-1]
                last = geno_lut[data[:, -1], :valid_last]
                last_valid = last >= 0
                sum_g[start : start + rows] = sum_lut[body].sum(axis=1, dtype=np.int64) + np.where(last_valid, last, 0).sum(axis=1)
                sumsq_g[start : start + rows] = sumsq_lut[body].sum(axis=1, dtype=np.int64) + np.where(last_valid, last * last, 0).sum(axis=1)
                missing[start : start + rows] = miss_lut[body].sum(axis=1, dtype=np.int64) + np.count_nonzero(~last_valid, axis=1)
    return BedDosageStats(sum_g=sum_g, sumsq_g=sumsq_g, missing=missing)


def _decode_bed_chunk(
    bed_path: Path,
    *,
    n_individuals: int,
    n_variants: int,
    start_row: int,
    rows: int,
) -> np.ndarray:
    path = Path(bed_path)
    row_bytes = _bed_row_bytes(int(n_individuals))
    _validate_bed_size(path, row_bytes=row_bytes, n_variants=int(n_variants))
    start = int(start_row)
    count = int(rows)
    if start < 0 or count < 0 or start + count > int(n_variants):
        raise ValueError("BED chunk is outside the variant range")
    geno_lut, _sum_lut, _sumsq_lut, _miss_lut = _bed_lookup_tables()
    with path.open("rb") as handle:
        magic = handle.read(3)
        if magic != b"\x6c\x1b\x01":
            raise ValueError(f"{path} is not a SNP-major PLINK BED file")
        handle.seek(3 + start * row_bytes)
        raw = handle.read(count * row_bytes)
    if len(raw) != count * row_bytes:
        raise ValueError(f"short read from {path} at SNP row {start}")
    data = np.frombuffer(raw, dtype=np.uint8).reshape(count, row_bytes)
    return geno_lut[data].reshape(count, row_bytes * 4)[:, : int(n_individuals)].astype(np.float64, copy=False)


def attach_bed_stats(
    manifest: ChromosomeManifest,
    stats: BedDosageStats,
    *,
    n_individuals: int,
) -> ChromosomeManifest:
    variants: list[Variant] = []
    n = float(n_individuals)
    rows = tuple(int(variant.bed_row) for variant in manifest.variants)
    for variant, row in zip(manifest.variants, rows, strict=True):
        missing = int(stats.missing[row])
        if missing:
            raise ValueError(f"chr{manifest.chrom} SNP {variant.snp_id} has {missing} missing PLINK hard calls; this comparison requires complete data")
        sum_g = float(stats.sum_g[row])
        sumsq_g = float(stats.sumsq_g[row])
        mean = sum_g / n
        mean_center_norm2 = sumsq_g - (sum_g * sum_g / n)
        norm_scale = 0.0 if mean_center_norm2 <= 0.0 else math.sqrt((n - 1.0) / mean_center_norm2)
        proj_norm2 = mean_center_norm2
        x_norm2 = proj_norm2 * norm_scale * norm_scale if norm_scale > 0.0 else 0.0
        variants.append(
            replace(
                variant,
                missing=missing,
                mean=mean,
                mean_center_norm2=mean_center_norm2,
                proj_norm2=proj_norm2,
                norm_scale=norm_scale,
                x_norm2=x_norm2,
                a1freq=sum_g / (2.0 * n),
            )
        )
    return replace(manifest, variants=tuple(variants))


def attach_projected_bed_stats(
    manifest: ChromosomeManifest,
    *,
    covariates: CovariateBasis,
    n_individuals: int,
    n_variants: int,
    chunk_rows: int = 64,
) -> ChromosomeManifest:
    """Recompute BOLT raw projected norms after the covariate basis is known."""

    variants = list(manifest.variants)
    if not variants:
        return manifest
    if covariates.nused != int(n_individuals):
        raise ValueError(f"covariate sample count {covariates.nused} does not match FAM sample count {n_individuals}")
    by_bed_row = {int(variant.bed_row): idx for idx, variant in enumerate(variants)}
    sorted_rows = sorted(by_bed_row)
    for run_start, run_len in _contiguous_runs(sorted_rows):
        for offset in range(0, run_len, int(chunk_rows)):
            count = min(int(chunk_rows), run_len - offset)
            chunk_start = run_start + offset
            genotypes = _decode_bed_chunk(
                manifest.bed_path,
                n_individuals=int(n_individuals),
                n_variants=int(n_variants),
                start_row=chunk_start,
                rows=count,
            )
            for local_offset in range(count):
                bed_row = chunk_start + local_offset
                variant_idx = by_bed_row.get(bed_row)
                if variant_idx is None:
                    continue
                variant = variants[variant_idx]
                if float(variant.mean_center_norm2) <= 0.0:
                    variants[variant_idx] = replace(variant, proj_norm2=0.0, norm_scale=0.0, x_norm2=0.0)
                    continue
                centered = genotypes[local_offset] - float(variant.mean)
                projected = covariates.project_host(centered)
                proj_norm2 = float(np.dot(projected, projected))
                norm_scale = math.sqrt(float(n_individuals - 1) / float(variant.mean_center_norm2))
                x_norm2 = proj_norm2 * norm_scale * norm_scale
                variants[variant_idx] = replace(
                    variant,
                    proj_norm2=proj_norm2,
                    norm_scale=norm_scale,
                    x_norm2=x_norm2,
                )
    return replace(manifest, variants=tuple(variants))


def assign_global_indices(manifests: Iterable[ChromosomeManifest]) -> tuple[ChromosomeManifest, ...]:
    output: list[ChromosomeManifest] = []
    global_idx = 0
    for manifest in manifests:
        variants: list[Variant] = []
        for variant in manifest.variants:
            variants.append(replace(variant, global_idx=global_idx))
            global_idx += 1
        output.append(replace(manifest, variants=tuple(variants)))
    return tuple(output)


def _require_projected_model_snps(manifests: Sequence[ChromosomeManifest]) -> int:
    # Official BOLT switches makeChunkAssignments() behavior in this case; this
    # comparison keeps one selected chromosome per LOCO chunk and rejects it.
    empty_chroms = [manifest.chrom for manifest in manifests if manifest.model_count <= 0]
    if empty_chroms:
        names = ", ".join(f"chr{int(chrom)}" for chrom in empty_chroms)
        raise ValueError(f"chromosomes have no projected model SNPs after covariate projection: {names}")
    total = sum(manifest.model_count for manifest in manifests)
    if total < 2:
        raise ValueError(f"calibration requires at least two eligible projected SNPs; found {total}")
    return total


def write_rewritten_bim(manifest: ChromosomeManifest, path: Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with out.open("w", encoding="utf-8") as handle:
        for variant in manifest.variants:
            if variant.snp_id in seen:
                raise ValueError(f"rewritten BIM ID would be duplicated: {variant.snp_id}")
            seen.add(variant.snp_id)
            handle.write(variant.plink_line + "\n")


def write_manifest(manifests: Iterable[ChromosomeManifest], path: Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "global_idx",
        "chrom",
        "local_grg_idx",
        "bed_row",
        "snp_id",
        "bp",
        "allele1",
        "allele0",
        "missing",
        "mean",
        "mean_center_norm2",
        "proj_norm2",
        "norm_scale",
        "x_norm2",
        "a1freq",
    )
    with out.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for manifest in manifests:
            for variant in manifest.variants:
                handle.write(
                    "\t".join(
                        (
                            str(variant.global_idx),
                            str(variant.chrom),
                            str(variant.local_idx),
                            str(variant.bed_row),
                            variant.snp_id,
                            str(variant.bp),
                            variant.allele1,
                            variant.allele0,
                            str(variant.missing),
                            f"{variant.mean:.17g}",
                            f"{variant.mean_center_norm2:.17g}",
                            f"{variant.proj_norm2:.17g}",
                            f"{variant.norm_scale:.17g}",
                            f"{variant.x_norm2:.17g}",
                            f"{variant.a1freq:.17g}",
                        )
                    )
                    + "\n"
                )


def ensure_artifacts(files: Iterable[ChromosomeFiles], artifact_cache: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for item in files:
        artifact = artifact_path_for_grg(item.grg, Path(artifact_cache).expanduser())
        if artifact.exists():
            try:
                scan_grg_spmv(artifact)
            except ValueError as exc:
                LOGGER.info("Rebuilding unsupported chr%s artifact %s: %s", item.chrom, artifact, exc)
                artifact.unlink(missing_ok=True)
        if not artifact.exists():
            LOGGER.info("Converting chr%s GRG to %s", item.chrom, artifact)
            artifact = convert(item.grg, Path(artifact_cache).expanduser(), dtype=DTYPE)
        paths.append(artifact)
    return tuple(paths)


def bolt_runtime_requirements() -> RuntimeRequirements:
    return RuntimeRequirements(
        max_k_up=1,
        max_k_down=1,
        need_down_miss_input=False,
        need_up_miss_output=False,
        need_init_vector=False,
        need_init_matrix=False,
        need_init_xtx=False,
    )


class BoltGrgOps:
    """GRG-backed BOLT-LMM-inf linear algebra using BOLT's SNP scaling."""

    def __init__(self, runtime, manifests: Sequence[ChromosomeManifest], covariates: CovariateBasis):
        import cupy as cp

        self.cp = cp
        self.runtime = runtime
        self.device = runtime.device
        self.stream = runtime.stream
        self.manifests = tuple(manifests)
        self.covariates = covariates
        self.states: dict[int, ChromOpsState] = {}
        self._managers: list[Any] = []
        self.n = 0
        self.dim = 0
        self.cindep = 0
        self.m_proj = 0
        self.xfro2 = 0.0
        self.test_m = sum(len(manifest.variants) for manifest in self.manifests)
        self.work = None
        self.basis_dev = None

    def __enter__(self) -> "BoltGrgOps":
        grgs = self.runtime.grgs
        if len(grgs) != len(self.manifests):
            raise ValueError(f"runtime has {len(grgs)} GRGs, but {len(self.manifests)} manifests were provided")
        self.n = int(grgs[0].num_individuals)
        if self.covariates.nused != self.n:
            raise ValueError(f"covariate sample count {self.covariates.nused} does not match GRG individual count {self.n}")
        self.cindep = int(self.covariates.cindep)
        self.dim = int(self.covariates.dim)
        if self.dim <= 0:
            raise ValueError(f"nonpositive projected dimension: Nused={self.n}, Cindep={self.cindep}")
        with self.device:
            self.work = self.cp.empty((self.n,), dtype=DTYPE)
            self.basis_dev = self.cp.asarray(self.covariates.basis, dtype=DTYPE)
            for grg, manifest in zip(grgs, self.manifests, strict=True):
                up_manager = grg.prepare_matmul_cuda(direction="up", k=1, by_individual=True)
                up_op = up_manager.__enter__()
                self._managers.append(up_manager)
                down_manager = grg.prepare_matmul_cuda(direction="down", k=1, by_individual=True)
                down_op = down_manager.__enter__()
                self._managers.append(down_manager)

                means = np.zeros(int(grg.num_mutations), dtype=DTYPE)
                scales = np.zeros(int(grg.num_mutations), dtype=DTYPE)
                local_indices = np.asarray([variant.local_idx for variant in manifest.variants if is_model_variant(variant)], dtype=np.int64)
                for variant in manifest.variants:
                    if is_model_variant(variant):
                        means[int(variant.local_idx)] = float(variant.mean)
                        scales[int(variant.local_idx)] = float(variant.norm_scale)
                self.m_proj += int(local_indices.size)
                self.xfro2 += float(sum(float(variant.x_norm2) for variant in manifest.variants if is_model_variant(variant)))
                scale_dev = self.cp.asarray(scales, dtype=DTYPE)
                mean_dev = self.cp.asarray(means, dtype=DTYPE)
                state = ChromOpsState(
                    manifest=manifest,
                    grg=grg,
                    up_op=up_op,
                    down_op=down_op,
                    views={
                        "up_input": self.cp.from_dlpack(up_op.input),
                        "up_output": self.cp.from_dlpack(up_op.output),
                        "down_input": self.cp.from_dlpack(down_op.input),
                        "down_output": self.cp.from_dlpack(down_op.output),
                    },
                    mean=mean_dev,
                    scale=scale_dev,
                    mean_scale=mean_dev * scale_dev,
                    local_indices=self.cp.asarray(local_indices, dtype=self.cp.int64),
                )
                self.states[int(manifest.chrom)] = state
        if self.m_proj <= 0:
            raise ValueError("eligible SNP set is empty after filtering")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        with self.device:
            for manager in reversed(self._managers):
                manager.__exit__(None, None, None)
        self._managers = []
        self.states.clear()
        self.work = None
        self.basis_dev = None

    def _run(self, op) -> None:
        with self.stream:
            op()

    def project(self, values):
        with self.device, self.stream:
            arr = self.cp.asarray(values, dtype=DTYPE)
            was_vector = arr.ndim == 1
            mat = arr.reshape((self.n, 1)) if was_vector else arr
            projected = mat - self.basis_dev @ (self.basis_dev.T @ mat)
            return projected[:, 0] if was_vector else projected

    def project_inplace(self, values):
        with self.device, self.stream:
            values[...] = self.project(values)
        return values

    def scores(self, chrom: int, vector):
        state = self.states[int(chrom)]
        with self.device, self.stream:
            v = self.project(vector)
            input_sum = v.sum()
            state.views["up_input"][0, :] = v
        self._run(state.up_op)
        with self.device, self.stream:
            out = state.views["up_output"][0]
            out -= state.mean * input_sum
            out *= state.scale
        return out

    def apply_x(self, chrom: int, weights, out):
        state = self.states[int(chrom)]
        with self.device, self.stream:
            w = self.cp.asarray(weights, dtype=DTYPE)
            state.views["down_input"][0, :] = w
            state.views["down_input"][0, :] *= state.scale
            constant = self.cp.dot(state.mean_scale, w)
        self._run(state.down_op)
        with self.device, self.stream:
            out[...] = state.views["down_output"][0]
            out -= constant
            self.project_inplace(out)
        return out

    def column(self, chrom: int, local_idx: int):
        state = self.states[int(chrom)]
        idx = int(local_idx)
        with self.device, self.stream:
            state.views["down_input"][0, :].fill(0.0)
            state.views["down_input"][0, idx] = state.scale[idx]
            constant = state.mean_scale[idx]
        self._run(state.down_op)
        with self.device, self.stream:
            out = state.views["down_output"][0].copy()
            out -= constant
            self.project_inplace(out)
        return out

    def apply_k(self, vector, *, exclude_chrom: int | None, out):
        with self.device, self.stream:
            out.fill(0.0)
        active_m_proj = 0
        for chrom, state in self.states.items():
            if exclude_chrom is not None and int(chrom) == int(exclude_chrom):
                continue
            scores = self.scores(chrom, vector)
            self.apply_x(chrom, scores, self.work)
            with self.device, self.stream:
                out += self.work
            active_m_proj += int(state.local_indices.size)
        if active_m_proj == 0:
            with self.device, self.stream:
                out.fill(0.0)
            return out
        with self.device, self.stream:
            out /= float(active_m_proj)
            self.project_inplace(out)
        return out


def bolt_conj_grad_solve(
    matvecs: Sequence[Any],
    rhs_columns: Sequence[Any],
    *,
    rel_tol: float,
    max_iter: int,
    stats: CgStats | None = None,
    project,
):
    if len(matvecs) != len(rhs_columns):
        raise ValueError("matvec and RHS counts differ")
    if not rhs_columns:
        return []
    xp = _array_module(rhs_columns[0])
    # Mirrors BOLT-LMM_v2.5 Bolt::conjGradSolve: full-batch CG, no active-column
    # mask, no denominator guard, and no exception when maxIters is reached.
    b_cols = [project(xp.asarray(rhs, dtype=DTYPE).copy()) for rhs in rhs_columns]
    b = xp.column_stack(b_cols)
    x = xp.zeros_like(b)
    r = b.copy()
    p = r.copy()
    hp = xp.empty_like(b)
    r2_orig = xp.sum(r * r, axis=0)
    r2_old = r2_orig.copy()
    rels = xp.sqrt(r2_old / r2_orig)
    for it in range(1, int(max_iter) + 1):
        for col, matvec in enumerate(matvecs):
            matvec(p[:, col], hp[:, col])
        denom = xp.sum(p * hp, axis=0)
        alpha = r2_old / denom
        x += p * alpha.reshape((1, -1))
        r -= hp * alpha.reshape((1, -1))
        r = project(r)
        r2_new = xp.sum(r * r, axis=0)
        rels = xp.sqrt(r2_new / r2_orig)
        if not bool(_as_float(xp.any(rels > float(rel_tol)))):
            if stats is not None:
                rel_values = xp.asnumpy(rels) if hasattr(xp, "asnumpy") else np.asarray(rels)
                for rel in rel_values:
                    stats.add(it, float(rel))
            return [project(x[:, idx].copy()) for idx in range(x.shape[1])]
        beta = r2_new / r2_old
        p *= beta.reshape((1, -1))
        p += r
        r2_old = r2_new
    if stats is not None:
        rel_values = xp.asnumpy(rels) if hasattr(xp, "asnumpy") else np.asarray(rels)
        for rel in rel_values:
            stats.add(int(max_iter), float(rel))
    return [project(x[:, idx].copy()) for idx in range(x.shape[1])]


def log_delta_from_h2(ops: BoltGrgOps, h2: float) -> float:
    return math.log(float(ops.xfro2) / (float(ops.m_proj) * float(ops.dim)) * (1.0 - float(h2)) / float(h2))


def h2_from_log_delta(ops: BoltGrgOps, log_delta: float) -> float:
    return float(ops.xfro2) / (float(ops.xfro2) + float(ops.m_proj) * float(ops.dim) * math.exp(float(log_delta)))


def _sum_score_squares(ops: BoltGrgOps, vector) -> float:
    total = 0.0
    for chrom, state in ops.states.items():
        scores = ops.scores(chrom, vector)
        with ops.device, ops.stream:
            total += _as_float(ops.cp.sum(scores[state.local_indices] * scores[state.local_indices]))
    return float(total)


def _generate_bolt_mc_components(
    ops: BoltGrgOps,
    y,
    *,
    trials: int,
    seed: int,
) -> tuple[list[Any], list[Any], Any]:
    rng = BoostMt19937(int(seed) + 1)
    randn = BoostNormalDistribution()
    inv_sqrt_m = 1.0 / math.sqrt(float(ops.m_proj))
    weights_by_chrom: dict[int, np.ndarray] = {
        int(chrom): np.zeros((int(trials), int(state.grg.num_mutations)), dtype=np.float64)
        for chrom, state in ops.states.items()
    }

    for manifest in ops.manifests:
        chrom_weights = weights_by_chrom[int(manifest.chrom)]
        for variant in manifest.variants:
            if not is_model_variant(variant):
                continue
            for trial in range(int(trials)):
                chrom_weights[trial, int(variant.local_idx)] = randn(rng) * inv_sqrt_m

    g_rand: list[Any] = []
    e_rand: list[Any] = []
    with ops.device:
        for trial in range(int(trials)):
            g = ops.cp.zeros((ops.n,), dtype=DTYPE)
            for chrom, weights in weights_by_chrom.items():
                w = ops.cp.asarray(weights[trial], dtype=DTYPE)
                ops.apply_x(chrom, w, ops.work)
                with ops.stream:
                    g += ops.work
            with ops.stream:
                ops.project_inplace(g)
            g_rand.append(g)

        for _trial in range(int(trials)):
            values = np.fromiter((randn(rng) for _ in range(int(ops.n))), dtype=np.float64, count=int(ops.n))
            e = ops.cp.asarray(values, dtype=DTYPE)
            with ops.stream:
                ops.project_inplace(e)
            e_rand.append(e)

        y_dev = ops.project(ops.cp.asarray(y, dtype=DTYPE).copy())
    return g_rand, e_rand, y_dev


def _compute_mc_scaling(
    ops: BoltGrgOps,
    y_dev,
    g_rand: Sequence[Any],
    e_rand: Sequence[Any],
    *,
    log_delta: float,
    rel_tol: float,
    max_iter: int,
    stats: CgStats,
) -> McScalingResult:
    # Corresponds to the official BOLT log-delta MC scaling solve path.
    trials = len(g_rand)
    delta = math.exp(float(log_delta))
    sqrt_delta = math.sqrt(delta)

    def h_into(src, dst) -> None:
        ops.apply_k(src, exclude_chrom=None, out=dst)
        with ops.stream:
            dst += delta * src

    rand_beta: list[float] = []
    rand_eps: list[float] = []
    with ops.device:
        rhs_columns = []
        for g_t, e_t in zip(g_rand, e_rand, strict=True):
            rhs = ops.cp.empty((ops.n,), dtype=DTYPE)
            with ops.stream:
                ops.cp.multiply(e_t, sqrt_delta, out=rhs)
                ops.cp.add(rhs, g_t, out=rhs)
                ops.project_inplace(rhs)
            rhs_columns.append(rhs)
        rhs_columns.append(y_dev)
        z_columns = bolt_conj_grad_solve(
            [h_into for _ in rhs_columns],
            rhs_columns,
            rel_tol=rel_tol,
            max_iter=max_iter,
            stats=stats,
            project=ops.project,
        )
        for z_t in z_columns[:-1]:
            rand_beta.append(_sum_score_squares(ops, z_t))
            rand_eps.append(_dot(z_t, z_t))

        z_data = z_columns[-1]
        data_beta = _sum_score_squares(ops, z_data)
        data_eps = _dot(z_data, z_data)

    if min([data_beta, data_eps, *rand_beta, *rand_eps]) <= 0.0:
        raise RuntimeError("invalid BOLT MC-scaling objective component")

    rand_beta_total = float(sum(rand_beta))
    rand_eps_total = float(sum(rand_eps))
    f_reml = math.log((data_beta / data_eps) / (rand_beta_total / rand_eps_total))

    f_jacks: list[float] = []
    for jack in range(trials + 1):
        jack_rand_beta = 0.0
        jack_rand_eps = 0.0
        for trial in range(trials):
            if trial != jack:
                jack_rand_beta += rand_beta[trial]
                jack_rand_eps += rand_eps[trial]
        if jack_rand_beta <= 0.0 or jack_rand_eps <= 0.0:
            f_jacks.append(float("nan"))
        else:
            f_jacks.append(math.log((data_beta / data_eps) / (jack_rand_beta / jack_rand_eps)))
    f_jacks[-1] = f_reml

    f_rands_as_data: list[float] = []
    for trial in range(trials):
        f_rands_as_data.append(math.log((rand_beta[trial] / rand_eps[trial]) / (rand_beta_total / rand_eps_total)))

    sigma2_k = _dot(y_dev, z_data) / float(max(ops.dim, 1))
    return McScalingResult(
        log_delta=float(log_delta),
        f_jacks=tuple(float(value) for value in f_jacks),
        f_rands_as_data=tuple(float(value) for value in f_rands_as_data),
        sigma2_k=float(sigma2_k),
        all_hinv_y=z_data,
    )


def fit_bolt_variance_components(
    ops: BoltGrgOps,
    y,
    *,
    mc_trials: int,
    seed: int,
    rel_tol: float,
    max_iter: int,
    stats: CgStats,
) -> VarianceFit:
    # Local analogue of BOLT's variance-component log-delta search for --lmmInfOnly.
    trials = max(2, int(mc_trials))
    g_rand, e_rand, y_dev = _generate_bolt_mc_components(ops, y, trials=trials, seed=int(seed))

    def evaluate(log_delta: float) -> McScalingResult:
        return _compute_mc_scaling(
            ops,
            y_dev,
            g_rand,
            e_rand,
            log_delta=float(log_delta),
            rel_tol=rel_tol,
            max_iter=max_iter,
            stats=stats,
        )

    prev = evaluate(log_delta_from_h2(ops, 0.25))
    cur = evaluate(log_delta_from_h2(ops, 0.125 if prev.f_reml < 0.0 else 0.5))
    best = prev if abs(prev.f_reml) <= abs(cur.f_reml) else cur
    if abs(prev.f_reml) < abs(cur.f_reml):
        prev, cur = cur, prev

    best_accepts_secant = False
    for _step in range(5):
        if abs(cur.f_reml - prev.f_reml) < 1e-300:
            break
        next_log_delta = (prev.log_delta * cur.f_reml - cur.log_delta * prev.f_reml) / (cur.f_reml - prev.f_reml)
        next_log_delta = float(np.clip(next_log_delta, -10.0, 10.0))
        if (not best_accepts_secant) and best.log_delta == cur.log_delta and abs(next_log_delta - cur.log_delta) < 0.01:
            break
        prev = cur
        cur = evaluate(next_log_delta)
        if (not best_accepts_secant) or abs(cur.f_reml) < abs(best.f_reml):
            best = cur
            best_accepts_secant = True

    delta = math.exp(float(best.log_delta))
    sigma_g2 = float(best.sigma2_k)
    sigma_e2 = delta * sigma_g2
    h2 = h2_from_log_delta(ops, best.log_delta)
    return VarianceFit(
        log_delta=float(best.log_delta),
        sigma_g2=sigma_g2,
        sigma_e2=float(sigma_e2),
        h2=float(h2),
        delta=float(delta),
        all_hinv_y=best.all_hinv_y,
    )


def solve_loco_hinv_y(
    ops: BoltGrgOps,
    y,
    *,
    fit: VarianceFit,
    rel_tol: float,
    max_iter: int,
    stats: CgStats,
) -> dict[int, Any]:
    residuals: dict[int, Any] = {}
    with ops.device:
        y_dev = ops.project(ops.cp.asarray(y, dtype=DTYPE))
        chroms = tuple(int(chrom) for chrom in ops.states)
        matvecs = []
        rhs_columns = []
        for chrom in chroms:
            def h_into(src, dst, left_out=chrom) -> None:
                ops.apply_k(src, exclude_chrom=int(left_out), out=dst)
                with ops.stream:
                    dst += float(fit.delta) * src

            matvecs.append(h_into)
            rhs_columns.append(y_dev)
        solved = bolt_conj_grad_solve(matvecs, rhs_columns, rel_tol=rel_tol, max_iter=max_iter, stats=stats, project=ops.project)
        for chrom, value in zip(chroms, solved, strict=True):
            residuals[int(chrom)] = value
    return residuals


def _ordered_variants(manifests: Sequence[ChromosomeManifest]) -> tuple[Variant, ...]:
    return tuple(variant for manifest in manifests for variant in manifest.variants)


def select_bolt_calibration_snps(
    ops: BoltGrgOps,
    *,
    fit: VarianceFit,
    count: int,
    seed: int,
) -> tuple[tuple[Variant, ...], int]:
    num_calib = int(count)
    if num_calib < 2:
        raise ValueError("at least two calibration SNPs are required")
    ordered = _ordered_variants(ops.manifests)
    proj_mask = [is_model_variant(variant) for variant in ordered]
    model_count = sum(proj_mask)
    if model_count <= 0:
        raise ValueError("no eligible SNPs available for calibration")
    if num_calib > model_count:
        raise ValueError(f"requested {num_calib} calibration SNPs but only {model_count} model SNPs are available")

    m_total = len(ordered)
    m_first = [m_total] * (num_calib + 1)
    m_good = 0
    for m, is_good in enumerate(proj_mask):
        if is_good:
            block = num_calib * m_good // model_count
            if m_first[block] == m_total:
                m_first[block] = m
            m_good += 1
    if any(start == m_total for start in m_first[:-1]):
        raise RuntimeError("failed to build BOLT calibration SNP blocks")

    all_hinv_norm2 = _dot(fit.all_hinv_y, fit.all_hinv_y)
    if all_hinv_norm2 <= 0.0:
        raise RuntimeError("all-chromosome H^-1 y has nonpositive norm")
    grammar_scores: dict[int, float] = {}
    with ops.device:
        for manifest in ops.manifests:
            chrom = int(manifest.chrom)
            state = ops.states[chrom]
            scores = ops.scores(chrom, fit.all_hinv_y)
            local_indices = ops.cp.asnumpy(state.local_indices)
            local_scores = ops.cp.asnumpy(scores[state.local_indices])
            score_by_local = {int(local): float(score) for local, score in zip(local_indices, local_scores, strict=True)}
            for variant in manifest.variants:
                if is_model_variant(variant):
                    grammar_scores[int(variant.global_idx)] = score_by_local[int(variant.local_idx)]

    rng = BoostMt19937(int(seed) + 321)
    selected: list[Variant] = []
    tried = 0
    for block in range(num_calib):
        block_start = int(m_first[block])
        block_end = int(m_first[block + 1])
        block_width = block_end - block_start
        if block_width <= 0:
            raise RuntimeError(f"empty calibration block {block}")
        attempts = 0
        while True:
            attempts += 1
            if attempts > 1_000_000:
                raise RuntimeError(f"could not select a non-outlier calibration SNP from block {block}")
            m = block_start + boost_uniform_int_0_2pow30(rng) % block_width
            if not proj_mask[m]:
                continue
            tried += 1
            variant = ordered[m]
            x_norm2 = float(variant.x_norm2)
            retro_stat = (grammar_scores[int(variant.global_idx)] ** 2) / all_hinv_norm2 / x_norm2 * float(ops.dim)
            if retro_stat < 5.0:
                selected.append(variant)
                break
    return tuple(selected), tried


def calibrate_lmm_inf(
    ops: BoltGrgOps,
    y,
    residuals: dict[int, Any],
    *,
    fit: VarianceFit,
    count: int,
    seed: int,
    rel_tol: float,
    max_iter: int,
    stats: CgStats,
) -> CalibrationResult:
    # Mirrors Bolt::computeLmmInf: batch LOCO phenotype solves with calibration-SNP
    # denominator solves, then calibrate retrospective stats to prospective stats.
    selected, tried = select_bolt_calibration_snps(ops, fit=fit, count=int(count), seed=int(seed))
    pro_stats: list[float] = []
    retro_stats: list[float] = []
    ratios: list[float] = []
    n_minus_c = float(max(ops.dim, 1))
    residuals.clear()
    with ops.device:
        y_dev = ops.project(ops.cp.asarray(y, dtype=DTYPE).copy())
        chroms = tuple(int(chrom) for chrom in ops.states)
        rhs_columns = []
        matvecs = []
        for chrom in chroms:
            def h_into(src, dst, left_out=chrom) -> None:
                ops.apply_k(src, exclude_chrom=int(left_out), out=dst)
                with ops.stream:
                    dst += float(fit.delta) * src

            matvecs.append(h_into)
            rhs_columns.append(y_dev)
        selected_columns = []
        for variant in selected:
            chrom = int(variant.chrom)
            x = ops.column(chrom, int(variant.local_idx))

            def h_into(src, dst, left_out=chrom) -> None:
                ops.apply_k(src, exclude_chrom=int(left_out), out=dst)
                with ops.stream:
                    dst += float(fit.delta) * src

            matvecs.append(h_into)
            rhs_columns.append(x)
            selected_columns.append(x)
        solved_columns = bolt_conj_grad_solve(
            matvecs,
            rhs_columns,
            rel_tol=rel_tol,
            max_iter=max_iter,
            stats=stats,
            project=ops.project,
        )
        for chrom, solved in zip(chroms, solved_columns[: len(chroms)], strict=True):
            residuals[int(chrom)] = solved
        q_by_global = {
            int(variant.global_idx): solved
            for variant, solved in zip(selected, solved_columns[len(chroms) :], strict=True)
        }
        x_by_global = {
            int(variant.global_idx): x
            for variant, x in zip(selected, selected_columns, strict=True)
        }
        h_norm2 = {chrom: _dot(value, value) for chrom, value in residuals.items()}
        phi_h_phi = {chrom: _dot(y_dev, value) for chrom, value in residuals.items()}

        for variant in selected:
            chrom = int(variant.chrom)
            x = x_by_global[int(variant.global_idx)]
            score_h = _dot(x, residuals[chrom])
            x_norm2 = _dot(x, x)
            if h_norm2[chrom] <= 0.0 or phi_h_phi[chrom] <= 0.0:
                raise RuntimeError(f"invalid LOCO H^-1 y moments for chr{chrom}")
            if x_norm2 <= 0.0:
                raise RuntimeError(f"selected calibration SNP has nonpositive projected norm: {variant.snp_id}")
            retro = n_minus_c * score_h * score_h / (h_norm2[chrom] * x_norm2)
            if retro <= 0.0:
                raise RuntimeError(f"selected calibration SNP has nonpositive retrospective stat: {variant.snp_id}")
            q = q_by_global[int(variant.global_idx)]
            denom_h = _dot(x, q)
            if denom_h <= 0.0:
                raise RuntimeError(f"selected calibration SNP has nonpositive prospective denominator: {variant.snp_id}")
            pro = n_minus_c * score_h * score_h / denom_h / phi_h_phi[chrom]
            pro_stats.append(float(pro))
            retro_stats.append(float(retro))
            ratios.append(float(pro / retro))

    total_pro = float(sum(pro_stats))
    total_retro = float(sum(retro_stats))
    if total_pro <= 0.0 or total_retro <= 0.0:
        raise RuntimeError("calibration failed: prospective or retrospective sum is nonpositive")
    factor = total_pro / total_retro
    calibration_jacks = [
        (total_pro - pro) / (total_retro - retro)
        for pro, retro in zip(pro_stats, retro_stats, strict=True)
    ]
    jack_count = len(calibration_jacks)
    jack_sum = float(sum(calibration_jacks))
    jack_sum2 = float(sum(value * value for value in calibration_jacks))
    calibration_std = math.sqrt(max(0.0, (jack_sum2 - jack_sum * jack_sum / jack_count) * (jack_count - 1) / jack_count))
    ratio_of_medians = float(np.median(np.asarray(pro_stats, dtype=np.float64)) / np.median(np.asarray(retro_stats, dtype=np.float64)))
    median_of_ratios = float(np.median(np.asarray(ratios, dtype=np.float64)))
    if calibration_std > 0.01:
        factor = ratio_of_medians
    if factor <= 0.0:
        raise RuntimeError(f"calibration factor is nonpositive: {factor}")
    with ops.device:
        vinv_scale_by_chrom = {}
        for chrom, residual in residuals.items():
            resid_norm2 = _dot(residual, residual)
            if resid_norm2 <= 0.0:
                raise RuntimeError(f"LOCO H^-1 y has nonpositive norm for chr{chrom}")
            resid_factor = math.sqrt(n_minus_c / resid_norm2 * factor)
            vinv_scale_by_chrom[int(chrom)] = 1.0 / (resid_factor * float(fit.sigma_g2))
    return CalibrationResult(
        factor=float(factor),
        std=float(calibration_std),
        ratio_of_medians=ratio_of_medians,
        median_of_ratios=median_of_ratios,
        selected_snps=tuple(variant.snp_id for variant in selected),
        tried_snps=int(tried),
        vinv_scale_by_chrom=vinv_scale_by_chrom,
    )


def write_pheno(samples: Sequence[FamSample], y: np.ndarray, path: Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(samples) != int(y.size):
        raise ValueError(f"phenotype length {y.size} does not match FAM sample count {len(samples)}")
    with out.open("w", encoding="utf-8") as handle:
        handle.write("FID IID PHENO\n")
        for sample, value in zip(samples, y, strict=True):
            handle.write(f"{sample.fid} {sample.iid} {float(value):.17g}\n")


def _json_dump_stable(data: Any, path: Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _json_load(path: Path) -> Any | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _file_fingerprint(path: Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    st = p.stat()
    return {"path": str(p), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _fam_hash(samples: Sequence[FamSample]) -> str:
    h = hashlib.sha256()
    for sample in samples:
        h.update(sample.fid.encode("utf-8"))
        h.update(b"\0")
        h.update(sample.iid.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _manifest_signature(manifests: Sequence[ChromosomeManifest]) -> dict[str, Any]:
    h = hashlib.sha256()
    count = 0
    for manifest in manifests:
        for variant in manifest.variants:
            h.update(str(variant.chrom).encode("ascii"))
            h.update(b"\t")
            h.update(str(variant.bed_row).encode("ascii"))
            h.update(b"\t")
            h.update(variant.snp_id.encode("utf-8"))
            h.update(b"\n")
            count += 1
    return {"count": count, "sha256": h.hexdigest()}


def _input_fingerprints(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [_file_fingerprint(path) for path in paths]


def _covariate_cache_meta(
    *,
    seed: int,
    samples: Sequence[FamSample],
    manifests: Sequence[ChromosomeManifest],
    source_files: Sequence[ChromosomeFiles],
) -> dict[str, Any]:
    input_paths: list[Path] = []
    for item in source_files:
        input_paths.extend((item.grg, item.bed, item.bim, item.fam))
    return {
        "schema_version": COVARIATE_CACHE_SCHEMA_VERSION,
        "code_version": BOLT_LMM_INF_CODE_VERSION,
        "seed": int(seed),
        "fam_hash": _fam_hash(samples),
        "selected_variants": _manifest_signature(manifests),
        "pca": {"pcs": len(GENERATED_Q_COVARS), "operator": "sample-space-grg-cusparse", "which": "LA", "tol": 1e-6, "maxiter": 200},
        "sex_rule": "balanced_1_2_seedsequence_seed_0x534558",
        "inputs": _input_fingerprints(input_paths),
    }


def _phenotype_cache_meta(
    *,
    seed: int,
    sim_h2: float,
    samples: Sequence[FamSample],
    grg_paths: Sequence[Path],
    num_causal_per_file: int,
) -> dict[str, Any]:
    try:
        version = importlib_metadata.version("grg-pheno-sim")
    except importlib_metadata.PackageNotFoundError:
        try:
            version = importlib_metadata.version("grg_pheno_sim")
        except importlib_metadata.PackageNotFoundError:
            version = "unknown"
    return {
        "schema_version": PHENOTYPE_CACHE_SCHEMA_VERSION,
        "code_version": BOLT_LMM_INF_CODE_VERSION,
        "seed": int(seed),
        "simH2": float(sim_h2),
        "fam_hash": _fam_hash(samples),
        "grg_inputs": _input_fingerprints(grg_paths),
        "grg_pheno_sim_version": version,
        "model": "normal(mean=0,var=1)",
        "num_causal_per_file": int(num_causal_per_file),
        "load_all_ram": False,
        "normalize_phenotype": True,
        "normalize_genetic_values_before_noise": False,
        "normalize_genetic_values_after": False,
    }


def _read_table_by_fam(path: Path, samples: Sequence[FamSample], columns: Sequence[str]) -> dict[str, list[str]]:
    requested = tuple(str(col) for col in columns)
    sample_keys = [(sample.fid, sample.iid) for sample in samples]
    sample_set = set(sample_keys)
    seen: set[tuple[str, str]] = set()
    values_by_key: dict[tuple[str, str], dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError(f"covariate file is empty: {path}")
        header = header_line.rstrip("\n").split()
        if len(header) < 2 or header[0] != "FID" or header[1] != "IID":
            raise ValueError(f"covariate/phenotype file must start with header: FID IID: {path}")
        index_by_name = {name: idx for idx, name in enumerate(header)}
        missing_columns = [col for col in requested if col not in index_by_name]
        if missing_columns:
            raise ValueError(f"{path} is missing requested column(s): {', '.join(missing_columns)}")
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split()
            if len(fields) != len(header):
                raise ValueError(f"{path}:{line_number} has {len(fields)} fields; expected {len(header)}")
            key = (fields[0], fields[1])
            if key not in sample_set:
                continue
            if key in seen:
                raise ValueError(f"duplicate FID/IID in {path}: {key[0]} {key[1]}")
            seen.add(key)
            values_by_key[key] = {col: fields[index_by_name[col]] for col in requested}
    missing_samples = [key for key in sample_keys if key not in seen]
    if missing_samples:
        fid, iid = missing_samples[0]
        raise ValueError(f"{path} is missing {len(missing_samples)} FAM sample(s); first missing FID/IID={fid} {iid}")
    return {col: [values_by_key[key][col] for key in sample_keys] for col in requested}


def read_covariate_basis(
    path: Path,
    *,
    samples: Sequence[FamSample],
    covar_cols: Sequence[str],
    q_covar_cols: Sequence[str],
    covar_max_levels: int,
) -> CovariateBasis:
    covar_cols_tuple = tuple(str(value) for value in covar_cols)
    q_covar_cols_tuple = tuple(str(value) for value in q_covar_cols)
    requested = (*covar_cols_tuple, *q_covar_cols_tuple)
    if not requested:
        return CovariateBasis.intercept_only(len(samples))
    table = _read_table_by_fam(path, samples, requested)
    columns: list[np.ndarray] = []
    for col in covar_cols_tuple:
        raw = table[col]
        if any(value in {"-9", "NA"} for value in raw):
            raise ValueError(f"categorical covariate {col} contains missing value -9/NA")
        levels = tuple(sorted(set(raw)))
        if len(levels) > int(covar_max_levels):
            raise ValueError(f"categorical covariate {col} has {len(levels)} levels, exceeding covarMaxLevels={covar_max_levels}")
        for level in levels:
            columns.append(np.asarray([1.0 if value == level else 0.0 for value in raw], dtype=np.float64))
    for col in q_covar_cols_tuple:
        raw = table[col]
        if any(value in {"-9", "NA"} for value in raw):
            raise ValueError(f"quantitative covariate {col} contains missing value -9/NA")
        try:
            columns.append(np.asarray([float(value) for value in raw], dtype=np.float64))
        except ValueError as exc:
            raise ValueError(f"quantitative covariate {col} contains a non-numeric value") from exc
    columns.append(np.ones(len(samples), dtype=np.float64))
    matrix = np.column_stack(columns)
    return CovariateBasis.from_matrix(
        matrix,
        covar_cols=covar_cols_tuple,
        q_covar_cols=q_covar_cols_tuple,
        covar_max_levels=int(covar_max_levels),
        source_path=path,
    )


def read_pheno_values(path: Path, samples: Sequence[FamSample], *, column: str = "PHENO") -> np.ndarray:
    table = _read_table_by_fam(path, samples, (column,))
    return np.asarray([float(value) for value in table[column]], dtype=np.float64)


@contextlib.contextmanager
def deterministic_grg_pheno_sim_rng(seed: int):
    legacy_state = np.random.get_state()
    original_default_rng = np.random.default_rng
    seed_sequence = np.random.SeedSequence(int(seed))
    child_counter = 0

    def patched_default_rng(seed_arg=None):
        nonlocal child_counter
        if seed_arg is not None:
            return original_default_rng(seed_arg)
        child = seed_sequence.spawn(child_counter + 1)[child_counter]
        child_counter += 1
        return original_default_rng(child)

    np.random.seed(int(seed))
    np.random.default_rng = patched_default_rng
    try:
        yield
    finally:
        np.random.default_rng = original_default_rng
        np.random.set_state(legacy_state)


def simulate_cached_phenotype(
    *,
    grg_paths: Sequence[Path],
    samples: Sequence[FamSample],
    pheno_path: Path,
    meta_path: Path,
    log_path: Path,
    seed: int,
    sim_h2: float,
    num_causal_per_file: int = 1000,
) -> tuple[np.ndarray, dict[str, float]]:
    expected_meta = _phenotype_cache_meta(
        seed=int(seed),
        sim_h2=float(sim_h2),
        samples=samples,
        grg_paths=grg_paths,
        num_causal_per_file=int(num_causal_per_file),
    )
    cached_meta = _json_load(meta_path)
    if Path(pheno_path).exists() and isinstance(cached_meta, dict) and cached_meta.get("cache_key") == expected_meta:
        y = read_pheno_values(pheno_path, samples)
        cached_metrics = cached_meta.get("metrics")
        metrics = {str(key): float(value) for key, value in cached_metrics.items()} if isinstance(cached_metrics, dict) else {}
        metrics.setdefault("phenotype.h2", float(sim_h2))
        metrics["phenotype.cache_hit"] = 1.0
        metrics["phenotype.var"] = float(np.var(y))
        return y, metrics

    import grg_pheno_sim.multi_grg_phenotype as multi_grg_phenotype

    out = Path(log_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            with deterministic_grg_pheno_sim_rng(int(seed)):
                df = multi_grg_phenotype.sim_phenotypes_multi_grg(
                    [str(Path(path)) for path in grg_paths],
                    model=multi_grg_phenotype.grg_causal_mutation_model("normal", mean=0, var=1),
                    num_causal_per_file=int(num_causal_per_file),
                    random_seed=int(seed),
                    heritability=float(sim_h2),
                    normalize_phenotype=True,
                    load_all_ram=False,
                )
    ids = np.asarray(df["individual_id"], dtype=np.int64)
    expected_ids = np.arange(len(samples), dtype=np.int64)
    if ids.shape != expected_ids.shape or not np.array_equal(ids, expected_ids):
        raise ValueError("grg_pheno_sim returned individual_id values that are not exactly 0..N-1 in order")
    y = np.asarray(df["phenotype"], dtype=np.float64)
    write_pheno(samples, y, pheno_path)
    metrics = {
        "phenotype.cache_hit": 0.0,
        "phenotype.h2": float(sim_h2),
        "phenotype.var": float(np.var(y)),
    }
    for source_col, metric_name in (("genetic_value", "phenotype.genetic_var"), ("environmental_noise", "phenotype.noise_var")):
        if source_col in df:
            values = np.asarray(df[source_col], dtype=np.float64)
            metrics[metric_name] = float(np.var(values))
    _json_dump_stable({"cache_key": expected_meta, "metrics": metrics}, meta_path)
    return y, metrics


def _standardize_pc(values: np.ndarray) -> np.ndarray:
    pc = np.asarray(values, dtype=np.float64).copy()
    if pc.size < 2:
        raise ValueError("at least two samples are required for PC standardization")
    max_idx = int(np.argmax(np.abs(pc)))
    if pc[max_idx] < 0.0:
        pc *= -1.0
    pc -= float(np.mean(pc))
    sd = float(np.std(pc, ddof=1))
    if sd <= 0.0:
        raise ValueError("principal component has nonpositive sample variance")
    pc /= sd
    return pc


def compute_pcs_from_grg(ops: BoltGrgOps, *, pcs: int, seed: int) -> np.ndarray:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator as CupyxLinearOperator
    from cupyx.scipy.sparse.linalg import eigsh as cupyx_eigsh

    k = int(pcs)
    if k < 1:
        raise ValueError("number of PCs must be positive")
    with ops.device:
        class SampleSpaceOperator(CupyxLinearOperator):
            def __init__(self) -> None:
                super().__init__(dtype=cp.dtype(DTYPE), shape=(ops.n, ops.n))

            def _matvec(self, values):
                vec = cp.asarray(values, dtype=DTYPE)
                out = cp.empty_like(vec)
                ops.apply_k(vec, exclude_chrom=None, out=out)
                return out

            def _matmat(self, values):
                cols = []
                for col in range(int(values.shape[1])):
                    cols.append(self._matvec(values[:, col]))
                return cp.column_stack(cols)

        operator = SampleSpaceOperator()
        rng = cp.random.default_rng(int(seed))
        v0 = rng.standard_normal((ops.n,), dtype=cp.float64)
        eigvals, eigvecs = cupyx_eigsh(operator, k=k, which="LA", tol=1e-6, maxiter=200, v0=v0)
        cp.cuda.runtime.deviceSynchronize()
    eigvals_host = cp.asnumpy(eigvals)
    eigvecs_host = cp.asnumpy(eigvecs)
    order = np.argsort(eigvals_host)[::-1]
    pcs_host = np.column_stack([_standardize_pc(eigvecs_host[:, idx]) for idx in order])
    return np.asarray(pcs_host, dtype=np.float64)


def compute_pcs_from_materialized_columns(ops: BoltGrgOps, *, pcs: int) -> np.ndarray:
    k = int(pcs)
    if k < 1:
        raise ValueError("number of PCs must be positive")
    if ops.m_proj < k:
        raise ValueError(f"cannot compute {k} PCs from {ops.m_proj} model SNPs")
    x = np.empty((int(ops.n), int(ops.m_proj)), dtype=np.float64)
    col = 0
    with ops.device:
        for manifest in ops.manifests:
            chrom = int(manifest.chrom)
            for variant in manifest.variants:
                if not is_model_variant(variant):
                    continue
                x[:, col] = ops.cp.asnumpy(ops.column(chrom, int(variant.local_idx)))
                col += 1
    if col != ops.m_proj:
        raise RuntimeError(f"materialized {col} SNP columns; expected {ops.m_proj}")
    u, _s, _vt = np.linalg.svd(x, full_matrices=False)
    return np.column_stack([_standardize_pc(u[:, idx]) for idx in range(k)])


def generate_sex_values(n: int, *, seed: int) -> np.ndarray:
    values = np.empty(int(n), dtype=np.int8)
    half = int(n) // 2
    values[:half] = 1
    values[half:] = 2
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x534558]))
    rng.shuffle(values)
    return values


def write_covariates(
    *,
    samples: Sequence[FamSample],
    pcs: np.ndarray,
    sex: np.ndarray,
    path: Path,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pc_arr = np.asarray(pcs, dtype=np.float64)
    sex_arr = np.asarray(sex)
    if pc_arr.shape != (len(samples), len(GENERATED_Q_COVARS)):
        raise ValueError(f"PC matrix has shape {pc_arr.shape}; expected {(len(samples), len(GENERATED_Q_COVARS))}")
    if sex_arr.shape != (len(samples),):
        raise ValueError(f"SEX vector has shape {sex_arr.shape}; expected {(len(samples),)}")
    with out.open("w", encoding="utf-8") as handle:
        handle.write("FID IID " + " ".join((*GENERATED_Q_COVARS, *GENERATED_COVARS)) + "\n")
        for row, sample in enumerate(samples):
            pc_values = " ".join(f"{float(pc_arr[row, col]):.17g}" for col in range(pc_arr.shape[1]))
            handle.write(f"{sample.fid} {sample.iid} {pc_values} {int(sex_arr[row])}\n")


def ensure_generated_covariates(
    *,
    ops: BoltGrgOps,
    samples: Sequence[FamSample],
    manifests: Sequence[ChromosomeManifest],
    source_files: Sequence[ChromosomeFiles],
    covar_path: Path,
    meta_path: Path,
    seed: int,
) -> Path:
    expected_meta = _covariate_cache_meta(seed=int(seed), samples=samples, manifests=manifests, source_files=source_files)
    if Path(covar_path).exists() and _json_load(meta_path) == expected_meta:
        read_covariate_basis(
            covar_path,
            samples=samples,
            covar_cols=GENERATED_COVARS,
            q_covar_cols=GENERATED_Q_COVARS,
            covar_max_levels=10,
        )
        return Path(covar_path)
    if int(ops.m_proj) <= 2_000:
        pcs = compute_pcs_from_materialized_columns(ops, pcs=len(GENERATED_Q_COVARS))
    else:
        pcs = compute_pcs_from_grg(ops, pcs=len(GENERATED_Q_COVARS), seed=int(seed))
    sex = generate_sex_values(len(samples), seed=int(seed))
    write_covariates(samples=samples, pcs=pcs, sex=sex, path=covar_path)
    _json_dump_stable(expected_meta, meta_path)
    return Path(covar_path)


def run_official_bolt(
    *,
    bolt_bin: Path,
    fam: Path,
    rewritten_bims: Sequence[Path],
    beds: Sequence[Path],
    pheno: Path,
    covar_file: Path | None,
    covar_cols: Sequence[str],
    q_covar_cols: Sequence[str],
    covar_max_levels: int,
    stats_file: Path,
    matched_snp_count: int,
    seed: int,
    num_threads: int,
    num_leave_out_chunks: int,
    num_calib_snps: int,
    h2_trials: int,
    cg_tol: float,
    max_iters: int,
) -> Path:
    if int(num_leave_out_chunks) < 1:
        raise ValueError("num_leave_out_chunks must be >= 1")
    cmd = [
        str(bolt_bin),
        "--fam",
        str(fam),
        "--phenoFile",
        str(pheno),
        "--phenoCol",
        "PHENO",
        "--lmmInfOnly",
        "--verboseStats",
        "--statsFile",
        str(stats_file),
        "--numThreads",
        str(int(num_threads)),
        "--numLeaveOutChunks",
        str(int(num_leave_out_chunks)),
        "--numCalibSnps",
        str(int(num_calib_snps)),
        "--h2EstMCtrials",
        str(int(h2_trials)),
        "--CGtol",
        str(float(cg_tol)),
        "--maxIters",
        str(int(max_iters)),
        "--seed",
        str(int(seed)),
        "--noMapCheck",
        "--maxModelSnps",
        str(max(1, int(matched_snp_count)) + 10),
    ]
    if covar_file is not None:
        cmd.extend(["--covarFile", str(covar_file)])
        for col in covar_cols:
            cmd.extend(["--covarCol", str(col)])
        for col in q_covar_cols:
            cmd.extend(["--qCovarCol", str(col)])
        cmd.extend(["--covarMaxLevels", str(int(covar_max_levels))])
    for bim in rewritten_bims:
        cmd.extend(["--bim", str(bim)])
    for bed in beds:
        cmd.extend(["--bed", str(bed)])
    LOGGER.info("Running official BOLT: %s", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    log_path = Path(stats_file).with_suffix(".log")
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"official BOLT failed with exit {result.returncode}; log written to {log_path}\n{result.stdout[-4000:]}")
    return Path(stats_file)


def _bolt_stream_float(value: float) -> str:
    return format(float(value), ".6g")


def _bolt_pvalue(value: float, stat: float) -> str:
    p_value = float(value)
    if p_value != 0.0:
        return f"{p_value:.1E}"
    log10p = math.log10(2.0) - math.log10(math.e) * float(stat) / 2.0 - 0.5 * math.log10(float(stat) * 2.0 * math.pi)
    exponent = math.floor(log10p)
    fraction = math.pow(10.0, log10p - exponent)
    if fraction >= 9.95:
        fraction = 1.0
        exponent += 1
    return f"{fraction:.1f}E{exponent:d}"


def write_grg_stats(
    *,
    ops: BoltGrgOps,
    y,
    residuals: dict[int, Any],
    fit: VarianceFit,
    calibration: CalibrationResult,
    path: Path,
) -> None:
    # Matches BOLT::printStatsHeader and BOLT::getSnpStats output shape.
    from scipy.stats import chi2 as scipy_chi2

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with ops.device:
        y_dev = ops.project(ops.cp.asarray(y, dtype=DTYPE).copy())
        y_norm2 = _dot(y_dev, y_dev)
    if y_norm2 <= 0.0:
        raise RuntimeError("phenotype has nonpositive projected norm")
    with out.open("w", encoding="utf-8") as handle:
        handle.write(
            "SNP\tCHR\tBP\tGENPOS\tALLELE1\tALLELE0\tA1FREQ\tF_MISS\t"
            "CHISQ_LINREG\tP_LINREG\tBETA\tSE\tCHISQ_BOLT_LMM_INF\tP_BOLT_LMM_INF\n"
        )
        for manifest in ops.manifests:
            chrom = int(manifest.chrom)
            state = ops.states[chrom]
            linreg_scores = ops.scores(chrom, y_dev)
            linreg_local_scores = ops.cp.asnumpy(linreg_scores[state.local_indices])
            lmm_scores = ops.scores(chrom, residuals[chrom])
            lmm_local_scores = ops.cp.asnumpy(lmm_scores[state.local_indices])
            local_indices = ops.cp.asnumpy(state.local_indices)
            linreg_score_by_local = {
                int(local): float(score) for local, score in zip(local_indices, linreg_local_scores, strict=True)
            }
            lmm_score_by_local = {
                int(local): float(score) for local, score in zip(local_indices, lmm_local_scores, strict=True)
            }
            vinv_scale = float(calibration.vinv_scale_by_chrom[chrom])
            if vinv_scale <= 0.0:
                raise RuntimeError(f"nonpositive VinvScaleFactor for chr{chrom}: {vinv_scale}")
            for variant in manifest.variants:
                if not is_model_variant(variant):
                    # Mirrors Bolt::getSnpStats(): bad SNPs keep BAD_SNP_STAT,
                    # p-value 1, beta 0, and SE streams as -nan.
                    linreg_chi2 = BOLT_BAD_SNP_STAT
                    linreg_p = 1.0
                    beta = 0.0
                    se_text = "-nan"
                    lmm_chi2 = BOLT_BAD_SNP_STAT
                    lmm_p = 1.0
                else:
                    linreg_score = linreg_score_by_local[int(variant.local_idx)]
                    linreg_chi2 = (linreg_score * linreg_score) / y_norm2 / float(variant.x_norm2) * float(ops.dim)
                    linreg_p = float(scipy_chi2.sf(linreg_chi2, df=1))

                    normalized_score = lmm_score_by_local[int(variant.local_idx)]
                    h_score_raw = normalized_score / float(variant.norm_scale)
                    vinv_score_raw = h_score_raw / float(fit.sigma_g2)
                    lmm_chi2 = ((vinv_score_raw / vinv_scale) ** 2) / float(variant.proj_norm2)
                    beta = vinv_score_raw / (float(variant.proj_norm2) * vinv_scale * vinv_scale)
                    se = 1.0 / (math.sqrt(float(variant.proj_norm2)) * vinv_scale)
                    se_text = _bolt_stream_float(se)
                    lmm_p = float(scipy_chi2.sf(lmm_chi2, df=1))
                handle.write(
                    "\t".join(
                        (
                            variant.snp_id,
                            str(variant.chrom),
                            str(variant.bp),
                            _bolt_stream_float(float(variant.genetic_pos)),
                            variant.allele1,
                            variant.allele0,
                            _bolt_stream_float(variant.a1freq),
                            _bolt_stream_float(float(variant.missing) / float(ops.dim)),
                            _bolt_stream_float(linreg_chi2),
                            _bolt_pvalue(linreg_p, linreg_chi2),
                            _bolt_stream_float(beta),
                            se_text,
                            _bolt_stream_float(lmm_chi2),
                            _bolt_pvalue(lmm_p, lmm_chi2),
                        )
                    )
                    + "\n"
                )


def read_stats_file(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        header: list[str] | None = None
        rows: list[dict[str, str]] = []
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if header is None:
                header = fields
                continue
            if len(fields) != len(header):
                raise ValueError(f"stats row in {path} has {len(fields)} fields; expected {len(header)}")
            rows.append(dict(zip(header, fields, strict=True)))
    if header is None:
        raise ValueError(f"stats file is empty: {path}")
    return rows


def _numeric_diff_summary(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    diff = actual - reference
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(reference), 1e-300)
    rel = abs_diff / denom
    return {
        "max_abs": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "median_abs": float(np.median(abs_diff)) if abs_diff.size else 0.0,
        "p99_abs": float(np.quantile(abs_diff, 0.99)) if abs_diff.size else 0.0,
        "max_rel": float(np.max(rel)) if rel.size else 0.0,
        "median_rel": float(np.median(rel)) if rel.size else 0.0,
        "p99_rel": float(np.quantile(rel, 0.99)) if rel.size else 0.0,
        "pearson": float(np.corrcoef(reference, actual)[0, 1]) if reference.size > 1 and np.std(reference) > 0 and np.std(actual) > 0 else float("nan"),
    }


def compare_stat_files(
    reference_path: Path,
    actual_path: Path,
    *,
    thresholds: StatComparisonThresholds | None = None,
) -> dict[str, Any]:
    ref_rows = read_stats_file(reference_path)
    got_rows = read_stats_file(actual_path)
    strict = thresholds is not None
    identity_cols = ("SNP", "CHR", "BP", "GENPOS", "ALLELE1", "ALLELE0", "F_MISS")
    numeric_cols = ("A1FREQ", "CHISQ_LINREG", "BETA", "SE", "CHISQ_BOLT_LMM_INF")
    pvalue_cols = ("P_LINREG", "P_BOLT_LMM_INF")
    required_cols = (*identity_cols, *numeric_cols, *pvalue_cols)
    summary: dict[str, Any] = {
        "status": "ok",
        "reference_rows": len(ref_rows),
        "actual_rows": len(got_rows),
        "identity_mismatches": [],
    }
    if strict:
        summary["passed"] = True
    if len(ref_rows) != len(got_rows):
        summary["status"] = "row_count_mismatch"
        if strict:
            summary["passed"] = False
        return summary

    ref_cols = set(ref_rows[0]) if ref_rows else set()
    got_cols = set(got_rows[0]) if got_rows else set()
    missing_columns = sorted(col for col in required_cols if col not in ref_cols or col not in got_cols)
    if missing_columns:
        summary["missing_columns"] = missing_columns
        if strict:
            summary["passed"] = False
        if any(col in identity_cols for col in missing_columns):
            summary["status"] = "missing_required_columns"
            summary["rows"] = len(ref_rows)
            return summary

    mismatches: list[dict[str, Any]] = []
    for idx, (ref, got) in enumerate(zip(ref_rows, got_rows, strict=True)):
        for col in identity_cols:
            if ref.get(col) != got.get(col):
                mismatches.append({"row": idx, "column": col, "reference": ref.get(col), "actual": got.get(col)})
                break
        if len(mismatches) >= 20:
            break
    if mismatches:
        summary["status"] = "identity_mismatch"
        summary["rows"] = len(ref_rows)
        summary["identity_mismatches"] = mismatches
        if strict:
            summary["passed"] = False
        return summary

    summary["rows"] = len(ref_rows)
    checks = (
        {}
        if thresholds is None
        else {
            "A1FREQ": (0.0, thresholds.freq_atol),
            "CHISQ_LINREG": (thresholds.chisq_rtol, thresholds.chisq_atol),
            "BETA": (thresholds.beta_rtol, thresholds.beta_atol),
            "SE": (thresholds.se_rtol, thresholds.se_atol),
            "CHISQ_BOLT_LMM_INF": (thresholds.chisq_rtol, thresholds.chisq_atol),
        }
    )
    for col in numeric_cols:
        if col in missing_columns:
            continue
        ref_values = np.asarray([float(row[col]) for row in ref_rows], dtype=np.float64)
        got_values = np.asarray([float(row[col]) for row in got_rows], dtype=np.float64)
        col_summary = _numeric_diff_summary(ref_values, got_values)
        if strict:
            rtol, atol = checks[col]
            col_passed = bool(np.allclose(got_values, ref_values, rtol=rtol, atol=atol, equal_nan=True))
            col_summary["passed"] = col_passed
            summary["passed"] = bool(summary["passed"] and col_passed)
        summary[col] = col_summary
    for col in pvalue_cols:
        if col in missing_columns:
            continue
        ref_values = np.asarray([float(row[col]) for row in ref_rows], dtype=np.float64)
        got_values = np.asarray([float(row[col]) for row in got_rows], dtype=np.float64)
        col_summary = _numeric_diff_summary(ref_values, got_values)
        col_summary["string_format_excluded_from_pass_fail"] = True
        summary[col] = col_summary
    return summary


@contextlib.contextmanager
def timed(timing: OrderedDict[str, float], name: str):
    start = perf_counter()
    try:
        yield
    finally:
        timing[str(name)] = timing.get(str(name), 0.0) + (perf_counter() - start)


def parse_chromosomes(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        raw_tokens = [token.strip() for token in value.split(",") if token.strip()]
    else:
        raw_tokens = [str(token) for token in value]
    if not raw_tokens:
        raise ValueError("at least one chromosome is required")
    chromosomes: list[int] = []
    seen: set[int] = set()
    for raw in raw_tokens:
        token = raw[3:] if raw.lower().startswith("chr") else raw
        try:
            chrom = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid chromosome {raw!r}; use comma-separated integer chromosomes") from exc
        if chrom < 1:
            raise ValueError(f"invalid chromosome {raw!r}; chromosomes must be positive integers")
        if chrom in seen:
            raise ValueError(f"duplicate chromosome {chrom}")
        seen.add(chrom)
        chromosomes.append(chrom)
    return tuple(chromosomes)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Compare GRG BOLT-LMM-inf output against official BOLT-LMM v2.5")
    parser.add_argument("--workDir", type=Path, required=True, help="Run workspace for generated inputs, logs, stats, and summary.json.")
    parser.add_argument("--artifactCache", type=Path, required=True, help="Cache directory for .grg_spmv artifacts.")
    parser.add_argument("--grgDir", type=Path, required=True, help=f"Directory containing source chromosome GRG files. Project-standard value: {STANDARD_GRG_DIR}.")
    parser.add_argument("--plinkDir", type=Path, required=True, help=f"Directory containing source PLINK BED/BIM/FAM files. Project-standard value: {STANDARD_PLINK_DIR}.")
    parser.add_argument("--chromosomes", required=True, help=f"Comma-separated integer chromosomes to compare. Project-standard value: {','.join(map(str, STANDARD_CHROMOSOMES))}.")
    parser.add_argument("--snpsPerChrom", type=int, required=True, help=f"Sampled singleton matches per chromosome; 0 uses all singleton matches. Recommended smoke value: {STANDARD_SNPS_PER_CHROM}.")
    parser.add_argument("--seed", type=int, required=True, help="Harness seed for SNP sampling, generated covariates, and phenotype cache keys.")
    parser.add_argument("--simH2", type=float, required=True, help="Target heritability for the simulated phenotype; must satisfy 0 < simH2 < 1.")
    parser.add_argument("--numThreads", type=int, required=True, help="Thread count passed to official BOLT-LMM.")
    parser.add_argument("--device", type=int, required=True, help="CUDA device index for the cuSPARSE runtime.")
    parser.add_argument("--vramBudgetBytes", type=int, required=True, help="cuSPARSE planner VRAM budget in bytes; 0 uses the required resident budget.")
    parser.add_argument("--ringBufferSize", type=int, required=True, help="cuSPARSE streaming ring slots; 0 requests a fully resident layout.")
    parser.add_argument("--logLevel", choices=("DEBUG", "INFO", "WARNING", "ERROR"), required=True, help="Python logging level for progress messages.")
    parser.add_argument("--covarFile", type=Path, default=None, help="Optional BOLT-style covariate file with FID/IID and requested covariate columns.")
    parser.add_argument("--covarCol", action="append", help="Categorical covariate column to pass to BOLT; may be repeated.")
    parser.add_argument("--qCovarCol", action="append", help="Quantitative covariate column to pass to BOLT; may be repeated.")
    parser.add_argument("--covarMaxLevels", type=int, required=True, help="Maximum allowed level count for each categorical --covarCol.")
    args = parser.parse_args(argv)
    args.covarCol = tuple(args.covarCol or ())
    args.qCovarCol = tuple(args.qCovarCol or ())
    if args.snpsPerChrom < 0:
        parser.error("--snpsPerChrom must be >= 0")
    if not (0.0 < args.simH2 < 1.0):
        parser.error("--simH2 must satisfy 0 < simH2 < 1")
    if args.numThreads < 1:
        parser.error("--numThreads must be >= 1")
    if args.covarMaxLevels < 1:
        parser.error("--covarMaxLevels must be >= 1")
    if args.covarFile is not None and not (args.covarCol or args.qCovarCol):
        parser.error("--covarFile requires at least one --covarCol or --qCovarCol")
    try:
        args.chromosomes = parse_chromosomes(args.chromosomes)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def prepare_inputs(args, timing: OrderedDict[str, float] | None = None) -> tuple[tuple[ChromosomeFiles, ...], tuple[ChromosomeManifest, ...], tuple[FamSample, ...], tuple[Path, ...], Path, Path, Path]:
    work_dir = Path(args.workDir).expanduser().resolve()
    input_dir = work_dir / "inputs"
    artifact_cache = Path(args.artifactCache).expanduser().resolve() if args.artifactCache is not None else work_dir / "artifacts"
    grg_dir = Path(args.grgDir).expanduser().resolve()
    plink_dir = Path(args.plinkDir).expanduser().resolve()
    chromosomes = tuple(int(chrom) for chrom in args.chromosomes)
    sampled_mode = int(args.snpsPerChrom) > 0
    LOGGER.info(
        "Preparing %s comparison inputs from GRG=%s PLINK=%s for chromosomes %s",
        f"{int(args.snpsPerChrom):,} sampled singleton GRG/BIM matches/chromosome" if sampled_mode else "all singleton GRG/BIM matches",
        grg_dir,
        plink_dir,
        ",".join(map(str, chromosomes)),
    )
    if timing is None:
        files = discover_chromosome_files(grg_dir, plink_dir, chromosomes)
        fam_samples = assert_same_fam(file.fam for file in files)
    else:
        with timed(timing, "input discovery"):
            files = discover_chromosome_files(grg_dir, plink_dir, chromosomes)
            fam_samples = assert_same_fam(file.fam for file in files)
    manifests: list[ChromosomeManifest] = []
    for file in files:
        with timed(timing, "SNP sampling/materialized PLINK") if timing is not None else contextlib.nullcontext():
            row_count, bim_repeated_bps = scan_bim_repeated_bps(file.bim)
            if sampled_mode:
                selected_matches = select_matched_bim_records(
                    file,
                    row_count=row_count,
                    bim_repeated_bps=bim_repeated_bps,
                    snps_per_chrom=int(args.snpsPerChrom),
                    seed=int(args.seed),
                )
                selected_records = tuple(record for record, _local_idx in selected_matches)
                selected_rows = np.asarray([record.row for record in selected_records], dtype=np.int64)
                LOGGER.info("chr%s selected %s/%s singleton-matched BIM rows", file.chrom, f"{len(selected_rows):,}", f"{row_count:,}")
                bed_out = input_dir / f"chr{file.chrom}.sample.bed"
                bim_out = input_dir / f"chr{file.chrom}.sample.bim"
                manifest = ChromosomeManifest(
                    chrom=int(file.chrom),
                    grg_path=file.grg,
                    bed_path=bed_out,
                    bim_path=bim_out,
                    fam_path=file.fam,
                    variants=tuple(
                        _variant_from_bim_record(file, record, local_idx=local_idx, bed_row=idx)
                        for idx, (record, local_idx) in enumerate(selected_matches)
                    ),
                )
            else:
                manifest = match_grg_to_bim(file, read_bim(file.bim), bim_repeated_bps=bim_repeated_bps)
                selected_rows = np.asarray([variant.bed_row for variant in manifest.variants], dtype=np.int64)
                bim_out = input_dir / f"chr{file.chrom}.full.bim"
                bed_out = input_dir / f"chr{file.chrom}.full.bed"
                manifest = replace(
                    manifest,
                    bed_path=bed_out,
                    bim_path=bim_out,
                    variants=tuple(replace(variant, bed_row=idx) for idx, variant in enumerate(manifest.variants)),
                )
                LOGGER.info("chr%s retained %s/%s singleton-matched BIM rows", file.chrom, f"{len(selected_rows):,}", f"{row_count:,}")
            write_rewritten_bim(manifest, bim_out)
            copied = write_selected_bed(
                file.bed,
                bed_out,
                selected_rows,
                n_individuals=len(fam_samples),
                source_variant_count=row_count,
            )
            LOGGER.info(
                "chr%s materialized filtered PLINK BED/BIM: %s rows, %.2f MiB copied",
                file.chrom,
                f"{len(selected_rows):,}",
                copied / float(1024 * 1024),
            )
        with timed(timing, "initial BED stats") if timing is not None else contextlib.nullcontext():
            stats = read_bed_snp_stats(manifest.bed_path, n_individuals=len(fam_samples), n_variants=len(manifest.variants))
            manifest = attach_bed_stats(manifest, stats, n_individuals=len(fam_samples))
        manifests.append(manifest)
    manifests_tuple = assign_global_indices(manifests)
    for manifest in manifests_tuple:
        if manifest.model_count <= 0:
            raise ValueError(f"chr{manifest.chrom} has no polymorphic matched SNPs")
    LOGGER.info("Using all %s eligible matched SNPs as model SNPs", f"{sum(manifest.model_count for manifest in manifests_tuple):,}")
    manifest_path = input_dir / "run_manifest.tsv"
    pheno_path = input_dir / "pheno.tsv"
    write_manifest(manifests_tuple, manifest_path)
    LOGGER.info("Ensuring GRG artifacts under %s", artifact_cache)
    with timed(timing, "artifact cache/conversion") if timing is not None else contextlib.nullcontext():
        artifacts = ensure_artifacts(files, artifact_cache)
    return files, manifests_tuple, fam_samples, artifacts, manifest_path, pheno_path, artifact_cache


def main(argv: list[str] | None = None) -> None:
    from pygrgl_spmv.backends.cusparse import CusparseRuntime, plan_cusparse_layout

    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.logLevel)), format="%(levelname)s:%(name)s:%(message)s")
    total_start = perf_counter()
    timing: OrderedDict[str, float] = OrderedDict()
    work_dir = Path(args.workDir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    files, manifests, fam_samples, artifacts, manifest_path, pheno_path, artifact_cache = prepare_inputs(args, timing)

    with timed(timing, "cuSPARSE layout"):
        LOGGER.info("Planning GRG/cuSPARSE artifact layout")
        layout = plan_cusparse_layout(
            artifacts=artifacts,
            pair=parse_cusparse_plan(DEFAULT_CUSPARSE_PLAN_NAME),
            dtype=DTYPE,
            requirements=bolt_runtime_requirements(),
            vram_budget_bytes=int(args.vramBudgetBytes),
            ring_buffer_size=int(args.ringBufferSize),
            allow_residency=True,
            device=int(args.device),
            stream=0,
        )

    input_dir = work_dir / "inputs"
    generated_covar_path = input_dir / "covariates.tsv"
    covar_meta_path = input_dir / "covariates.meta.json"
    pheno_meta_path = input_dir / "pheno.meta.json"
    pheno_log_path = input_dir / "pheno_sim.log"

    if args.covarFile is None:
        covar_path = generated_covar_path
        covar_cols = GENERATED_COVARS
        q_covar_cols = GENERATED_Q_COVARS
        with timed(timing, "PCA/covariate cache"):
            expected_covar_meta = _covariate_cache_meta(
                seed=int(args.seed),
                samples=fam_samples,
                manifests=manifests,
                source_files=files,
            )
            if Path(covar_path).exists() and _json_load(covar_meta_path) == expected_covar_meta:
                read_covariate_basis(
                    covar_path,
                    samples=fam_samples,
                    covar_cols=GENERATED_COVARS,
                    q_covar_cols=GENERATED_Q_COVARS,
                    covar_max_levels=10,
                )
            else:
                with CusparseRuntime(layout) as runtime:
                    with BoltGrgOps(runtime, manifests, CovariateBasis.intercept_only(len(fam_samples))) as pca_ops:
                        ensure_generated_covariates(
                            ops=pca_ops,
                            samples=fam_samples,
                            manifests=manifests,
                            source_files=files,
                            covar_path=covar_path,
                            meta_path=covar_meta_path,
                            seed=int(args.seed),
                        )
    else:
        covar_path = Path(args.covarFile).expanduser().resolve()
        covar_cols = tuple(str(value) for value in args.covarCol)
        q_covar_cols = tuple(str(value) for value in args.qCovarCol)
        with timed(timing, "PCA/covariate cache"):
            read_covariate_basis(
                covar_path,
                samples=fam_samples,
                covar_cols=covar_cols,
                q_covar_cols=q_covar_cols,
                covar_max_levels=int(args.covarMaxLevels),
            )

    covariates = read_covariate_basis(
        covar_path,
        samples=fam_samples,
        covar_cols=covar_cols,
        q_covar_cols=q_covar_cols,
        covar_max_levels=int(args.covarMaxLevels),
    )
    projected_manifests: list[ChromosomeManifest] = []
    with timed(timing, "covariate projection stats"):
        for file, manifest in zip(files, manifests, strict=True):
            projected_manifests.append(
                attach_projected_bed_stats(
                    manifest,
                    covariates=covariates,
                    n_individuals=len(fam_samples),
                    n_variants=len(manifest.variants),
                )
            )
    manifests = assign_global_indices(projected_manifests)
    write_manifest(manifests, manifest_path)
    model_snps_count = _require_projected_model_snps(manifests)
    rewritten_bims = tuple(manifest.bim_path for manifest in manifests)
    beds = tuple(manifest.bed_path for manifest in manifests)

    with timed(timing, "phenotype cache/simulation"):
        y, phenotype_metrics = simulate_cached_phenotype(
            grg_paths=tuple(file.grg for file in files),
            samples=fam_samples,
            pheno_path=pheno_path,
            meta_path=pheno_meta_path,
            log_path=pheno_log_path,
            seed=int(args.seed),
            sim_h2=float(args.simH2),
        )

    with timed(timing, "BOLT build/preflight"):
        LOGGER.info("Checking cached BOLT-LMM build")
        from scripts.bolt_lmm_inf.build_bolt import ensure_cached_bolt

        bolt_bin = ensure_cached_bolt(cache_dir=artifact_cache / "bolt_lmm_cache", jobs=max(1, min(8, os.cpu_count() or 1)))

    grg_stats = work_dir / "grg.stats"
    bolt_stats = work_dir / "bolt.stats"
    local_metrics: dict[str, float] = {}

    with timed(timing, "official BOLT run"):
        LOGGER.info("Starting official BOLT-LMM --lmmInfOnly run")
        run_official_bolt(
            bolt_bin=bolt_bin,
            fam=files[0].fam,
            rewritten_bims=rewritten_bims,
            beds=beds,
            pheno=pheno_path,
            covar_file=covar_path,
            covar_cols=covar_cols,
            q_covar_cols=q_covar_cols,
            covar_max_levels=int(args.covarMaxLevels),
            stats_file=bolt_stats,
            matched_snp_count=sum(len(manifest.variants) for manifest in manifests),
            seed=BOLT_RANDOM_SEED,
            num_threads=int(args.numThreads),
            num_leave_out_chunks=len(manifests),
            num_calib_snps=min(DEFAULT_NUM_CALIB_SNPS, model_snps_count),
            h2_trials=DEFAULT_H2_EST_MC_TRIALS,
            cg_tol=DEFAULT_CG_TOL,
            max_iters=DEFAULT_MAX_ITERS,
        )

    runtime = None
    ops = None
    try:
        with timed(timing, "GRG runtime init"):
            runtime = CusparseRuntime(layout)
            runtime.__enter__()
            ops = BoltGrgOps(runtime, manifests, covariates)
            ops.__enter__()
        LOGGER.info("Starting GRG BOLT-LMM-inf run")
        reml_stats = CgStats()
        calib_stats = CgStats()
        with timed(timing, "REML fit"):
            fit = fit_bolt_variance_components(
                ops,
                y,
                mc_trials=DEFAULT_H2_EST_MC_TRIALS,
                seed=BOLT_RANDOM_SEED,
                rel_tol=10.0 * DEFAULT_CG_TOL,
                max_iter=DEFAULT_MAX_ITERS,
                stats=reml_stats,
            )
        residuals: dict[int, Any] = {}
        with timed(timing, "calibration"):
            calibration = calibrate_lmm_inf(
                ops,
                y,
                residuals,
                fit=fit,
                count=min(DEFAULT_NUM_CALIB_SNPS, model_snps_count),
                seed=BOLT_RANDOM_SEED,
                rel_tol=DEFAULT_CG_TOL,
                max_iter=DEFAULT_MAX_ITERS,
                stats=calib_stats,
            )
        with timed(timing, "GRG stats write"):
            write_grg_stats(ops=ops, y=y, residuals=residuals, fit=fit, calibration=calibration, path=grg_stats)
        local_metrics = {
            "grg.log_delta": fit.log_delta,
            "grg.sigma_g2": fit.sigma_g2,
            "grg.sigma_e2": fit.sigma_e2,
            "grg.h2": fit.h2,
            "grg.delta": fit.delta,
            "grg.calibration.factor": calibration.factor,
            "grg.calibration.std": calibration.std,
            "grg.calibration.ratio_of_medians": calibration.ratio_of_medians,
            "grg.calibration.median_of_ratios": calibration.median_of_ratios,
            "grg.calibration.tried_snps": calibration.tried_snps,
            "grg.cg.reml.solves": reml_stats.solves,
            "grg.cg.calibration.solves": calib_stats.solves,
            "grg.Cindep": covariates.cindep,
            "grg.dim": covariates.dim,
            "grg.Xfro2": ops.xfro2,
        }
    finally:
        if ops is not None:
            ops.__exit__(None, None, None)
        if runtime is not None:
            runtime.__exit__(None, None, None)

    with timed(timing, "comparison"):
        if not bolt_stats.exists():
            raise FileNotFoundError(bolt_stats)
        if not grg_stats.exists():
            raise FileNotFoundError(grg_stats)
        comparison = compare_stat_files(bolt_stats, grg_stats, thresholds=None)
    print("Comparison:", file=sys.stderr)
    pprint.pprint(comparison, stream=sys.stderr, sort_dicts=True, width=120)

    timing["total"] = perf_counter() - total_start
    official_bolt_seconds = timing.get("official BOLT run", 0.0)
    grg_bolt_seconds = sum(
        timing.get(name, 0.0)
        for name in ("GRG runtime init", "REML fit", "calibration", "GRG stats write")
    )
    timing["official_bolt_seconds"] = official_bolt_seconds
    timing["grg_bolt_lmm_inf_seconds"] = grg_bolt_seconds
    print("Timing (seconds)", file=sys.stderr)
    for name, seconds in timing.items():
        print(f"  {name}: {seconds:.3f}", file=sys.stderr)

    summary_json = work_dir / "summary.json"
    summary: dict[str, Any] = {
        "work_dir": str(work_dir),
        "summary_json": str(summary_json),
        "bolt_bin": None if bolt_bin is None else str(bolt_bin),
        "manifest": str(manifest_path),
        "pheno_file": str(pheno_path),
        "pheno_meta": str(pheno_meta_path),
        "pheno_log": str(pheno_log_path),
        "covar_file": str(covar_path),
        "covar_meta": None if args.covarFile is not None else str(covar_meta_path),
        "covar_cols": list(covar_cols),
        "q_covar_cols": list(q_covar_cols),
        "covarMaxLevels": int(args.covarMaxLevels),
        "Cindep": int(covariates.cindep),
        "bolt_stats": str(bolt_stats),
        "bolt_log": str(bolt_stats.with_suffix(".log")),
        "grg_stats": str(grg_stats),
        "grgDir": str(Path(args.grgDir).expanduser().resolve()),
        "plinkDir": str(Path(args.plinkDir).expanduser().resolve()),
        "chromosomes": [manifest.chrom for manifest in manifests],
        "numLeaveOutChunks": len(manifests),
        "matched_snps": sum(len(manifest.variants) for manifest in manifests),
        "model_snps_count": model_snps_count,
        "snpsPerChrom": int(args.snpsPerChrom),
        "elapsed_seconds": timing["total"],
        "official_bolt_seconds": official_bolt_seconds,
        "grg_bolt_lmm_inf_seconds": grg_bolt_seconds,
        "timing": timing,
        "phenotype": phenotype_metrics,
        "local": local_metrics,
    }
    summary["comparison"] = comparison
    summary_json.write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, allow_nan=True), flush=True)
