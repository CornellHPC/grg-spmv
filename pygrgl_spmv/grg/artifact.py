"""Artifact persistence, scanning, and incremental block loading."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import logging
import os
from pathlib import Path
import re
import tempfile
import time
import warnings

import numpy as np

from pygrgl_spmv.grg.compile import CompiledOperatorState, _invert_permutation
from pygrgl_spmv.grg.sparse import binary_csr_from_parts

GRG_SPMV_FORMAT_MAGIC = "grg_spmv"
GRG_SPMV_FORMAT_VERSION = 6
_FORMAT_MAGIC_KEY = "grg_spmv_magic"
_FORMAT_VERSION_KEY = "grg_spmv_format_version"
_BLOCK_SHAPE_RE = re.compile(r"^A_blocks_(\d+)_(\d+)_shape$")
_LOGGER = logging.getLogger(__name__)


def artifact_path_for_grg(grg_path: Path, artifact_root: Path) -> Path:
    resolved = grg_path.expanduser().resolve()
    if resolved.is_absolute():
        if resolved.drive:
            drive = resolved.drive.replace(":", "")
            rel_parts = ["_drive_" + drive, *resolved.parts[1:]]
        else:
            rel_parts = ["_abs", *resolved.parts[1:]]
    else:
        rel_parts = ["_rel", *resolved.parts]
    return artifact_root.joinpath(*rel_parts).with_suffix(".grg_spmv")


@dataclass(frozen=True)
class ArtifactBlockScan:
    dst_level: int
    src_level: int
    shape: tuple[int, int]
    nnz: int
    indices_dtype: np.dtype
    indptr_dtype: np.dtype


@dataclass(frozen=True)
class ArtifactBlock:
    dst_level: int
    src_level: int
    shape: tuple[int, int]
    indices: np.ndarray
    indptr: np.ndarray

    @property
    def nnz(self) -> int:
        return int(self.indices.size)


@dataclass(frozen=True)
class ArtifactScan:
    path: Path
    num_samples: int
    num_mutations: int
    num_nodes: int
    num_individuals: int
    num_edges: int
    ploidy: int
    has_missing_data: bool
    has_xtx_bias: bool
    level_offsets: np.ndarray
    level_sizes: tuple[int, ...]
    num_levels: int
    selector_mut_nnz: int
    selector_miss_nnz: int
    has_individual_coals: bool
    sel_mut_nnz_by_level: np.ndarray
    sel_miss_nnz_by_level: np.ndarray
    init_vector_up_bias_size: int
    init_vector_down_bias_size: int
    blocks: tuple[ArtifactBlockScan, ...]

    @property
    def max_block_nnz(self) -> int:
        return max((block.nnz for block in self.blocks), default=0)


def save_grg_spmv(state: CompiledOperatorState, artifact_path) -> None:
    if state.A_blocks is None:
        raise RuntimeError("compiled operator blocks are required for .grg_spmv save")
    if state.init_vector_up_bias is None or state.init_vector_down_bias is None:
        raise RuntimeError("init vector bias cache must be populated before .grg_spmv save")

    t0 = time.perf_counter()
    sel_mut_indices = np.asarray(state.sel_mut.indices)
    sel_miss_indices = np.asarray(state.sel_miss.indices)
    level_offsets_arr = np.asarray(state.level_offsets)
    level_spans = [(int(level_offsets_arr[i]), int(level_offsets_arr[i + 1])) for i in range(len(level_offsets_arr) - 1)]
    sel_mut_nnz_by_level = np.asarray([int(state.sel_mut[:, lo:hi].nnz) for lo, hi in level_spans], dtype=np.int64)
    sel_miss_nnz_by_level = np.asarray([int(state.sel_miss[:, lo:hi].nnz) for lo, hi in level_spans], dtype=np.int64)
    save_dict = {
        _FORMAT_MAGIC_KEY: np.asarray(GRG_SPMV_FORMAT_MAGIC),
        _FORMAT_VERSION_KEY: np.asarray(GRG_SPMV_FORMAT_VERSION, dtype=np.int32),
        "num_samples": np.asarray(state.num_samples, dtype=np.int64),
        "num_mutations": np.asarray(state.num_mutations, dtype=np.int64),
        "num_nodes": np.asarray(state.num_nodes, dtype=np.int64),
        "ploidy": np.asarray(state.ploidy, dtype=np.int64),
        "num_individuals": np.asarray(state.num_individuals, dtype=np.int64),
        "num_edges": np.asarray(state.num_edges, dtype=np.int64),
        "has_missing_data": np.asarray(state.has_missing_data, dtype=bool),
        "level_offsets": np.asarray(state.level_offsets),
        "node_perm": np.asarray(state.node_perm),
        "sample_to_individual": np.asarray(state.sample_to_individual),
        "mutation_positions": np.asarray(state.mutation_positions, dtype=np.float64),
        "mutation_times": np.asarray(state.mutation_times, dtype=np.float64),
        "mutation_alleles": np.asarray(state.mutation_alleles),
        "mutation_allele_offsets": np.asarray(state.mutation_allele_offsets),
        "mutation_ref_alleles": np.asarray(state.mutation_ref_alleles),
        "mutation_ref_allele_offsets": np.asarray(state.mutation_ref_allele_offsets),
        "has_individual_coals": np.asarray(state.coalescence_counts is not None, dtype=bool),
        "init_vector_up_bias": np.asarray(state.init_vector_up_bias),
        "init_vector_down_bias": np.asarray(state.init_vector_down_bias),
        "has_xtx_bias": np.asarray(
            state.init_xtx_up_bias is not None and state.init_xtx_down_bias is not None,
            dtype=bool,
        ),
        "sel_mut_indices": sel_mut_indices,
        "sel_mut_indptr": np.asarray(state.sel_mut.indptr),
        "sel_mut_nnz": np.asarray(sel_mut_indices.size, dtype=np.int64),
        "sel_mut_nnz_by_level": sel_mut_nnz_by_level,
        "sel_miss_indices": sel_miss_indices,
        "sel_miss_indptr": np.asarray(state.sel_miss.indptr),
        "sel_miss_nnz": np.asarray(sel_miss_indices.size, dtype=np.int64),
        "sel_miss_nnz_by_level": sel_miss_nnz_by_level,
        "init_vector_up_bias_size": np.asarray(np.asarray(state.init_vector_up_bias).size, dtype=np.int64),
        "init_vector_down_bias_size": np.asarray(np.asarray(state.init_vector_down_bias).size, dtype=np.int64),
    }
    if state.coalescence_counts is not None:
        save_dict["coalescence_counts"] = np.asarray(state.coalescence_counts, dtype=np.int64)
    if state.init_xtx_up_bias is not None:
        save_dict["init_xtx_up_bias"] = np.asarray(state.init_xtx_up_bias)
    if state.init_xtx_down_bias is not None:
        save_dict["init_xtx_down_bias"] = np.asarray(state.init_xtx_down_bias)
    num_blocks = 0
    for dst_level, blocks in enumerate(state.A_blocks):
        for src_level, block in enumerate(blocks):
            indices_arr = np.asarray(block.indices)
            indptr_arr = np.asarray(block.indptr)
            save_dict[f"A_blocks_{dst_level}_{src_level}_indices"] = indices_arr
            save_dict[f"A_blocks_{dst_level}_{src_level}_indptr"] = indptr_arr
            save_dict[f"A_blocks_{dst_level}_{src_level}_shape"] = np.asarray(block.shape, dtype=np.int64)
            save_dict[f"A_blocks_{dst_level}_{src_level}_nnz"] = np.asarray(indices_arr.size, dtype=np.int64)
            save_dict[f"A_blocks_{dst_level}_{src_level}_indices_dtype"] = np.asarray(str(indices_arr.dtype))
            save_dict[f"A_blocks_{dst_level}_{src_level}_indptr_dtype"] = np.asarray(str(indptr_arr.dtype))
            num_blocks += 1

    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(raw_tmp_path)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **save_dict)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    _scan_grg_spmv_cached.cache_clear()
    _LOGGER.debug(
        "save_grg_spmv path=%s blocks=%d size=%.1fMB elapsed=%.3fs",
        path, num_blocks, path.stat().st_size / 1e6, time.perf_counter() - t0,
    )


def _open_archive(artifact_path) -> np.lib.npyio.NpzFile:
    return np.load(artifact_path, allow_pickle=False)


def _validate_archive(data: np.lib.npyio.NpzFile, artifact_path) -> None:
    if _FORMAT_MAGIC_KEY not in data.files:
        raise ValueError(f"unsupported .grg_spmv artifact in {artifact_path}: missing {_FORMAT_MAGIC_KEY}")
    magic = str(np.asarray(data[_FORMAT_MAGIC_KEY]).item())
    if magic != GRG_SPMV_FORMAT_MAGIC:
        raise ValueError(f"unsupported .grg_spmv artifact in {artifact_path}: got magic {magic!r}")
    if _FORMAT_VERSION_KEY not in data.files:
        raise ValueError(f"unsupported .grg_spmv artifact in {artifact_path}: missing {_FORMAT_VERSION_KEY}")
    version = int(np.asarray(data[_FORMAT_VERSION_KEY]).item())
    if version != GRG_SPMV_FORMAT_VERSION:
        raise ValueError(
            f"unsupported .grg_spmv artifact in {artifact_path}: got version {version}, expected {GRG_SPMV_FORMAT_VERSION}"
        )


def _log_struct_array(key: str, arr: np.ndarray) -> None:
    if not _LOGGER.isEnabledFor(logging.DEBUG):
        return
    _LOGGER.debug(
        "artifact array key=%s dtype=%s shape=%s nbytes=%d",
        key,
        np.asarray(arr).dtype,
        tuple(int(v) for v in np.asarray(arr).shape),
        int(np.asarray(arr).nbytes),
    )


def _load_struct_array(data: np.lib.npyio.NpzFile, key: str, *, non_negative: bool = True) -> np.ndarray:
    arr = np.asarray(data[key])
    if arr.dtype not in {np.dtype(np.int32), np.dtype(np.int64)}:
        raise ValueError(f"unsupported structural dtype for {key}: {arr.dtype}")
    if non_negative and arr.size and int(arr.min()) < 0:
        raise ValueError(f"structural array {key} must be non-negative")
    _log_struct_array(key, arr)
    return arr


def _scan_block_keys(data: np.lib.npyio.NpzFile) -> list[tuple[int, int]]:
    keys: list[tuple[int, int]] = []
    for key in data.files:
        match = _BLOCK_SHAPE_RE.fullmatch(key)
        if match is None:
            continue
        keys.append((int(match.group(1)), int(match.group(2))))
    keys.sort()
    return keys


@functools.cache
def _scan_grg_spmv_cached(artifact_path: Path) -> ArtifactScan:
    t0 = time.perf_counter()
    with _open_archive(artifact_path) as data:
        _validate_archive(data, artifact_path)
        level_offsets = _load_struct_array(data, "level_offsets")
        level_offsets.setflags(write=False)
        level_sizes = tuple(
            int(level_offsets[level + 1]) - int(level_offsets[level])
            for level in range(len(level_offsets) - 1)
        )
        blocks: list[ArtifactBlockScan] = []
        for dst_level, src_level in _scan_block_keys(data):
            shape = tuple(int(v) for v in np.asarray(data[f"A_blocks_{dst_level}_{src_level}_shape"], dtype=np.int64))
            nnz = int(np.asarray(data[f"A_blocks_{dst_level}_{src_level}_nnz"]).item())
            indices_dtype = np.dtype(str(np.asarray(data[f"A_blocks_{dst_level}_{src_level}_indices_dtype"]).item()))
            indptr_dtype = np.dtype(str(np.asarray(data[f"A_blocks_{dst_level}_{src_level}_indptr_dtype"]).item()))
            blocks.append(
                ArtifactBlockScan(
                    dst_level=dst_level,
                    src_level=src_level,
                    shape=(shape[0], shape[1]),
                    nnz=nnz,
                    indices_dtype=indices_dtype,
                    indptr_dtype=indptr_dtype,
                )
            )
        sel_mut_nnz_by_level = np.asarray(data["sel_mut_nnz_by_level"], dtype=np.int64)
        sel_miss_nnz_by_level = np.asarray(data["sel_miss_nnz_by_level"], dtype=np.int64)
        sel_mut_nnz_by_level.setflags(write=False)
        sel_miss_nnz_by_level.setflags(write=False)
        result = ArtifactScan(
            path=artifact_path,
            num_samples=int(np.asarray(data["num_samples"]).item()),
            num_mutations=int(np.asarray(data["num_mutations"]).item()),
            num_nodes=int(np.asarray(data["num_nodes"]).item()),
            num_individuals=int(np.asarray(data["num_individuals"]).item()),
            num_edges=int(np.asarray(data["num_edges"]).item()),
            ploidy=int(np.asarray(data["ploidy"]).item()),
            has_missing_data=bool(np.asarray(data["has_missing_data"]).item()),
            has_xtx_bias=bool(np.asarray(data["has_xtx_bias"]).item()),
            level_offsets=np.asarray(level_offsets),
            level_sizes=level_sizes,
            num_levels=len(level_sizes),
            selector_mut_nnz=int(np.asarray(data["sel_mut_nnz"]).item()),
            selector_miss_nnz=int(np.asarray(data["sel_miss_nnz"]).item()),
            has_individual_coals=bool(np.asarray(data["has_individual_coals"]).item()),
            sel_mut_nnz_by_level=sel_mut_nnz_by_level,
            sel_miss_nnz_by_level=sel_miss_nnz_by_level,
            init_vector_up_bias_size=int(np.asarray(data["init_vector_up_bias_size"]).item()),
            init_vector_down_bias_size=int(np.asarray(data["init_vector_down_bias_size"]).item()),
            blocks=tuple(blocks),
        )
    _LOGGER.debug(
        "scan_grg_spmv path=%s blocks=%d elapsed=%.3fs",
        artifact_path, len(blocks), time.perf_counter() - t0,
    )
    return result


def scan_grg_spmv(path) -> ArtifactScan:
    artifact_path = Path(path).resolve()
    return _scan_grg_spmv_cached(artifact_path)


def iter_artifact_blocks(path):
    artifact_path = Path(path)
    with _open_archive(artifact_path) as data:
        _validate_archive(data, artifact_path)
        for dst_level, src_level in _scan_block_keys(data):
            indices = _load_struct_array(data, f"A_blocks_{dst_level}_{src_level}_indices")
            indptr = _load_struct_array(data, f"A_blocks_{dst_level}_{src_level}_indptr")
            shape = tuple(int(v) for v in np.asarray(data[f"A_blocks_{dst_level}_{src_level}_shape"], dtype=np.int64))
            yield ArtifactBlock(
                dst_level=dst_level,
                src_level=src_level,
                shape=(shape[0], shape[1]),
                indices=indices,
                indptr=indptr,
            )


def _load_grg_spmv_host(artifact_path, dtype) -> CompiledOperatorState:
    t0 = time.perf_counter()
    with _open_archive(artifact_path) as data:
        _validate_archive(data, artifact_path)

        num_samples = int(np.asarray(data["num_samples"]).item())
        num_mutations = int(np.asarray(data["num_mutations"]).item())
        num_nodes = int(np.asarray(data["num_nodes"]).item())
        level_offsets = _load_struct_array(data, "level_offsets")
        node_perm = _load_struct_array(data, "node_perm")
        sample_to_individual = _load_struct_array(data, "sample_to_individual")
        init_vector_up_bias = np.asarray(data["init_vector_up_bias"], dtype=dtype)
        init_vector_down_bias = np.asarray(data["init_vector_down_bias"], dtype=dtype)

        coalescence_counts = None
        if bool(np.asarray(data["has_individual_coals"]).item()):
            coalescence_counts = np.asarray(data["coalescence_counts"], dtype=np.int64)

        init_xtx_up_bias = None
        init_xtx_down_bias = None
        if bool(np.asarray(data["has_xtx_bias"]).item()):
            init_xtx_up_bias = np.asarray(data["init_xtx_up_bias"], dtype=dtype)
            init_xtx_down_bias = np.asarray(data["init_xtx_down_bias"], dtype=dtype)

        inv_node_perm = _invert_permutation(node_perm)
        sel_mut = binary_csr_from_parts(
            indices=_load_struct_array(data, "sel_mut_indices"),
            indptr=_load_struct_array(data, "sel_mut_indptr"),
            shape=(num_mutations, num_nodes),
        )
        sel_miss = binary_csr_from_parts(
            indices=_load_struct_array(data, "sel_miss_indices"),
            indptr=_load_struct_array(data, "sel_miss_indptr"),
            shape=(num_mutations, num_nodes),
        )

        result = CompiledOperatorState(
            A_blocks=None,
            level_offsets=level_offsets,
            node_perm=node_perm,
            inv_node_perm=inv_node_perm,
            sel_mut=sel_mut,
            sel_miss=sel_miss,
            num_samples=num_samples,
            num_mutations=num_mutations,
            num_nodes=num_nodes,
            ploidy=int(np.asarray(data["ploidy"]).item()),
            num_individuals=int(np.asarray(data["num_individuals"]).item()),
            num_edges=int(np.asarray(data["num_edges"]).item()),
            has_missing_data=bool(np.asarray(data["has_missing_data"]).item()),
            sample_to_individual=sample_to_individual,
            mutation_positions=np.asarray(data["mutation_positions"], dtype=np.float64),
            mutation_times=np.asarray(data["mutation_times"], dtype=np.float64),
            mutation_alleles=np.asarray(data["mutation_alleles"]),
            mutation_allele_offsets=np.asarray(data["mutation_allele_offsets"]),
            mutation_ref_alleles=np.asarray(data["mutation_ref_alleles"]),
            mutation_ref_allele_offsets=np.asarray(data["mutation_ref_allele_offsets"]),
            coalescence_counts=coalescence_counts,
            init_vector_up_bias=init_vector_up_bias,
            init_vector_down_bias=init_vector_down_bias,
            init_xtx_up_bias=init_xtx_up_bias,
            init_xtx_down_bias=init_xtx_down_bias,
        )
    _LOGGER.debug(
        "_load_grg_spmv_host path=%s dtype=%s elapsed=%.3fs",
        artifact_path, dtype, time.perf_counter() - t0,
    )
    return result


def load_grg_spmv(artifact_path, dtype) -> CompiledOperatorState:
    t0 = time.perf_counter()
    header = _load_grg_spmv_host(artifact_path, dtype)
    num_levels = len(header.level_offsets) - 1
    grids: list[list[object]] = [[] for _ in range(num_levels)]
    wasted_struct_bytes = False
    num_blocks_loaded = 0
    for block in iter_artifact_blocks(artifact_path):
        num_blocks_loaded += 1
        while len(grids[block.dst_level]) < block.src_level + 1:
            grids[block.dst_level].append(None)
        loaded = binary_csr_from_parts(
            indices=np.asarray(block.indices),
            indptr=np.asarray(block.indptr),
            shape=block.shape,
            shared_data=True,
        )
        if (
            np.dtype(loaded.indices.dtype).itemsize < np.dtype(block.indices.dtype).itemsize
            or np.dtype(loaded.indptr.dtype).itemsize < np.dtype(block.indptr.dtype).itemsize
        ):
            wasted_struct_bytes = True
        grids[block.dst_level][block.src_level] = loaded
    for dst_level in range(num_levels):
        row = grids[dst_level]
        if len(row) < dst_level:
            row.extend([None] * (dst_level - len(row)))
        grids[dst_level] = [
            binary_csr_from_parts(
                indices=np.empty(0, dtype=np.int32),
                indptr=np.zeros(int(header.level_offsets[dst_level + 1] - header.level_offsets[dst_level]) + 1, dtype=np.int32),
                shape=(
                    int(header.level_offsets[dst_level + 1] - header.level_offsets[dst_level]),
                    int(header.level_offsets[src_level + 1] - header.level_offsets[src_level]),
                ),
            )
            if block is None
            else block
            for src_level, block in enumerate(row)
        ]
    header.A_blocks = grids
    if wasted_struct_bytes:
        warnings.warn(
            (
                f"{artifact_path} stores some structural arrays wider than necessary; "
                "load_grg_spmv() rebuilt smaller in-memory dtypes, so the artifact uses more disk space than needed"
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    _LOGGER.debug(
        "load_grg_spmv path=%s dtype=%s blocks=%d elapsed=%.3fs",
        artifact_path, dtype, num_blocks_loaded, time.perf_counter() - t0,
    )
    return header


__all__ = [
    "ArtifactBlock",
    "ArtifactBlockScan",
    "ArtifactScan",
    "GRG_SPMV_FORMAT_MAGIC",
    "GRG_SPMV_FORMAT_VERSION",
    "iter_artifact_blocks",
    "load_grg_spmv",
    "save_grg_spmv",
    "scan_grg_spmv",
]
