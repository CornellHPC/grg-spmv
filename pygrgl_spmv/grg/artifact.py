"""Artifact persistence, scanning, and incremental block loading."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import logging
import os
from pathlib import Path
import tempfile
import time
import warnings

import numpy as np

from pygrgl_spmv.grg.compile import CompiledOperatorState, _invert_permutation
from pygrgl_spmv.grg.sparse import binary_csr_from_parts

GRG_SPMV_FORMAT_MAGIC = "grg_spmv"
GRG_SPMV_FORMAT_VERSION = 8
_FORMAT_MAGIC_KEY = "grg_spmv_magic"
_FORMAT_VERSION_KEY = "grg_spmv_format_version"
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
    selector_nnz_by_level: np.ndarray
    node_perm_dtype: np.dtype
    sample_to_individual_dtype: np.dtype
    blocks: tuple[ArtifactBlockScan, ...]

    @property
    def max_block_nnz(self) -> int:
        return max((block.nnz for block in self.blocks), default=0)

    @property
    def sel_mut_nnz_by_level(self) -> np.ndarray:
        return self.selector_nnz_by_level[:, 0]

    @property
    def sel_miss_nnz_by_level(self) -> np.ndarray:
        return self.selector_nnz_by_level[:, 1]


def _selector_nnz_by_level(selector_indices: np.ndarray, level_offsets: np.ndarray) -> np.ndarray:
    offsets = np.asarray(level_offsets)
    num_levels = max(int(offsets.size) - 1, 0)
    indices = np.asarray(selector_indices)
    if indices.size == 0:
        return np.zeros((num_levels,), dtype=np.int64)
    levels = np.searchsorted(offsets, indices, side="right") - 1
    if np.any(levels < 0) or np.any(levels >= num_levels):
        raise ValueError("selector indices are outside level_offsets")
    return np.bincount(levels, minlength=num_levels).astype(np.int64, copy=False)


def save_grg_spmv(state: CompiledOperatorState, artifact_path) -> None:
    if state.A_blocks is None:
        raise RuntimeError("compiled operator blocks are required for .grg_spmv save")
    if state.init_vector_up_bias is None or state.init_vector_down_bias is None:
        raise RuntimeError("init vector bias cache must be populated before .grg_spmv save")

    t0 = time.perf_counter()
    sel_mut_indices = np.asarray(state.sel_mut.indices)
    sel_miss_indices = np.asarray(state.sel_miss.indices)
    level_offsets_arr = np.asarray(state.level_offsets)
    sel_mut_nnz_by_level = _selector_nnz_by_level(sel_mut_indices, level_offsets_arr)
    sel_miss_nnz_by_level = _selector_nnz_by_level(sel_miss_indices, level_offsets_arr)
    selector_nnz_by_level = np.column_stack((sel_mut_nnz_by_level, sel_miss_nnz_by_level)).astype(np.int64, copy=False)
    node_perm = np.asarray(state.node_perm)
    sample_to_individual = np.asarray(state.sample_to_individual)
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
        "node_perm": node_perm,
        "sample_to_individual": sample_to_individual,
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
        "sel_miss_indices": sel_miss_indices,
        "sel_miss_indptr": np.asarray(state.sel_miss.indptr),
    }
    if state.coalescence_counts is not None:
        save_dict["coalescence_counts"] = np.asarray(state.coalescence_counts, dtype=np.int64)
    if state.init_xtx_up_bias is not None:
        save_dict["init_xtx_up_bias"] = np.asarray(state.init_xtx_up_bias)
    if state.init_xtx_down_bias is not None:
        save_dict["init_xtx_down_bias"] = np.asarray(state.init_xtx_down_bias)
    scan_block_levels: list[tuple[int, int]] = []
    scan_block_shapes: list[tuple[int, int]] = []
    scan_block_nnzs: list[int] = []
    scan_block_struct_itemsize: list[tuple[int, int]] = []
    for dst_level, blocks in enumerate(state.A_blocks):
        for src_level, block in enumerate(blocks):
            indices = np.asarray(block.indices)
            indptr = np.asarray(block.indptr)
            save_dict[f"A_blocks_{dst_level}_{src_level}_indices"] = indices
            save_dict[f"A_blocks_{dst_level}_{src_level}_indptr"] = indptr
            scan_block_levels.append((int(dst_level), int(src_level)))
            scan_block_shapes.append((int(block.shape[0]), int(block.shape[1])))
            scan_block_nnzs.append(int(indices.size))
            scan_block_struct_itemsize.append((int(indices.dtype.itemsize), int(indptr.dtype.itemsize)))
    save_dict["scan_block_levels"] = np.asarray(scan_block_levels, dtype=np.int32).reshape((-1, 2))
    save_dict["scan_block_shapes"] = np.asarray(scan_block_shapes, dtype=np.int64).reshape((-1, 2))
    save_dict["scan_block_nnzs"] = np.asarray(scan_block_nnzs, dtype=np.int64)
    save_dict["scan_block_struct_itemsize"] = np.asarray(scan_block_struct_itemsize, dtype=np.uint8).reshape((-1, 2))
    save_dict["scan_selector_nnzs"] = np.asarray((int(state.sel_mut.nnz), int(state.sel_miss.nnz)), dtype=np.int64)
    save_dict["scan_selector_nnz_by_level"] = selector_nnz_by_level
    save_dict["scan_metadata_struct_itemsize"] = np.asarray(
        (int(node_perm.dtype.itemsize), int(sample_to_individual.dtype.itemsize)),
        dtype=np.uint8,
    )
    num_blocks = len(scan_block_levels)

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


def _load_struct_array(data: np.lib.npyio.NpzFile, key: str, *, non_negative: bool = True) -> np.ndarray:
    arr = np.asarray(data[key])
    if arr.dtype not in {np.dtype(np.int32), np.dtype(np.int64)}:
        raise ValueError(f"unsupported structural dtype for {key}: {arr.dtype}")
    if non_negative and arr.size and int(arr.min()) < 0:
        raise ValueError(f"structural array {key} must be non-negative")
    return arr


def _struct_dtype_from_itemsize(itemsize: int) -> np.dtype:
    match int(itemsize):
        case 4:
            return np.dtype(np.int32)
        case 8:
            return np.dtype(np.int64)
        case _:
            raise ValueError(f"unsupported structural itemsize: {itemsize}")


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
        block_levels = np.asarray(data["scan_block_levels"])
        block_shapes = np.asarray(data["scan_block_shapes"])
        block_nnzs = np.asarray(data["scan_block_nnzs"])
        block_struct_itemsize = np.asarray(data["scan_block_struct_itemsize"])
        selector_nnzs = np.asarray(data["scan_selector_nnzs"])
        selector_nnz_by_level = np.asarray(data["scan_selector_nnz_by_level"], dtype=np.int64)
        metadata_struct_itemsize = np.asarray(data["scan_metadata_struct_itemsize"])
        if block_levels.ndim != 2 or block_levels.shape[1] != 2:
            raise ValueError(f"scan_block_levels must have shape (num_blocks, 2), got {block_levels.shape}")
        if block_shapes.ndim != 2 or block_shapes.shape[1] != 2:
            raise ValueError(f"scan_block_shapes must have shape (num_blocks, 2), got {block_shapes.shape}")
        if block_struct_itemsize.ndim != 2 or block_struct_itemsize.shape[1] != 2:
            raise ValueError(
                f"scan_block_struct_itemsize must have shape (num_blocks, 2), got {block_struct_itemsize.shape}"
            )
        if block_nnzs.ndim != 1:
            raise ValueError(f"scan_block_nnzs must be one-dimensional, got {block_nnzs.shape}")
        num_blocks = int(block_levels.shape[0])
        if block_shapes.shape[0] != num_blocks or block_nnzs.shape[0] != num_blocks or block_struct_itemsize.shape[0] != num_blocks:
            raise ValueError("scan block metadata arrays must have matching lengths")
        if selector_nnzs.shape != (2,):
            raise ValueError(f"scan_selector_nnzs must have shape (2,), got {selector_nnzs.shape}")
        if selector_nnz_by_level.shape != (len(level_sizes), 2):
            raise ValueError(
                f"scan_selector_nnz_by_level must have shape ({len(level_sizes)}, 2), got {selector_nnz_by_level.shape}"
            )
        if metadata_struct_itemsize.shape != (2,):
            raise ValueError(f"scan_metadata_struct_itemsize must have shape (2,), got {metadata_struct_itemsize.shape}")
        if (
            (block_levels.size and np.any(block_levels < 0))
            or (block_shapes.size and np.any(block_shapes < 0))
            or (block_nnzs.size and np.any(block_nnzs < 0))
            or np.any(selector_nnzs < 0)
            or (selector_nnz_by_level.size and np.any(selector_nnz_by_level < 0))
        ):
            raise ValueError("scan metadata arrays must be non-negative")
        if np.any(selector_nnz_by_level.sum(axis=0) != selector_nnzs):
            raise ValueError("scan selector metadata arrays must have matching nnz totals")
        selector_nnz_by_level.setflags(write=False)
        blocks: list[ArtifactBlockScan] = []
        for (dst_level, src_level), shape, nnz, itemsize in zip(
            block_levels,
            block_shapes,
            block_nnzs,
            block_struct_itemsize,
            strict=True,
        ):
            blocks.append(
                ArtifactBlockScan(
                    dst_level=int(dst_level),
                    src_level=int(src_level),
                    shape=(int(shape[0]), int(shape[1])),
                    nnz=int(nnz),
                    indices_dtype=_struct_dtype_from_itemsize(int(itemsize[0])),
                    indptr_dtype=_struct_dtype_from_itemsize(int(itemsize[1])),
                )
            )
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
            selector_mut_nnz=int(selector_nnzs[0]),
            selector_miss_nnz=int(selector_nnzs[1]),
            has_individual_coals=bool(np.asarray(data["has_individual_coals"]).item()),
            selector_nnz_by_level=selector_nnz_by_level,
            node_perm_dtype=_struct_dtype_from_itemsize(int(metadata_struct_itemsize[0])),
            sample_to_individual_dtype=_struct_dtype_from_itemsize(int(metadata_struct_itemsize[1])),
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
        block_levels = np.asarray(data["scan_block_levels"])
        block_shapes = np.asarray(data["scan_block_shapes"])
        if block_levels.ndim != 2 or block_levels.shape[1] != 2:
            raise ValueError(f"scan_block_levels must have shape (num_blocks, 2), got {block_levels.shape}")
        if block_shapes.ndim != 2 or block_shapes.shape[1] != 2:
            raise ValueError(f"scan_block_shapes must have shape (num_blocks, 2), got {block_shapes.shape}")
        if block_levels.shape[0] != block_shapes.shape[0]:
            raise ValueError("scan block metadata arrays must have matching lengths")
        for (dst_level, src_level), shape in zip(block_levels, block_shapes, strict=True):
            dst_level = int(dst_level)
            src_level = int(src_level)
            indices = _load_struct_array(data, f"A_blocks_{dst_level}_{src_level}_indices")
            indptr = _load_struct_array(data, f"A_blocks_{dst_level}_{src_level}_indptr")
            yield ArtifactBlock(
                dst_level=dst_level,
                src_level=src_level,
                shape=(int(shape[0]), int(shape[1])),
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
