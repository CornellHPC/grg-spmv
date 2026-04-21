"""Small shared helpers for runtime-owned backends."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Iterator
from typing import Any

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends.types import Direction, SparseFormat, StoredMatrix


@dataclass(frozen=True)
class BudgetItem:
    kind: str
    name: str
    nbytes: int
    artifact_index: int | None = None
    dst_level: int | None = None
    src_level: int | None = None
    slot: int | None = None


def _require_struct_dtype(dtype: np.dtype, *, label: str) -> np.dtype:
    dt = np.dtype(dtype)
    if dt not in {np.dtype(np.int32), np.dtype(np.int64)}:
        raise TypeError(f"{label} must use int32 or int64, got {dt}")
    return dt


def _struct_dtype_for_bound(max_value: int) -> np.dtype:
    value = int(max_value)
    if value < 0:
        raise ValueError(f"structural bound must be non-negative, got {value}")
    if value > int(np.iinfo(np.int32).max):
        return np.dtype(np.int64)
    return np.dtype(np.int32)


def _layout_struct_dtypes(fmt: str | SparseFormat, *, nrows: int, ncols: int, nnz: int) -> tuple[np.dtype, np.dtype]:
    token = str(getattr(fmt, "value", fmt)).strip().lower()
    if token == "csr":
        return _struct_dtype_for_bound(max(int(nnz), 0)), _struct_dtype_for_bound(max(int(ncols) - 1, 0))
    if token == "csc":
        return _struct_dtype_for_bound(max(int(nnz), 0)), _struct_dtype_for_bound(max(int(nrows) - 1, 0))
    if token == "coo":
        common = _struct_dtype_for_bound(max(int(nrows) - 1, int(ncols) - 1, 0))
        return common, common
    raise ValueError(f"unknown sparse format for structural dtype bounds: {fmt!r}")


def sparse_structure_lengths(fmt: str | SparseFormat, *, nrows: int, ncols: int, nnz: int) -> tuple[int, int]:
    token = str(getattr(fmt, "value", fmt)).strip().lower()
    if token == "csr":
        return int(nrows) + 1, int(nnz)
    if token == "csc":
        return int(ncols) + 1, int(nnz)
    if token == "coo":
        return int(nnz), int(nnz)
    raise ValueError(f"unknown sparse format for structural lengths: {fmt!r}")


def sparse_structure_nbytes(
    fmt: str | SparseFormat,
    *,
    nrows: int,
    ncols: int,
    nnz: int,
    struct0_dtype: np.dtype | None = None,
    struct1_dtype: np.dtype | None = None,
) -> int:
    if struct0_dtype is None or struct1_dtype is None:
        inferred0, inferred1 = _layout_struct_dtypes(fmt, nrows=nrows, ncols=ncols, nnz=nnz)
        if struct0_dtype is None:
            struct0_dtype = inferred0
        if struct1_dtype is None:
            struct1_dtype = inferred1
    len0, len1 = sparse_structure_lengths(fmt, nrows=nrows, ncols=ncols, nnz=nnz)
    return int(len0 * np.dtype(struct0_dtype).itemsize + len1 * np.dtype(struct1_dtype).itemsize)


def _copy_struct_checked(dst: Any, values: Any, *, label: str) -> None:
    dst_arr = np.asarray(dst)
    src_arr = np.asarray(values)
    source = _require_struct_dtype(src_arr.dtype, label=label)
    target = _require_struct_dtype(dst_arr.dtype, label=label)
    if dst_arr.shape != src_arr.shape:
        raise ValueError(f"{label} shape mismatch: expected {dst_arr.shape}, got {src_arr.shape}")
    if source == target:
        np.copyto(dst_arr, src_arr, casting="no")
        return
    if source.itemsize < target.itemsize:
        np.copyto(dst_arr, src_arr, casting="safe")
        return
    if src_arr.size:
        lo = int(src_arr.min())
        hi = int(src_arr.max())
        info = np.iinfo(target)
        if lo < int(info.min) or hi > int(info.max):
            raise ValueError(f"{label} exceeds {target} range")
    np.copyto(dst_arr, src_arr, casting="unsafe")


def iter_direction_level_pairs(direction: Direction, height: int) -> Iterator[tuple[int, int, int]]:
    if height < 0:
        raise ValueError(f"number of levels must be non-negative, got {height}")
    match direction:
        case Direction.UP:
            for dst_level in range(height):
                for src_level in range(dst_level):
                    yield dst_level, src_level, src_level
        case Direction.DOWN:
            for dst_level in range(height):
                for src_level in range(height - 1, dst_level, -1):
                    yield dst_level, src_level, src_level - dst_level - 1
        case _:
            raise ValueError(f"unknown direction {direction!r}")


def stored_block_shape(nrows: int, ncols: int, *, store: StoredMatrix) -> tuple[int, int]:
    if store == StoredMatrix.N:
        return int(nrows), int(ncols)
    return int(ncols), int(nrows)


def materialize_sparse_block(
    block: sp.csr_matrix,
    *,
    store: StoredMatrix,
    fmt: SparseFormat,
) -> sp.spmatrix:
    stored = block if store == StoredMatrix.N else block.T
    if fmt == SparseFormat.CSR:
        csr = stored.tocsr()
        csr.sum_duplicates()
        csr.sort_indices()
        return csr
    if fmt == SparseFormat.CSC:
        csc = stored.tocsc()
        csc.sum_duplicates()
        csc.sort_indices()
        return csc
    if fmt == SparseFormat.COO:
        coo = stored.tocoo()
        coo.sum_duplicates()
        return coo
    raise ValueError(f"unsupported sparse format: {fmt.value}")


def split_selector_by_level(
    selector: sp.csr_matrix,
    level_offsets: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    row_dtype = _struct_dtype_for_bound(max(int(selector.shape[0]) - 1, 0))
    offsets = np.asarray(level_offsets, dtype=np.int64)
    for level in range(len(offsets) - 1):
        lo = int(offsets[level])
        hi = int(offsets[level + 1])
        block = selector[:, lo:hi].tocoo()
        col_dtype = _struct_dtype_for_bound(max(hi - lo - 1, 0))
        rows = np.asarray(block.row, dtype=row_dtype)
        cols = np.asarray(block.col, dtype=col_dtype)
        out.append((rows, cols))
    return out


__all__ = [
    "BudgetItem",
    "_copy_struct_checked",
    "_layout_struct_dtypes",
    "_require_struct_dtype",
    "_struct_dtype_for_bound",
    "iter_direction_level_pairs",
    "materialize_sparse_block",
    "relink_stream_dependencies",
    "sparse_structure_lengths",
    "sparse_structure_nbytes",
    "split_selector_by_level",
    "stored_block_shape",
]


def relink_stream_dependencies(ops_by_level, *, direction: Direction):
    prev_by_slot: dict[int, tuple[int, int]] = {}
    height = len(ops_by_level)
    level_iter = range(1, height) if direction == Direction.UP else range(height - 2, -1, -1)
    for dst_level in level_iter:
        for op_idx, op in enumerate(ops_by_level[dst_level]):
            slot = getattr(op, "slot", None)
            if slot is None:
                continue
            ops_by_level[dst_level][op_idx] = replace(op, prev_in_slot=prev_by_slot.get(int(slot)))
            prev_by_slot[int(slot)] = (dst_level, op_idx)
    return ops_by_level
