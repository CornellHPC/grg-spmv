"""Adapted Triton naive CSR/CSC kernels for unit-weight SpMM."""

from __future__ import annotations

from dataclasses import dataclass

import triton
import triton.language as tl

from pygrgl_spmv.backends.types import SparseFormat


@dataclass(frozen=True)
class CsrKernelConfig:
    rows_per_program: int
    block_nnz: int
    num_warps: int


@dataclass(frozen=True)
class CscKernelConfig:
    cols_per_program: int
    block_nnz: int
    num_warps: int


CSR_CANDIDATE_CONFIGS = (
    CsrKernelConfig(rows_per_program=1, block_nnz=32, num_warps=2),
    CsrKernelConfig(rows_per_program=1, block_nnz=64, num_warps=2),
    CsrKernelConfig(rows_per_program=1, block_nnz=128, num_warps=4),
    CsrKernelConfig(rows_per_program=1, block_nnz=256, num_warps=4),
    CsrKernelConfig(rows_per_program=1, block_nnz=512, num_warps=8),
    CsrKernelConfig(rows_per_program=2, block_nnz=64, num_warps=2),
    CsrKernelConfig(rows_per_program=2, block_nnz=128, num_warps=4),
    CsrKernelConfig(rows_per_program=2, block_nnz=256, num_warps=4),
    CsrKernelConfig(rows_per_program=4, block_nnz=64, num_warps=2),
    CsrKernelConfig(rows_per_program=4, block_nnz=128, num_warps=4),
    CsrKernelConfig(rows_per_program=4, block_nnz=256, num_warps=4),
)

CSC_CANDIDATE_CONFIGS = (
    CscKernelConfig(cols_per_program=1, block_nnz=32, num_warps=2),
    CscKernelConfig(cols_per_program=1, block_nnz=64, num_warps=2),
    CscKernelConfig(cols_per_program=1, block_nnz=128, num_warps=4),
    CscKernelConfig(cols_per_program=1, block_nnz=256, num_warps=4),
    CscKernelConfig(cols_per_program=2, block_nnz=64, num_warps=2),
    CscKernelConfig(cols_per_program=2, block_nnz=128, num_warps=4),
    CscKernelConfig(cols_per_program=2, block_nnz=256, num_warps=4),
    CscKernelConfig(cols_per_program=4, block_nnz=64, num_warps=2),
    CscKernelConfig(cols_per_program=4, block_nnz=128, num_warps=4),
    CscKernelConfig(cols_per_program=4, block_nnz=256, num_warps=4),
)

_BLOCK_K = 32


@triton.jit
def csr_spmm_add_kernel(
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    num_rows,
    k,
    x_row_stride,
    x_col_stride,
    y_row_stride,
    y_col_stride,
    BLOCK_NNZ: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    FP64_ACC: tl.constexpr,
):
    row_base = tl.program_id(0) * ROWS_PER_PROGRAM
    k_base = tl.program_id(1) * BLOCK_K
    nz_offsets = tl.arange(0, BLOCK_NNZ)
    k_offsets = k_base + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < k
    acc_dtype = tl.float64 if FP64_ACC else tl.float32

    for row_offset in range(ROWS_PER_PROGRAM):
        row = row_base + row_offset
        row_mask = row < num_rows
        row_start = tl.load(indptr_ptr + row, mask=row_mask, other=0)
        row_end = tl.load(indptr_ptr + row + 1, mask=row_mask, other=0)
        acc = tl.zeros((BLOCK_K,), dtype=acc_dtype)

        for start in tl.range(row_start, row_end, BLOCK_NNZ):
            block_offsets = start + nz_offsets
            nz_mask = row_mask & (block_offsets < row_end)
            cols = tl.load(indices_ptr + block_offsets, mask=nz_mask, other=0)
            x_ptrs = x_ptr + cols[:, None] * x_row_stride + k_offsets[None, :] * x_col_stride
            x_vals = tl.load(x_ptrs, mask=nz_mask[:, None] & k_mask[None, :], other=0).to(acc_dtype)
            acc += tl.sum(x_vals, axis=0)

        y_ptrs = y_ptr + row * y_row_stride + k_offsets * y_col_stride
        prev = tl.load(y_ptrs, mask=row_mask & k_mask, other=0).to(acc_dtype)
        tl.store(y_ptrs, prev + acc, mask=row_mask & k_mask)


@triton.jit
def csc_spmm_add_kernel(
    rowidx_ptr,
    colptr_ptr,
    x_ptr,
    y_ptr,
    num_cols,
    k,
    x_row_stride,
    x_col_stride,
    y_row_stride,
    y_col_stride,
    BLOCK_NNZ: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COLS_PER_PROGRAM: tl.constexpr,
    FP64_ACC: tl.constexpr,
):
    col_base = tl.program_id(0) * COLS_PER_PROGRAM
    k_base = tl.program_id(1) * BLOCK_K
    nz_offsets = tl.arange(0, BLOCK_NNZ)
    k_offsets = k_base + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < k
    acc_dtype = tl.float64 if FP64_ACC else tl.float32

    for col_offset in range(COLS_PER_PROGRAM):
        col = col_base + col_offset
        col_mask = col < num_cols
        nz_start = tl.load(colptr_ptr + col, mask=col_mask, other=0)
        nz_end = tl.load(colptr_ptr + col + 1, mask=col_mask, other=0)
        x_ptrs = x_ptr + col * x_row_stride + k_offsets * x_col_stride
        x_vals = tl.load(x_ptrs, mask=col_mask & k_mask, other=0).to(acc_dtype)

        for start in tl.range(nz_start, nz_end, BLOCK_NNZ):
            block_offsets = start + nz_offsets
            nz_mask = col_mask & (block_offsets < nz_end)
            rows = tl.load(rowidx_ptr + block_offsets, mask=nz_mask, other=0)
            y_ptrs = y_ptr + rows[:, None] * y_row_stride + k_offsets[None, :] * y_col_stride
            tl.atomic_add(y_ptrs, x_vals[None, :], mask=nz_mask[:, None] & k_mask[None, :])


def launch_block(
    *,
    block,
    x,
    y,
    config: CsrKernelConfig | CscKernelConfig,
    fp64_acc: bool,
) -> None:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Triton SpMM expects 2D x/y, got x.ndim={x.ndim}, y.ndim={y.ndim}")
    if int(x.shape[1]) != int(y.shape[1]):
        raise ValueError(f"Triton SpMM k mismatch: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}")
    k = int(x.shape[1])
    if block.fmt == SparseFormat.CSR:
        if not isinstance(config, CsrKernelConfig):
            raise TypeError(f"CSR block requires CsrKernelConfig, got {type(config).__name__}")
        grid = (
            (block.nrows + config.rows_per_program - 1) // config.rows_per_program,
            (k + _BLOCK_K - 1) // _BLOCK_K,
        )
        csr_spmm_add_kernel[grid](
            block.indices,
            block.indptr,
            x,
            y,
            block.nrows,
            k,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_NNZ=config.block_nnz,
            BLOCK_K=_BLOCK_K,
            ROWS_PER_PROGRAM=config.rows_per_program,
            FP64_ACC=fp64_acc,
            num_warps=config.num_warps,
        )
        return

    if block.fmt == SparseFormat.CSC:
        if not isinstance(config, CscKernelConfig):
            raise TypeError(f"CSC block requires CscKernelConfig, got {type(config).__name__}")
        grid = (
            (block.ncols + config.cols_per_program - 1) // config.cols_per_program,
            (k + _BLOCK_K - 1) // _BLOCK_K,
        )
        csc_spmm_add_kernel[grid](
            block.indices,
            block.indptr,
            x,
            y,
            block.ncols,
            k,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_NNZ=config.block_nnz,
            BLOCK_K=_BLOCK_K,
            COLS_PER_PROGRAM=config.cols_per_program,
            FP64_ACC=fp64_acc,
            num_warps=config.num_warps,
        )
        return

    raise ValueError(f"Unsupported Triton sparse format: {block.fmt}")


__all__ = [
    "CSC_CANDIDATE_CONFIGS",
    "CSR_CANDIDATE_CONFIGS",
    "CscKernelConfig",
    "CsrKernelConfig",
    "launch_block",
]
