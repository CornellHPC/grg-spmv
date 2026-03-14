"""Adapted Triton naive CSR/CSC kernels for unit-weight SpMV."""

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


@triton.jit
def csr_spmv_add_kernel(
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    num_rows,
    BLOCK_NNZ: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    FP64_ACC: tl.constexpr,
):
    row_base = tl.program_id(0) * ROWS_PER_PROGRAM
    offsets = tl.arange(0, BLOCK_NNZ)
    acc_dtype = tl.float64 if FP64_ACC else tl.float32

    for row_offset in range(ROWS_PER_PROGRAM):
        row = row_base + row_offset
        row_mask = row < num_rows
        row_start = tl.load(indptr_ptr + row, mask=row_mask, other=0)
        row_end = tl.load(indptr_ptr + row + 1, mask=row_mask, other=0)
        acc = tl.zeros((), dtype=acc_dtype)

        for start in tl.range(row_start, row_end, BLOCK_NNZ):
            block_offsets = start + offsets
            mask = row_mask & (block_offsets < row_end)
            cols = tl.load(indices_ptr + block_offsets, mask=mask, other=0)
            x_vals = tl.load(x_ptr + cols, mask=mask, other=0).to(acc_dtype)
            acc += tl.sum(x_vals, axis=0)

        prev = tl.load(y_ptr + row, mask=row_mask, other=0).to(acc_dtype)
        tl.store(y_ptr + row, prev + acc, mask=row_mask)


@triton.jit
def csc_spmv_add_kernel(
    rowidx_ptr,
    colptr_ptr,
    x_ptr,
    y_ptr,
    num_cols,
    BLOCK_NNZ: tl.constexpr,
    COLS_PER_PROGRAM: tl.constexpr,
    FP64_ACC: tl.constexpr,
):
    col_base = tl.program_id(0) * COLS_PER_PROGRAM
    offsets = tl.arange(0, BLOCK_NNZ)
    acc_dtype = tl.float64 if FP64_ACC else tl.float32

    for col_offset in range(COLS_PER_PROGRAM):
        col = col_base + col_offset
        col_mask = col < num_cols
        nz_start = tl.load(colptr_ptr + col, mask=col_mask, other=0)
        nz_end = tl.load(colptr_ptr + col + 1, mask=col_mask, other=0)
        x_val = tl.load(x_ptr + col, mask=col_mask, other=0).to(acc_dtype)

        for start in tl.range(nz_start, nz_end, BLOCK_NNZ):
            nz_offsets = start + offsets
            mask = col_mask & (nz_offsets < nz_end)
            rows = tl.load(rowidx_ptr + nz_offsets, mask=mask, other=0)
            contrib = tl.full((BLOCK_NNZ,), x_val, acc_dtype)
            tl.atomic_add(y_ptr + rows, contrib, mask=mask)


def format_config(config: CsrKernelConfig | CscKernelConfig) -> str:
    if isinstance(config, CsrKernelConfig):
        return (
            f"ROWS_PER_PROGRAM={config.rows_per_program}, "
            f"BLOCK_NNZ={config.block_nnz}, num_warps={config.num_warps}"
        )
    return (
        f"COLS_PER_PROGRAM={config.cols_per_program}, "
        f"BLOCK_NNZ={config.block_nnz}, num_warps={config.num_warps}"
    )


def launch_block(
    *,
    block,
    x,
    y,
    config: CsrKernelConfig | CscKernelConfig,
    fp64_acc: bool,
) -> None:
    if block.fmt == SparseFormat.CSR:
        if not isinstance(config, CsrKernelConfig):
            raise TypeError(f"CSR block requires CsrKernelConfig, got {type(config).__name__}")
        grid = ((block.nrows + config.rows_per_program - 1) // config.rows_per_program,)
        csr_spmv_add_kernel[grid](
            block.indices,
            block.indptr,
            x,
            y,
            block.nrows,
            BLOCK_NNZ=config.block_nnz,
            ROWS_PER_PROGRAM=config.rows_per_program,
            FP64_ACC=fp64_acc,
            num_warps=config.num_warps,
        )
        return

    if block.fmt == SparseFormat.CSC:
        if not isinstance(config, CscKernelConfig):
            raise TypeError(f"CSC block requires CscKernelConfig, got {type(config).__name__}")
        grid = ((block.ncols + config.cols_per_program - 1) // config.cols_per_program,)
        csc_spmv_add_kernel[grid](
            block.indices,
            block.indptr,
            x,
            y,
            block.ncols,
            BLOCK_NNZ=config.block_nnz,
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
    "format_config",
    "launch_block",
]
