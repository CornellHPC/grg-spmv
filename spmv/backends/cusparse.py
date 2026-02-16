"""
cuSPARSE Backend - GPU-accelerated block-wise SpMM using NVIDIA cuSPARSE.

Uses per-level dense buffers (one per height level).  In graph mode,
captures the entire forward/backward wavefront pipeline as a CUDA graph
with multi-stream parallelism; sel gather/scatter runs outside the graph
since CuPy fancy indexing is not reliably graph-capture-safe.

Requires CuPy for GPU memory management and CUDA stream/graph APIs.
Uses CuSparseLib (cuda_utils.py) for all cuSPARSE FFI calls.
"""

from typing import List, Optional

import numpy as np
import scipy.sparse as sp

from spmv.backends import Backend
from spmv.backends.cuda_utils import (
    CuSparseLib, cuda_dtype, parse_algorithm, GpuTimer,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_TRANSPOSE,
)

_OP_N = CUSPARSE_OPERATION_NON_TRANSPOSE
_OP_T = CUSPARSE_OPERATION_TRANSPOSE

# Whitelist of (fmt, algorithm) combinations that cuSPARSE actually supports.
# Determined empirically and from NVIDIA docs:
#   - CSR_ALG1/2: valid for CSR and CSC only (NOT COO — cuSPARSE status 3)
#   - CSR_ALG3: valid for CSR only
#   - COO_ALG*: valid for COO only
#   - SPMM_ALG_DEFAULT: valid for all formats
VALID_FMT_ALG_COMBOS = frozenset({
    ('csr', 'default'),
    ('csr', 'csr_alg1'),
    ('csr', 'csr_alg2'),
    ('csc', 'default'),
    ('csc', 'csr_alg1'),
    ('csc', 'csr_alg2'),
    ('coo', 'default'),
    ('coo', 'coo_alg1'),
    ('coo', 'coo_alg2'),
    ('coo', 'coo_alg3'),
    ('coo', 'coo_alg4'),
})


def is_valid_combo(fmt: str, alg: str) -> bool:
    """Return True if (fmt, alg) is a supported cuSPARSE SpMM combination."""
    return (fmt, alg) in VALID_FMT_ALG_COMBOS


class CusparseBackend(Backend):
    """
    GPU backend using NVIDIA cuSPARSE for block-wise SpMM.

    Parameters
    ----------
    fmt : str
        Sparse storage format: 'csr', 'csc', or 'coo'.
    k : int or None
        If set, CUDA graphs are captured for this many dense columns.
    algorithm : str or int
        cuSPARSE SpMM algorithm name or integer constant.
    verbose : bool
        Print timing and profiling info.
    """

    def __init__(self, fmt: str = 'csr', k: Optional[int] = None,
                 algorithm: str = 'default', verbose: bool = False):
        self._fmt = fmt
        self._graph_k = k
        self._algorithm = parse_algorithm(algorithm)
        self._verbose = verbose

        # Algorithm-format validation (whitelist-based)
        if not is_valid_combo(fmt, algorithm):
            raise ValueError(
                f"(fmt={fmt!r}, algorithm={algorithm!r}) is not a supported "
                f"cuSPARSE SpMM combination. Valid combinations: {sorted(VALID_FMT_ALG_COMBOS)}"
            )

        try:
            import cupy as cp
            self._cp = cp
        except ImportError:
            raise ImportError("CuPy required: pip install cupy-cuda12x")

        self._cslib = CuSparseLib()

        self._stream = cp.cuda.Stream(non_blocking=True)

        if verbose:
            print(f"cuSPARSE version: {self._cslib.version}")

        # Will be populated in setup()
        self._H = self._n = self._K = self._m = 0
        self._dtype = np.float64
        self._level_offsets = None

        # Graph mode state
        self._fwd_exec = None
        self._bwd_exec = None
        self._k = None

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(
        self,
        A_blocks: List[List[sp.csr_matrix]],
        level_offsets: np.ndarray,
        n: int,
        K: int,
        sel: sp.csr_matrix,
        sample_perm: np.ndarray,
        inv_sample_perm: np.ndarray,
        dtype: np.dtype,
    ) -> None:
        cp = self._cp
        off = level_offsets
        self._level_offsets = off
        self._H = len(off) - 1
        self._n = n
        self._K = K
        self._m = sel.shape[0]
        self._dtype = np.dtype(dtype)

        # Device scalars (DEVICE pointers for graph-safe SpMM)
        self._alpha = cp.ones(1, dtype=dtype)
        self._beta_zero = cp.zeros(1, dtype=dtype)
        self._beta_one = cp.ones(1, dtype=dtype)

        # --- Upload forward blocks only (backward uses _fwd_sp with TRANSPOSE) ---
        self._fwd_sp, self._fwd_gpu = self._upload_blocks(A_blocks)

        # --- Sel index arrays (gather/scatter instead of SpMM) ---
        self._sel_mut_idx = []
        self._sel_node_local = []
        for h in range(self._H):
            lo, hi = int(off[h]), int(off[h + 1])
            sb = sel[:, lo:hi].tocoo()
            self._sel_mut_idx.append(cp.array(sb.row.astype(np.int32)))
            self._sel_node_local.append(cp.array(sb.col.astype(np.int32)))

        # --- Permutation indices on GPU ---
        # Forward scatter: level_bufs[h][:ns] = X_gpu[src_indices]
        self._fwd_scatter_ns = []
        self._fwd_scatter_src = []
        for h in range(self._H):
            lo, hi = int(off[h]), int(off[h + 1])
            ns = max(0, min(hi, n) - lo)
            src = cp.array(sample_perm[lo:lo + ns]) if ns > 0 else None
            self._fwd_scatter_ns.append(ns)
            self._fwd_scatter_src.append(src)

        # Backward gather: result[dst] = level_bufs[h][src]
        self._bwd_gather_dst = []
        self._bwd_gather_src = []
        for h in range(self._H):
            lo, hi = int(off[h]), int(off[h + 1])
            mask = (inv_sample_perm >= lo) & (inv_sample_perm < hi)
            dst = cp.array(np.where(mask)[0])
            src = cp.array(inv_sample_perm[mask] - lo)
            self._bwd_gather_dst.append(dst)
            self._bwd_gather_src.append(src)

        # --- Per-level CUDA streams (for wavefront parallelism) ---
        self._level_streams = [cp.cuda.Stream(non_blocking=True)
                               for _ in range(self._H)]

        if self._verbose:
            nblk_fwd = sum(1 for h in range(self._H) for d in self._fwd_sp[h] if d is not None)
            nnz_fwd = sum(b.nnz for blocks in A_blocks for b in blocks)
            print(f"CusparseBackend setup: H={self._H}, K={K}, n={n}, m={self._m}, "
                  f"fmt={self._fmt}, graph_k={self._graph_k}")
            print(f"  Forward: {nblk_fwd} active blocks, {nnz_fwd:,} total nnz")
            print(f"  Backward: {nblk_fwd} active blocks (transpose of fwd), {nnz_fwd:,} total nnz")

        if self._graph_k is not None:
            self._prepare_graphs(self._graph_k, self._dtype)

    # ------------------------------------------------------------------
    # upload helpers
    # ------------------------------------------------------------------

    def _upload_blocks(self, blocks_list):
        """Upload sparse blocks to GPU, returning (descriptors, gpu_arrays) per level."""
        all_descs, all_arrs = [], []
        for h in range(self._H):
            descs, arrs = [], []
            for blk in blocks_list[h]:
                if blk.nnz > 0:
                    d, a = self._upload_sparse(blk)
                    descs.append(d)
                    arrs.append(a)
                else:
                    descs.append(None)
                    arrs.append(None)
            all_descs.append(descs)
            all_arrs.append(arrs)
        return all_descs, all_arrs

    def _upload_sparse(self, A_scipy):
        """Upload a scipy sparse matrix to GPU and create cuSPARSE descriptor."""
        cp = self._cp
        cslib = self._cslib
        cdt = cuda_dtype(self._dtype)

        match self._fmt:
            case 'csr':
                A = sp.csr_matrix(A_scipy).astype(self._dtype)
                indptr = cp.array(A.indptr.astype(np.int32))
                indices = cp.array(A.indices.astype(np.int32))
                data = cp.array(A.data)
                desc = cslib.create_csr(
                    A.shape[0], A.shape[1], A.nnz,
                    indptr.data.ptr, indices.data.ptr, data.data.ptr, cdt,
                )
                return desc, (indptr, indices, data)
            case 'csc':
                A = A_scipy.tocsc()
                indptr = cp.array(A.indptr.astype(np.int32))
                indices = cp.array(A.indices.astype(np.int32))
                data = cp.array(A.data.astype(self._dtype))
                desc = cslib.create_csc(
                    A.shape[0], A.shape[1], A.nnz,
                    indptr.data.ptr, indices.data.ptr, data.data.ptr, cdt,
                )
                return desc, (indptr, indices, data)
            case 'coo':
                A = A_scipy.tocoo()
                row_idx = cp.array(A.row.astype(np.int32))
                col_idx = cp.array(A.col.astype(np.int32))
                data = cp.array(A.data.astype(self._dtype))
                desc = cslib.create_coo(
                    A.shape[0], A.shape[1], A.nnz,
                    row_idx.data.ptr, col_idx.data.ptr, data.data.ptr, cdt,
                )
                return desc, (row_idx, col_idx, data)
            case _:
                raise ValueError(f"Unknown sparse format: {self._fmt!r}")

    # ------------------------------------------------------------------
    # SpMM helpers
    # ------------------------------------------------------------------

    def _spmm_preprocess(self, sp_desc, B_desc, C_desc, alpha_ptr, beta_ptr,
                         cdt, ext_buf, op_a=_OP_N):
        """Run cusparseSpMM_preprocess on an already-allocated workspace."""
        self._cslib.spmm_preprocess(
            self._algorithm, op_a, _OP_N,
            alpha_ptr, sp_desc, B_desc, beta_ptr, C_desc,
            cdt, ext_buf.data.ptr,
        )

    def _preprocess_one(self, sp_desc, B_desc, C_desc, alpha_ptr, beta_ptr,
                        cdt, op_a=_OP_N):
        """Allocate workspace, preprocess, and return the buffer."""
        buf = self._cslib.spmm_buffer_size(
            self._cp, self._algorithm, op_a, _OP_N,
            alpha_ptr, sp_desc, B_desc, beta_ptr, C_desc, cdt,
        )
        self._spmm_preprocess(sp_desc, B_desc, C_desc, alpha_ptr, beta_ptr,
                              cdt, buf, op_a=op_a)
        return buf

    def _spmm(self, sp_desc, B_desc, C_desc, alpha_ptr, beta_ptr, cdt, ext_buf_ptr,
              op_a=_OP_N):
        """Launch a single cusparseSpMM kernel."""
        self._cslib.spmm(
            self._algorithm, op_a, _OP_N,
            alpha_ptr, sp_desc, B_desc, beta_ptr, C_desc,
            cdt, ext_buf_ptr,
        )

    # ------------------------------------------------------------------
    # Dense descriptor helpers
    # ------------------------------------------------------------------

    def _create_level_dnmat(self, gpu_buf, k, dtype, ld=None):
        """Create DnMat descriptor for a per-level buffer."""
        if ld is None:
            ld = gpu_buf.shape[1]
        return self._cslib.create_dnmat(
            gpu_buf.shape[0], k, ld, gpu_buf.data.ptr, cuda_dtype(dtype),
        )

    # ------------------------------------------------------------------
    # sel gather / scatter_add  (run OUTSIDE CUDA graph)
    # ------------------------------------------------------------------

    def _sel_gather(self, result_gpu, level_bufs):
        """Forward sel: result[mut] = level_bufs[h][node].  (Outside graph.)"""
        with self._stream:
            result_gpu.fill(0)
            for h in range(self._H):
                idx = self._sel_mut_idx[h]
                if len(idx) > 0:
                    result_gpu[idx] = level_bufs[h][self._sel_node_local[h]]

    def _sel_scatter_add(self, X_gpu, level_bufs):
        """Backward sel_T: level_bufs[h][node] += X[mut].  (Outside graph.)"""
        cp = self._cp
        with self._stream:
            for h in range(self._H):
                level_bufs[h].fill(0)
                idx = self._sel_mut_idx[h]
                if len(idx) > 0:
                    cp.add.at(level_bufs[h], self._sel_node_local[h], X_gpu[idx])

    def _fwd_scatter(self):
        """Zero level_bufs and scatter X_gpu rows into them.  (Outside graph.)"""
        cp = self._cp
        with self._stream:
            for h in range(self._H):
                self._level_bufs[h].fill(0)
                ns = self._fwd_scatter_ns[h]
                if ns > 0:
                    cp.take(self._fwd_X_gpu, self._fwd_scatter_src[h], axis=0,
                            out=self._level_bufs[h][:ns])

    def _bwd_gather(self):
        """Gather from level_bufs into bwd_result_gpu.  (Outside graph.)"""
        cp = self._cp
        with self._stream:
            self._bwd_result_gpu.fill(0)
            for h in range(self._H):
                if self._bwd_gather_temp[h] is not None:
                    cp.take(self._level_bufs[h], self._bwd_gather_src[h], axis=0,
                            out=self._bwd_gather_temp[h])
                    self._bwd_result_gpu[self._bwd_gather_dst[h]] = \
                        self._bwd_gather_temp[h]

    # ------------------------------------------------------------------
    # CUDA graph preparation (multi-stream wavefront)
    # ------------------------------------------------------------------

    def _prepare_graphs(self, k, dtype):
        """Allocate persistent buffers, preprocess, warm up, capture graphs."""
        cp = self._cp
        off = self._level_offsets
        cdt = cuda_dtype(dtype)
        self._k = k

        a1 = self._alpha.data.ptr
        b1 = self._beta_one.data.ptr

        # --- Per-level dense buffers ---
        self._level_bufs = []
        for h in range(self._H):
            sz = int(off[h + 1]) - int(off[h])
            self._level_bufs.append(cp.zeros((sz, k), dtype=dtype, order='C'))

        # --- I/O buffers ---
        self._fwd_X_gpu = cp.zeros((self._n, k), dtype=dtype, order='C')
        self._fwd_result_gpu = cp.zeros((self._m, k), dtype=dtype, order='C')
        self._bwd_X_gpu = cp.zeros((self._m, k), dtype=dtype, order='C')
        self._bwd_result_gpu = cp.zeros((self._n, k), dtype=dtype, order='C')

        # Pre-allocate backward gather temps (persistent, reusable across replays)
        self._bwd_gather_temp = []
        for h in range(self._H):
            ng = len(self._bwd_gather_dst[h])
            self._bwd_gather_temp.append(
                cp.zeros((ng, k), dtype=dtype, order='C') if ng > 0 else None)

        # --- Dense descriptors for level buffers ---
        self._level_dn = [self._create_level_dnmat(self._level_bufs[h], k, dtype)
                          for h in range(self._H)]

        if self._verbose:
            mp = cp.get_default_memory_pool()
            print(f"  GPU memory after level bufs: {mp.used_bytes() / 1e9:.2f} GB")

        # --- bufferSize + preprocess for EVERY block ---
        # Forward blocks (beta=1, accumulate)
        self._fwd_ext = []
        for h in range(self._H):
            row = []
            for j in range(h):
                if self._fwd_sp[h][j] is None:
                    row.append(None)
                    continue
                row.append(self._preprocess_one(
                    self._fwd_sp[h][j], self._level_dn[j], self._level_dn[h],
                    a1, b1, cdt))
            self._fwd_ext.append(row)

        # Backward blocks (beta=1, accumulate) — use _fwd_sp[src][h]^T
        # AT_blocks[h][j] == A_blocks[src][h]^T  where src = h+1+j
        self._bwd_ext = []
        for h in range(self._H):
            row = []
            for src in range(h + 1, self._H):
                fwd_desc = self._fwd_sp[src][h]
                if fwd_desc is None:
                    row.append(None)
                    continue
                row.append(self._preprocess_one(
                    fwd_desc, self._level_dn[src], self._level_dn[h],
                    a1, b1, cdt, op_a=_OP_T))
            self._bwd_ext.append(row)

        if self._verbose:
            mp = cp.get_default_memory_pool()
            print(f"  GPU memory after preprocess: {mp.used_bytes() / 1e9:.2f} GB")

        # --- Warm up + capture graphs ---
        self._fwd_exec = self._capture_fwd_graph(dtype)
        self._bwd_exec = self._capture_bwd_graph(dtype)

        if self._verbose:
            print(f"  CUDA graphs captured for k={k}")

    # ------------------------------------------------------------------
    # graph capture (forward wavefront only — sel gather runs outside)
    # ------------------------------------------------------------------

    def _capture_fwd_graph(self, dtype):
        """Warm up forward wavefront, then capture as CUDA graph.

        The graph covers: fork -> wavefront SpMMs -> join.
        Scatter (zero + cp.take) and sel gather run outside the graph
        in forward_matmat() via _fwd_scatter() and _sel_gather().
        """
        cp = self._cp
        cslib = self._cslib
        cdt = cuda_dtype(dtype)
        a1, b1 = self._alpha.data.ptr, self._beta_one.data.ptr
        streams = self._level_streams

        # --- Warmup pass (per-call sync, outside graph capture) ---
        if self._verbose:
            print("  Warming up forward...")
        fwd_level_times = []

        for h in range(self._H):
            s = streams[h]
            cslib.set_stream(s.ptr)
            with s:
                self._level_bufs[h].fill(0)
                ns = self._fwd_scatter_ns[h]
                if ns > 0:
                    cp.take(self._fwd_X_gpu, self._fwd_scatter_src[h], axis=0,
                            out=self._level_bufs[h][:ns])
            s.synchronize()

            n_blocks = sum(1 for j in range(h) if self._fwd_sp[h][j] is not None)
            if self._verbose and n_blocks > 0:
                ev_start = cp.cuda.Event()
                ev_end = cp.cuda.Event()
                ev_start.record(s)

            for j in range(h):
                if self._fwd_sp[h][j] is None:
                    continue
                cslib.set_stream(s.ptr)
                self._spmm(self._fwd_sp[h][j],
                           self._level_dn[j], self._level_dn[h],
                           a1, b1, cdt, self._fwd_ext[h][j].data.ptr)

            if self._verbose and n_blocks > 0:
                ev_end.record(s)
                ev_end.synchronize()
                fwd_level_times.append(
                    (h, n_blocks, cp.cuda.get_elapsed_time(ev_start, ev_end)))
            else:
                s.synchronize()

        # Sel gather warmup (outside graph, exercises CuPy fancy indexing JIT)
        self._sel_gather(self._fwd_result_gpu, self._level_bufs)
        self._stream.synchronize()

        # Exercise _fwd_scatter on main stream (used in forward_matmat)
        self._fwd_scatter()
        self._stream.synchronize()

        if self._verbose:
            print("    Forward warmup OK")
            if fwd_level_times:
                fwd_level_times.sort(key=lambda x: x[2], reverse=True)
                print(f"    Forward warmup per-level (top {min(10, len(fwd_level_times))}):")
                for h, nb, t in fwd_level_times[:10]:
                    print(f"      Level {h:3d}:  {nb:3d} blocks,  {t:8.3f}ms")

        # --- Reset and capture ---
        for h in range(self._H):
            self._level_bufs[h].fill(0)
        cp.cuda.Device().synchronize()

        # Begin capture on main stream
        cslib.set_stream(self._stream.ptr)
        self._stream.begin_capture()

        # Fork: main -> all level streams
        fork_ev = cp.cuda.Event()
        fork_ev.record(self._stream)
        for h in range(self._H):
            streams[h].wait_event(fork_ev)

        # NOTE: scatter (zero + cp.take) runs OUTSIDE the graph in
        # forward_matmat() via _fwd_scatter().  Level bufs are already
        # initialized before graph launch; fork event guarantees visibility.

        # ready_events[h]: fired when level_bufs[h] is fully computed
        ready = [cp.cuda.Event() for _ in range(self._H)]

        # Level 0: ready after fork (scatter done before graph launch)
        ready[0].record(streams[0])

        # --- Wavefront: for each level h, queue blocks with event waits ---
        for h in range(1, self._H):
            cslib.set_stream(streams[h].ptr)
            for j in range(h):
                if self._fwd_sp[h][j] is None:
                    continue
                streams[h].wait_event(ready[j])
                self._spmm(self._fwd_sp[h][j],
                           self._level_dn[j], self._level_dn[h],
                           a1, b1, cdt, self._fwd_ext[h][j].data.ptr)
            ready[h].record(streams[h])

        # --- Join all level streams -> main stream ---
        for h in range(self._H):
            self._stream.wait_event(ready[h])

        # NOTE: sel gather is NOT captured — runs outside in forward_matmat()

        return self._stream.end_capture()

    # ------------------------------------------------------------------
    # graph capture (backward wavefront only — sel_T runs outside)
    # ------------------------------------------------------------------

    def _capture_bwd_graph(self, dtype):
        """Warm up backward wavefront, then capture as CUDA graph.

        The graph covers: fork -> wavefront SpMMs -> join.
        sel_T scatter_add and perm gather run outside the graph in
        backward_matmat() via _sel_scatter_add() and _bwd_gather().
        Level bufs must be pre-filled with sel_T data before graph launch.
        """
        cp = self._cp
        cslib = self._cslib
        cdt = cuda_dtype(dtype)
        a1, b1 = self._alpha.data.ptr, self._beta_one.data.ptr
        streams = self._level_streams

        # --- Warmup pass ---
        if self._verbose:
            print("  Warming up backward...")
        bwd_level_times = []

        # sel_T scatter_add warmup (outside graph, exercises CuPy add.at JIT)
        self._sel_scatter_add(self._bwd_X_gpu, self._level_bufs)
        self._stream.synchronize()

        # Backward blocks (reverse level order)
        for h in range(self._H - 2, -1, -1):
            s = streams[h]
            n_blocks = sum(1 for src in range(h + 1, self._H)
                           if self._fwd_sp[src][h] is not None)
            if self._verbose and n_blocks > 0:
                ev_start = cp.cuda.Event()
                ev_end = cp.cuda.Event()
                ev_start.record(s)

            for src in reversed(range(h + 1, self._H)):
                if self._fwd_sp[src][h] is None:
                    continue
                cslib.set_stream(s.ptr)
                j = src - h - 1
                self._spmm(self._fwd_sp[src][h],
                           self._level_dn[src], self._level_dn[h],
                           a1, b1, cdt, self._bwd_ext[h][j].data.ptr, op_a=_OP_T)

            if self._verbose and n_blocks > 0:
                ev_end.record(s)
                ev_end.synchronize()
                bwd_level_times.append(
                    (h, n_blocks, cp.cuda.get_elapsed_time(ev_start, ev_end)))
            else:
                s.synchronize()

        # Perm gather warmup (exercises CuPy JIT kernels)
        with self._stream:
            self._bwd_result_gpu.fill(0)
            for h in range(self._H):
                if self._bwd_gather_temp[h] is not None:
                    cp.take(self._level_bufs[h], self._bwd_gather_src[h], axis=0,
                            out=self._bwd_gather_temp[h])
                    self._bwd_result_gpu[self._bwd_gather_dst[h]] = \
                        self._bwd_gather_temp[h]
        self._stream.synchronize()

        # Also warm up _bwd_gather() on main stream (used after graph launch)
        self._bwd_gather()
        self._stream.synchronize()

        if self._verbose:
            print("    Backward warmup OK")
            if bwd_level_times:
                bwd_level_times.sort(key=lambda x: x[2], reverse=True)
                print(f"    Backward warmup per-level (top {min(10, len(bwd_level_times))}):")
                for h, nb, t in bwd_level_times[:10]:
                    print(f"      Level {h:3d}:  {nb:3d} blocks,  {t:8.3f}ms")

        # --- Reset and capture ---
        # Level bufs will be pre-filled by sel_T scatter_add before graph launch.
        # During capture they're zero — this is fine, the graph records kernel
        # code + addresses, not data values.
        for h in range(self._H):
            self._level_bufs[h].fill(0)
        self._bwd_result_gpu.fill(0)
        cp.cuda.Device().synchronize()

        # Begin capture on main stream
        cslib.set_stream(self._stream.ptr)
        self._stream.begin_capture()

        # Fork: main -> all level streams
        fork_ev = cp.cuda.Event()
        fork_ev.record(self._stream)
        for h in range(self._H):
            streams[h].wait_event(fork_ev)

        # Level H-1: immediately ready (sel_T data already in level_bufs
        # from before graph launch; no backward SpMMs target it)
        ready = [cp.cuda.Event() for _ in range(self._H)]
        ready[self._H - 1].record(streams[self._H - 1])

        # --- Wavefront (reverse): process from H-2 down to 0 ---
        # SpMMs accumulate (beta=1) into level_bufs which already hold sel_T data
        for h in range(self._H - 2, -1, -1):
            cslib.set_stream(streams[h].ptr)
            for src in reversed(range(h + 1, self._H)):
                if self._fwd_sp[src][h] is None:
                    continue
                j = src - h - 1
                streams[h].wait_event(ready[src])
                self._spmm(self._fwd_sp[src][h],
                           self._level_dn[src], self._level_dn[h],
                           a1, b1, cdt, self._bwd_ext[h][j].data.ptr, op_a=_OP_T)
            ready[h].record(streams[h])

        # --- Join all level streams -> main stream ---
        for h in range(self._H):
            self._stream.wait_event(ready[h])

        # NOTE: perm gather is NOT captured — runs outside in backward_matmat()

        return self._stream.end_capture()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def forward_matmat(self, X: np.ndarray) -> np.ndarray:
        cp = self._cp
        X = np.atleast_2d(X)
        k = X.shape[1]
        dtype = self._dtype
        use_graph = (self._fwd_exec is not None and k == self._k)

        if use_graph:
            timer = GpuTimer(cp, self._stream) if self._verbose else None
            if timer:
                timer.mark()
            self._fwd_X_gpu.set(X, stream=self._stream)
            self._stream.synchronize()
            if timer:
                timer.mark('H2D')
            # Scatter into level_bufs (outside graph — CuPy ops not capture-safe)
            self._fwd_scatter()
            self._stream.synchronize()
            if timer:
                timer.mark('scatter')
            # Graph: SpMM wavefront only (no CuPy ops inside)
            self._fwd_exec.launch(stream=self._stream)
            self._stream.synchronize()
            if timer:
                timer.mark('wavefront')
            # Sel gather (outside graph)
            self._sel_gather(self._fwd_result_gpu, self._level_bufs)
            self._stream.synchronize()
            if timer:
                timer.mark('sel')
            result = self._fwd_result_gpu.get()
            if timer:
                timer.mark('D2H')
                timer.report('forward (graph)')
            return result
        else:
            return self._run_fwd_dynamic(X, k, dtype)

    def backward_matmat(self, X: np.ndarray) -> np.ndarray:
        cp = self._cp
        X = np.atleast_2d(X)
        k = X.shape[1]
        dtype = self._dtype
        use_graph = (self._bwd_exec is not None and k == self._k)

        if use_graph:
            timer = GpuTimer(cp, self._stream) if self._verbose else None
            if timer:
                timer.mark()
            self._bwd_X_gpu.set(X, stream=self._stream)
            self._stream.synchronize()
            if timer:
                timer.mark('H2D')
            # sel_T scatter_add (outside graph — not capture-safe)
            self._sel_scatter_add(self._bwd_X_gpu, self._level_bufs)
            self._stream.synchronize()
            if timer:
                timer.mark('sel_T')
            self._bwd_exec.launch(stream=self._stream)
            self._stream.synchronize()
            if timer:
                timer.mark('wavefront')
            self._bwd_gather()
            self._stream.synchronize()
            if timer:
                timer.mark('perm')
            result = self._bwd_result_gpu.get()
            if timer:
                timer.mark('D2H')
                timer.report('backward (graph)')
            return result
        else:
            return self._run_bwd_dynamic(X, k, dtype)

    # ------------------------------------------------------------------
    # dynamic (non-graph) execution
    # ------------------------------------------------------------------

    def _run_fwd_dynamic(self, X, k, dtype):
        """Forward pass without CUDA graphs (any k)."""
        cp = self._cp
        cslib = self._cslib
        off = self._level_offsets
        cdt = cuda_dtype(dtype)
        a1, b1 = self._alpha.data.ptr, self._beta_one.data.ptr

        timer = GpuTimer(cp, self._stream) if self._verbose else None
        if timer:
            timer.mark()

        # All CuPy ops on self._stream to avoid race with non-blocking stream
        with self._stream:
            level_bufs = []
            for h in range(self._H):
                sz = int(off[h + 1]) - int(off[h])
                level_bufs.append(cp.zeros((sz, k), dtype=dtype, order='C'))

            X_gpu = cp.asarray(X, dtype=dtype, order='C')

            # Scatter permuted input
            for h in range(self._H):
                ns = self._fwd_scatter_ns[h]
                if ns > 0:
                    cp.take(X_gpu, self._fwd_scatter_src[h], axis=0,
                            out=level_bufs[h][:ns])

        if timer:
            timer.mark('H2D+scatter')

        # Block SpMMs (single stream)
        cslib.set_stream(self._stream.ptr)
        with self._stream:
            for h in range(1, self._H):
                for j in range(h):
                    if self._fwd_sp[h][j] is None:
                        continue
                    B_desc = self._create_level_dnmat(level_bufs[j], k, dtype, ld=k)
                    C_desc = self._create_level_dnmat(level_bufs[h], k, dtype, ld=k)
                    ext = cslib.spmm_buffer_size(
                        self._cp, self._algorithm, _OP_N, _OP_N,
                        a1, self._fwd_sp[h][j], B_desc, b1, C_desc, cdt,
                    )
                    self._spmm(self._fwd_sp[h][j], B_desc, C_desc,
                               a1, b1, cdt, ext.data.ptr)
                    cslib.destroy_dn_mat(B_desc)
                    cslib.destroy_dn_mat(C_desc)
        self._stream.synchronize()

        if timer:
            timer.mark('wavefront')

        # Sel gather
        with self._stream:
            result_gpu = cp.zeros((self._m, k), dtype=dtype, order='C')
            for h in range(self._H):
                idx = self._sel_mut_idx[h]
                if len(idx) > 0:
                    result_gpu[idx] = level_bufs[h][self._sel_node_local[h]]
        self._stream.synchronize()

        if timer:
            timer.mark('sel')

        result = result_gpu.get()
        if timer:
            timer.mark('D2H')
            timer.report('forward (dynamic)')
        return result

    def _run_bwd_dynamic(self, X, k, dtype):
        """Backward pass without CUDA graphs (any k)."""
        cp = self._cp
        cslib = self._cslib
        off = self._level_offsets
        cdt = cuda_dtype(dtype)
        a1, b1 = self._alpha.data.ptr, self._beta_one.data.ptr

        timer = GpuTimer(cp, self._stream) if self._verbose else None
        if timer:
            timer.mark()

        # All CuPy ops on self._stream to avoid race with non-blocking stream
        with self._stream:
            level_bufs = []
            for h in range(self._H):
                sz = int(off[h + 1]) - int(off[h])
                level_bufs.append(cp.zeros((sz, k), dtype=dtype, order='C'))

            X_gpu = cp.asarray(X, dtype=dtype, order='C')
            for h in range(self._H):
                idx = self._sel_mut_idx[h]
                if len(idx) > 0:
                    cp.add.at(level_bufs[h], self._sel_node_local[h], X_gpu[idx])
        self._stream.synchronize()

        if timer:
            timer.mark('H2D+sel_T')

        # Backward block SpMMs (single stream, reverse level order)
        # AT_blocks[h][j] == A_blocks[src][h]^T  where src = h+1+j
        cslib.set_stream(self._stream.ptr)
        with self._stream:
            for h in range(self._H - 2, -1, -1):
                for src in reversed(range(h + 1, self._H)):
                    sp_desc = self._fwd_sp[src][h]
                    if sp_desc is None:
                        continue
                    B_desc = self._create_level_dnmat(level_bufs[src], k, dtype, ld=k)
                    C_desc = self._create_level_dnmat(level_bufs[h], k, dtype, ld=k)
                    ext = cslib.spmm_buffer_size(
                        self._cp, self._algorithm, _OP_T, _OP_N,
                        a1, sp_desc, B_desc, b1, C_desc, cdt,
                    )
                    self._spmm(sp_desc, B_desc, C_desc, a1, b1, cdt, ext.data.ptr, op_a=_OP_T)
                    cslib.destroy_dn_mat(B_desc)
                    cslib.destroy_dn_mat(C_desc)
        self._stream.synchronize()

        if timer:
            timer.mark('wavefront')

        # Inverse permutation (gather)
        with self._stream:
            result_gpu = cp.zeros((self._n, k), dtype=dtype, order='C')
            for h in range(self._H):
                if len(self._bwd_gather_dst[h]) > 0:
                    temp = cp.take(level_bufs[h], self._bwd_gather_src[h], axis=0)
                    result_gpu[self._bwd_gather_dst[h]] = temp
        self._stream.synchronize()

        result = result_gpu.get()
        if timer:
            timer.mark('perm+D2H')
            timer.report('backward (dynamic)')
        return result

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------

    def __del__(self):
        cslib = getattr(self, '_cslib', None)
        if cslib is None:
            return

        # Dense descriptors (graph mode)
        for d in getattr(self, '_level_dn', []):
            if d is not None:
                cslib._lib.cusparseDestroyDnMat(d)

        # Sparse descriptors
        for sp_list in getattr(self, '_fwd_sp', []):
            for d in sp_list:
                if d is not None:
                    cslib._lib.cusparseDestroySpMat(d)

        cslib.destroy()
