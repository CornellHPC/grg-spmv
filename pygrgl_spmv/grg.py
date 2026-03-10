"""SpmvGRG - Level-wise sparse matmul for GRG-based genotype matrices."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pygrgl
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee

from pygrgl_spmv.backends import Backend
from pygrgl_spmv.backends.types import Direction, InitMode, parse_direction
from pygrgl_spmv.io import load_operator_npz, save_operator_npz


_COMMON_BACKEND_KEYS = {"type", "log_level", "plan_up", "plan_down"}
_MKL_BACKEND_KEYS = _COMMON_BACKEND_KEYS
_CUSPARSE_BACKEND_KEYS = _COMMON_BACKEND_KEYS


def _cache_path_for_grg(grg_path: Path, cache_root: Path) -> Path:
    resolved = grg_path.expanduser().resolve()
    if resolved.is_absolute():
        if resolved.drive:
            drive = resolved.drive.replace(":", "")
            rel_parts = ["_drive_" + drive, *resolved.parts[1:]]
        else:
            rel_parts = ["_abs", *resolved.parts[1:]]
    else:
        rel_parts = ["_rel", *resolved.parts]
    return cache_root.joinpath(*rel_parts).with_suffix(".pygrgl_spmv.npz")


def _rcm_bipartite(A: sp.spmatrix, index_dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """
    Bipartite RCM for a rectangular block A_{h,h-1}.

    Builds B = [[0, A], [A^T, 0]] and applies Reverse Cuthill-McKee.
    Returns both row and column permutations.
    """
    nrows, ncols = A.shape
    A_csr = sp.csr_matrix(A)

    top = sp.hstack([sp.csr_matrix((nrows, nrows)), A_csr], format="csr")
    bottom = sp.hstack([A_csr.T, sp.csr_matrix((ncols, ncols))], format="csr")
    B = sp.vstack([top, bottom], format="csr")

    perm_full = reverse_cuthill_mckee(B, symmetric_mode=True)

    row_indices = perm_full[perm_full < nrows]
    col_indices = perm_full[perm_full >= nrows] - nrows
    if row_indices.size != nrows or col_indices.size != ncols:
        raise RuntimeError(
            "reverse_cuthill_mckee returned incomplete bipartite permutation "
            f"(rows={row_indices.size}/{nrows}, cols={col_indices.size}/{ncols})"
        )
    if not np.array_equal(np.sort(row_indices), np.arange(nrows)):
        raise RuntimeError("Row permutation from reverse_cuthill_mckee is invalid")
    if not np.array_equal(np.sort(col_indices), np.arange(ncols)):
        raise RuntimeError("Column permutation from reverse_cuthill_mckee is invalid")
    return row_indices.astype(index_dtype, copy=False), col_indices.astype(index_dtype, copy=False)


def _validate_backend_config_keys(config: dict[str, Any], allowed_keys: set[str], backend_type: str) -> None:
    unknown = sorted(set(config) - allowed_keys)
    if unknown:
        raise ValueError(
            f"Unknown backend_config key(s) for backend={backend_type!r}: {unknown}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )


def _create_backend(config: dict[str, Any]):
    """
    Create a backend from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Backend configuration with keys:
        - 'type': str — Backend type ('mkl', 'cusparse')
        - Additional backend-specific parameters
    """
    backend_type = config.get("type", "mkl")
    log_level = str(config.get("log_level", "WARNING"))
    match backend_type:
        case "mkl":
            from pygrgl_spmv.backends.mkl import MklBackend, MklPlan

            _validate_backend_config_keys(config, _MKL_BACKEND_KEYS, backend_type)
            return MklBackend(
                plan_up=MklPlan.from_any(config["plan_up"]),
                plan_down=MklPlan.from_any(config["plan_down"]),
                log_level=log_level,
            )
        case "cusparse":
            from pygrgl_spmv.backends.cusparse import CusparseBackend, CusparsePlan

            _validate_backend_config_keys(config, _CUSPARSE_BACKEND_KEYS, backend_type)
            return CusparseBackend(
                plan_up=CusparsePlan.from_any(config["plan_up"]),
                plan_down=CusparsePlan.from_any(config["plan_down"]),
                log_level=log_level,
            )
        case _:
            raise ValueError(f"No config handler for backend: {backend_type!r}")


class SpmvGRG:
    """
    Matmul-focused GRG operator for genotype matrix G (n x m).

    Parameters
    ----------
    path : str or Path
        Path to a .grg file.
    backend_config : dict
        Backend configuration with keys:
        - 'type': str — Backend type ('mkl', 'cusparse')
        - Additional backend-specific parameters
    dtype : numpy dtype
        Data type for computation.
    index_dtype : numpy dtype
        Index data type for permutation arrays and indices.
    Public Attributes
    -----------------
    n : int
        Number of samples (rows of G).
    m : int
        Number of mutations (columns of G).
    K : int
        Total number of nodes in the GRG.
    """

    def __init__(self, path, backend_config: dict[str, Any], dtype, index_dtype, cache_dir: str | Path = "pygrgl_spmv_cache"):
        self._dtype = np.dtype(dtype)
        self._index_dtype = np.dtype(index_dtype)
        log_level = str(backend_config.get("log_level", "WARNING"))
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(getattr(logging, log_level.upper(), logging.WARNING))
        self._backend = _create_backend(backend_config)

        path = Path(path)
        assert path.suffix == ".grg", f"Not a GRG file: {path}"
        cache_root = Path(cache_dir).expanduser()
        npz_path = _cache_path_for_grg(path, cache_root)
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        if npz_path.exists():
            self._logger.info("Loading cached SpmvGRG from %s", npz_path)
            try:
                self._load_from_npz(npz_path)
            except (KeyError, ValueError) as exc:
                # Stale/incomplete cache files are expected during active refactors.
                # Rebuild from the source GRG instead of carrying legacy load paths.
                self._logger.warning(
                    "Cached SpmvGRG at %s is invalid (%s); rebuilding from %s",
                    npz_path,
                    exc,
                    path,
                )
                grg = pygrgl.load_immutable_grg(str(path), load_up_edges=True)
                self._build_from_grg(grg)
                self._build_init_bias_cache()
                save_operator_npz(self, self.A_blocks, npz_path)
        else:
            self._logger.info("Building SpmvGRG with %s", path)
            grg = pygrgl.load_immutable_grg(str(path), load_up_edges=True)
            self._build_from_grg(grg)
            self._build_init_bias_cache()
            save_operator_npz(self, self.A_blocks, npz_path)

        # Initialize backend with block-wise matrices and permutations.
        self._backend.setup(
            self.A_blocks,
            self.level_offsets,
            self.n,
            self.K,
            self.sel_mut,
            self.sel_miss,
            self.sample_perm,
            self._inv_sample_perm,
            self.coalescence_counts,
            self._dtype,
        )
        del self.A_blocks

    def _build_init_bias_cache(self) -> None:
        """
        Precompute per-output init bias vectors for fast init='xtx' and init=vector.

        This removes per-matmul O(K) init additions and uses setup-time work instead.
        """
        helper = Backend(
            plan_up=Backend.plan(fmt="CSR", store="N", k_hint=None),
            plan_down=Backend.plan(fmt="CSC", store="T", k_hint=None),
            log_level="WARNING",
        )
        helper.setup(
            self.A_blocks,
            self.level_offsets,
            self.n,
            self.K,
            self.sel_mut,
            self.sel_miss,
            self.sample_perm,
            self._inv_sample_perm,
            self.coalescence_counts,
            self._dtype,
        )
        zeros_up = np.zeros((self.n, 1), dtype=self._dtype)
        zeros_down = np.zeros((self.m, 1), dtype=self._dtype)
        init_vec = np.ones(1, dtype=self._dtype)

        up_bias, _ = helper.run_up(zeros_up, init_mode=InitMode.VECTOR, init=init_vec, need_miss_output=False)
        down_bias = helper.run_down(zeros_down, init_mode=InitMode.VECTOR, init=init_vec, miss=None)
        self._init_vector_up_bias = np.asarray(up_bias[:, 0], dtype=self._dtype).reshape(self.m)
        self._init_vector_down_bias = np.asarray(down_bias[:, 0], dtype=self._dtype).reshape(self.n)
        self._init_xtx_up_bias = None
        self._init_xtx_down_bias = None
        if self.coalescence_counts is not None:
            up_xtx, _ = helper.run_up(zeros_up, init_mode=InitMode.XTX, init=None, need_miss_output=False)
            down_xtx = helper.run_down(zeros_down, init_mode=InitMode.XTX, init=None, miss=None)
            self._init_xtx_up_bias = np.asarray(up_xtx[:, 0], dtype=self._dtype).reshape(self.m)
            self._init_xtx_down_bias = np.asarray(down_xtx[:, 0], dtype=self._dtype).reshape(self.n)

    def _extract_edges_and_heights(self, grg, K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract directed edges (parent -> child) and node heights.

        Heights are computed in node-id order, assuming children are already finalized.
        """
        index_dtype = self._index_dtype
        heights = np.zeros(K, dtype=np.int64)
        row_chunks: list[np.ndarray] = []
        col_chunks: list[np.ndarray] = []

        for node_id in range(K):
            children = np.asarray(grg.get_down_edges(node_id), dtype=index_dtype)
            if children.size == 0:
                continue
            row_chunks.append(np.full(int(children.size), node_id, dtype=index_dtype))
            col_chunks.append(children)
            heights[node_id] = int(heights[children].max()) + 1

        if row_chunks:
            row_arr = np.concatenate(row_chunks)
            col_arr = np.concatenate(col_chunks)
        else:
            row_arr = np.empty(0, dtype=index_dtype)
            col_arr = np.empty(0, dtype=index_dtype)
        return row_arr, col_arr, heights

    def _build_level_order(self, heights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build height-sorted node order and level offsets."""
        index_dtype = self._index_dtype
        K = int(heights.shape[0])

        perm_height = np.argsort(heights, kind="stable").astype(index_dtype, copy=False)
        inv_perm_height = np.empty(K, dtype=index_dtype)
        inv_perm_height[perm_height] = np.arange(K, dtype=index_dtype)

        sorted_heights = heights[perm_height]
        if np.any(np.diff(sorted_heights) < 0):
            raise RuntimeError("Heights must be non-decreasing after sorting")
        changes = np.flatnonzero(np.diff(sorted_heights)) + 1
        level_offsets = np.concatenate(
            [
                np.array([0], dtype=index_dtype),
                changes.astype(index_dtype, copy=False),
                np.array([K], dtype=index_dtype),
            ]
        )

        # Map original node id -> compact level id.
        num_levels = int(level_offsets.size - 1)
        node_levels = np.empty(K, dtype=np.int32)
        for h in range(num_levels):
            lo, hi = int(level_offsets[h]), int(level_offsets[h + 1])
            node_levels[perm_height[lo:hi]] = h

        return perm_height, inv_perm_height, level_offsets, node_levels

    def _build_level_perms(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        inv_perm_height: np.ndarray,
        level_offsets: np.ndarray,
        dst_levels: np.ndarray,
        src_order: np.ndarray,
        src_offsets: np.ndarray,
    ) -> list[np.ndarray]:
        """Compute always-on per-level permutations (RCM + min-column ordering)."""
        index_dtype = self._index_dtype
        num_levels = int(level_offsets.size - 1)

        level_perms = [
            np.arange(int(level_offsets[h + 1] - level_offsets[h]), dtype=index_dtype)
            for h in range(num_levels)
        ]

        for h in range(1, num_levels):
            lo_e = int(src_offsets[h])
            hi_e = int(src_offsets[h + 1])
            if lo_e == hi_e:
                continue
            edge_idx = src_order[lo_e:hi_e]
            prev_idx = edge_idx[dst_levels[edge_idx] == (h - 1)]
            if prev_idx.size == 0:
                continue

            row_local = (inv_perm_height[rows[prev_idx]] - level_offsets[h]).astype(np.int64, copy=False)
            col_local = (inv_perm_height[cols[prev_idx]] - level_offsets[h - 1]).astype(np.int64, copy=False)

            row_size = int(level_offsets[h + 1] - level_offsets[h])
            col_size = int(level_offsets[h] - level_offsets[h - 1])

            if h == 1:
                data = np.ones(prev_idx.size, dtype=self._dtype)
                A_block = sp.csr_matrix((data, (row_local, col_local)), shape=(row_size, col_size))
                A_block.sum_duplicates()
                if A_block.nnz > 0:
                    A_block.data.fill(1)
                row_perm, col_perm = _rcm_bipartite(A_block, index_dtype)
                level_perms[1] = row_perm
                level_perms[0] = col_perm
                continue

            # Levels >=2 have fixed previous-level columns; sort by minimum predecessor column.
            min_col = np.full(row_size, col_size, dtype=np.int64)
            np.minimum.at(min_col, row_local, col_local)
            level_perms[h] = np.argsort(min_col, kind="stable").astype(index_dtype, copy=False)

        return level_perms

    def _build_blocks_from_edges(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        inv_final_perm: np.ndarray,
        level_offsets: np.ndarray,
        dst_levels: np.ndarray,
        src_order: np.ndarray,
        src_offsets: np.ndarray,
    ) -> list[list[sp.csr_matrix]]:
        """
        Build block CSR matrices directly from edge buckets.

        This avoids materializing a full KxK adjacency matrix and then slicing it.
        """
        dtype = self._dtype
        num_levels = int(level_offsets.size - 1)
        A_blocks: list[list[sp.csr_matrix]] = []

        for h in range(num_levels):
            row_size = int(level_offsets[h + 1] - level_offsets[h])
            level_blocks = [
                sp.csr_matrix((row_size, int(level_offsets[j + 1] - level_offsets[j])), dtype=dtype)
                for j in range(h)
            ]
            lo_e = int(src_offsets[h])
            hi_e = int(src_offsets[h + 1])
            if lo_e == hi_e:
                A_blocks.append(level_blocks)
                continue

            edge_idx = src_order[lo_e:hi_e]
            dst_sub = dst_levels[edge_idx]
            order_dst = np.argsort(dst_sub, kind="stable")
            edge_sorted = edge_idx[order_dst]
            dst_sorted = dst_sub[order_dst]

            splits = np.flatnonzero(np.diff(dst_sorted)) + 1
            bounds = np.concatenate([np.array([0]), splits, np.array([edge_sorted.size])])
            for b0, b1 in zip(bounds[:-1], bounds[1:]):
                j = int(dst_sorted[int(b0)])
                if j >= h:
                    raise RuntimeError(
                        f"Invalid GRG edge bucket (source level {h}, target level {j}); expected target < source"
                    )
                pair_idx = edge_sorted[int(b0) : int(b1)]
                if pair_idx.size == 0:
                    continue
                row_local = (inv_final_perm[rows[pair_idx]] - level_offsets[h]).astype(np.int64, copy=False)
                col_local = (inv_final_perm[cols[pair_idx]] - level_offsets[j]).astype(np.int64, copy=False)
                col_size = int(level_offsets[j + 1] - level_offsets[j])
                data = np.ones(pair_idx.size, dtype=dtype)
                blk = sp.csr_matrix((data, (row_local, col_local)), shape=(row_size, col_size))
                blk.sum_duplicates()
                if blk.nnz > 0:
                    blk.data.fill(1)
                level_blocks[j] = blk

            A_blocks.append(level_blocks)

        return A_blocks

    def _build_selectors(self, grg, inv_final_perm: np.ndarray, m: int, K: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Build mutation and missingness selector matrices."""
        index_dtype = self._index_dtype
        dtype = self._dtype
        mut_rows: list[int] = []
        mut_cols_orig: list[int] = []
        miss_rows: list[int] = []
        miss_cols_orig: list[int] = []

        for mut_id, mut_node, miss_node in grg.get_mutation_node_miss():
            if mut_node != pygrgl.INVALID_NODE:
                mut_rows.append(int(mut_id))
                mut_cols_orig.append(int(mut_node))
            if miss_node != pygrgl.INVALID_NODE:
                miss_rows.append(int(mut_id))
                miss_cols_orig.append(int(miss_node))

        if mut_rows:
            mut_rows_arr = np.asarray(mut_rows, dtype=index_dtype)
            mut_cols = inv_final_perm[np.asarray(mut_cols_orig, dtype=index_dtype)]
            sel_mut = sp.csr_matrix(
                (np.ones(mut_rows_arr.size, dtype=dtype), (mut_rows_arr, mut_cols)),
                shape=(m, K),
            )
            sel_mut.sum_duplicates()
            if sel_mut.nnz > 0:
                sel_mut.data.fill(1)
        else:
            sel_mut = sp.csr_matrix((m, K), dtype=dtype)

        if miss_rows:
            miss_rows_arr = np.asarray(miss_rows, dtype=index_dtype)
            miss_cols = inv_final_perm[np.asarray(miss_cols_orig, dtype=index_dtype)]
            sel_miss = sp.csr_matrix(
                (np.ones(miss_rows_arr.size, dtype=dtype), (miss_rows_arr, miss_cols)),
                shape=(m, K),
            )
            sel_miss.sum_duplicates()
            if sel_miss.nnz > 0:
                sel_miss.data.fill(1)
        else:
            sel_miss = sp.csr_matrix((m, K), dtype=dtype)

        return sel_mut, sel_miss

    def _build_coalescence_counts(self, grg) -> np.ndarray | None:
        """Load and validate per-node coalescence counts in internal node order."""
        counts_orig = np.array(
            [grg.get_num_individual_coals(i) for i in range(grg.num_nodes)],
            dtype=np.int64,
        )
        not_set = int(pygrgl.COAL_COUNT_NOT_SET)
        has_individual_coals = getattr(grg, "has_individual_coals", None)
        if has_individual_coals is None:
            # Older pygrgl releases do not expose has_individual_coals.
            # Treat "all internal nodes are NOT_SET" as "coalescences absent".
            has_individual_coals = bool(np.any(counts_orig[grg.num_samples :] != not_set))
        if not bool(has_individual_coals):
            return None

        missing_mask = counts_orig == not_set
        internal_missing_count = int(np.sum(missing_mask[grg.num_samples :]))
        if internal_missing_count > 0:
            raise ValueError(
                "GRG has missing coalescence counts for "
                f"{internal_missing_count} internal nodes; init='xtx' requires complete internal counts."
            )
        sample_missing = missing_mask[: grg.num_samples]
        if np.any(sample_missing):
            counts_orig[np.flatnonzero(sample_missing)] = 0

        return counts_orig[self.node_perm]

    def _build_from_grg(self, grg):
        """Build SpmvGRG from a GRG object."""
        n, m, K = grg.num_samples, grg.num_mutations, grg.num_nodes
        index_dtype = self._index_dtype

        rows, cols, heights = self._extract_edges_and_heights(grg, K)
        perm_height, inv_perm_height, level_offsets, node_levels = self._build_level_order(heights)
        self.level_offsets = level_offsets
        num_levels = int(level_offsets.size - 1)
        E = int(rows.size)

        src_levels = node_levels[rows] if E else np.empty(0, dtype=np.int32)
        dst_levels = node_levels[cols] if E else np.empty(0, dtype=np.int32)
        if E and np.any(src_levels <= dst_levels):
            raise RuntimeError("Invalid GRG topology: expected every edge to point from higher to lower level")

        # Group edges by source level once; reused for both level ordering and block build.
        src_order = np.argsort(src_levels, kind="stable") if E else np.empty(0, dtype=np.int64)
        src_counts = np.bincount(src_levels, minlength=num_levels) if E else np.zeros(num_levels, dtype=np.int64)
        src_offsets = np.zeros(num_levels + 1, dtype=np.int64)
        src_offsets[1:] = np.cumsum(src_counts, dtype=np.int64)

        level_perms = self._build_level_perms(
            rows=rows,
            cols=cols,
            inv_perm_height=inv_perm_height,
            level_offsets=level_offsets,
            dst_levels=dst_levels,
            src_order=src_order,
            src_offsets=src_offsets,
        )

        within_perm = np.arange(K, dtype=index_dtype)
        for h in range(num_levels):
            lo, hi = int(level_offsets[h]), int(level_offsets[h + 1])
            perm = level_perms[h]
            if perm.shape[0] != (hi - lo):
                raise RuntimeError(f"Invalid level permutation for level {h}: got {perm.shape[0]}, expected {hi - lo}")
            within_perm[lo:hi] = lo + perm

        final_perm = perm_height[within_perm]
        inv_final_perm = np.empty(K, dtype=index_dtype)
        inv_final_perm[final_perm] = np.arange(K, dtype=index_dtype)

        self.node_perm = final_perm.copy()
        self._inv_node_perm = inv_final_perm.copy()
        self.sample_perm = final_perm[:n].copy()
        self._inv_sample_perm = inv_final_perm[:n].copy()

        self.A_blocks = self._build_blocks_from_edges(
            rows=rows,
            cols=cols,
            inv_final_perm=inv_final_perm,
            level_offsets=level_offsets,
            dst_levels=dst_levels,
            src_order=src_order,
            src_offsets=src_offsets,
        )

        for h in range(num_levels):
            if len(self.A_blocks[h]) != h:
                raise RuntimeError(f"Invalid number of blocks at level {h}: got {len(self.A_blocks[h])}, expected {h}")
            for j, blk in enumerate(self.A_blocks[h]):
                expected_shape = (int(level_offsets[h + 1] - level_offsets[h]), int(level_offsets[j + 1] - level_offsets[j]))
                if blk.shape != expected_shape:
                    raise RuntimeError(
                        f"Invalid block shape for A_blocks[{h}][{j}]: got {blk.shape}, expected {expected_shape}"
                    )

        self.sel_mut, self.sel_miss = self._build_selectors(grg, inv_final_perm, m, K)

        self.n = n
        self.m = m
        self.K = K
        self.ploidy = int(grg.ploidy)
        self.num_individuals = int(grg.num_individuals)
        self.sample_to_individual = np.arange(n, dtype=index_dtype) // max(self.ploidy, 1)
        self.coalescence_counts = self._build_coalescence_counts(grg)

    def _load_from_npz(self, npz_path):
        """Load operator state from NPZ file."""
        data = load_operator_npz(npz_path, self._dtype, self._index_dtype)
        self.n = data["n"]
        self.m = data["m"]
        self.K = data["K"]
        self.ploidy = data["ploidy"]
        self.num_individuals = data["num_individuals"]
        self.level_offsets = data["level_offsets"]
        self.sample_perm = data["sample_perm"]
        self._inv_sample_perm = data["_inv_sample_perm"]
        self.node_perm = data["node_perm"]
        self._inv_node_perm = data["_inv_node_perm"]
        self.sample_to_individual = data["sample_to_individual"]
        self.coalescence_counts = data["coalescence_counts"]
        self._init_vector_up_bias = np.asarray(data["init_vector_up_bias"], dtype=self._dtype)
        self._init_vector_down_bias = np.asarray(data["init_vector_down_bias"], dtype=self._dtype)
        self._init_xtx_up_bias = (
            None if data["init_xtx_up_bias"] is None else np.asarray(data["init_xtx_up_bias"], dtype=self._dtype)
        )
        self._init_xtx_down_bias = (
            None if data["init_xtx_down_bias"] is None else np.asarray(data["init_xtx_down_bias"], dtype=self._dtype)
        )
        self.sel_mut = data["sel_mut"]
        self.sel_miss = data["sel_miss"]
        self.A_blocks = data["A_blocks"]

    @property
    def shape(self) -> tuple[int, int]:
        """Return (num_samples, num_mutations)."""
        return (self.n, self.m)

    def _parse_direction(self, direction: Any) -> Direction:
        match direction:
            case str():
                return parse_direction(direction)
            case pygrgl.TraversalDirection.UP:
                return Direction.UP
            case pygrgl.TraversalDirection.DOWN:
                return Direction.DOWN
            case _:
                raise ValueError(
                    f"Unknown direction: {direction!r}. Expected 'up', 'down', "
                    "pygrgl.TraversalDirection.UP, or pygrgl.TraversalDirection.DOWN"
                )

    def _parse_init(
        self,
        init: Any,
        rows: int,
        input_dtype: np.dtype,
    ) -> tuple[InitMode, np.ndarray | None]:
        if init is None:
            return InitMode.NONE, None
        if isinstance(init, str):
            if init != "xtx":
                raise ValueError(f"Unexpected init value: {init}")
            if self.coalescence_counts is None:
                raise ValueError(
                    "init='xtx' requires per-node coalescence counts in the GRG. "
                    "This GRG was loaded without coalescence counts."
                )
            return InitMode.XTX, None

        init_arr = np.asarray(init)
        # Project policy: bool init is valid only when the input dtype is also bool.
        if init_arr.dtype != input_dtype:
            raise TypeError(
                f"The init matrix must match the dtype of the input matrix. Got: {init_arr.dtype}"
            )

        if init_arr.ndim == 1:
            if init_arr.shape[0] != rows:
                raise ValueError(
                    "If init has a single dimension, it must match the number of rows in the input matrix"
                )
            return InitMode.VECTOR, np.asarray(init_arr, dtype=self._dtype, order="C")

        if init_arr.ndim == 2:
            if init_arr.shape != (rows, self.K):
                raise ValueError("If init is a matrix, it must match the dimensions ROW x NODES")
            # Backends expect node-major (K x rows) in internal node order.
            init_nodes = init_arr[:, self.node_perm].T
            return InitMode.MATRIX, np.asarray(init_nodes, dtype=self._dtype, order="C")

        raise ValueError("init must be None, 'xtx', a vector, or a matrix")

    def _validate_miss(
        self,
        miss: Any,
        rows: int,
        direction: Direction,
        input_dtype: np.dtype,
    ) -> np.ndarray:
        miss_arr = np.asarray(miss)
        if miss_arr.dtype != input_dtype:
            raise TypeError(
                f'The "miss" input must match the dtype of the input matrix. Got: {miss_arr.dtype}'
            )
        if miss_arr.ndim != 2:
            raise ValueError(f'"miss" must be a two-dimension numpy array (matrix). ndim={miss_arr.ndim}')
        if miss_arr.shape[0] != rows:
            raise ValueError(
                f'"miss" has {miss_arr.shape[0]} rows, but must match the input/output matrices ({rows})'
            )
        if miss_arr.shape[1] != self.m:
            match direction:
                case Direction.DOWN:
                    raise ValueError(
                        'The "miss" matrix must match the number of columns in the input matrix. '
                        f"Got: {miss_arr.shape[1]}"
                    )
                case Direction.UP:
                    raise ValueError(
                        'The "miss" matrix must match the number of columns in the output matrix. '
                        f"Got: {miss_arr.shape[1]}"
                    )
                case _:
                    raise ValueError(f"Unsupported direction for miss validation: {direction!r}")
        return miss_arr

    def matmul(
        self,
        input_matrix: np.ndarray,
        direction: Any,
        by_individual: bool = False,
        init: Any = None,
        miss: Any = None,
    ) -> np.ndarray:
        """
        pygrgl-compatible matrix multiplication.

        Input and output are row-major (rows x cols), matching pygrgl.matmul.
        """
        timing_pairs: list[tuple[str, float]] = []
        total_t0 = perf_counter()

        def _record(name: str, t0: float) -> None:
            if self._logger.isEnabledFor(logging.INFO):
                timing_pairs.append((name, (perf_counter() - t0) * 1000.0))

        t0 = perf_counter()
        X_in = np.asarray(input_matrix)
        if X_in.ndim != 2:
            raise ValueError("matmul() only supports two-dimensional numpy arrays as input.")
        rows, cols = X_in.shape
        if rows == 0 or cols == 0:
            raise ValueError("matmul() requires non-zero dimensions.")

        direction_name = self._parse_direction(direction)
        expect_sample_cols = self.num_individuals if by_individual else self.n
        if direction_name == Direction.UP:
            if cols != expect_sample_cols:
                raise ValueError(
                    "Input matrix has wrong number of columns for UP direction "
                    "(numSamples or numIndividuals depending on by_individual)"
                )
        else:
            if cols != self.m:
                raise ValueError("Input matrix has wrong number of columns for DOWN direction (numMutations)")
        _record("validate_input", t0)

        if init is not None and miss is not None:
            raise ValueError('The "miss" parameter cannot be mixed with the "init" parameter')

        t0 = perf_counter()
        init_mode, init_payload = self._parse_init(init, rows, X_in.dtype)
        _record("parse_init", t0)

        backend_init_mode = init_mode
        backend_init_payload = init_payload
        if init_mode in (InitMode.VECTOR, InitMode.XTX):
            # Fast path: use precomputed biases instead of node-space init each call.
            backend_init_mode = InitMode.NONE
            backend_init_payload = None

        t0 = perf_counter()
        X = np.asarray(X_in, dtype=self._dtype, order="C")
        X_col = X.T
        _record("cast_and_transpose", t0)

        if direction_name == Direction.UP:
            miss_arr = None
            if miss is not None:
                t0 = perf_counter()
                miss_arr = self._validate_miss(miss, rows, direction_name, X_in.dtype)
                _record("validate_miss", t0)

            if by_individual:
                t0 = perf_counter()
                X_col = X_col[self.sample_to_individual]
                _record("by_individual_expand", t0)

            t0 = perf_counter()
            result_col, miss_col = self._backend.run_up(
                X_col,
                init_mode=backend_init_mode,
                init=backend_init_payload,
                need_miss_output=(miss_arr is not None),
            )
            _record("backend_run", t0)

            if init_mode == InitMode.XTX:
                t0 = perf_counter()
                result_col += self._init_xtx_up_bias[:, None]
                _record("init_bias_add", t0)
            elif init_mode == InitMode.VECTOR:
                assert init_payload is not None
                t0 = perf_counter()
                result_col += self._init_vector_up_bias[:, None] * init_payload[None, :]
                _record("init_bias_add", t0)

            if miss_arr is not None and miss_col is not None:
                t0 = perf_counter()
                miss_arr += miss_col.T.astype(miss_arr.dtype, copy=False)
                _record("write_miss", t0)

            t0 = perf_counter()
            out = result_col.T.astype(self._dtype, copy=False)
            _record("output_cast", t0)

            if self._logger.isEnabledFor(logging.INFO):
                self._print_matmul_timings(
                    direction=direction_name,
                    rows=rows,
                    cols=cols,
                    by_individual=by_individual,
                    init_mode=init_mode.value,
                    miss=(miss is not None),
                    timings=timing_pairs,
                    total_ms=(perf_counter() - total_t0) * 1000.0,
                )
            return out

        # direction == down
        miss_col = None
        if miss is not None:
            t0 = perf_counter()
            miss_arr = self._validate_miss(miss, rows, direction_name, X_in.dtype)
            miss_col = np.asarray(miss_arr.T, dtype=self._dtype, order="C")
            _record("validate_miss", t0)

        t0 = perf_counter()
        result_col = self._backend.run_down(
            X_col,
            miss=miss_col,
            init_mode=backend_init_mode,
            init=backend_init_payload,
        )
        _record("backend_run", t0)

        if init_mode == InitMode.XTX:
            t0 = perf_counter()
            result_col += self._init_xtx_down_bias[:, None]
            _record("init_bias_add", t0)
        elif init_mode == InitMode.VECTOR:
            assert init_payload is not None
            t0 = perf_counter()
            result_col += self._init_vector_down_bias[:, None] * init_payload[None, :]
            _record("init_bias_add", t0)

        if by_individual:
            t0 = perf_counter()
            result_by_individual = np.zeros((self.num_individuals, rows), dtype=self._dtype)
            np.add.at(result_by_individual, self.sample_to_individual, result_col)
            result_col = result_by_individual
            _record("by_individual_reduce", t0)

        t0 = perf_counter()
        out = result_col.T.astype(self._dtype, copy=False)
        _record("output_cast", t0)

        if self._logger.isEnabledFor(logging.INFO):
            self._print_matmul_timings(
                direction=direction_name,
                rows=rows,
                cols=cols,
                by_individual=by_individual,
                init_mode=init_mode.value,
                miss=(miss is not None),
                timings=timing_pairs,
                total_ms=(perf_counter() - total_t0) * 1000.0,
            )
        return out

    def _print_matmul_timings(
        self,
        direction: Direction,
        rows: int,
        cols: int,
        by_individual: bool,
        init_mode: str,
        miss: bool,
        timings: list[tuple[str, float]],
        total_ms: float,
    ) -> None:
        miss_mode = "on" if miss else "off"
        timing_body = " ".join(f"{stage}={ms:.3f}ms" for stage, ms in timings)
        self._logger.info(
            "SpmvGRG.matmul[%s] rows=%d cols=%d by_individual=%s init=%s miss=%s "
            "path=fused_single_traversal=on %s total=%.3fms",
            direction.value,
            rows,
            cols,
            by_individual,
            init_mode,
            miss_mode,
            timing_body,
            total_ms,
        )
