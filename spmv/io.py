"""
I/O utilities for loading/saving SpMVOperator state.
"""
import numpy as np
import scipy.sparse as sp


def save_operator_npz(op, A_blocks, AT_blocks, npz_path, algo="none"):
    """
    Save SpMVOperator state to NPZ file for fast loading.
    Stores extracted A_blocks and AT_blocks (list-of-lists) directly.

    Parameters
    ----------
    op : SpMVOperator
        The operator to save (for metadata).
    A_blocks : list of list of sp.csr_matrix
        Forward blocks: A_blocks[h][j] for j in range(h).
    AT_blocks : list of list of sp.csr_matrix
        Backward blocks: AT_blocks[h][j] for j in range(num_levels-1-h).
    npz_path : str or Path
        Path to save the NPZ file.
    algo : str
        Algorithm used (stored in file for validation).
    """
    save_dict = {
        'n': op.n,
        'm': op.m,
        'K': op.K,
        'algo': algo,
        'level_offsets': op.level_offsets,
        'sample_perm': op.sample_perm,
        'sel_indices': op.sel.indices,
        'sel_indptr': op.sel.indptr,
        'sel_T_indices': op.sel_T.indices,
        'sel_T_indptr': op.sel_T.indptr,
    }
    for h, blocks in enumerate(A_blocks):
        for j, A in enumerate(blocks):
            save_dict[f'A_blocks_{h}_{j}_indices'] = A.indices
            save_dict[f'A_blocks_{h}_{j}_indptr'] = A.indptr
            save_dict[f'A_blocks_{h}_{j}_shape'] = np.array(A.shape)
    for h, blocks in enumerate(AT_blocks):
        for j, AT in enumerate(blocks):
            save_dict[f'AT_blocks_{h}_{j}_indices'] = AT.indices
            save_dict[f'AT_blocks_{h}_{j}_indptr'] = AT.indptr
            save_dict[f'AT_blocks_{h}_{j}_shape'] = np.array(AT.shape)
    np.savez(npz_path, **save_dict)


def load_operator_npz(npz_path, dtype, index_dtype):
    """
    Load SpMVOperator state from NPZ file.
    Loads extracted A_blocks and AT_blocks (list-of-lists) directly.

    Parameters
    ----------
    npz_path : str or Path
        Path to the NPZ file.
    dtype : numpy dtype
        Data type for sparse matrix values.
    index_dtype : numpy dtype
        Index dtype for permutation arrays.

    Returns
    -------
    dict
        Dictionary containing all operator attributes.
    """
    data = np.load(npz_path, allow_pickle=False)
    result = {
        'n': int(data['n']),
        'm': int(data['m']),
        'K': int(data['K']),
        'algo': str(data['algo']) if 'algo' in data else 'unknown',
        'level_offsets': data['level_offsets'],
        'sample_perm': data['sample_perm'],
    }
    n = result['n']
    inv_sample_perm = np.empty(n, dtype=index_dtype)
    inv_sample_perm[result['sample_perm']] = np.arange(n)
    result['_inv_sample_perm'] = inv_sample_perm

    n_levels = len(result['level_offsets']) - 1
    sel_nnz = len(data['sel_indices'])
    sel_T_nnz = len(data['sel_T_indices'])

    result['sel'] = sp.csr_matrix(
        (np.ones(sel_nnz, dtype=dtype), data['sel_indices'], data['sel_indptr']),
        shape=(result['m'], result['K'])
    )

    result['sel_T'] = sp.csr_matrix(
        (np.ones(sel_T_nnz, dtype=dtype), data['sel_T_indices'], data['sel_T_indptr']),
        shape=(result['K'], result['m'])
    )

    # Reconstruct list-of-lists: A_blocks[h] has h blocks, AT_blocks[h] has (n_levels-1-h) blocks
    result['A_blocks'] = []
    for h in range(n_levels):
        level_blocks = []
        for j in range(h):
            indices = data[f'A_blocks_{h}_{j}_indices']
            indptr = data[f'A_blocks_{h}_{j}_indptr']
            shape = tuple(data[f'A_blocks_{h}_{j}_shape'])
            nnz = len(indices)
            level_blocks.append(sp.csr_matrix(
                (np.ones(nnz, dtype=dtype), indices, indptr),
                shape=shape
            ))
        result['A_blocks'].append(level_blocks)

    result['AT_blocks'] = []
    for h in range(n_levels):
        level_blocks = []
        for j in range(n_levels - 1 - h):
            indices = data[f'AT_blocks_{h}_{j}_indices']
            indptr = data[f'AT_blocks_{h}_{j}_indptr']
            shape = tuple(data[f'AT_blocks_{h}_{j}_shape'])
            nnz = len(indices)
            level_blocks.append(sp.csr_matrix(
                (np.ones(nnz, dtype=dtype), indices, indptr),
                shape=shape
            ))
        result['AT_blocks'].append(level_blocks)

    return result


def _extract_block(M, row_lo, row_hi, col_lo, col_hi, dtype, index_dtype):
    """
    Extract a block M[row_lo:row_hi, col_lo:col_hi] as an independent CSR matrix.
    """
    if col_lo >= M.shape[1] or col_hi <= 0:
        nrows = min(row_hi, M.shape[0]) - max(row_lo, 0)
        return sp.csr_matrix((nrows, 0), dtype=dtype)
    nrows = row_hi - row_lo
    ncols = col_hi - col_lo
    if ncols == 0:
        return sp.csr_matrix((nrows, 0), dtype=dtype)
    start = M.indptr[row_lo]
    end = M.indptr[row_hi]
    indices = M.indices[start:end]
    indptr = M.indptr[row_lo:row_hi + 1] - start
    mask = (indices >= col_lo) & (indices < col_hi)
    new_indices = indices[mask] - col_lo
    mask_uint = mask.astype(index_dtype)
    cumsum_full = np.zeros(len(mask) + 1, dtype=index_dtype)
    np.cumsum(mask_uint, out=cumsum_full[1:])
    row_counts = cumsum_full[indptr[1:]] - cumsum_full[indptr[:-1]]
    new_indptr = np.zeros(nrows + 1, dtype=index_dtype)
    new_indptr[1:] = np.cumsum(row_counts)
    new_data = np.ones(len(new_indices), dtype=dtype)
    return sp.csr_matrix(
        (new_data, new_indices, new_indptr),
        shape=(nrows, ncols),
    )
