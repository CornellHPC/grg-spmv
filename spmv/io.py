"""
I/O utilities for loading/saving SpMVOperator state.
"""
import numpy as np
import scipy.sparse as sp


def save_operator_npz(op, A_fwd, AT_bwd, npz_path, algo="none"):
    """
    Save SpMVOperator state to NPZ file for fast loading.
    Stores extracted A_fwd and AT_bwd blocks directly to avoid extraction during load.
    Parameters
    ----------
    op : SpMVOperator
        The operator to save (for metadata).
    A_fwd : list of sp.csr_matrix
        Forward matrices for each level.
    AT_bwd : list of sp.csr_matrix
        Backward matrices for each level.
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
    for h, A in enumerate(A_fwd):
        save_dict[f'A_fwd_{h}_indices'] = A.indices
        save_dict[f'A_fwd_{h}_indptr'] = A.indptr
        save_dict[f'A_fwd_{h}_shape'] = np.array(A.shape)
    for h, AT in enumerate(AT_bwd):
        save_dict[f'AT_bwd_{h}_indices'] = AT.indices
        save_dict[f'AT_bwd_{h}_indptr'] = AT.indptr
        save_dict[f'AT_bwd_{h}_shape'] = np.array(AT.shape)
    np.savez(npz_path, **save_dict)


def load_operator_npz(npz_path, dtype):
    """
    Load SpMVOperator state from NPZ file.
    Loads extracted A_fwd and AT_bwd blocks directly.
    Parameters
    ----------
    npz_path : str or Path
        Path to the NPZ file.
    dtype : numpy dtype
        Data type for sparse matrix values.
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
    inv_sample_perm = np.empty(n, dtype=np.uint32)
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

    result['A_fwd'] = []
    for h in range(n_levels):
        indices = data[f'A_fwd_{h}_indices']
        indptr = data[f'A_fwd_{h}_indptr']
        shape = tuple(data[f'A_fwd_{h}_shape'])
        nnz = len(indices)
        result['A_fwd'].append(sp.csr_matrix(
            (np.ones(nnz, dtype=dtype), indices, indptr),
            shape=shape
        ))

    result['AT_bwd'] = []
    for h in range(n_levels):
        indices = data[f'AT_bwd_{h}_indices']
        indptr = data[f'AT_bwd_{h}_indptr']
        shape = tuple(data[f'AT_bwd_{h}_shape'])
        nnz = len(indices)
        result['AT_bwd'].append(sp.csr_matrix(
            (np.ones(nnz, dtype=dtype), indices, indptr),
            shape=shape
        ))

    return result


def _extract_block(M, row_lo, row_hi, col_lo, col_hi, dtype):
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
    mask_uint = mask.astype(np.uint32)
    cumsum_full = np.zeros(len(mask) + 1, dtype=np.uint32)
    np.cumsum(mask_uint, out=cumsum_full[1:])
    row_counts = cumsum_full[indptr[1:]] - cumsum_full[indptr[:-1]]
    new_indptr = np.zeros(nrows + 1, dtype=np.uint32)
    new_indptr[1:] = np.cumsum(row_counts)
    new_data = np.ones(len(new_indices), dtype=dtype)
    return sp.csr_matrix(
        (new_data, new_indices, new_indptr),
        shape=(nrows, ncols),
    )
