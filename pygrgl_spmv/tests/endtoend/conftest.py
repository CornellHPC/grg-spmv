"""Shared fixtures/helpers for end-to-end matmul tests."""

from __future__ import annotations

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.tests.conftest import DATA_DTYPE, INDEX_DTYPE


def _backend_params():
    params = [
        pytest.param(
            {'type': 'mkl', 'n_threads': 0, 'fmt_up': 'csr', 'log_level': 'INFO'},
            id='mkl',
            marks=pytest.mark.mkl,
        ),
    ]
    try:
        import cupy  # noqa: F401
    except ImportError:
        return params
    params.extend(
        [
            pytest.param(
                {
                    'type': 'cusparse',
                    'fmt_up': 'csr',
                    'algo_up': 'default',
                    'algo_down': 'default',
                    'k_hint': None,
                    'log_level': 'INFO',
                },
                id='cusparse-dyn',
                marks=pytest.mark.gpu,
            ),
            pytest.param(
                {
                    'type': 'cusparse',
                    'fmt_up': 'csr',
                    'algo_up': 'default',
                    'algo_down': 'default',
                    'k_hint': 4,
                    'log_level': 'INFO',
                },
                id='cusparse-graph-k4',
                marks=pytest.mark.gpu,
            ),
        ]
    )
    return params


@pytest.fixture(scope="session", params=_backend_params())
def backend_config(request, backend_filter):
    cfg = request.param
    btype = cfg['type']
    if backend_filter == "mkl" and btype != "mkl":
        pytest.skip("filtered to mkl backend")
    if backend_filter == "cusparse" and btype != "cusparse":
        pytest.skip("filtered to cusparse backend")
    return cfg


@pytest.fixture(scope="session")
def basic_grg_path(primary_grg_path):
    return primary_grg_path


@pytest.fixture(scope="session")
def basic_grg(basic_grg_path):
    return pygrgl.load_immutable_grg(basic_grg_path, load_up_edges=True)


@pytest.fixture(scope="session")
def missing_grg(missing_grg_path):
    return pygrgl.load_immutable_grg(missing_grg_path, load_up_edges=True)


@pytest.fixture
def make_operator(backend_config, spmv_cache_dir):
    def _make(path: str, dtype=DATA_DTYPE):
        return SpmvGRG(path, backend_config, dtype, INDEX_DTYPE, cache_dir=spmv_cache_dir)

    return _make


def allele_frequencies(grg: pygrgl.GRG) -> np.ndarray:
    """Allele frequencies with missing-aware denominator C_i."""
    kwargs = {}
    miss = None
    if grg.has_missing_data:
        miss = np.zeros((1, grg.num_mutations), dtype=np.int32)
        kwargs["miss"] = miss
    counts = pygrgl.matmul(
        grg,
        np.ones((1, grg.num_samples), dtype=np.int32),
        pygrgl.TraversalDirection.UP,
        **kwargs,
    )[0]
    miss_counts = np.zeros(grg.num_mutations, dtype=np.int32) if miss is None else miss[0]
    denom = grg.num_samples - miss_counts
    return np.divide(
        counts,
        denom,
        out=np.zeros_like(counts, dtype=np.float64),
        where=(denom != 0),
    )


def grg_to_matrix(grg: pygrgl.GRG, diploid: bool = False) -> np.ndarray:
    """
    Build explicit genotype matrix.

    When missing data exists, values are mean-imputed using per-site frequency.
    """
    n_rows = grg.num_individuals if diploid else grg.num_samples
    result = np.zeros((n_rows, grg.num_mutations), dtype=np.float64)
    samples_below = [list() for _ in range(grg.num_nodes)]
    for node_id in range(grg.num_nodes):
        below = []
        if grg.is_sample(node_id):
            below.append(node_id)
        for child_id in grg.get_down_edges(node_id):
            below.extend(samples_below[child_id])
        samples_below[node_id] = below

        muts = grg.get_mutations_for_node(node_id)
        if muts:
            for sample_id in below:
                row = sample_id // grg.ploidy if diploid else sample_id
                for mut_id in muts:
                    if diploid:
                        result[row, mut_id] += 1.0
                    else:
                        result[row, mut_id] = 1.0

    if grg.has_missing_data:
        freqs = allele_frequencies(grg)
        for mut_id, mut_node, miss_node in grg.get_mutation_node_miss():
            if miss_node == pygrgl.INVALID_NODE:
                continue
            for sample_id in samples_below[miss_node]:
                row = sample_id // grg.ploidy if diploid else sample_id
                if diploid:
                    result[row, mut_id] += freqs[mut_id]
                else:
                    result[row, mut_id] = freqs[mut_id]
    return result


def samples_below_node(grg: pygrgl.GRG, node_id: int) -> list[int]:
    """Return all sample nodes reachable under node_id."""
    out = []
    for child in grg.get_down_edges(node_id):
        if grg.is_sample(child):
            out.append(child)
        else:
            out.extend(samples_below_node(grg, child))
    return out
