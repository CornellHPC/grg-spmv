"""Helpers for explicit-matrix end-to-end runtime tests."""

from __future__ import annotations

import numpy as np
import pygrgl


def allele_frequencies(grg: pygrgl.GRG) -> np.ndarray:
    kwargs = {}
    miss = None
    if grg.has_missing_data:
        miss = np.zeros((1, grg.num_mutations), dtype=np.int32)
        kwargs["miss"] = miss
    counts = np.asarray(
        pygrgl.matmul(
            grg,
            np.ones((1, grg.num_samples), dtype=np.int32),
            pygrgl.TraversalDirection.UP,
            **kwargs,
        )
    )[0]
    miss_counts = np.zeros(grg.num_mutations, dtype=np.int32) if miss is None else miss[0]
    denom = grg.num_samples - miss_counts
    return np.divide(counts, denom, out=np.zeros_like(counts, dtype=np.float64), where=(denom != 0))


def samples_below_node(grg: pygrgl.GRG, node_id: int) -> list[int]:
    out: list[int] = []
    for child in grg.get_down_edges(node_id):
        if grg.is_sample(child):
            out.append(child)
        else:
            out.extend(samples_below_node(grg, child))
    return out


def grg_to_matrix(grg: pygrgl.GRG, diploid: bool = False) -> np.ndarray:
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
        for mut_id, _mut_node, miss_node in grg.get_mutation_node_miss():
            if miss_node == pygrgl.INVALID_NODE:
                continue
            for sample_id in samples_below[miss_node]:
                row = sample_id // grg.ploidy if diploid else sample_id
                if diploid:
                    result[row, mut_id] += freqs[mut_id]
                else:
                    result[row, mut_id] = freqs[mut_id]
    return result

