"""
Benchmarking utilities for SpMVOperator.
"""

import argparse
from time import perf_counter

import numpy as np

from spmv import SpMVOperator


def benchmark_configs(grg_path, k, n_trials, worker_counts, chunk_size):
    """
    Compare SpMV performance with different configurations.

    Parameters
    ----------
    grg_path : str
        Path to GRG file.
    k : int
        Number of columns in the dense matrix.
    n_trials : int
        Number of timing trials.
    worker_counts : list
        Worker counts to test.
    chunk_size : int
        Rows per chunk for parallel execution.
    """
    print(f"GRG file: {grg_path}")

    results = []

    use_rcm = True
    for n_workers in worker_counts:
        print(f"\n{'='*60}")
        print(f"Loading operator: use_rcm={use_rcm}, n_workers={n_workers}")
        t0 = perf_counter()
        backend_config = {'type': 'multithread', 'n_workers': n_workers, 'chunk_size': chunk_size, 'verbose': True}
        op = SpMVOperator(grg_path, backend_config=backend_config, use_rcm=use_rcm)
        build_time = perf_counter() - t0

        n, m = op.n, op.m
        rng = np.random.default_rng(42)
        if k == 1:
            W = rng.standard_normal(m)
            V = rng.standard_normal(n)
        else:
            W = rng.standard_normal((m, k))
            V = rng.standard_normal((n, k))

        # Time G @ W
        times_fwd = []
        for _ in range(n_trials):
            t0 = perf_counter()
            _ = op @ W
            times_fwd.append(perf_counter() - t0)

        # Time G^T @ V
        times_bwd = []
        for _ in range(n_trials):
            t0 = perf_counter()
            _ = op.H @ V
            times_bwd.append(perf_counter() - t0)
        
        results.append({
            'rcm': use_rcm,
            'workers': n_workers,
            'build': build_time,
            'fwd_mean': np.mean(times_fwd) * 1000,
            'fwd_std': np.std(times_fwd) * 1000,
            'bwd_mean': np.mean(times_bwd) * 1000,
            'bwd_std': np.std(times_bwd) * 1000,
        })
        
        print(f"  Build: {build_time:.2f}s")
        print(f"  G @ W:   {np.mean(times_fwd)*1000:.0f} ± {np.std(times_fwd)*1000:.0f} ms")
        print(f"  G^T @ V: {np.mean(times_bwd)*1000:.0f} ± {np.std(times_bwd)*1000:.0f} ms")
    
    # Summary table
    print("\n" + "=" * 70)
    print(f"SUMMARY (k={k})")
    print("=" * 70)
    print(f"{'RCM':>5} {'Workers':>8} {'G@W (ms)':>12} {'G^T@V (ms)':>12} {'Build (s)':>10}")
    print("-" * 70)
    for r in results:
        print(f"{str(r['rcm']):>5} {r['workers']:>8} "
              f"{r['fwd_mean']:>8.0f}±{r['fwd_std']:<3.0f} "
              f"{r['bwd_mean']:>8.0f}±{r['bwd_std']:<3.0f} "
              f"{r['build']:>10.2f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grg", default="inputs/simulation-source-stdpopsim-500k.trees.v4.igd.final.grg")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--workers", type=str, default="1,4,16")
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()

    worker_counts = [int(w) for w in args.workers.split(",")]
    benchmark_configs(
        grg_path=args.grg,
        k=args.k,
        n_trials=args.trials,
        worker_counts=worker_counts,
        chunk_size=args.chunk_size
    )
