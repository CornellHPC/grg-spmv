"""
Benchmarking utilities for SpMVOperator.
"""

import argparse
from time import perf_counter

import numpy as np

from spmv import SpMVOperator
from spmv.backends.cusparse import is_valid_combo

DTYPE = np.float64
INDEX_DTYPE = np.uintp


def _make_vectors(op, k, seed=42):
    """Create random test vectors."""
    rng = np.random.default_rng(seed)
    n, m = op.n, op.m
    W = rng.standard_normal((m, k), dtype=DTYPE)
    V = rng.standard_normal((n, k), dtype=DTYPE)
    return W, V


def _time_op(op, W, V, n_warmup, n_trials):
    """Warmup then time forward and backward matmul."""
    # Warmup (not timed)
    for _ in range(n_warmup):
        _ = op @ W
        _ = op.H @ V

    # Time G @ W (backward_matmat in the backend)
    times_fwd = []
    result_fwd = None
    for _ in range(n_trials):
        t0 = perf_counter()
        result_fwd = op @ W
        times_fwd.append(perf_counter() - t0)

    # Time G^T @ V (forward_matmat in the backend)
    times_bwd = []
    result_bwd = None
    for _ in range(n_trials):
        t0 = perf_counter()
        result_bwd = op.H @ V
        times_bwd.append(perf_counter() - t0)

    return times_fwd, times_bwd, result_fwd, result_bwd


def benchmark_spsparse(grg_path, k, n_trials, n_warmup, worker_counts, chunk_size):
    """Benchmark spsparse backend across worker counts."""
    print(f"GRG file: {grg_path}")

    results = []
    for n_workers in worker_counts:
        print(f"\n{'='*60}")
        print(f"Loading operator: n_workers={n_workers}")
        t0 = perf_counter()
        backend_config = {
            'type': 'spsparse', 'n_workers': n_workers,
            'chunk_size': chunk_size, 'verbose': True
        }
        op = SpMVOperator(grg_path, backend_config, DTYPE, INDEX_DTYPE, use_rcm=True)
        build_time = perf_counter() - t0

        W, V = _make_vectors(op, k)
        times_fwd, times_bwd, result_fwd, result_bwd = _time_op(op, W, V, n_warmup, n_trials)

        results.append({
            'label': f'spsparse-{n_workers}w',
            'build': build_time,
            'fwd_mean': np.mean(times_fwd) * 1000,
            'fwd_std': np.std(times_fwd) * 1000,
            'bwd_mean': np.mean(times_bwd) * 1000,
            'bwd_std': np.std(times_bwd) * 1000,
            'fwd_cksum': np.linalg.norm(result_fwd),
            'bwd_cksum': np.linalg.norm(result_bwd),
        })

        print(f"  Build: {build_time:.2f}s")
        print(f"  G @ W:   {np.mean(times_fwd)*1000:.2f} +/- {np.std(times_fwd)*1000:.2f} ms")
        print(f"  G^T @ V: {np.mean(times_bwd)*1000:.2f} +/- {np.std(times_bwd)*1000:.2f} ms")

    return results


def benchmark_mkl(grg_path, k, n_trials, n_warmup, thread_counts):
    """Benchmark MKL backend across thread counts."""
    print(f"GRG file: {grg_path}")

    results = []
    for n_threads in thread_counts:
        print(f"\n{'='*60}")
        print(f"Loading operator: n_threads={n_threads}")
        t0 = perf_counter()
        backend_config = {
            'type': 'mkl', 'n_threads': n_threads, 'verbose': True
        }
        op = SpMVOperator(grg_path, backend_config, DTYPE, INDEX_DTYPE, use_rcm=True)
        build_time = perf_counter() - t0

        W, V = _make_vectors(op, k)
        times_fwd, times_bwd, result_fwd, result_bwd = _time_op(op, W, V, n_warmup, n_trials)

        results.append({
            'label': f'mkl-{n_threads}t',
            'build': build_time,
            'fwd_mean': np.mean(times_fwd) * 1000,
            'fwd_std': np.std(times_fwd) * 1000,
            'bwd_mean': np.mean(times_bwd) * 1000,
            'bwd_std': np.std(times_bwd) * 1000,
            'fwd_cksum': np.linalg.norm(result_fwd),
            'bwd_cksum': np.linalg.norm(result_bwd),
        })

        print(f"  Build: {build_time:.2f}s")
        print(f"  G @ W:   {np.mean(times_fwd)*1000:.2f} +/- {np.std(times_fwd)*1000:.2f} ms")
        print(f"  G^T @ V: {np.mean(times_bwd)*1000:.2f} +/- {np.std(times_bwd)*1000:.2f} ms")

    return results


def filter_valid_combinations(fmts, algorithms):
    """Filter out invalid (format, algorithm) combinations."""
    return [(fmt, alg) for fmt in fmts for alg in algorithms
            if is_valid_combo(fmt, alg)]


def _parse_verbose_timing(line):
    """Extract H2D, kernel, D2H, total timing from verbose output line."""
    import re
    # Example: "forward_matmat (graph): H2D=58.51ms kernel=166.16ms D2H=681.34ms total=906.02ms"
    match = re.search(r'H2D=([\d.]+)ms.*kernel=([\d.]+)ms.*D2H=([\d.]+)ms.*total=([\d.]+)ms', line)
    if match:
        return {
            'h2d': float(match.group(1)),
            'kernel': float(match.group(2)),
            'd2h': float(match.group(3)),
            'total': float(match.group(4))
        }
    return None


def benchmark_gpu(grg_path, k, n_trials, n_warmup, fmts, graph_ks, algorithms):
    """
    Benchmark GPU cusparse backend across (fmt, graph_k, algorithm) combinations.

    Uses verbose=True during a single diagnostic warmup run to print the
    H2D / kernel / D2H breakdown, then switches to verbose=False for the
    timed trials so that perf_counter and print I/O don't pollute timing.
    """
    import sys
    from io import StringIO

    print(f"GRG file: {grg_path}")

    # Filter invalid combinations
    valid_combos = filter_valid_combinations(fmts, algorithms)

    results = []
    for fmt, algorithm in valid_combos:
        for graph_k in graph_ks:
            mode = f"graph-k{graph_k}" if graph_k is not None else "dynamic"
            label = f"gpu-{fmt}-{algorithm}-{mode}"
            print(f"\n{'='*60}")
            print(f"Config: fmt={fmt}, algorithm={algorithm}, {mode}")

            # --- Build with verbose to see setup diagnostics ---
            t0 = perf_counter()

            # Capture stdout to parse verbose output
            old_stdout = sys.stdout
            sys.stdout = captured = StringIO()

            try:
                op_verbose = SpMVOperator(grg_path, {
                    'type': 'cusparse', 'fmt': fmt, 'k': graph_k,
                    'algorithm': algorithm, 'verbose': True
                }, DTYPE, INDEX_DTYPE, use_rcm=True)
                build_time = perf_counter() - t0

                W, V = _make_vectors(op_verbose, k)

                # --- One verbose trial to show H2D / kernel / D2H breakdown ---
                _ = op_verbose @ W
                _ = op_verbose.H @ V

                # Get captured output
                verbose_output = captured.getvalue()
            finally:
                sys.stdout = old_stdout

            # Print build diagnostics
            print(f"  Build: {build_time:.2f}s")
            print("\n  Diagnostic run (verbose, not timed):")
            print(verbose_output, end='')

            # Parse timing components from verbose output
            timing_fwd = None
            timing_bwd = None
            for line in verbose_output.split('\n'):
                if 'forward_matmat' in line:
                    timing_fwd = _parse_verbose_timing(line)
                elif 'backward_matmat' in line:
                    timing_bwd = _parse_verbose_timing(line)

            del op_verbose

            # --- Build again without verbose for clean timing ---
            op = SpMVOperator(grg_path, {
                'type': 'cusparse', 'fmt': fmt, 'k': graph_k,
                'algorithm': algorithm, 'verbose': False
            }, DTYPE, INDEX_DTYPE, use_rcm=True)
            W, V = _make_vectors(op, k)
            times_fwd, times_bwd, result_fwd, result_bwd = _time_op(op, W, V, n_warmup, n_trials)

            result = {
                'label': label,
                'build': build_time,
                'fwd_mean': np.mean(times_fwd) * 1000,
                'fwd_std': np.std(times_fwd) * 1000,
                'bwd_mean': np.mean(times_bwd) * 1000,
                'bwd_std': np.std(times_bwd) * 1000,
                'fwd_cksum': np.linalg.norm(result_fwd),
                'bwd_cksum': np.linalg.norm(result_bwd),
            }

            # Add timing components if available
            if timing_fwd:
                result.update({
                    'fwd_h2d': timing_fwd['h2d'],
                    'fwd_kernel': timing_fwd['kernel'],
                    'fwd_d2h': timing_fwd['d2h'],
                })
            if timing_bwd:
                result.update({
                    'bwd_h2d': timing_bwd['h2d'],
                    'bwd_kernel': timing_bwd['kernel'],
                    'bwd_d2h': timing_bwd['d2h'],
                })

            results.append(result)

            print(f"\n  Timed ({n_trials} trials, {n_warmup} warmup):")
            print(f"  G @ W:   {np.mean(times_fwd)*1000:.2f} +/- {np.std(times_fwd)*1000:.2f} ms")
            print(f"  G^T @ V: {np.mean(times_bwd)*1000:.2f} +/- {np.std(times_bwd)*1000:.2f} ms")
            del op

    return results


def print_summary(results, k):
    """Print a summary table of all benchmark results."""
    print("\n" + "=" * 75)
    print(f"SUMMARY (k={k})")
    print("=" * 75)
    print(f"{'Config':<22} {'G@W (ms)':>14} {'G^T@V (ms)':>14} {'Build (s)':>10}")
    print("-" * 75)
    for r in results:
        print(f"{r['label']:<22} "
              f"{r['fwd_mean']:>8.2f}+/-{r['fwd_std']:<4.2f} "
              f"{r['bwd_mean']:>8.2f}+/-{r['bwd_std']:<4.2f} "
              f"{r['build']:>10.2f}")


def print_summary_detailed(results, k):
    """Print 4 separate tables: H2D, Kernel, D2H, Total."""
    # Filter results that have timing components
    detailed_results = [r for r in results if 'fwd_h2d' in r and 'bwd_h2d' in r]

    if not detailed_results:
        print("\nNo detailed timing data available (verbose output not parsed)")
        return

    print("\n" + "=" * 75)
    print(f"DETAILED TIMING BREAKDOWN (k={k})")
    print("=" * 75)

    # Table 1: H2D Transfer Times
    print("\n--- H2D Transfer Times (Host to Device) ---")
    print(f"{'Config':<32} {'G@W (ms)':>14} {'G^T@V (ms)':>14}")
    print("-" * 75)
    for r in detailed_results:
        print(f"{r['label']:<32} "
              f"{r['fwd_h2d']:>14.2f} "
              f"{r['bwd_h2d']:>14.2f}")

    # Table 2: Kernel Times
    print("\n--- Kernel Execution Times ---")
    print(f"{'Config':<32} {'G@W (ms)':>14} {'G^T@V (ms)':>14}")
    print("-" * 75)
    for r in detailed_results:
        print(f"{r['label']:<32} "
              f"{r['fwd_kernel']:>14.2f} "
              f"{r['bwd_kernel']:>14.2f}")

    # Table 3: D2H Transfer Times
    print("\n--- D2H Transfer Times (Device to Host) ---")
    print(f"{'Config':<32} {'G@W (ms)':>14} {'G^T@V (ms)':>14}")
    print("-" * 75)
    for r in detailed_results:
        print(f"{r['label']:<32} "
              f"{r['fwd_d2h']:>14.2f} "
              f"{r['bwd_d2h']:>14.2f}")

    # Table 4: Total Times (from timing breakdown, for comparison)
    print("\n--- Total Times (H2D + Kernel + D2H) ---")
    print(f"{'Config':<32} {'G@W (ms)':>14} {'G^T@V (ms)':>14}")
    print("-" * 75)
    for r in detailed_results:
        total_fwd = r['fwd_h2d'] + r['fwd_kernel'] + r['fwd_d2h']
        total_bwd = r['bwd_h2d'] + r['bwd_kernel'] + r['bwd_d2h']
        print(f"{r['label']:<32} "
              f"{total_fwd:>14.2f} "
              f"{total_bwd:>14.2f}")


def print_checksums(results, k):
    """Print checksum comparison table for sanity checking results."""
    print("\n" + "=" * 75)
    print(f"CHECKSUMS (k={k})")
    print("=" * 75)
    print(f"{'Config':<32} {'G@W norm':>16} {'G^T@V norm':>16}")
    print("-" * 75)
    for r in results:
        print(f"{r['label']:<32} "
              f"{r['fwd_cksum']:>16.4f} "
              f"{r['bwd_cksum']:>16.4f}")


def _parse_graph_ks(s):
    """Parse comma-separated graph_k values ('none,1,4' -> [None, 1, 4])."""
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if tok in ("none", "dynamic"):
            out.append(None)
        else:
            out.append(int(tok))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark SpMVOperator (CPU and GPU backends)")
    parser.add_argument("--grg",
        default="/pscratch/sd/q/qys/grg/simulation-mutation-200m.trees.v4.igd.final.grg")
    parser.add_argument("--k", type=int, default=4,
        help="Number of dense columns")
    parser.add_argument("--trials", type=int, default=10,
        help="Number of timed trials")
    parser.add_argument("--warmup", type=int, default=3,
        help="Number of warmup runs (not timed)")
    parser.add_argument("--backend", type=str, default="cusparse",
        choices=["spsparse", "mkl", "cusparse", "cpu", "all"],
        help="Which backend(s) to benchmark")

    # Spsparse-specific
    parser.add_argument("--workers", type=str, default="1,4,16",
        help="Comma-separated worker counts for spsparse backend")
    parser.add_argument("--chunk-size", type=int, default=4096)

    # MKL-specific
    parser.add_argument("--mkl-threads", type=str, default="0,1,4,16",
        help="Comma-separated thread counts for MKL backend")

    # GPU-specific
    parser.add_argument("--fmt", type=str, default="csr,csc",
        help="Comma-separated sparse formats for cusparse backend (csr,csc,coo)")
    parser.add_argument("--algorithm", type=str, default="default",
        help="Comma-separated algorithms for cusparse backend (default,csr_alg1,csr_alg2,csr_alg3,coo_alg1,coo_alg2,coo_alg3,coo_alg4)")
    parser.add_argument("--graph-k", type=str, default="none,4",
        help="Comma-separated graph_k values (use 'none' for dynamic mode)")

    args = parser.parse_args()

    all_results = []

    if args.backend in ("spsparse", "cpu", "all"):
        worker_counts = [int(w) for w in args.workers.split(",")]
        spsparse_results = benchmark_spsparse(
            grg_path=args.grg, k=args.k,
            n_trials=args.trials, n_warmup=args.warmup,
            worker_counts=worker_counts, chunk_size=args.chunk_size,
        )
        all_results.extend(spsparse_results)

    if args.backend in ("mkl", "cpu", "all"):
        thread_counts = [int(t) for t in args.mkl_threads.split(",")]
        mkl_results = benchmark_mkl(
            grg_path=args.grg, k=args.k,
            n_trials=args.trials, n_warmup=args.warmup,
            thread_counts=thread_counts,
        )
        all_results.extend(mkl_results)

    if args.backend in ("cusparse", "all"):
        fmts = [f.strip() for f in args.fmt.split(",")]
        algorithms = [a.strip() for a in args.algorithm.split(",")]
        graph_ks = _parse_graph_ks(args.graph_k)
        gpu_results = benchmark_gpu(
            grg_path=args.grg, k=args.k,
            n_trials=args.trials, n_warmup=args.warmup,
            fmts=fmts, graph_ks=graph_ks, algorithms=algorithms,
        )
        all_results.extend(gpu_results)

    print_summary(all_results, args.k)
    print_checksums(all_results, args.k)

    # Print detailed breakdown for GPU results if available
    if args.backend in ("cusparse", "all"):
        print_summary_detailed(all_results, args.k)
