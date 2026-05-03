from __future__ import annotations

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv.grg import RuntimeRequirements
from pygrgl_spmv.grg.artifact import save_grg_spmv
from pygrgl_spmv.grg.sparse import binary_csr_from_parts
from pygrgl_spmv.tests.runtime._streaming_cases import _synthetic_state
from scripts.bolt_lmm_inf.__main__ import (
    CgBucket,
    EffectCheckSnp,
    GrgBoltOps,
    TimingBucket,
    _effect_check_metrics,
    _log_delta_from_h2,
    _reml_log_delta_bounds,
    _resolve_artifacts,
    _scan_metrics,
    _simulate_phenotype,
    cg_solve,
    main as bolt_main,
)


def test_cg_solve_matches_tiny_spd_system():
    matrix = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float64)
    b = np.array([1.0, -1.0], dtype=np.float64)
    matvec_calls = 0

    def matvec_into(src, dst) -> None:
        nonlocal matvec_calls
        matvec_calls += 1
        dst[...] = matrix @ src

    bucket = CgBucket()
    actual = cg_solve(matvec_into, b, rel_tol=1e-12, max_iter=20, bucket=bucket)
    expected = np.linalg.solve(matrix, b)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert bucket.solves == 1
    assert bucket.iterations == 1
    assert matvec_calls == 1


def _standardized_from_counts(
    counts: np.ndarray,
    *,
    num_samples: int,
    ploidy: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=np.float64)
    freq = counts.sum(axis=0) / float(num_samples)
    mu = float(ploidy) * freq
    sigma = np.sqrt(float(ploidy) * freq * (1.0 - freq))
    valid = sigma > 0.0
    standardized = np.zeros_like(counts, dtype=np.float64)
    np.divide(counts - mu, sigma, out=standardized, where=valid[None, :])
    return standardized, valid


def _dense_standardized_matrix(grg) -> tuple[np.ndarray, np.ndarray]:
    basis = np.eye(int(grg.num_mutations), dtype=np.float64)
    counts = np.asarray(
        pygrgl.matmul(grg, basis, pygrgl.TraversalDirection.DOWN, by_individual=True),
        dtype=np.float64,
    ).T
    return _standardized_from_counts(counts, num_samples=int(grg.num_samples), ploidy=int(grg.ploidy))


def _cupy_device_id(array) -> int:
    return int(array.device.id)


def _bolt_requirements() -> RuntimeRequirements:
    return RuntimeRequirements(
        max_k_up=1,
        max_k_down=1,
        need_down_miss_input=False,
        need_up_miss_output=False,
        need_init_vector=False,
        need_init_matrix=False,
        need_init_xtx=False,
    )


def test_reml_bounds_allow_sub_percent_h2():
    lo, hi = _reml_log_delta_bounds()
    assert lo < _log_delta_from_h2(0.001) < hi


def test_resolve_artifacts_rejects_mixed_labeled_paths(tmp_path):
    with pytest.raises(
        ValueError,
        match="either all chr<num>-labeled or all unlabeled",
    ):
        _resolve_artifacts(
            grg_dir=None,
            artifact_cache=None,
            artifacts=(tmp_path / "chr22.grg_spmv", tmp_path / "custom.grg_spmv"),
            chromosomes="21,22",
        )


def test_resolve_artifacts_maps_labeled_and_unlabeled_cases(tmp_path):
    chr22 = tmp_path / "chr22.grg_spmv"
    chr21 = tmp_path / "chr21-sim.grg_spmv"
    labeled = _resolve_artifacts(
        grg_dir=None,
        artifact_cache=None,
        artifacts=(chr22, chr21),
        chromosomes="all",
    )
    assert labeled == ((21, chr21), (22, chr22))

    first = tmp_path / "first.grg_spmv"
    second = tmp_path / "second.grg_spmv"
    with pytest.raises(
        ValueError,
        match="must contain chr<num> labels, or --chromosomes must provide one label per artifact",
    ):
        _resolve_artifacts(
            grg_dir=None,
            artifact_cache=None,
            artifacts=(first, second),
            chromosomes="all",
        )

    unlabeled = _resolve_artifacts(
        grg_dir=None,
        artifact_cache=None,
        artifacts=(first, second),
        chromosomes="21,22",
    )
    assert unlabeled == ((21, first), (22, second))


@pytest.mark.gpu
@pytest.mark.cusparse
def test_bolt_lmm_inf_main_writes_valid_summary(primary_artifact, tmp_path):
    pytest.importorskip("cupy")
    summary_file = tmp_path / "bolt_summary.tsv"

    bolt_main(
        [
            "--artifacts",
            str(primary_artifact),
            "--chromosomes",
            "21",
            "--summary-file",
            str(summary_file),
            "--vram-budget-bytes",
            "1000000000",
            "--n-calib",
            "1",
            "--n-effect-check",
            "0",
            "--scan-top-k",
            "0",
            "--cg-tol",
            "1e-3",
            "--cg-max-iter",
            "200",
            "--log-level",
            "WARNING",
        ]
    )

    lines = summary_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "metric\tvalue"
    metrics = dict(line.split("\t", 1) for line in lines[1:])

    def as_float(key: str) -> float:
        return float(metrics[key])

    def as_int(key: str) -> int:
        return int(metrics[key])

    assert metrics["backend"] == "cusparse"
    assert metrics["chromosomes"] == "21"
    assert as_int("artifact_count") == 1
    assert as_int("num_individuals") > 0
    assert as_int("num_samples") > 0
    assert as_int("num_mutations.raw") > 0
    assert as_int("num_mutations.used") > 0
    assert as_int("chr21.num_mutations.raw") > 0
    assert as_int("chr21.num_mutations.used") == as_int("num_mutations.used")

    for key in ("reml.sigma_g2", "reml.sigma_e2", "reml.delta", "reml.h2"):
        value = as_float(key)
        assert np.isfinite(value)
        assert value > 0.0
    assert as_float("reml.h2") == pytest.approx(1.0 / (1.0 + as_float("reml.delta")))

    assert as_float("calibration.c_inf") > 0.0
    assert as_int("calibration.num_snps_effective") == min(1, as_int("num_mutations.used"))

    hist_total = sum(as_int(f"scan.p_hist.bin{idx}.count") for idx in range(as_int("scan.p_hist.bin_count")))
    assert hist_total == as_int("scan.genome.num_snps")

    assert as_int("cg.reml.solves") > 0
    assert as_int("cg.loco.solves") > 0
    assert as_int("cg.calibration.solves") > 0


def _synthetic_counts_artifact(tmp_path, name: str, counts: np.ndarray) -> tuple[object, np.ndarray]:
    path = tmp_path / name
    struct_dtype = np.dtype(np.int32)
    counts = np.asarray(counts, dtype=np.float64)
    num_samples, num_mutations = counts.shape
    num_nodes = num_samples + num_mutations
    indices = []
    indptr = [0]
    for mut_idx in range(num_mutations):
        indices.extend(np.flatnonzero(counts[:, mut_idx]).tolist())
        indptr.append(len(indices))
    block = binary_csr_from_parts(
        indices=np.asarray(indices, dtype=struct_dtype),
        indptr=np.asarray(indptr, dtype=struct_dtype),
        shape=(num_mutations, num_samples),
        shared_data=True,
    )
    sel_mut = binary_csr_from_parts(
        indices=np.arange(num_samples, num_nodes, dtype=struct_dtype),
        indptr=np.arange(num_mutations + 1, dtype=struct_dtype),
        shape=(num_mutations, num_nodes),
        shared_data=True,
    )
    sel_miss = binary_csr_from_parts(
        indices=np.empty(0, dtype=struct_dtype),
        indptr=np.zeros(num_mutations + 1, dtype=struct_dtype),
        shape=(num_mutations, num_nodes),
        shared_data=True,
    )
    state = _synthetic_state(
        blocks=[[], [block]],
        level_offsets=np.asarray([0, num_samples, num_nodes], dtype=struct_dtype),
        num_samples=num_samples,
        num_mutations=num_mutations,
        num_nodes=num_nodes,
        sel_mut=sel_mut,
        sel_miss=sel_miss,
    )
    save_grg_spmv(state, path)
    return path, counts


def _synthetic_monomorphic_artifact(tmp_path) -> tuple[object, np.ndarray]:
    counts = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return _synthetic_counts_artifact(tmp_path, "chr7-monomorphic.grg_spmv", counts)


def _bolt_layout(artifacts):
    from pygrgl_spmv.backends.cusparse import plan_cusparse_layout
    from scripts.bench.cusparse import DEFAULT_CUSPARSE_PLAN_NAME, parse_cusparse_plan

    return plan_cusparse_layout(
        artifacts=list(artifacts),
        pair=parse_cusparse_plan(DEFAULT_CUSPARSE_PLAN_NAME),
        dtype=np.float64,
        requirements=_bolt_requirements(),
        vram_budget_bytes=1_000_000_000,
        ring_buffer_size=0,
        allow_residency=True,
        device=0,
        stream=0,
    )


@pytest.mark.gpu
@pytest.mark.cusparse
def test_grg_bolt_ops_matches_dense_formulas(primary_artifact, primary_grg):
    cp = pytest.importorskip("cupy")

    from pygrgl_spmv.backends.cusparse import CusparseRuntime, plan_cusparse_layout
    from scripts.bench.cusparse import DEFAULT_CUSPARSE_PLAN_NAME, parse_cusparse_plan

    x_dense, valid_mask = _dense_standardized_matrix(primary_grg)
    if not np.any(valid_mask):
        pytest.skip("small GRG fixture contains no polymorphic mutations")
    valid_count = int(np.count_nonzero(valid_mask))
    requirements = _bolt_requirements()
    layout = plan_cusparse_layout(
        artifacts=[primary_artifact],
        pair=parse_cusparse_plan(DEFAULT_CUSPARSE_PLAN_NAME),
        dtype=np.float64,
        requirements=requirements,
        vram_budget_bytes=1_000_000_000,
        ring_buffer_size=0,
        allow_residency=True,
        device=0,
        stream=0,
    )
    timing = {"up": TimingBucket(), "down": TimingBucket()}
    rng = np.random.default_rng(2042)
    v_host = rng.standard_normal(int(primary_grg.num_individuals))
    weights_host = rng.standard_normal(int(primary_grg.num_mutations))
    valid_indices = np.flatnonzero(valid_mask)
    local_idx = int(valid_indices[min(2, valid_count - 1)])

    with CusparseRuntime(layout) as runtime:
        with GrgBoltOps(runtime, [21], timing) as ops:
            ops.initialize_frequencies()
            assert ops.raw_m == int(primary_grg.num_mutations)
            assert ops.used_m == valid_count
            with ops.device:
                v_dev = cp.asarray(v_host, dtype=cp.float64)
                weights_dev = cp.asarray(weights_host, dtype=cp.float64)

            with ops.device:
                scores = cp.asnumpy(ops.scores_view(21, v_dev).copy())
            expected_scores = x_dense.T @ (v_host - v_host.mean())
            np.testing.assert_allclose(scores, expected_scores, atol=1e-8, rtol=1e-8)

            with ops.device:
                out = cp.empty((int(primary_grg.num_individuals),), dtype=cp.float64)
                actual_x = cp.asnumpy(ops.apply_x(21, weights_dev, out).copy())
            expected_x = x_dense @ weights_host
            expected_x -= expected_x.mean()
            np.testing.assert_allclose(actual_x, expected_x, atol=1e-8, rtol=1e-8)

            with ops.device:
                actual_col = cp.asnumpy(ops.column(21, local_idx))
            expected_col = x_dense[:, local_idx].copy()
            expected_col -= expected_col.mean()
            np.testing.assert_allclose(actual_col, expected_col, atol=1e-8, rtol=1e-8)

            with ops.device:
                actual_k = cp.asnumpy(ops.apply_k(v_dev, exclude_label=None, out=out).copy())
            centered_v = v_host - v_host.mean()
            expected_k = x_dense @ (x_dense.T @ centered_v) / float(valid_count)
            expected_k -= expected_k.mean()
            np.testing.assert_allclose(actual_k, expected_k, atol=1e-8, rtol=1e-8)

    assert timing["up"].calls >= 3
    assert timing["down"].calls >= 3


@pytest.mark.gpu
@pytest.mark.cusparse
def test_grg_bolt_ops_skips_monomorphic_snps_with_used_index_space(tmp_path):
    cp = pytest.importorskip("cupy")

    from pygrgl_spmv.backends.cusparse import CusparseRuntime, plan_cusparse_layout
    from scripts.bench.cusparse import DEFAULT_CUSPARSE_PLAN_NAME, parse_cusparse_plan

    artifact, counts = _synthetic_monomorphic_artifact(tmp_path)
    x_dense, valid_mask = _standardized_from_counts(counts, num_samples=counts.shape[0], ploidy=1)
    layout = plan_cusparse_layout(
        artifacts=[artifact],
        pair=parse_cusparse_plan(DEFAULT_CUSPARSE_PLAN_NAME),
        dtype=np.float64,
        requirements=_bolt_requirements(),
        vram_budget_bytes=1_000_000_000,
        ring_buffer_size=0,
        allow_residency=True,
        device=0,
        stream=0,
    )

    with CusparseRuntime(layout) as runtime:
        with GrgBoltOps(runtime, [7], {"up": TimingBucket(), "down": TimingBucket()}) as ops:
            ops.initialize_frequencies()
            state = ops.states[7]
            assert ops.raw_m == 5
            assert ops.used_m == 3
            assert state.num_used == 3
            assert state.num_monomorphic_ref == 1
            assert state.num_monomorphic_alt == 1
            np.testing.assert_array_equal(cp.asnumpy(state.used_local_indices), np.asarray([0, 3, 4]))
            np.testing.assert_array_equal(cp.asnumpy(state.used_mask), valid_mask)

            v_host = np.asarray([0.25, -1.0, 0.75, 2.0], dtype=np.float64)
            centered_v = v_host - v_host.mean()
            with ops.device:
                v_dev = cp.asarray(v_host, dtype=cp.float64)
                scores = cp.asnumpy(ops.scores_view(7, v_dev).copy())
                out = cp.empty((counts.shape[0],), dtype=cp.float64)

            expected_scores = x_dense.T @ centered_v
            np.testing.assert_allclose(scores, expected_scores, atol=1e-8, rtol=1e-8)
            np.testing.assert_array_equal(scores[~valid_mask], np.zeros(2, dtype=np.float64))
            assert [ops.used_global_to_local(idx) for idx in range(3)] == [(7, 0), (7, 3), (7, 4)]

            with pytest.raises(ValueError, match="chromosome 7 SNP 1 is monomorphic and was skipped"):
                ops.column(7, 1)

            with ops.device:
                actual_k = cp.asnumpy(ops.apply_k(v_dev, exclude_label=None, out=out).copy())
            expected_k = x_dense @ (x_dense.T @ centered_v) / 3.0
            expected_k -= expected_k.mean()
            np.testing.assert_allclose(actual_k, expected_k, atol=1e-8, rtol=1e-8)


@pytest.mark.gpu
@pytest.mark.cusparse
def test_initialize_frequencies_uses_runtime_stream_when_ambient_stream_differs(tmp_path):
    cp = pytest.importorskip("cupy")

    from pygrgl_spmv.backends.cusparse import CusparseRuntime, plan_cusparse_layout
    from scripts.bench.cusparse import DEFAULT_CUSPARSE_PLAN_NAME, parse_cusparse_plan

    artifact, counts = _synthetic_monomorphic_artifact(tmp_path)
    _x_dense, valid_mask = _standardized_from_counts(counts, num_samples=counts.shape[0], ploidy=1)
    allele_counts = counts.sum(axis=0)
    with cp.cuda.Device(0):
        requested = cp.cuda.Stream(non_blocking=True)
        ambient = cp.cuda.Stream(non_blocking=True)
    layout = plan_cusparse_layout(
        artifacts=[artifact],
        pair=parse_cusparse_plan(DEFAULT_CUSPARSE_PLAN_NAME),
        dtype=np.float64,
        requirements=_bolt_requirements(),
        vram_budget_bytes=1_000_000_000,
        ring_buffer_size=0,
        allow_residency=True,
        device=0,
        stream=requested,
    )

    with CusparseRuntime(layout) as runtime:
        assert runtime.stream_ptr == int(requested.ptr)
        with GrgBoltOps(runtime, [7], {"up": TimingBucket(), "down": TimingBucket()}) as ops:
            with ambient:
                ambient_ptr = int(cp.cuda.get_current_stream().ptr)
                ops.initialize_frequencies()
                assert int(cp.cuda.get_current_stream().ptr) == ambient_ptr

            state = ops.states[7]
            assert ops.raw_m == int(counts.shape[1])
            assert ops.used_m == int(np.count_nonzero(valid_mask))
            assert state.num_used == int(np.count_nonzero(valid_mask))
            assert state.num_monomorphic_ref == int(np.count_nonzero(allele_counts == 0.0))
            assert state.num_monomorphic_alt == int(np.count_nonzero(allele_counts == counts.shape[0]))
            np.testing.assert_array_equal(cp.asnumpy(state.used_local_indices), np.flatnonzero(valid_mask))
            np.testing.assert_array_equal(cp.asnumpy(state.used_mask), valid_mask)


@pytest.mark.gpu
@pytest.mark.cusparse
def test_simulate_null_phenotype_is_centered_unit_variance(tmp_path):
    cp = pytest.importorskip("cupy")

    from pygrgl_spmv.backends.cusparse import CusparseRuntime

    artifact, _counts = _synthetic_monomorphic_artifact(tmp_path)
    layout = _bolt_layout([artifact])

    with CusparseRuntime(layout) as runtime:
        with GrgBoltOps(runtime, [7], {"up": TimingBucket(), "down": TimingBucket()}) as ops:
            ops.initialize_frequencies()
            simulation = _simulate_phenotype(
                ops,
                mode="null",
                sim_h2=0.3,
                n_effect_check=30,
                phenotype_rng=np.random.default_rng(100),
                validation_rng=np.random.default_rng(200),
            )
            y = cp.asnumpy(simulation.y)

    assert simulation.metrics["phenotype.mode"] == "null"
    assert simulation.metrics["phenotype.true_h2"] == 0.0
    assert simulation.metrics["phenotype.var"] == pytest.approx(1.0, abs=1e-12)
    assert float(y.mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(np.mean(y * y)) == pytest.approx(1.0, abs=1e-12)
    assert simulation.effect_snps == ()


@pytest.mark.gpu
@pytest.mark.cusparse
def test_simulate_infinitesimal_phenotype_has_requested_empirical_h2(tmp_path):
    cp = pytest.importorskip("cupy")

    from pygrgl_spmv.backends.cusparse import CusparseRuntime

    artifact, _counts = _synthetic_monomorphic_artifact(tmp_path)
    layout = _bolt_layout([artifact])

    with CusparseRuntime(layout) as runtime:
        with GrgBoltOps(runtime, [7], {"up": TimingBucket(), "down": TimingBucket()}) as ops:
            ops.initialize_frequencies()
            simulation = _simulate_phenotype(
                ops,
                mode="infinitesimal",
                sim_h2=0.65,
                n_effect_check=10,
                phenotype_rng=np.random.default_rng(101),
                validation_rng=np.random.default_rng(201),
            )
            y = cp.asnumpy(simulation.y)
            state = ops.states[7]
            used_mask = cp.asnumpy(state.used_mask)

    assert simulation.metrics["phenotype.mode"] == "infinitesimal"
    assert simulation.metrics["phenotype.requested_h2"] == pytest.approx(0.65)
    assert simulation.metrics["phenotype.empirical_h2"] == pytest.approx(0.65, abs=1e-12)
    assert simulation.metrics["phenotype.var"] == pytest.approx(1.0, abs=1e-12)
    assert float(y.mean()) == pytest.approx(0.0, abs=1e-12)
    assert len(simulation.effect_snps) == 3
    for snp in simulation.effect_snps:
        assert snp.label == 7
        assert bool(used_mask[snp.local_idx])
        assert snp.local_idx not in {1, 2}


@pytest.mark.gpu
@pytest.mark.cusparse
def test_scan_metrics_reports_distribution_and_top_hits(tmp_path):
    cp = pytest.importorskip("cupy")

    from pygrgl_spmv.backends.cusparse import CusparseRuntime

    artifact, _counts = _synthetic_monomorphic_artifact(tmp_path)
    layout = _bolt_layout([artifact])

    with CusparseRuntime(layout) as runtime:
        with GrgBoltOps(runtime, [7], {"up": TimingBucket(), "down": TimingBucket()}) as ops:
            ops.initialize_frequencies()
            residual = ops.column(7, 3)
            metrics = _scan_metrics(ops, {7: residual}, 1.0, top_k=2)

    hist_total = sum(int(metrics[f"scan.p_hist.bin{i}.count"]) for i in range(int(metrics["scan.p_hist.bin_count"])))
    assert hist_total == 3
    assert metrics["scan.chr7.num_snps"] == 3
    assert metrics["scan.genome.num_snps"] == 3
    assert metrics["scan.top1.chr"] == 7
    assert metrics["scan.top1.local_idx"] == 3
    assert metrics["scan.top2.local_idx"] in {0, 4}
    assert metrics["scan.top2.local_idx"] not in {1, 2}
    assert 0.0 <= metrics["scan.p_hist.ks_approx"] <= 1.0
    assert np.isfinite(metrics["scan.lambda_gc_approx"])


@pytest.mark.gpu
@pytest.mark.cusparse
def test_effect_check_matches_controlled_dense_math(tmp_path):
    cp = pytest.importorskip("cupy")

    from pygrgl_spmv.backends.cusparse import CusparseRuntime

    counts = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    artifact, _counts = _synthetic_counts_artifact(tmp_path, "chr11-controlled.grg_spmv", counts)
    x_dense, valid_mask = _standardized_from_counts(counts, num_samples=counts.shape[0], ploidy=1)
    assert valid_mask.tolist() == [True, True, False]
    beta = np.asarray([0.25, -0.4], dtype=np.float64)
    y_host = x_dense[:, :2] @ beta
    layout = _bolt_layout([artifact])

    with CusparseRuntime(layout) as runtime:
        with GrgBoltOps(runtime, [11], {"up": TimingBucket(), "down": TimingBucket()}) as ops:
            ops.initialize_frequencies()
            with ops.device:
                y = cp.asarray(y_host, dtype=cp.float64)
            metrics = _effect_check_metrics(
                ops,
                {11: y},
                (
                    EffectCheckSnp(global_idx=0, label=11, local_idx=0, true_beta=float(beta[0])),
                    EffectCheckSnp(global_idx=1, label=11, local_idx=1, true_beta=float(beta[1])),
                ),
                sigma_g2=0.0,
                sigma_e2=1.0,
                rel_tol=1e-12,
                max_iter=20,
                bucket=CgBucket(),
            )

    assert metrics["effect_check.count"] == 2
    assert metrics["effect_check.beta_slope"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["effect_check.beta_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["effect_check.sign_concordance"] == 1.0

@pytest.mark.gpu
@pytest.mark.cusparse
def test_grg_bolt_ops_runs_on_runtime_device_when_current_device_differs(primary_artifact, primary_grg):
    cp = pytest.importorskip("cupy")

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs at least two CUDA devices")

    from pygrgl_spmv.backends.cusparse import CusparseRuntime, plan_cusparse_layout
    from scripts.bench.cusparse import DEFAULT_CUSPARSE_PLAN_NAME, parse_cusparse_plan

    _x_dense, valid_mask = _dense_standardized_matrix(primary_grg)
    if not np.any(valid_mask):
        pytest.skip("small GRG fixture contains no polymorphic mutations")
    valid_indices = np.flatnonzero(valid_mask)
    requirements = _bolt_requirements()
    layout = plan_cusparse_layout(
        artifacts=[primary_artifact],
        pair=parse_cusparse_plan(DEFAULT_CUSPARSE_PLAN_NAME),
        dtype=np.float64,
        requirements=requirements,
        vram_budget_bytes=1_000_000_000,
        ring_buffer_size=0,
        allow_residency=True,
        device=0,
        stream=0,
    )
    timing = {"up": TimingBucket(), "down": TimingBucket()}
    rng = np.random.default_rng(2043)
    v_host = rng.standard_normal(int(primary_grg.num_individuals))
    weights_host = rng.standard_normal(int(primary_grg.num_mutations))
    local_idx = int(valid_indices[min(2, int(valid_indices.size) - 1)])

    with cp.cuda.Device(1):
        with CusparseRuntime(layout) as runtime:
            with GrgBoltOps(runtime, [21], timing) as ops:
                ops.initialize_frequencies()
                assert cp.cuda.runtime.getDevice() == 1

                (state,) = tuple(ops.states.values())
                for view in state.views.values():
                    assert _cupy_device_id(view) == 0
                assert ops.weights_work is not None
                assert ops.sample_work is not None
                assert _cupy_device_id(ops.weights_work) == 0
                assert _cupy_device_id(ops.sample_work) == 0
                for array in (
                    state.used_mask,
                    state.used_local_indices,
                    state.mu,
                    state.sigma,
                    state.inv_sigma,
                    state.mu_over_sigma,
                ):
                    assert array is not None
                    assert _cupy_device_id(array) == 0

                with ops.device:
                    v_dev = cp.asarray(v_host, dtype=cp.float64)
                    weights_dev = cp.asarray(weights_host, dtype=cp.float64)
                    out_x = cp.empty((int(primary_grg.num_individuals),), dtype=cp.float64)
                    out_k = cp.empty_like(out_x)
                assert cp.cuda.runtime.getDevice() == 1

                scores = ops.scores_view(21, v_dev)
                column = ops.column(21, local_idx)
                x_result = ops.apply_x(21, weights_dev, out_x)
                k_result = ops.apply_k(v_dev, exclude_label=None, out=out_k)

                assert cp.cuda.runtime.getDevice() == 1
                for array in (scores, column, x_result, k_result):
                    assert _cupy_device_id(array) == 0
