"""Cache lifecycle and observability tests for GRG/backend plumbing."""

from __future__ import annotations

import logging
import shutil

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.backends import ReferenceBackend, ReferencePlanPair
from pygrgl_spmv.grg.artifact import load_grg_spmv
from pygrgl_spmv.tests.conftest import (
    DATA_DTYPE,
    HAS_MKL_RUNTIME,
    make_mkl_backend,
    matmul_expect_k_hint_warning,
    tol,
)

MKL_ONLY = pytest.mark.skipif(not HAS_MKL_RUNTIME, reason="MKL runtime unavailable (libmkl_rt.so not found)")


@pytest.mark.smoke
def test_wavefront_debug_logging_preserves_values(backend_builder, primary_grg_path, spmv_cache_dir):
    base_backend = backend_builder()
    wave_backend = backend_builder()
    wave_backend._logger.setLevel(logging.DEBUG)

    op_base = SpmvGRG(primary_grg_path, base_backend, DATA_DTYPE, artifact_dir=spmv_cache_dir)
    op_wave = SpmvGRG(primary_grg_path, wave_backend, DATA_DTYPE, artifact_dir=spmv_cache_dir)

    rng = np.random.default_rng(4404)
    rows = 2
    x_up = rng.standard_normal((rows, op_base.num_samples), dtype=DATA_DTYPE)
    x_down = rng.standard_normal((rows, op_base.num_mutations), dtype=DATA_DTYPE)

    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(
        matmul_expect_k_hint_warning(op_wave, x_up, pygrgl.TraversalDirection.UP),
        matmul_expect_k_hint_warning(op_base, x_up, pygrgl.TraversalDirection.UP),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        matmul_expect_k_hint_warning(op_wave, x_down, pygrgl.TraversalDirection.DOWN),
        matmul_expect_k_hint_warning(op_base, x_down, pygrgl.TraversalDirection.DOWN),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.smoke
def test_reusing_backend_instance_does_not_leak_call_buffers_into_next_setup(primary_grg_path, spmv_cache_dir):
    backend = ReferenceBackend(
        pair=ReferencePlanPair(
            plan_up=ReferenceBackend.plan(fmt="CSR", store="N", k_hint=None),
            plan_down=ReferenceBackend.plan(fmt="CSC", store="T", k_hint=None),
        ),
    )
    first = SpmvGRG(primary_grg_path, backend, DATA_DTYPE, artifact_dir=spmv_cache_dir)
    x = np.ones((8, first.num_samples), dtype=DATA_DTYPE)
    _ = first.matmul(x, pygrgl.TraversalDirection.UP)

    second = SpmvGRG(primary_grg_path, backend, DATA_DTYPE, artifact_dir=spmv_cache_dir)
    assert second.memory.retained is not None
    assert not any(row.retention == "call" and row.owner == "backend" for row in second.memory.retained.allocations)


@pytest.mark.smoke
def test_validation_error_before_backend_run_does_not_leave_capture_active(primary_grg_path, spmv_cache_dir):
    backend = ReferenceBackend(
        pair=ReferencePlanPair(
            plan_up=ReferenceBackend.plan(fmt="CSR", store="N", k_hint=None),
            plan_down=ReferenceBackend.plan(fmt="CSC", store="T", k_hint=None),
        ),
    )
    op = SpmvGRG(primary_grg_path, backend, DATA_DTYPE, artifact_dir=spmv_cache_dir)
    op._compiled.coalescence_counts = None
    op._retained_mem.coalescence_counts = None
    x = np.ones((2, op.num_samples), dtype=DATA_DTYPE)
    with pytest.raises(ValueError, match="coalescence counts"):
        op.matmul(x, pygrgl.TraversalDirection.UP, init="xtx")
    reused = SpmvGRG(primary_grg_path, backend, DATA_DTYPE, artifact_dir=spmv_cache_dir)
    assert reused.memory.retained is not None


@pytest.mark.mkl
@MKL_ONLY
def test_wavefront_debug_logs_all_levels_when_instrumented(primary_grg_path, spmv_cache_dir, caplog):
    with caplog.at_level(logging.DEBUG, logger="pygrgl_spmv.backends.mkl.backend.MklBackend"):
        op = SpmvGRG(
            primary_grg_path,
            make_mkl_backend(fmt_up="csr", fmt_down=None, n_threads=1, log_level="DEBUG", instrumentation=True),
            DATA_DTYPE,
            artifact_dir=spmv_cache_dir,
        )
        x = np.ones((2, op.num_samples), dtype=DATA_DTYPE)
        _ = op.matmul(x, pygrgl.TraversalDirection.UP)
    messages = [rec.getMessage() for rec in caplog.records]
    assert any(msg.startswith("wavefront[up]") for msg in messages)
    assert any("  level=" in msg for msg in messages)


@pytest.mark.mkl
@MKL_ONLY
def test_wavefront_debug_logging_is_quiet_without_instrumentation(primary_grg_path, spmv_cache_dir, caplog):
    with caplog.at_level(logging.DEBUG, logger="pygrgl_spmv.backends.mkl.backend.MklBackend"):
        op = SpmvGRG(
            primary_grg_path,
            make_mkl_backend(fmt_up="csr", fmt_down=None, n_threads=1, log_level="DEBUG", instrumentation=False),
            DATA_DTYPE,
            artifact_dir=spmv_cache_dir,
        )
        x = np.ones((2, op.num_samples), dtype=DATA_DTYPE)
        _ = op.matmul(x, pygrgl.TraversalDirection.UP)
    messages = [rec.getMessage() for rec in caplog.records]
    assert not any(msg.startswith("wavefront[up]") for msg in messages)


def test_cache_miss_build_logs_rss_checkpoints_at_info(primary_grg_path, tmp_path, caplog):
    dst = tmp_path / "rss-build.grg"
    shutil.copy2(primary_grg_path, dst)
    artifact_dir = tmp_path / "artifact-root"
    backend = ReferenceBackend(
        pair=ReferencePlanPair(
            plan_up=ReferenceBackend.plan(fmt="CSR", store="N", k_hint=None),
            plan_down=ReferenceBackend.plan(fmt="CSC", store="T", k_hint=None),
        ),
    )

    with caplog.at_level(logging.INFO):
        op = SpmvGRG(dst, backend, DATA_DTYPE, artifact_dir=artifact_dir)

    operator_messages = [
        rec.getMessage()
        for rec in caplog.records
        if rec.name == "pygrgl_spmv.grg.SpmvGRG"
    ]
    compile_messages = [
        rec.getMessage()
        for rec in caplog.records
        if rec.name == "pygrgl_spmv.grg.compile"
    ]
    compile_labels = [
        msg.split()[1]
        for msg in compile_messages
        if msg.startswith("rss compile:")
    ]
    expected_labels = [
        "compile:start",
        "compile:blocks_ready",
        "compile:mutation_rows_loaded",
        "compile:selectors_built",
        "compile:selectors_ready",
        "compile:mutation_table_ready",
    ]
    matched = 0
    for label in compile_labels:
        if matched < len(expected_labels) and label == expected_labels[matched]:
            matched += 1

    assert any("Building SpmvGRG from" in msg for msg in operator_messages)
    assert any("rss artifact:grg_loaded" in msg and "rss_bytes=" in msg for msg in compile_messages)
    assert matched == len(expected_labels)
    assert any("rss artifact:saved" in msg and "delta_bytes=" in msg for msg in compile_messages)
    assert op.artifact_path.exists()


def test_artifact_load_rehydrates_shared_nonempty_block_data(primary_grg_path, tmp_path):
    dst = tmp_path / "artifact-shared-data.grg"
    shutil.copy2(primary_grg_path, dst)
    artifact_dir = tmp_path / "artifact-root"
    backend = ReferenceBackend(
        pair=ReferencePlanPair(
            plan_up=ReferenceBackend.plan(fmt="CSR", store="N", k_hint=None),
            plan_down=ReferenceBackend.plan(fmt="CSC", store="T", k_hint=None),
        ),
    )
    op = SpmvGRG(
        dst,
        backend,
        DATA_DTYPE,
        artifact_dir=artifact_dir,
    )

    state = load_grg_spmv(op.artifact_path, DATA_DTYPE)
    assert state.A_blocks is not None
    data_arrays = [block.data for level_blocks in state.A_blocks for block in level_blocks if block.nnz > 0]
    assert data_arrays
    first = data_arrays[0]
    assert first.strides == (0,)
    assert not first.flags.writeable
    for data in data_arrays[1:]:
        assert data.strides == (0,)
        assert not data.flags.writeable
        assert np.shares_memory(data, first)


@pytest.mark.mkl
@MKL_ONLY
def test_cache_miss_build_uses_down_edges_only_and_fails_on_missing_coals(primary_grg_path, tmp_path, monkeypatch):
    dst = tmp_path / "coal-build.grg"
    shutil.copy2(primary_grg_path, dst)
    cache_dir = tmp_path / "cache"

    import pygrgl_spmv.grg as grg_module

    real_loader = grg_module.pygrgl.load_immutable_grg
    calls = {"loader": 0, "load_up_edges": [], "calculate_missing_coals": 0}

    class _GrgProxy:
        def __init__(self, inner):
            self._inner = inner

        def calculate_missing_coals(self):
            calls["calculate_missing_coals"] += 1
            return self._inner.calculate_missing_coals()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def _wrapped_loader(path, *args, **kwargs):
        calls["loader"] += 1
        calls["load_up_edges"].append(bool(kwargs.get("load_up_edges", False)))
        g = real_loader(path, *args, **kwargs)
        counts = np.array([g.get_num_individual_coals(i) for i in range(g.num_nodes)], dtype=np.int64)
        internal_positive = np.flatnonzero(counts[g.num_samples :] > 0) + g.num_samples
        assert internal_positive.size > 0
        g.set_num_individual_coals(int(internal_positive[0]), grg_module.pygrgl.COAL_COUNT_NOT_SET)
        return _GrgProxy(g)

    monkeypatch.setattr(grg_module.pygrgl, "load_immutable_grg", _wrapped_loader)

    with pytest.raises(ValueError, match="missing coalescence counts"):
        SpmvGRG(
            dst,
            make_mkl_backend(fmt_up="csr", fmt_down=None, n_threads=1),
            DATA_DTYPE,
            artifact_dir=cache_dir,
        )
    assert calls["loader"] == 1
    assert calls["load_up_edges"] == [False]
    assert calls["calculate_missing_coals"] == 0


@pytest.mark.mkl
@MKL_ONLY
def test_cache_hit_does_not_reload_grg(primary_grg_path, tmp_path, monkeypatch):
    dst = tmp_path / "cache-hit.grg"
    shutil.copy2(primary_grg_path, dst)
    cache_dir = tmp_path / "cache"
    first = SpmvGRG(
        dst,
        make_mkl_backend(fmt_up="csr", fmt_down=None, n_threads=1),
        DATA_DTYPE,
        artifact_dir=cache_dir,
    )
    import pygrgl_spmv.grg as grg_module

    def _forbidden_loader(*_args, **_kwargs):
        raise AssertionError("cache hit should not call pygrgl.load_immutable_grg")

    monkeypatch.setattr(grg_module.pygrgl, "load_immutable_grg", _forbidden_loader)
    second = SpmvGRG(
        dst,
        make_mkl_backend(fmt_up="csr", fmt_down=None, n_threads=1),
        DATA_DTYPE,
        artifact_dir=cache_dir,
    )
    assert second.shape == first.shape
    np.testing.assert_array_equal(second.coalescence_counts, first.coalescence_counts)
    np.testing.assert_allclose(second.init_vector_up_bias, first.init_vector_up_bias)
    np.testing.assert_allclose(second.init_vector_down_bias, first.init_vector_down_bias)
    if first.init_xtx_up_bias is None:
        assert second.init_xtx_up_bias is None
        assert second.init_xtx_down_bias is None
    else:
        np.testing.assert_allclose(second.init_xtx_up_bias, first.init_xtx_up_bias)
        np.testing.assert_allclose(second.init_xtx_down_bias, first.init_xtx_down_bias)


@pytest.mark.mkl
@MKL_ONLY
def test_cache_hit_skips_init_bias_rebuild(primary_grg_path, tmp_path, monkeypatch):
    dst = tmp_path / "cache-hit-no-rebuild.grg"
    shutil.copy2(primary_grg_path, dst)
    cache_dir = tmp_path / "cache"
    _ = SpmvGRG(
        dst,
        make_mkl_backend(fmt_up="csr", fmt_down=None, n_threads=1),
        DATA_DTYPE,
        artifact_dir=cache_dir,
    )

    import pygrgl_spmv.grg as grg_module

    def _forbidden_rebuild(*_args, **_kwargs):
        raise AssertionError("cache hit should not rebuild init bias cache")

    monkeypatch.setattr(grg_module, "_build_init_biases", _forbidden_rebuild)
    second = SpmvGRG(
        dst,
        make_mkl_backend(fmt_up="csr", fmt_down=None, n_threads=1),
        DATA_DTYPE,
        artifact_dir=cache_dir,
    )
    assert second.shape[0] > 0


@pytest.mark.mkl
@MKL_ONLY
def test_artifact_path_uses_grg_spmv_suffix(primary_grg_path, tmp_path):
    dst = tmp_path / "artifact-path.grg"
    shutil.copy2(primary_grg_path, dst)
    artifact_dir = tmp_path / "artifact-root"
    op = SpmvGRG(
        dst,
        make_mkl_backend(fmt_up="csr", fmt_down=None, n_threads=1),
        DATA_DTYPE,
        artifact_dir=artifact_dir,
    )
    assert op.artifact_path.suffix == ".grg_spmv"
    assert op.artifact_path.exists()


@pytest.mark.mkl
@MKL_ONLY
def test_direct_artifact_load_skips_grg_loader(primary_grg_path, tmp_path, monkeypatch):
    dst = tmp_path / "direct-artifact.grg"
    shutil.copy2(primary_grg_path, dst)
    artifact_dir = tmp_path / "artifact-root"
    first = SpmvGRG(
        dst,
        make_mkl_backend(fmt_up="csr", fmt_down=None, n_threads=1),
        DATA_DTYPE,
        artifact_dir=artifact_dir,
    )

    import pygrgl_spmv.grg as grg_module

    def _forbidden_loader(*_args, **_kwargs):
        raise AssertionError("direct .grg_spmv load should not call pygrgl.load_immutable_grg")

    monkeypatch.setattr(grg_module.pygrgl, "load_immutable_grg", _forbidden_loader)
    second = SpmvGRG(
        first.artifact_path,
        make_mkl_backend(fmt_up="csr", fmt_down=None, n_threads=1),
        DATA_DTYPE,
        artifact_dir=artifact_dir,
    )

    assert second.shape == first.shape
    assert second.num_samples == first.num_samples
    assert second.num_individuals == first.num_individuals
    assert second.num_mutations == first.num_mutations
    assert second.num_nodes == first.num_nodes
    assert second.num_edges == first.num_edges
    assert second.has_missing_data == first.has_missing_data
    for mutation_id in range(first.num_mutations):
        left = first.get_mutation_by_id(mutation_id)
        right = second.get_mutation_by_id(mutation_id)
        assert right.position == left.position
        assert right.time == left.time
        assert right.allele == left.allele
        assert right.ref_allele == left.ref_allele


@pytest.mark.mkl
@MKL_ONLY
def test_direct_artifact_load_logs_structural_array_dtypes(primary_grg_path, tmp_path, caplog):
    dst = tmp_path / "direct-artifact-logs.grg"
    shutil.copy2(primary_grg_path, dst)
    artifact_dir = tmp_path / "artifact-root"
    first = SpmvGRG(
        dst,
        make_mkl_backend(fmt_up="csr", fmt_down=None, n_threads=1),
        DATA_DTYPE,
        artifact_dir=artifact_dir,
    )

    with caplog.at_level(logging.INFO, logger="pygrgl_spmv.grg.artifact"):
        _ = SpmvGRG(
            first.artifact_path,
            make_mkl_backend(fmt_up="csr", fmt_down=None, n_threads=1),
            DATA_DTYPE,
            artifact_dir=artifact_dir,
        )

    messages = [rec.getMessage() for rec in caplog.records if rec.name == "pygrgl_spmv.grg.artifact"]
    assert any("artifact array key=level_offsets" in msg and "dtype=" in msg for msg in messages)
    assert any("artifact array key=sel_mut_indices" in msg and "dtype=" in msg for msg in messages)
    assert any("artifact array key=A_blocks_1_0_indptr" in msg and "dtype=" in msg for msg in messages)


@pytest.mark.smoke
@pytest.mark.mkl
@MKL_ONLY
def test_mem_usage_tracks_retained_and_last_call(primary_grg_path, spmv_cache_dir):
    op = SpmvGRG(
        primary_grg_path,
        make_mkl_backend(fmt_up="csr", fmt_down=None, n_threads=1, log_level="WARNING"),
        DATA_DTYPE,
        artifact_dir=spmv_cache_dir,
    )
    assert op.memory.retained is not None
    assert op.memory.last_call is None
    assert not any(row.retention == "call" for row in op.memory.retained.allocations)

    x_up = np.ones((2, op.num_samples), dtype=DATA_DTYPE)
    x_down = np.ones((2, op.num_mutations), dtype=DATA_DTYPE)
    _ = op.matmul(x_up, pygrgl.TraversalDirection.UP)
    first_call = op.memory.last_call
    assert first_call is not None
    assert first_call.stage == "run_up"
    assert int(first_call.runtime_k) == 2
    assert all(row.retention == "call" for row in first_call.allocations)
    _ = op.matmul(x_down, pygrgl.TraversalDirection.DOWN)
    second_call = op.memory.last_call
    assert second_call is not None
    assert second_call is not first_call
    assert second_call.stage == "run_down"
    assert int(second_call.runtime_k) == 2
    assert all(row.retention == "call" for row in second_call.allocations)
