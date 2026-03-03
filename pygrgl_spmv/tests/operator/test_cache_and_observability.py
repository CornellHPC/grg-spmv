"""Cache lifecycle and observability tests for operator/backends."""

from __future__ import annotations

import logging
import shutil

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.tests.conftest import DATA_DTYPE, HAS_MKL_RUNTIME, INDEX_DTYPE, tol

MKL_ONLY = pytest.mark.skipif(not HAS_MKL_RUNTIME, reason="MKL runtime unavailable (libmkl_rt.so not found)")


@pytest.mark.smoke
def test_wavefront_debug_logging_preserves_values(backend_config, primary_grg_path, spmv_cache_dir):
    cfg_base = dict(backend_config)
    cfg_wave = dict(backend_config)
    cfg_wave["log_level"] = "DEBUG"

    op_base = SpmvGRG(primary_grg_path, cfg_base, DATA_DTYPE, INDEX_DTYPE, cache_dir=spmv_cache_dir)
    op_wave = SpmvGRG(primary_grg_path, cfg_wave, DATA_DTYPE, INDEX_DTYPE, cache_dir=spmv_cache_dir)

    rng = np.random.default_rng(4404)
    rows = 2
    x_up = rng.standard_normal((rows, op_base.n), dtype=DATA_DTYPE)
    x_down = rng.standard_normal((rows, op_base.m), dtype=DATA_DTYPE)

    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(
        op_wave.matmul(x_up, pygrgl.TraversalDirection.UP),
        op_base.matmul(x_up, pygrgl.TraversalDirection.UP),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        op_wave.matmul(x_down, pygrgl.TraversalDirection.DOWN),
        op_base.matmul(x_down, pygrgl.TraversalDirection.DOWN),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.mkl
@MKL_ONLY
def test_wavefront_debug_logs_all_levels(primary_grg_path, spmv_cache_dir, caplog):
    cfg = {"type": "mkl", "n_threads": 1, "log_level": "DEBUG"}
    with caplog.at_level(logging.DEBUG, logger="pygrgl_spmv.backends.mkl.MklBackend"):
        op = SpmvGRG(primary_grg_path, cfg, DATA_DTYPE, INDEX_DTYPE, cache_dir=spmv_cache_dir)
        x = np.ones((2, op.n), dtype=DATA_DTYPE)
        _ = op.matmul(x, pygrgl.TraversalDirection.UP)
    messages = [rec.getMessage() for rec in caplog.records]
    assert any(msg.startswith("wavefront[up]") for msg in messages)
    assert any("  level=" in msg for msg in messages)


@pytest.mark.mkl
@MKL_ONLY
def test_cache_miss_build_uses_up_edges_and_fails_on_missing_coals(primary_grg_path, tmp_path, monkeypatch):
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

    cfg = {"type": "mkl", "n_threads": 1}
    with pytest.raises(ValueError, match="missing coalescence counts"):
        SpmvGRG(dst, cfg, DATA_DTYPE, INDEX_DTYPE, cache_dir=cache_dir)
    assert calls["loader"] == 1
    assert calls["load_up_edges"] == [True]
    assert calls["calculate_missing_coals"] == 0


@pytest.mark.mkl
@MKL_ONLY
def test_cache_hit_does_not_reload_grg(primary_grg_path, tmp_path, monkeypatch):
    dst = tmp_path / "cache-hit.grg"
    shutil.copy2(primary_grg_path, dst)
    cache_dir = tmp_path / "cache"
    cfg = {"type": "mkl", "n_threads": 1}

    first = SpmvGRG(dst, cfg, DATA_DTYPE, INDEX_DTYPE, cache_dir=cache_dir)
    import pygrgl_spmv.grg as grg_module

    def _forbidden_loader(*_args, **_kwargs):
        raise AssertionError("cache hit should not call pygrgl.load_immutable_grg")

    monkeypatch.setattr(grg_module.pygrgl, "load_immutable_grg", _forbidden_loader)
    second = SpmvGRG(dst, cfg, DATA_DTYPE, INDEX_DTYPE, cache_dir=cache_dir)
    assert second.shape == first.shape
    np.testing.assert_array_equal(second.coalescence_counts, first.coalescence_counts)
    np.testing.assert_allclose(second._init_vector_up_bias, first._init_vector_up_bias)
    np.testing.assert_allclose(second._init_vector_down_bias, first._init_vector_down_bias)
    if first._init_xtx_up_bias is None:
        assert second._init_xtx_up_bias is None
        assert second._init_xtx_down_bias is None
    else:
        np.testing.assert_allclose(second._init_xtx_up_bias, first._init_xtx_up_bias)
        np.testing.assert_allclose(second._init_xtx_down_bias, first._init_xtx_down_bias)


@pytest.mark.mkl
@MKL_ONLY
def test_cache_hit_skips_init_bias_rebuild(primary_grg_path, tmp_path, monkeypatch):
    dst = tmp_path / "cache-hit-no-rebuild.grg"
    shutil.copy2(primary_grg_path, dst)
    cache_dir = tmp_path / "cache"
    cfg = {"type": "mkl", "n_threads": 1}

    _ = SpmvGRG(dst, cfg, DATA_DTYPE, INDEX_DTYPE, cache_dir=cache_dir)

    import pygrgl_spmv.grg as grg_module

    def _forbidden_rebuild(self):
        raise AssertionError("cache hit should not rebuild init bias cache")

    monkeypatch.setattr(grg_module.SpmvGRG, "_build_init_bias_cache", _forbidden_rebuild)
    second = SpmvGRG(dst, cfg, DATA_DTYPE, INDEX_DTYPE, cache_dir=cache_dir)
    assert second.shape[0] > 0


@pytest.mark.smoke
@pytest.mark.mkl
@MKL_ONLY
def test_mem_usage_records_setup_and_calls(primary_grg_path, spmv_cache_dir):
    cfg = {"type": "mkl", "n_threads": 1, "log_level": "WARNING"}
    op = SpmvGRG(primary_grg_path, cfg, DATA_DTYPE, INDEX_DTYPE, cache_dir=spmv_cache_dir)
    assert op._backend.mem_usage.host_static.level_offsets > 0
    assert len(op._backend.mem_usage.calls) == 0

    x_up = np.ones((2, op.n), dtype=DATA_DTYPE)
    x_down = np.ones((2, op.m), dtype=DATA_DTYPE)
    _ = op.matmul(x_up, pygrgl.TraversalDirection.UP)
    _ = op.matmul(x_down, pygrgl.TraversalDirection.DOWN)

    stages = [call.stage for call in op._backend.mem_usage.calls]
    assert stages == ["run_up", "run_down"]
    assert int(op._backend.mem_usage.calls[0].runtime_k) == 2
    assert int(op._backend.mem_usage.calls[1].runtime_k) == 2
