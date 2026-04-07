"""Tests for the convert() function and SpmvGRG construction from different input types."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv import SpmvGRG, convert
from pygrgl_spmv.grg.compile import CompiledOperatorState
from pygrgl_spmv.tests.conftest import (
    DATA_DTYPE,
    HAS_MKL_RUNTIME,
    make_mkl_backend,
    matmul_expect_k_hint_warning,
    tol,
)

MKL_ONLY = pytest.mark.skipif(not HAS_MKL_RUNTIME, reason="MKL runtime unavailable (libmkl_rt.so not found)")


# ---------------------------------------------------------------------------
# convert() — no backend required
# ---------------------------------------------------------------------------


def test_convert_from_path_saves_artifact(primary_grg_path, tmp_path):
    state = convert(primary_grg_path, tmp_path)
    expected = tmp_path / f"{Path(primary_grg_path).stem}.grg_spmv"
    assert expected.exists(), f"Expected artifact at {expected}"
    assert isinstance(state, CompiledOperatorState)


def test_convert_from_path_in_memory(primary_grg_path, tmp_path):
    state = convert(primary_grg_path)
    assert isinstance(state, CompiledOperatorState)
    assert not any(tmp_path.iterdir()), "In-memory convert should not write any files"


def test_convert_from_grg_object_in_memory(grg_ref, tmp_path):
    state = convert(grg_ref)
    assert isinstance(state, CompiledOperatorState)
    assert state.num_samples == grg_ref.num_samples
    assert state.num_mutations == grg_ref.num_mutations


def test_convert_from_grg_object_with_save(grg_ref, tmp_path):
    state = convert(grg_ref, tmp_path, name="my_grg")
    artifact = tmp_path / "my_grg.grg_spmv"
    assert artifact.exists(), f"Expected artifact at {artifact}"
    assert isinstance(state, CompiledOperatorState)


def test_convert_init_biases_present(primary_grg_path):
    state = convert(primary_grg_path)
    assert state.init_vector_up_bias is not None
    assert state.init_vector_down_bias is not None
    assert state.init_vector_up_bias.shape == (state.num_mutations,)
    assert state.init_vector_down_bias.shape == (state.num_samples,)


def test_convert_init_biases_path_vs_grg_object(primary_grg_path, grg_ref):
    state_path = convert(primary_grg_path)
    state_obj = convert(grg_ref)
    np.testing.assert_array_equal(state_path.init_vector_up_bias, state_obj.init_vector_up_bias)
    np.testing.assert_array_equal(state_path.init_vector_down_bias, state_obj.init_vector_down_bias)
    if state_path.init_xtx_up_bias is not None:
        np.testing.assert_array_equal(state_path.init_xtx_up_bias, state_obj.init_xtx_up_bias)
        np.testing.assert_array_equal(state_path.init_xtx_down_bias, state_obj.init_xtx_down_bias)
    else:
        assert state_obj.init_xtx_up_bias is None
        assert state_obj.init_xtx_down_bias is None


def test_convert_error_grg_object_save_requires_name(grg_ref, tmp_path):
    with pytest.raises(ValueError, match="name"):
        convert(grg_ref, tmp_path)


def test_convert_error_invalid_suffix(tmp_path):
    with pytest.raises(ValueError):
        convert(str(tmp_path / "file.txt"), tmp_path)


# ---------------------------------------------------------------------------
# SpmvGRG construction — backend-parametrized
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_spmvgrg_from_grg_path(backend_builder, primary_grg_path, spmv_cache_dir):
    op = SpmvGRG(primary_grg_path, backend_builder(), DATA_DTYPE, artifact_dir=spmv_cache_dir)
    assert op.artifact_path is not None
    assert op.artifact_path.exists()
    assert op.artifact_path.suffix == ".grg_spmv"


@pytest.mark.smoke
def test_spmvgrg_from_spmv_path(backend_builder, primary_grg_path, tmp_path, spmv_cache_dir):
    # First build the artifact via a path-based op
    first = SpmvGRG(primary_grg_path, backend_builder(), DATA_DTYPE, artifact_dir=spmv_cache_dir)
    artifact = first.artifact_path
    # Load directly from the .grg_spmv artifact
    second = SpmvGRG(artifact, backend_builder(), DATA_DTYPE)
    assert second.artifact_path == artifact
    assert second.num_samples == first.num_samples
    assert second.num_mutations == first.num_mutations


@pytest.mark.smoke
def test_spmvgrg_from_grg_object(backend_builder, grg_ref, primary_grg_path, spmv_cache_dir):
    op = SpmvGRG(grg_ref, backend_builder(), DATA_DTYPE)
    assert op.artifact_path is None
    assert op.num_samples == grg_ref.num_samples
    assert op.num_mutations == grg_ref.num_mutations


def test_spmvgrg_invalid_backend_type(grg_ref):
    with pytest.raises(TypeError):
        SpmvGRG(grg_ref, "not_a_backend", DATA_DTYPE)


def test_spmvgrg_invalid_path_suffix(backend_builder, tmp_path):
    with pytest.raises(ValueError):
        SpmvGRG(str(tmp_path / "file.csv"), backend_builder(), DATA_DTYPE)


# ---------------------------------------------------------------------------
# Init bias consistency: path-based vs GRG-object construction
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_init_biases_consistent_grg_vs_path(backend_builder, primary_grg_path, grg_ref, spmv_cache_dir):
    op_path = SpmvGRG(primary_grg_path, backend_builder(), DATA_DTYPE, artifact_dir=spmv_cache_dir)
    op_obj = SpmvGRG(grg_ref, backend_builder(), DATA_DTYPE)

    np.testing.assert_array_equal(op_path.init_vector_up_bias, op_obj.init_vector_up_bias)
    np.testing.assert_array_equal(op_path.init_vector_down_bias, op_obj.init_vector_down_bias)
    if op_path.init_xtx_up_bias is not None:
        np.testing.assert_array_equal(op_path.init_xtx_up_bias, op_obj.init_xtx_up_bias)
        np.testing.assert_array_equal(op_path.init_xtx_down_bias, op_obj.init_xtx_down_bias)
    else:
        assert op_obj.init_xtx_up_bias is None

    rng = np.random.default_rng(42)
    x = rng.standard_normal((3, grg_ref.num_samples), dtype=DATA_DTYPE)
    init_vec = rng.standard_normal(3, dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    result_path = matmul_expect_k_hint_warning(op_path, x, pygrgl.TraversalDirection.UP, init=init_vec)
    result_obj = matmul_expect_k_hint_warning(op_obj, x, pygrgl.TraversalDirection.UP, init=init_vec)
    np.testing.assert_allclose(result_path, result_obj, atol=atol, rtol=rtol)
