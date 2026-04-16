from __future__ import annotations

import os
import shutil
from pathlib import Path, PurePath

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv import convert
from pygrgl_spmv.grg.artifact import artifact_path_for_grg, load_grg_spmv


class _PathWrapper(os.PathLike[str]):
    def __init__(self, value: str) -> None:
        self._value = value

    def __fspath__(self) -> str:
        return self._value


def _pure_path(value: str):
    return PurePath(value)


def _wrapped_path(value: str):
    return _PathWrapper(value)


def test_convert_from_path_writes_artifact(primary_grg_path, tmp_path):
    artifact = convert(primary_grg_path, tmp_path)
    assert artifact.exists()
    assert artifact.suffix == ".grg_spmv"
    assert artifact == artifact_path_for_grg(Path(primary_grg_path), tmp_path)


def test_convert_avoids_basename_collisions(primary_grg_path, tmp_path):
    src_a = tmp_path / "a" / "sample.grg"
    src_b = tmp_path / "b" / "sample.grg"
    src_a.parent.mkdir(parents=True, exist_ok=True)
    src_b.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(primary_grg_path, src_a)
    shutil.copyfile(primary_grg_path, src_b)
    out_dir = tmp_path / "artifacts"
    artifact_a = convert(src_a, out_dir)
    artifact_b = convert(src_b, out_dir)
    assert artifact_a == artifact_path_for_grg(src_a, out_dir)
    assert artifact_b == artifact_path_for_grg(src_b, out_dir)
    assert artifact_a != artifact_b
    assert artifact_a.exists()
    assert artifact_b.exists()


@pytest.mark.parametrize(
    "path_factory",
    [
        pytest.param(_pure_path, id="purepath"),
        pytest.param(_wrapped_path, id="pathlike-wrapper"),
    ],
)
def test_convert_accepts_generic_pathlikes(primary_grg_path, tmp_path, path_factory):
    artifact = convert(path_factory(primary_grg_path), tmp_path)
    assert artifact.exists()
    assert artifact.suffix == ".grg_spmv"


def test_convert_from_grg_object_requires_name(primary_grg, tmp_path):
    with pytest.raises(ValueError, match="name"):
        convert(primary_grg, tmp_path)


def test_convert_from_grg_object_writes_artifact(primary_grg, tmp_path):
    artifact = convert(primary_grg, tmp_path, name="from-object")
    assert artifact == tmp_path / "from-object.grg_spmv"
    assert artifact.exists()


def test_convert_requires_output_dir(primary_grg_path):
    with pytest.raises(ValueError, match="output_dir"):
        convert(primary_grg_path, None)


def test_convert_rejects_invalid_suffix(tmp_path):
    with pytest.raises(ValueError, match=r"\.grg"):
        convert(str(tmp_path / "file.txt"), tmp_path)


def test_convert_init_biases_present(primary_grg_path, tmp_path):
    artifact = convert(primary_grg_path, tmp_path)
    state = load_grg_spmv(artifact, np.float64)
    assert state.init_vector_up_bias is not None
    assert state.init_vector_down_bias is not None
    assert state.init_vector_up_bias.shape == (state.num_mutations,)
    assert state.init_vector_down_bias.shape == (state.num_samples,)


def test_convert_init_biases_match_path_and_object(primary_grg_path, primary_grg, tmp_path):
    path_artifact = convert(primary_grg_path, tmp_path / "path")
    obj_artifact = convert(primary_grg, tmp_path / "obj", name="primary")
    state_path = load_grg_spmv(path_artifact, np.float64)
    state_obj = load_grg_spmv(obj_artifact, np.float64)
    np.testing.assert_array_equal(state_path.init_vector_up_bias, state_obj.init_vector_up_bias)
    np.testing.assert_array_equal(state_path.init_vector_down_bias, state_obj.init_vector_down_bias)
    if state_path.init_xtx_up_bias is None:
        assert state_obj.init_xtx_up_bias is None
        assert state_obj.init_xtx_down_bias is None
    else:
        np.testing.assert_array_equal(state_path.init_xtx_up_bias, state_obj.init_xtx_up_bias)
        np.testing.assert_array_equal(state_path.init_xtx_down_bias, state_obj.init_xtx_down_bias)


def test_convert_loads_grg_with_down_edges_only_and_does_not_compute_missing_coals(
    primary_grg_path,
    monkeypatch,
    tmp_path,
):
    import pygrgl_spmv.grg as grg_module

    real_loader = grg_module.pygrgl.load_immutable_grg
    calls = {"load_up_edges": [], "calculate_missing_coals": 0}

    class _Proxy:
        def __init__(self, inner):
            self._inner = inner

        def calculate_missing_coals(self):
            calls["calculate_missing_coals"] += 1
            return self._inner.calculate_missing_coals()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def _wrapped_loader(path, *args, **kwargs):
        calls["load_up_edges"].append(bool(kwargs.get("load_up_edges", False)))
        return _Proxy(real_loader(path, *args, **kwargs))

    monkeypatch.setattr(grg_module.pygrgl, "load_immutable_grg", _wrapped_loader)

    artifact = convert(primary_grg_path, tmp_path)
    assert artifact.exists()
    assert calls["load_up_edges"] == [False]
    assert calls["calculate_missing_coals"] == 0
