"""Tests for the load() convenience function."""

from __future__ import annotations

import importlib.resources as resources
import json
import logging
from pathlib import Path
import warnings

import pytest

import pygrgl_spmv
from pygrgl_spmv import SpmvGRG, load
from pygrgl_spmv.tests.conftest import HAS_MKL_RUNTIME, HAS_TRITON_RUNTIME

MKL_ONLY = pytest.mark.skipif(not HAS_MKL_RUNTIME, reason="MKL runtime unavailable (libmkl_rt.so not found)")
TRITON_ONLY = pytest.mark.skipif(not HAS_TRITON_RUNTIME, reason="Triton runtime unavailable (torch+triton CUDA not found)")


def _write_config(tmp_path: Path, config: object) -> Path:
    path = tmp_path / "spmv.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _sample_resource(name: str):
    return resources.files("pygrgl_spmv.configs").joinpath(name)


def _sample_config(name: str) -> dict[str, object]:
    return json.loads(_sample_resource(name).read_text(encoding="utf-8"))


def _require_cupy_runtime():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        pytest.skip(f"CuPy runtime unavailable: {exc}")


def test_load_requires_explicit_env_var(monkeypatch, primary_grg_path):
    monkeypatch.delenv("PYGRGL_SPMV_CONFIG", raising=False)
    with pytest.raises(ValueError, match="PYGRGL_SPMV_CONFIG"):
        load(primary_grg_path)

    monkeypatch.setenv("PYGRGL_SPMV_CONFIG", "   ")
    with pytest.raises(ValueError, match="PYGRGL_SPMV_CONFIG"):
        load(primary_grg_path)


@pytest.mark.parametrize(
    ("sample_name", "backend_name"),
    [
        pytest.param("reference-default.json", "reference", id="reference"),
        pytest.param("mkl-default.json", "mkl", id="mkl"),
        pytest.param("cusparse-default.json", "cusparse", id="cusparse"),
        pytest.param("triton-default.json", "triton", id="triton"),
    ],
)
def test_parse_shipped_sample_configs(sample_name, backend_name):
    selected, section, plan_up, plan_down = pygrgl_spmv._parse_backend_config(_sample_config(sample_name))
    assert selected == backend_name
    assert section["log_level"] == "WARNING"
    assert plan_up is not None
    assert plan_down is not None


def test_parse_backend_config_ignores_unselected_sections():
    config = {
        "backend": "reference",
        "reference": {
            "log_level": "WARNING",
            "up": {"store": "N", "fmt": "CSR", "k_hint": None},
            "down": {"store": "T", "fmt": "CSC", "k_hint": None},
        },
        "mkl": 123,
    }
    selected, _, plan_up, plan_down = pygrgl_spmv._parse_backend_config(config)
    assert selected == "reference"
    assert plan_up is not None
    assert plan_down is not None


@pytest.mark.parametrize(
    ("config", "match"),
    [
        pytest.param([], "config must be a JSON object", id="root-not-object"),
        pytest.param({}, "missing required field", id="missing-backend"),
        pytest.param({"backend": "unknown"}, "must be one of", id="unknown-backend"),
        pytest.param({"backend": "reference"}, "missing required 'reference' section", id="missing-section"),
        pytest.param({"backend": "reference", "reference": 1}, "reference config must be a JSON object", id="section-not-object"),
        pytest.param(
            {"backend": "reference", "reference": {"log_level": "WARNING", "up": None}},
            "missing required field",
            id="missing-section-key",
        ),
        pytest.param(
            {
                "backend": "reference",
                "reference": {
                    "log_level": "WARNING",
                    "up": None,
                    "down": {"store": "T", "fmt": "CSC", "k_hint": None},
                    "extra": True,
                },
            },
            "unknown field",
            id="unknown-section-key",
        ),
        pytest.param(
            {
                "backend": "reference",
                "reference": {
                    "log_level": "WARNING",
                    "up": 1,
                    "down": {"store": "T", "fmt": "CSC", "k_hint": None},
                },
            },
            "reference.up must be a JSON object",
            id="bad-up-type",
        ),
        pytest.param(
            {
                "backend": "reference",
                "reference": {
                    "log_level": "WARNING",
                    "up": None,
                    "down": None,
                },
            },
            "at least one of up/down",
            id="both-disabled",
        ),
        pytest.param(
            {
                "backend": "reference",
                "reference": {
                    "log_level": "WARNING",
                    "up": {"store": "N", "fmt": "CSR"},
                    "down": None,
                },
            },
            "missing required field",
            id="missing-plan-field",
        ),
        pytest.param(
            {
                "backend": "reference",
                "reference": {
                    "log_level": "WARNING",
                    "up": {"store": "N", "fmt": "CSR", "k_hint": None, "algo": "DEFAULT"},
                    "down": None,
                },
            },
            "unknown field",
            id="unknown-plan-field",
        ),
        pytest.param(
            {
                "backend": "cusparse",
                "cusparse": {
                    "device": 0,
                    "stream": 0,
                    "log_level": "WARNING",
                    "up": {
                        "k_hint": None,
                        "store": "N",
                        "fmt": "CSR",
                        "opA": "N",
                        "opB": "N",
                        "orderB": "ROW",
                        "orderC": "ROW",
                        "algo": "DEFAULT",
                        "scratch": "none",
                    },
                    "down": None,
                },
            },
            "missing required field",
            id="missing-gpu-section-field",
        ),
        pytest.param(
            {
                "backend": "triton",
                "triton": {
                    "device": 0,
                    "stream": 0,
                    "ring_buffer_size": 2,
                    "log_level": "WARNING",
                    "up": {"k_hint": 1, "store": "N", "fmt": "CSR"},
                    "down": None,
                },
            },
            "missing required field",
            id="missing-triton-scratch",
        ),
    ],
)
def test_parse_backend_config_rejects_malformed_inputs(config, match):
    with pytest.raises(ValueError, match=match):
        pygrgl_spmv._parse_backend_config(config)


def test_load_reference_sample_logs_info_without_warning(monkeypatch, primary_grg_path, tmp_path, caplog):
    resource = _sample_resource("reference-default.json")
    with resources.as_file(resource) as path:
        monkeypatch.setenv("PYGRGL_SPMV_CONFIG", str(path))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with caplog.at_level(logging.INFO, logger="pygrgl_spmv"):
                op = load(primary_grg_path, artifact_dir=tmp_path / "artifacts")
    assert isinstance(op, SpmvGRG)
    messages = [rec.getMessage() for rec in caplog.records if rec.name == "pygrgl_spmv"]
    assert any(msg.startswith("Selected reference backend from ") for msg in messages)


def test_load_accepts_reference_one_sided_config(monkeypatch, primary_grg_path, tmp_path):
    config = {
        "backend": "reference",
        "reference": {
            "log_level": "WARNING",
            "up": {"store": "N", "fmt": "CSR", "k_hint": None},
            "down": None,
        },
    }
    monkeypatch.setenv("PYGRGL_SPMV_CONFIG", str(_write_config(tmp_path, config)))
    op = load(primary_grg_path, artifact_dir=tmp_path / "artifacts")
    assert isinstance(op, SpmvGRG)
    assert op._backend._plan_up is not None
    assert op._backend._plan_down is None


@MKL_ONLY
@pytest.mark.mkl
def test_load_from_mkl_sample(monkeypatch, primary_grg_path, tmp_path):
    resource = _sample_resource("mkl-default.json")
    with resources.as_file(resource) as path:
        monkeypatch.setenv("PYGRGL_SPMV_CONFIG", str(path))
        op = load(primary_grg_path, artifact_dir=tmp_path / "artifacts")
    assert isinstance(op, SpmvGRG)
    assert op.num_samples > 0


@pytest.mark.gpu
@pytest.mark.cusparse
def test_load_from_cusparse_sample(monkeypatch, primary_grg_path, tmp_path):
    _require_cupy_runtime()
    resource = _sample_resource("cusparse-default.json")
    with resources.as_file(resource) as path:
        monkeypatch.setenv("PYGRGL_SPMV_CONFIG", str(path))
        op = load(primary_grg_path, artifact_dir=tmp_path / "artifacts")
    assert isinstance(op, SpmvGRG)
    assert op.num_samples > 0


@TRITON_ONLY
@pytest.mark.gpu
@pytest.mark.triton
def test_load_from_triton_sample(monkeypatch, primary_grg_path, tmp_path):
    resource = _sample_resource("triton-default.json")
    with resources.as_file(resource) as path:
        monkeypatch.setenv("PYGRGL_SPMV_CONFIG", str(path))
        op = load(primary_grg_path, artifact_dir=tmp_path / "artifacts")
    assert isinstance(op, SpmvGRG)
    assert op.num_samples > 0


def test_load_missing_config_file_raises(monkeypatch, primary_grg_path, tmp_path):
    monkeypatch.setenv("PYGRGL_SPMV_CONFIG", str(tmp_path / "nonexistent.json"))
    with pytest.raises((FileNotFoundError, OSError)):
        load(primary_grg_path)


def test_load_malformed_json_raises(monkeypatch, primary_grg_path, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    monkeypatch.setenv("PYGRGL_SPMV_CONFIG", str(path))
    with pytest.raises(json.JSONDecodeError):
        load(primary_grg_path)
