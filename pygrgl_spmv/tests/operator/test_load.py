"""Tests for the load() convenience function."""

from __future__ import annotations

import json

import pytest

from pygrgl_spmv import SpmvGRG, load
from pygrgl_spmv.tests.conftest import HAS_MKL_RUNTIME

MKL_ONLY = pytest.mark.skipif(not HAS_MKL_RUNTIME, reason="MKL runtime unavailable (libmkl_rt.so not found)")

_MKL_CONFIG = {
    "backend": "mkl",
    "mkl": {
        "n_threads": 0,
        "log_level": "WARNING",
        "up": {"fmt": "csr", "k_hint": None},
        "down": {"fmt": "csc", "k_hint": None},
    },
}


# ---------------------------------------------------------------------------
# Env var absent — should warn and use default backend
# ---------------------------------------------------------------------------


def test_load_warns_when_env_unset(monkeypatch, primary_grg_path):
    monkeypatch.delenv("PYGRGL_SPMV_CONFIG", raising=False)
    with pytest.warns(UserWarning, match="PYGRGL_SPMV_CONFIG"):
        op = load(primary_grg_path)
    assert isinstance(op, SpmvGRG)
    assert op.num_samples > 0


# ---------------------------------------------------------------------------
# Loading from a JSON config file
# ---------------------------------------------------------------------------


@MKL_ONLY
def test_load_from_mkl_config(monkeypatch, primary_grg_path, tmp_path):
    cfg = tmp_path / "spmv.json"
    cfg.write_text(json.dumps(_MKL_CONFIG))
    monkeypatch.setenv("PYGRGL_SPMV_CONFIG", str(cfg))
    op = load(primary_grg_path)
    assert isinstance(op, SpmvGRG)
    assert op.num_samples > 0


@MKL_ONLY
def test_load_saves_artifact(monkeypatch, primary_grg_path, tmp_path):
    cfg = tmp_path / "spmv.json"
    cfg.write_text(json.dumps(_MKL_CONFIG))
    monkeypatch.setenv("PYGRGL_SPMV_CONFIG", str(cfg))
    artifact_dir = tmp_path / "artifacts"
    op = load(primary_grg_path, artifact_dir=artifact_dir)
    assert op.artifact_path is not None
    assert op.artifact_path.exists()


@MKL_ONLY
def test_load_both_backends_in_config(monkeypatch, primary_grg_path, tmp_path):
    """Both backend sections can coexist; 'backend' key selects the active one."""
    cfg = tmp_path / "spmv.json"
    config = {
        "backend": "mkl",
        "mkl": {"up": {"fmt": "csr"}, "down": {"fmt": "csc"}},
        "cusparse": {"device": 0, "up": {"fmt": "csr"}, "down": {"fmt": "csc"}},
    }
    cfg.write_text(json.dumps(config))
    monkeypatch.setenv("PYGRGL_SPMV_CONFIG", str(cfg))
    op = load(primary_grg_path)
    assert isinstance(op, SpmvGRG)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_load_unknown_backend_raises(monkeypatch, primary_grg_path, tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text(json.dumps({"backend": "unknown"}))
    monkeypatch.setenv("PYGRGL_SPMV_CONFIG", str(cfg))
    with pytest.raises(ValueError, match="unknown"):
        load(primary_grg_path)


def test_load_missing_config_file_raises(monkeypatch, primary_grg_path, tmp_path):
    monkeypatch.setenv("PYGRGL_SPMV_CONFIG", str(tmp_path / "nonexistent.json"))
    with pytest.raises((FileNotFoundError, OSError)):
        load(primary_grg_path)
