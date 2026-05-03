from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pytest

from pygrgl_spmv import ReferenceRuntime
from pygrgl_spmv.backends.base import split_selector_by_level
import pygrgl_spmv.grg.artifact as artifact_module
from pygrgl_spmv.grg.artifact import (
    GRG_SPMV_FORMAT_MAGIC,
    GRG_SPMV_FORMAT_VERSION,
    iter_artifact_blocks,
    load_grg_spmv,
    save_grg_spmv,
    scan_grg_spmv,
)
from pygrgl_spmv.tests.runtime._runtime_builders import build_reference_layout
from pygrgl_spmv.tests.runtime._streaming_cases import write_three_level_band_artifact


def test_scan_matches_loaded_state(primary_artifact):
    scan = scan_grg_spmv(primary_artifact)
    state = load_grg_spmv(primary_artifact, np.float64)

    assert scan.path == primary_artifact
    assert scan.num_samples == state.num_samples
    assert scan.num_mutations == state.num_mutations
    assert scan.num_nodes == state.num_nodes
    assert scan.num_individuals == state.num_individuals
    assert scan.num_edges == state.num_edges
    assert scan.ploidy == state.ploidy
    assert scan.has_missing_data == state.has_missing_data
    assert scan.has_xtx_bias == bool(state.init_xtx_up_bias is not None and state.init_xtx_down_bias is not None)
    np.testing.assert_array_equal(scan.level_offsets, state.level_offsets)
    assert scan.num_levels == len(state.level_offsets) - 1
    assert scan.has_individual_coals == (state.coalescence_counts is not None)
    assert scan.node_perm_dtype == state.node_perm.dtype
    assert scan.sample_to_individual_dtype == state.sample_to_individual.dtype
    assert scan.selector_nnz_by_level.shape == (scan.num_levels, 2)
    assert int(scan.selector_nnz_by_level[:, 0].sum()) == scan.selector_mut_nnz
    assert int(scan.selector_nnz_by_level[:, 1].sum()) == scan.selector_miss_nnz
    mut_pairs = split_selector_by_level(state.sel_mut, state.level_offsets)
    miss_pairs = split_selector_by_level(state.sel_miss, state.level_offsets)
    for level, (rows, _) in enumerate(mut_pairs):
        assert int(scan.selector_nnz_by_level[level, 0]) == rows.size
    for level, (rows, _) in enumerate(miss_pairs):
        assert int(scan.selector_nnz_by_level[level, 1]) == rows.size

    loaded_blocks = {(dst_level, src_level): block for dst_level, row in enumerate(state.A_blocks or []) for src_level, block in enumerate(row)}
    assert len(scan.blocks) == len(loaded_blocks)
    for block in scan.blocks:
        loaded = loaded_blocks[(block.dst_level, block.src_level)]
        assert block.shape == loaded.shape
        assert block.nnz == loaded.nnz
        assert block.indices_dtype == loaded.indices.dtype
        assert block.indptr_dtype == loaded.indptr.dtype


def test_iter_artifact_blocks_matches_loaded_blocks(primary_artifact):
    state = load_grg_spmv(primary_artifact, np.float64)
    loaded = {(dst_level, src_level): block for dst_level, row in enumerate(state.A_blocks or []) for src_level, block in enumerate(row)}
    blocks = list(iter_artifact_blocks(primary_artifact))
    assert len(blocks) == len(loaded)
    for block in blocks:
        ref = loaded[(block.dst_level, block.src_level)]
        assert block.shape == ref.shape
        np.testing.assert_array_equal(block.indices, ref.indices)
        np.testing.assert_array_equal(block.indptr, ref.indptr)


def test_load_grg_spmv_rehydrates_shared_nonempty_block_data(primary_artifact):
    state = load_grg_spmv(primary_artifact, np.float64)
    assert state.A_blocks is not None
    data_arrays = [block.data for row in state.A_blocks for block in row if block.nnz > 0]
    assert data_arrays
    first = data_arrays[0]
    assert first.strides == (0,)
    assert not first.flags.writeable
    for data in data_arrays[1:]:
        assert data.strides == (0,)
        assert not data.flags.writeable
        assert np.shares_memory(data, first)


def test_runtime_consumes_artifact_without_grg_loader(primary_artifact, monkeypatch):
    import pygrgl_spmv.grg as grg_module

    def _forbidden_loader(*_args, **_kwargs):
        raise AssertionError("artifact-only runtime flow must not call pygrgl.load_immutable_grg")

    monkeypatch.setattr(grg_module.pygrgl, "load_immutable_grg", _forbidden_loader)

    layout = build_reference_layout([primary_artifact])
    with ReferenceRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        assert grg.artifact_path == primary_artifact
        assert grg.num_samples > 0


def test_bound_operator_round_trips_artifact_metadata(primary_artifact, primary_grg):
    layout = build_reference_layout([primary_artifact])
    with ReferenceRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        assert grg.shape == (primary_grg.num_samples, primary_grg.num_mutations)
        assert grg.num_individuals == primary_grg.num_individuals
        assert grg.num_nodes == primary_grg.num_nodes
        assert grg.num_edges == primary_grg.num_edges
        assert grg.has_missing_data == primary_grg.has_missing_data
        np.testing.assert_array_equal(grg.sample_to_individual, np.arange(grg.num_samples, dtype=grg.sample_to_individual.dtype) // grg.ploidy)

        for mutation_id in (0, grg.num_mutations // 2, grg.num_mutations - 1):
            left = primary_grg.get_mutation_by_id(int(mutation_id))
            right = grg.get_mutation_by_id(int(mutation_id))
            assert right.position == left.position
            assert right.time == left.time
            assert right.allele == left.allele
            assert right.ref_allele == left.ref_allele


def test_save_grg_spmv_is_atomic_on_failure(primary_artifact, tmp_path, monkeypatch):
    state = load_grg_spmv(primary_artifact, np.float64)
    target = Path(tmp_path) / primary_artifact.name
    target.write_bytes(Path(primary_artifact).read_bytes())
    before = target.read_bytes()

    def _broken_savez(handle, *args, **kwargs):
        handle.write(b"partial-artifact")
        handle.flush()
        raise RuntimeError("save failed")

    monkeypatch.setattr(np, "savez", _broken_savez)

    with pytest.raises(RuntimeError, match="save failed"):
        save_grg_spmv(state, target)

    assert target.read_bytes() == before
    scan = scan_grg_spmv(target)
    loaded = load_grg_spmv(target, np.float64)
    assert scan.num_nodes == loaded.num_nodes
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_saved_artifact_is_uncompressed_and_has_direct_scan_metadata(primary_artifact):
    with zipfile.ZipFile(primary_artifact) as archive:
        infos = archive.infolist()
        assert infos
        assert {info.compress_type for info in infos} == {zipfile.ZIP_STORED}
        names = set(archive.namelist())

    with np.load(primary_artifact, allow_pickle=False) as data:
        for key in (
            "scan_block_levels",
            "scan_block_shapes",
            "scan_block_nnzs",
            "scan_block_struct_itemsize",
            "scan_selector_nnzs",
            "scan_selector_nnz_by_level",
            "scan_metadata_struct_itemsize",
        ):
            assert key in data.files
        assert np.asarray(data["scan_block_levels"]).ndim == 2
        assert np.asarray(data["scan_block_levels"]).shape[1] == 2
        assert np.asarray(data["scan_block_shapes"]).shape[1] == 2
        assert np.asarray(data["scan_block_struct_itemsize"]).shape[1] == 2
        assert np.asarray(data["scan_selector_nnz_by_level"]).shape[1] == 2
        assert np.asarray(data["scan_metadata_struct_itemsize"]).shape == (2,)
    assert not any(name.startswith("A_blocks_") and name.endswith("_shape.npy") for name in names)
    assert not any(name.startswith("A_blocks_") and name.endswith("_nnz.npy") for name in names)
    assert not any(name.startswith("A_blocks_") and name.endswith("_indices_dtype.npy") for name in names)
    assert not any(name.startswith("A_blocks_") and name.endswith("_indptr_dtype.npy") for name in names)
    assert "sel_mut_nnz.npy" not in names
    assert "sel_miss_nnz.npy" not in names
    assert "sel_mut_nnz_by_level.npy" not in names
    assert "sel_miss_nnz_by_level.npy" not in names
    assert "init_vector_up_bias_size.npy" not in names
    assert "init_vector_down_bias_size.npy" not in names


def test_scan_uses_direct_metadata_without_loading_block_or_selector_arrays(tmp_path, monkeypatch):
    artifact = write_three_level_band_artifact(tmp_path, "direct-scan", n=4, bandwidth=1)
    real_load_struct_array = artifact_module._load_struct_array

    def _guarded_load_struct_array(data, key: str, *, non_negative: bool = True):
        if key != "level_offsets":
            raise AssertionError(f"scan loaded structural array {key}")
        return real_load_struct_array(data, key, non_negative=non_negative)

    artifact_module._scan_grg_spmv_cached.cache_clear()
    monkeypatch.setattr(artifact_module, "_load_struct_array", _guarded_load_struct_array)

    scan = scan_grg_spmv(artifact)

    assert scan.num_levels == 3
    assert scan.selector_mut_nnz == 4
    assert scan.selector_miss_nnz == 0
    assert [(block.dst_level, block.src_level, block.nnz) for block in scan.blocks] == [
        (1, 0, 4),
        (2, 0, 4),
        (2, 1, 4),
    ]


@pytest.mark.parametrize("version", [5, 6])
def test_old_artifacts_are_rejected(tmp_path, version):
    artifact = tmp_path / "old.grg_spmv"
    with artifact.open("wb") as handle:
        np.savez(
            handle,
            grg_spmv_magic=np.asarray(GRG_SPMV_FORMAT_MAGIC),
            grg_spmv_format_version=np.asarray(version, dtype=np.int32),
        )

    with pytest.raises(ValueError, match=f"got version {version}, expected {GRG_SPMV_FORMAT_VERSION}"):
        scan_grg_spmv(artifact)


def test_load_grg_spmv_warns_when_stored_struct_arrays_are_wider_than_loaded(tmp_path):
    artifact = write_three_level_band_artifact(tmp_path, "wide-storage", n=4, bandwidth=1, struct_dtype=np.int64)
    scan = scan_grg_spmv(artifact)
    assert all(block.indices_dtype == np.dtype(np.int64) and block.indptr_dtype == np.dtype(np.int64) for block in scan.blocks)
    with pytest.warns(RuntimeWarning, match="wider than necessary|more disk space"):
        state = load_grg_spmv(artifact, np.float64)
    assert state.A_blocks is not None
    for row in state.A_blocks:
        for block in row:
            assert block.indices.dtype == np.dtype(np.int32)
            assert block.indptr.dtype == np.dtype(np.int32)
