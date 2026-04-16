from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pygrgl
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
import pygrgl_spmv.backends.triton.backend as triton_backend_mod
from pygrgl_spmv.backends.reference import ReferenceRuntime
from pygrgl_spmv.backends.triton import TritonPlan, TritonPlanPair, TritonRuntime
from pygrgl_spmv.backends.triton.backend import _block_plan
from pygrgl_spmv.backends.types import Direction
from pygrgl_spmv.grg.artifact import scan_grg_spmv
from pygrgl_spmv.tests.conftest import DATA_DTYPE, tol
from pygrgl_spmv.tests.runtime._runtime_builders import (
    build_reference_layout,
    build_triton_layout,
    full_requirements,
)
from pygrgl_spmv.tests.runtime._streaming_cases import (
    THREE_BLOCK_EXACTNESS_MODES,
    THREE_BLOCK_TRANSITION_MODES,
    ThreeBlockMode,
    _equal_block_budget_components,
    clear_torch_state,
    expected_down,
    expected_up,
    prepare_triton_stream_stress_case,
    three_block_mode_budget_bytes,
    triton_ring_thresholds,
    write_overlap_band_artifact,
    write_three_level_band_artifact,
)

pytestmark = [pytest.mark.gpu, pytest.mark.triton]

_MODE_BY_NAME = {mode.name: mode for mode in THREE_BLOCK_TRANSITION_MODES}


def _requirements(runtime_k: int):
    return full_requirements(max_k_up=int(runtime_k), max_k_down=int(runtime_k))


def _assert_triton_reset(runtime: TritonRuntime) -> None:
    assert runtime._caller_stream is None
    assert runtime._root_stream is None
    assert runtime._caller_to_root_event is None
    assert runtime._root_to_caller_event is None
    assert runtime._level_streams == []
    assert runtime._slot_copy_streams == []
    assert runtime._scratch_streams_up == []
    assert runtime._scratch_streams_down == []
    assert runtime._slot_indices == []
    assert runtime._slot_indptr == []
    assert runtime._up_state is None
    assert runtime._down_state is None
    assert runtime._up_scratch == []
    assert runtime._down_scratch == []
    assert runtime._artifacts == ()
    assert runtime._grgs == ()
    assert runtime._config_up is None
    assert runtime._config_down is None
    assert runtime._staging_up == {}
    assert runtime._staging_down == {}
    assert runtime.stream is None
    assert not runtime._entered
    assert not runtime._active_call


def _owner_keys(layout, *, resident: bool) -> tuple[tuple[int, int], ...]:
    keys = [
        (int(block.dst_level), int(block.src_level))
        for artifact in layout.artifacts
        for block in (*artifact.blocks_up, *artifact.blocks_down)
        if bool(block.resident) == bool(resident)
    ]
    return tuple(sorted(keys))


def _nonzero_slot_count(layout) -> int:
    return sum(1 for slot in layout.slot_plans if slot.nbytes > 0)


def _build_layout(artifact, *, runtime_k: int, ring_buffer_size: int, budget_bytes: int):
    return build_triton_layout(
        [artifact],
        requirements=_requirements(runtime_k),
        ring_buffer_size=int(ring_buffer_size),
        vram_budget_bytes=int(budget_bytes),
    )


def _full_budget_components(artifact, *, runtime_k: int) -> tuple[int, int, int]:
    layout = build_triton_layout(
        [artifact],
        requirements=_requirements(runtime_k),
        ring_buffer_size=0,
        vram_budget_bytes=1_000_000_000_000,
    )
    return _equal_block_budget_components(layout)


def _assert_budget_accounting(layout, *, chosen_budget: int, fixed_bytes: int, block_bytes: int) -> None:
    assert sum(item.nbytes for item in layout.budget_items) == layout.bytes_total
    assert layout.required_budget_for_full_residency == int(fixed_bytes + 3 * block_bytes)
    assert layout.bytes_total <= int(chosen_budget) <= layout.required_budget_for_full_residency


def _assert_three_block_mode(layout, mode: ThreeBlockMode, *, chosen_budget: int, fixed_bytes: int, block_bytes: int) -> None:
    assert layout.requested_ring_buffer_size == int(mode.requested_ring_buffer_size)
    assert layout.allocated_ring_buffer_size == int(mode.slot_count)
    assert _owner_keys(layout, resident=True) == tuple(sorted(mode.resident_blocks))
    assert _owner_keys(layout, resident=False) == tuple(sorted(mode.streamed_blocks))
    assert _nonzero_slot_count(layout) == int(mode.slot_count)
    resident_items = [item for item in layout.budget_items if item.kind == "resident_sparse"]
    slot_items = [item for item in layout.budget_items if item.kind == "ring_slot"]
    assert len(resident_items) == len(mode.resident_blocks)
    assert len(slot_items) == int(mode.slot_count)
    _assert_budget_accounting(layout, chosen_budget=chosen_budget, fixed_bytes=fixed_bytes, block_bytes=block_bytes)


def _below_transition(mode: ThreeBlockMode) -> ThreeBlockMode | None:
    mapping = {
        "ring0-resident": None,
        "ring1-stream": None,
        "ring1-hybrid": _MODE_BY_NAME["ring1-stream"],
        "ring1-resident": _MODE_BY_NAME["ring1-hybrid"],
        "ring2-stream": None,
        "ring2-resident": _MODE_BY_NAME["ring2-stream"],
        "ring3-resident": None,
    }
    return mapping[mode.name]


def _build_with_mode_warning(artifact, *, runtime_k: int, mode: ThreeBlockMode, fixed_bytes: int, block_bytes: int):
    budget_bytes = three_block_mode_budget_bytes(mode, fixed_bytes=fixed_bytes, block_bytes=block_bytes)
    if mode.expect_warning:
        with pytest.warns(RuntimeWarning, match="requested ring_buffer_size"):
            layout = _build_layout(
                artifact,
                runtime_k=runtime_k,
                ring_buffer_size=mode.requested_ring_buffer_size,
                budget_bytes=budget_bytes,
            )
    else:
        layout = _build_layout(
            artifact,
            runtime_k=runtime_k,
            ring_buffer_size=mode.requested_ring_buffer_size,
            budget_bytes=budget_bytes,
        )
    return layout, budget_bytes


def _assert_planner_transition(artifact, *, runtime_k: int, mode: ThreeBlockMode) -> None:
    fixed_bytes, block_bytes, owner_block_count = _full_budget_components(artifact, runtime_k=runtime_k)
    assert owner_block_count == 3
    layout, budget_bytes = _build_with_mode_warning(
        artifact,
        runtime_k=runtime_k,
        mode=mode,
        fixed_bytes=fixed_bytes,
        block_bytes=block_bytes,
    )
    _assert_three_block_mode(layout, mode, chosen_budget=budget_bytes, fixed_bytes=fixed_bytes, block_bytes=block_bytes)

    predecessor = _below_transition(mode)
    lower_budget = int(budget_bytes - 1)
    if predecessor is None:
        with pytest.raises(ValueError):
            _build_layout(
                artifact,
                runtime_k=runtime_k,
                ring_buffer_size=mode.requested_ring_buffer_size,
                budget_bytes=lower_budget,
            )
        return
    if predecessor.expect_warning:
        with pytest.warns(RuntimeWarning, match="requested ring_buffer_size"):
            lower_layout = _build_layout(
                artifact,
                runtime_k=runtime_k,
                ring_buffer_size=mode.requested_ring_buffer_size,
                budget_bytes=lower_budget,
            )
    else:
        lower_layout = _build_layout(
            artifact,
            runtime_k=runtime_k,
            ring_buffer_size=mode.requested_ring_buffer_size,
            budget_bytes=lower_budget,
        )
    _assert_three_block_mode(lower_layout, predecessor, chosen_budget=lower_budget, fixed_bytes=fixed_bytes, block_bytes=block_bytes)


def _run_exactness_case(artifact, case, *, runtime_k: int, mode: ThreeBlockMode, order) -> None:
    fixed_bytes, block_bytes, owner_block_count = _full_budget_components(artifact, runtime_k=runtime_k)
    assert owner_block_count == 3
    layout, budget_bytes = _build_with_mode_warning(
        artifact,
        runtime_k=runtime_k,
        mode=mode,
        fixed_bytes=fixed_bytes,
        block_bytes=block_bytes,
    )
    _assert_three_block_mode(layout, mode, chosen_budget=budget_bytes, fixed_bytes=fixed_bytes, block_bytes=block_bytes)
    with TritonRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        for run_idx, direction in enumerate(order):
            seed = 61_000 + 1_000 * runtime_k + 100 * run_idx + sum(ord(ch) for ch in mode.name)
            rng = np.random.default_rng(seed)
            primary = rng.choice(np.array([-1.0, 1.0], dtype=DATA_DTYPE), size=(int(runtime_k), case.n))
            if direction == "up":
                expected = expected_up(primary.T, shifts=case.shifts, bandwidth=case.bandwidth).T
            else:
                expected = expected_down(primary.T, shifts=case.shifts, bandwidth=case.bandwidth).T
            actual = grg.matmul(primary, direction)
            np.testing.assert_array_equal(actual, expected)


def _run_overlap_case(tmp_path, order, requested_ring_buffer_size, monkeypatch, *, n: int, bandwidth: int) -> None:
    clear_torch_state()
    artifact = write_overlap_band_artifact(tmp_path, f"triton-overlap-{n}-{bandwidth}", n=n, bandwidth=bandwidth)
    budget_bytes = triton_ring_thresholds(
        artifact,
        requirements=_requirements(1),
        total_vram_bytes=prepare_triton_stream_stress_case().total_vram_bytes,
        max_ring_buffer_size=2,
    )[int(requested_ring_buffer_size) - 1]
    layout = _build_layout(
        artifact,
        runtime_k=1,
        ring_buffer_size=requested_ring_buffer_size,
        budget_bytes=budget_bytes,
    )
    ref_layout = build_reference_layout([artifact], requirements=_requirements(1))
    state = {"active": True, "compute": [], "copy": []}
    original_launch = triton_backend_mod.launch_block

    def _wrapped_launch(*, block, x, y, config, fp64_acc):
        if state["active"]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(device=x.device)
            start.record(stream)
            original_launch(block=block, x=x, y=y, config=config, fp64_acc=fp64_acc)
            torch.cuda._sleep(10_000_000)
            end.record(stream)
            state["compute"].append((start, end))
            return
        return original_launch(block=block, x=x, y=y, config=config, fp64_acc=fp64_acc)

    def _wrapped_copy(self, copy_done, compute_done, dst_level: int, op_idx: int, op) -> None:
        assert op.slot is not None
        assert op.indices_host is not None and op.indptr_host is not None
        stream = self._slot_copy_streams[op.slot]
        with torch.cuda.stream(stream):
            prev_event = self._prev_compute_event(compute_done, op)
            if prev_event is not None:
                stream.wait_event(prev_event)
            start = None
            end = None
            if state["active"]:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
            self._slot_indices[op.slot][: op.indices_host.numel()].copy_(op.indices_host, non_blocking=True)
            self._slot_indptr[op.slot][: op.indptr_host.numel()].copy_(op.indptr_host, non_blocking=True)
            if state["active"]:
                assert start is not None and end is not None
                end.record(stream)
                state["copy"].append((start, end))
            copy_done[dst_level][op_idx].record(stream)

    monkeypatch.setattr(triton_backend_mod, "launch_block", _wrapped_launch)
    monkeypatch.setattr(TritonRuntime, "_copy_host_block_to_slot", _wrapped_copy)

    def _has_overlap() -> bool:
        for compute_start, compute_end in state["compute"]:
            if compute_start.elapsed_time(compute_end) <= 0.0:
                continue
            for copy_start, copy_end in state["copy"]:
                if copy_start.elapsed_time(copy_end) <= 0.0:
                    continue
                if copy_start.elapsed_time(compute_end) > 0.0 and compute_start.elapsed_time(copy_end) > 0.0:
                    return True
        return False

    with ReferenceRuntime(ref_layout) as ref_runtime, TritonRuntime(layout) as runtime:
        (ref_grg,) = ref_runtime.grgs
        (grg,) = runtime.grgs
        for run_idx, direction in enumerate(order):
            size = ref_grg.num_samples if direction == "up" else ref_grg.num_mutations
            rng = np.random.default_rng(71_000 + 1_000 * requested_ring_buffer_size + 10 * run_idx + (0 if direction == "up" else 1))
            primary = rng.choice(np.array([-1.0, 1.0], dtype=DATA_DTYPE), size=(1, size))
            expected = ref_grg.matmul(primary, direction)
            actual = grg.matmul(primary, direction)
            np.testing.assert_array_equal(actual, expected)
            torch.cuda.synchronize(runtime.device)
            if run_idx == 0:
                if int(requested_ring_buffer_size) == 1:
                    assert not _has_overlap()
                else:
                    assert _has_overlap()
                state["active"] = False
    clear_torch_state()


@pytest.fixture(scope="module")
def triton_stream_stress_case():
    return prepare_triton_stream_stress_case()


@pytest.fixture(scope="module")
def gpu_small_stream_case():
    return SimpleNamespace(n=64, bandwidth=8, shifts=(0, 8, 16))


@pytest.fixture(scope="module")
def triton_small_stream_artifact(artifact_cache_dir, gpu_small_stream_case):
    return write_three_level_band_artifact(
        artifact_cache_dir,
        f"gpu-small-stream-n{gpu_small_stream_case.n}",
        n=gpu_small_stream_case.n,
        bandwidth=gpu_small_stream_case.bandwidth,
        shifts=gpu_small_stream_case.shifts,
    )


@pytest.fixture(scope="module")
def triton_stream_stress_artifact(artifact_cache_dir, triton_stream_stress_case):
    return write_three_level_band_artifact(
        artifact_cache_dir,
        f"gpu-stream-stress-n{triton_stream_stress_case.n}",
        n=triton_stream_stress_case.n,
        bandwidth=triton_stream_stress_case.bandwidth,
        shifts=triton_stream_stress_case.shifts,
    )


def test_triton_runtime_exposes_requested_stream_and_stream_ptr(primary_artifact):
    with torch.cuda.device(0):
        requested = torch.cuda.Stream()
    layout = build_triton_layout([primary_artifact], stream=requested)
    with TritonRuntime(layout) as runtime:
        assert runtime.device.type == "cuda"
        assert runtime.stream is not None
        assert runtime.stream_ptr == layout.stream_ptr == int(requested.cuda_stream)
        assert int(runtime.stream.cuda_stream) == int(requested.cuda_stream)
        assert len(runtime.grgs) == 1


def test_triton_runtime_enter_failure_releases_owned_state(primary_artifact, monkeypatch):
    layout = build_triton_layout([primary_artifact], requirements=_requirements(1))
    runtime = TritonRuntime(layout)

    def _boom(self, direction):
        raise RuntimeError(f"boom-{direction.value}")

    monkeypatch.setattr(TritonRuntime, "_tune_direction", _boom)
    with pytest.raises(RuntimeError, match="boom-up"):
        runtime.__enter__()
    _assert_triton_reset(runtime)


def test_triton_caller_root_scope_uses_explicit_handoff_streams(primary_artifact, monkeypatch):
    with torch.cuda.device(0):
        requested = torch.cuda.Stream()
    layout = build_triton_layout([primary_artifact], stream=requested, requirements=_requirements(1))
    runtime = TritonRuntime(layout)
    with torch.cuda.device(runtime.device):
        runtime._caller_stream = torch.cuda.Stream()
        runtime._root_stream = torch.cuda.Stream()
        runtime._caller_to_root_event = torch.cuda.Event()
        runtime._root_to_caller_event = torch.cuda.Event()

    log: list[tuple[object, ...]] = []
    original_stream_ctx = torch.cuda.stream

    @contextmanager
    def _wrapped_stream_ctx(stream):
        ptr = int(stream.cuda_stream)
        log.append(("enter_ctx_for", ptr))
        with original_stream_ctx(stream):
            yield
        log.append(("exit_ctx_for", ptr))

    monkeypatch.setattr(torch.cuda, "stream", _wrapped_stream_ctx)
    ambient = torch.cuda.Stream(device=runtime.device)
    caller_ptr = int(runtime._caller_stream.cuda_stream)
    root_ptr = int(runtime._root_stream.cuda_stream)
    with torch.cuda.stream(ambient):
        ambient_ptr = int(torch.cuda.current_stream(runtime.device).cuda_stream)
        log.append(("ambient_current_ptr", ambient_ptr))
        with runtime._caller_root_scope():
            log.append(("body_current_ptr", int(torch.cuda.current_stream(runtime.device).cuda_stream)))
    assert log == [
        ("enter_ctx_for", ambient_ptr),
        ("ambient_current_ptr", ambient_ptr),
        ("enter_ctx_for", caller_ptr),
        ("exit_ctx_for", caller_ptr),
        ("body_current_ptr", ambient_ptr),
        ("enter_ctx_for", root_ptr),
        ("exit_ctx_for", root_ptr),
        ("exit_ctx_for", ambient_ptr),
    ]


def test_triton_distinct_external_streams_use_distinct_runtime_handoffs(primary_artifact, primary_grg):
    ref_layout = build_reference_layout([primary_artifact], requirements=_requirements(2))
    with torch.cuda.device(0):
        requested_a = torch.cuda.Stream()
        requested_b = torch.cuda.Stream()
    layout_a = build_triton_layout([primary_artifact], stream=requested_a, requirements=_requirements(2))
    layout_b = build_triton_layout([primary_artifact], stream=requested_b, requirements=_requirements(2))
    with ReferenceRuntime(ref_layout) as ref_runtime, TritonRuntime(layout_a) as runtime_a, TritonRuntime(layout_b) as runtime_b:
        assert runtime_a.stream_ptr == int(requested_a.cuda_stream)
        assert runtime_b.stream_ptr == int(requested_b.cuda_stream)
        assert runtime_a.stream_ptr != runtime_b.stream_ptr
        (ref_grg,) = ref_runtime.grgs
        (grg_a,) = runtime_a.grgs
        (grg_b,) = runtime_b.grgs
        rng = np.random.default_rng(9201)
        x_up = rng.standard_normal((2, primary_grg.num_samples), dtype=DATA_DTYPE)
        x_down = rng.standard_normal((2, primary_grg.num_mutations), dtype=DATA_DTYPE)
        expected_up = ref_grg.matmul(x_up, "up")
        expected_down = ref_grg.matmul(x_down, "down")
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg_a.matmul(x_up, "up"), expected_up, atol=atol, rtol=rtol)
        np.testing.assert_allclose(grg_b.matmul(x_up, "up"), expected_up, atol=atol, rtol=rtol)
        np.testing.assert_allclose(grg_a.matmul(x_down, "down"), expected_down, atol=atol, rtol=rtol)
        np.testing.assert_allclose(grg_b.matmul(x_down, "down"), expected_down, atol=atol, rtol=rtol)


def test_triton_executes_under_current_device_switch(primary_artifact, primary_grg):
    if torch.cuda.device_count() < 2:
        pytest.skip("needs at least two CUDA devices")
    layout = build_triton_layout([primary_artifact], device=0, requirements=_requirements(2))
    with torch.cuda.device(1):
        with TritonRuntime(layout) as runtime:
            (grg,) = runtime.grgs
            rng = np.random.default_rng(7201)
            x_up = rng.standard_normal((2, primary_grg.num_samples), dtype=DATA_DTYPE)
            x_down = rng.standard_normal((2, primary_grg.num_mutations), dtype=DATA_DTYPE)
            atol, rtol = tol(DATA_DTYPE)
            np.testing.assert_allclose(
                grg.matmul(x_up, "up"),
                np.asarray(pygrgl.matmul(primary_grg, x_up, pygrgl.TraversalDirection.UP)),
                atol=atol,
                rtol=rtol,
            )
            np.testing.assert_allclose(
                grg.matmul(x_down, "down"),
                np.asarray(pygrgl.matmul(primary_grg, x_down, pygrgl.TraversalDirection.DOWN)),
                atol=atol,
                rtol=rtol,
            )


def test_triton_executes_on_foreign_cupy_stream(primary_artifact, primary_grg):
    cp = pytest.importorskip("cupy")
    with cp.cuda.Device(0):
        requested = cp.cuda.Stream(non_blocking=True)
    layout = build_triton_layout([primary_artifact], device=0, stream=requested, requirements=_requirements(2))
    with TritonRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(7202)
        x_up = rng.standard_normal((2, primary_grg.num_samples), dtype=DATA_DTYPE)
        x_down = rng.standard_normal((2, primary_grg.num_mutations), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(
            grg.matmul(x_up, "up"),
            np.asarray(pygrgl.matmul(primary_grg, x_up, pygrgl.TraversalDirection.UP)),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            grg.matmul(x_down, "down"),
            np.asarray(pygrgl.matmul(primary_grg, x_down, pygrgl.TraversalDirection.DOWN)),
            atol=atol,
            rtol=rtol,
        )


def test_triton_csc_block_plan_uses_separate_struct_dtypes_for_large_row_bound():
    huge = int(np.iinfo(np.int32).max) + 2
    fake = SimpleNamespace(dst_level=1, src_level=0, shape=(huge, 1), nnz=1)
    plan = TritonPlan(store="N", fmt="CSC", scratch="none")
    block = _block_plan(0, Direction.UP, False, fake, plan)
    assert block.stored_shape == (huge, 1)
    assert block.indptr_dtype == np.dtype(np.int32)
    assert block.indices_dtype == np.dtype(np.int64)


def test_triton_down_only_block_plan_uses_stored_shape():
    huge = int(np.iinfo(np.int32).max) + 2
    fake = SimpleNamespace(dst_level=1, src_level=0, shape=(1, huge), nnz=1)
    plan = TritonPlan(store="T", fmt="CSC", scratch="none")
    block = _block_plan(0, Direction.DOWN, False, fake, plan)
    assert block.stored_shape == (huge, 1)
    assert block.indptr_dtype == np.dtype(np.int32)
    assert block.indices_dtype == np.dtype(np.int64)


def test_triton_small_int64_artifact_compacts_slot_dtypes(tmp_path):
    artifact = write_three_level_band_artifact(tmp_path, "tiny-triton-i64", n=4, bandwidth=1, struct_dtype=np.int64)
    scan = scan_grg_spmv(artifact)
    assert all(block.indices_dtype == np.dtype(np.int64) and block.indptr_dtype == np.dtype(np.int64) for block in scan.blocks)
    full = build_triton_layout([artifact], ring_buffer_size=0, vram_budget_bytes=1_000_000_000, requirements=_requirements(2))
    block_bytes = min(block.nbytes for a in full.artifacts for block in (*a.blocks_up, *a.blocks_down))
    streamed = build_triton_layout(
        [artifact],
        ring_buffer_size=1,
        vram_budget_bytes=full.required_budget_for_full_residency - block_bytes,
        requirements=_requirements(2),
    )
    assert streamed.allocated_ring_buffer_size == 1
    assert streamed.slot_plans[0].indices_dtype == np.dtype(np.int32)
    assert streamed.slot_plans[0].indptr_dtype == np.dtype(np.int32)
    with TritonRuntime(streamed) as runtime:
        assert runtime._slot_indices[0].dtype == torch.int32
        assert runtime._slot_indptr[0].dtype == torch.int32


@pytest.mark.parametrize("runtime_k", [1, 2, 4], ids=["k1", "k2", "k4"])
def test_triton_primary_grg_exact_binary(primary_artifact, primary_grg, runtime_k):
    layout = build_triton_layout([primary_artifact], requirements=_requirements(runtime_k))
    with TritonRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(8002 + runtime_k)
        x_up = rng.choice(np.array([-1.0, 1.0], dtype=DATA_DTYPE), size=(runtime_k, primary_grg.num_samples))
        x_down = rng.choice(np.array([-1.0, 1.0], dtype=DATA_DTYPE), size=(runtime_k, primary_grg.num_mutations))
        np.testing.assert_array_equal(grg.matmul(x_up, "up"), np.asarray(pygrgl.matmul(primary_grg, x_up, pygrgl.TraversalDirection.UP)))
        np.testing.assert_array_equal(grg.matmul(x_down, "down"), np.asarray(pygrgl.matmul(primary_grg, x_down, pygrgl.TraversalDirection.DOWN)))


@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
def test_triton_scratch_enabled_up_matches_reference(primary_artifact, primary_grg, runtime_k):
    pair = TritonPlanPair(plan_up=TritonPlan(store="N", fmt="CSR", scratch="1"), plan_down=None)
    layout = build_triton_layout([primary_artifact], pair=pair, requirements=_requirements(runtime_k))
    with TritonRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(123 + runtime_k)
        x = rng.standard_normal((runtime_k, primary_grg.num_samples), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, "up"), np.asarray(pygrgl.matmul(primary_grg, x, pygrgl.TraversalDirection.UP)), atol=atol, rtol=rtol)


@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
def test_triton_scratch_enabled_down_matches_reference(primary_artifact, primary_grg, runtime_k):
    pair = TritonPlanPair(plan_up=None, plan_down=TritonPlan(store="T", fmt="CSC", scratch="0"))
    layout = build_triton_layout([primary_artifact], pair=pair, requirements=_requirements(runtime_k))
    with TritonRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(223 + runtime_k)
        x = rng.standard_normal((runtime_k, primary_grg.num_mutations), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, "down"), np.asarray(pygrgl.matmul(primary_grg, x, pygrgl.TraversalDirection.DOWN)), atol=atol, rtol=rtol)


@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
@pytest.mark.parametrize("mode", THREE_BLOCK_TRANSITION_MODES, ids=[mode.name for mode in THREE_BLOCK_TRANSITION_MODES])
def test_triton_small_three_block_planner_transitions(triton_small_stream_artifact, runtime_k, mode):
    _assert_planner_transition(triton_small_stream_artifact, runtime_k=runtime_k, mode=mode)


@pytest.mark.stress
@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
@pytest.mark.parametrize("mode", THREE_BLOCK_TRANSITION_MODES, ids=[mode.name for mode in THREE_BLOCK_TRANSITION_MODES])
def test_triton_large_three_block_planner_transitions(triton_stream_stress_artifact, runtime_k, mode):
    _assert_planner_transition(triton_stream_stress_artifact, runtime_k=runtime_k, mode=mode)


@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
@pytest.mark.parametrize("mode", THREE_BLOCK_EXACTNESS_MODES, ids=[mode.name for mode in THREE_BLOCK_EXACTNESS_MODES])
def test_triton_small_three_block_exactness(order, runtime_k, mode, gpu_small_stream_case, triton_small_stream_artifact):
    clear_torch_state()
    _run_exactness_case(triton_small_stream_artifact, gpu_small_stream_case, runtime_k=runtime_k, mode=mode, order=order)
    clear_torch_state()


@pytest.mark.stress
@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
@pytest.mark.parametrize("mode", THREE_BLOCK_EXACTNESS_MODES, ids=[mode.name for mode in THREE_BLOCK_EXACTNESS_MODES])
def test_triton_large_three_block_exactness(order, runtime_k, mode, triton_stream_stress_case, triton_stream_stress_artifact):
    clear_torch_state()
    _run_exactness_case(triton_stream_stress_artifact, triton_stream_stress_case, runtime_k=runtime_k, mode=mode, order=order)
    clear_torch_state()


@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("requested_ring_buffer_size", [1, 2], ids=["ring1", "ring2"])
def test_triton_small_stream_copy_overlaps_compute(tmp_path, order, requested_ring_buffer_size, monkeypatch):
    _run_overlap_case(tmp_path, order, requested_ring_buffer_size, monkeypatch, n=64, bandwidth=8)


@pytest.mark.stress
@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("requested_ring_buffer_size", [1, 2], ids=["ring1", "ring2"])
def test_triton_stream_copy_overlaps_compute(tmp_path, order, requested_ring_buffer_size, monkeypatch):
    _run_overlap_case(tmp_path, order, requested_ring_buffer_size, monkeypatch, n=4096, bandwidth=64)
