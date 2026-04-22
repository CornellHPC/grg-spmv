from __future__ import annotations

from contextlib import ExitStack, contextmanager
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
_THREE_BLOCK_KEYS = ((1, 0), (2, 0), (2, 1))
_MIXED_HEIGHT_GRAPH_ORDERS = (("up", "down"), ("down", "up"))


def _requirements(runtime_k: int):
    return full_requirements(max_k_up=int(runtime_k), max_k_down=int(runtime_k))


def _assert_triton_reset(runtime: TritonRuntime) -> None:
    assert runtime._caller_stream is None
    assert runtime._root_stream is None
    assert runtime._caller_to_root_event is None
    assert runtime._root_to_caller_event is None
    assert runtime._level_streams == []
    assert runtime._slot_copy_streams == []
    assert runtime._scratch_streams == []
    assert runtime._slot_indices == []
    assert runtime._slot_indptr == []
    assert runtime._state is None
    assert runtime._scratch == []
    assert runtime._artifacts == ()
    assert runtime._grgs == ()
    assert runtime._config_up is None
    assert runtime._config_down is None
    assert runtime.stream is None
    assert not runtime._entered
    assert not runtime._active_call


def _torch_device_nbytes(value, seen: set[int]) -> int:
    if value is None or not isinstance(value, torch.Tensor) or value.device.type != "cuda":
        return 0
    storage = value.untyped_storage()
    nbytes = int(storage.nbytes())
    if nbytes == 0:
        return 0
    ptr = int(storage.data_ptr())
    if ptr in seen:
        return 0
    seen.add(ptr)
    return nbytes


def _triton_owned_device_nbytes(runtime: TritonRuntime) -> int:
    seen: set[int] = set()
    total = 0
    for value in (runtime._state, runtime._io0, runtime._io1, runtime._aux):
        total += _torch_device_nbytes(value, seen)
    for values in (runtime._slot_indices, runtime._slot_indptr):
        total += sum(_torch_device_nbytes(value, seen) for value in values)
    for row in runtime._scratch:
        total += sum(_torch_device_nbytes(value, seen) for value in row)
    for artifact in runtime._artifacts:
        for value in (
            artifact.sel_mut_rows,
            artifact.sel_mut_cols,
            artifact.sel_miss_rows,
            artifact.sel_miss_cols,
            artifact.node_perm,
            artifact.sample_to_individual,
            artifact.xtx_bias,
            artifact.init_vector_up_bias,
            artifact.init_vector_down_bias,
            artifact.init_xtx_up_bias,
            artifact.init_xtx_down_bias,
        ):
            total += _torch_device_nbytes(value, seen)
        for ops_by_level in (artifact.up_ops, artifact.down_ops):
            for ops in ops_by_level:
                for op in ops:
                    total += _torch_device_nbytes(op.launch_block.indices, seen)
                    total += _torch_device_nbytes(op.launch_block.indptr, seen)
    return total


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


def _build_layout(artifact, *, runtime_k: int, ring_buffer_size: int, budget_bytes: int, allow_residency: bool = True, stream=0):
    return build_triton_layout(
        [artifact],
        requirements=_requirements(runtime_k),
        ring_buffer_size=int(ring_buffer_size),
        vram_budget_bytes=int(budget_bytes),
        allow_residency=bool(allow_residency),
        stream=stream,
    )


def _full_budget_components(artifact, *, runtime_k: int) -> tuple[int, int, int]:
    layout = build_triton_layout(
        [artifact],
        requirements=_requirements(runtime_k),
        ring_buffer_size=0,
        vram_budget_bytes=1_000_000_000_000,
    )
    return _equal_block_budget_components(layout)


def _source_cols(grg, direction: str) -> int:
    return int(grg.num_samples if direction == "up" else grg.num_mutations)


def _expected_prepared_chain(ref_grg, order, src: np.ndarray, init_first: np.ndarray, init_second: np.ndarray) -> np.ndarray:
    mid = ref_grg.matmul(src, order[0], init=init_first)
    return ref_grg.matmul(mid, order[1], init=init_second)


def _enter_prepared_chain(stack: ExitStack, grg, order, *, k: int):
    first = stack.enter_context(grg.prepare_matmul_cuda(direction=order[0], k=int(k), init_mode="vector"))
    second = stack.enter_context(grg.prepare_matmul_cuda(direction=order[1], k=int(k), init_mode="vector"))
    assert first.output.data_ptr() == second.input.data_ptr()
    return first, second


def _capture_prepared_chain(first, second, src, init_first, init_second, result=None) -> None:
    first.input.copy_(src)
    first.init_vector.copy_(init_first)
    first()
    second.init_vector.copy_(init_second)
    second()
    if result is not None:
        result.copy_(second.output)


def _mixed_height_graph_artifacts(tmp_path) -> tuple[object, object]:
    tall = write_overlap_band_artifact(tmp_path, "tall-h4", n=16, bandwidth=4)
    short = write_three_level_band_artifact(tmp_path, "short-h3", n=16, bandwidth=4)
    assert scan_grg_spmv(tall).num_levels == 4
    assert scan_grg_spmv(short).num_levels == 3
    return tall, short


def _build_multi_grg_graph_layout(artifacts, *, streamed: bool, stream):
    layout = build_triton_layout(
        artifacts,
        requirements=_requirements(1),
        ring_buffer_size=2 if streamed else 0,
        vram_budget_bytes=1_000_000_000,
        allow_residency=not bool(streamed),
        stream=stream,
    )
    if streamed:
        assert layout.allow_residency is False
        assert layout.requested_ring_buffer_size == 2
        assert layout.allocated_ring_buffer_size == 2
        assert all(
            not block.resident
            for artifact in layout.artifacts
            for block in (*artifact.blocks_up, *artifact.blocks_down)
        )
    else:
        assert layout.allocated_ring_buffer_size == 0
    return layout


def _multi_grg_graph_cases(artifacts) -> list[SimpleNamespace]:
    out = []
    ref_layout = build_reference_layout(artifacts, requirements=_requirements(1))
    with ReferenceRuntime(ref_layout) as ref_runtime:
        for idx, (ref_grg, order) in enumerate(zip(ref_runtime.grgs, _MIXED_HEIGHT_GRAPH_ORDERS, strict=True)):
            cols = _source_cols(ref_grg, order[0])
            src = np.arange(cols, dtype=DATA_DTYPE).reshape(1, cols) + DATA_DTYPE(idx * 100)
            init_first = np.asarray([0.25 + idx], dtype=DATA_DTYPE)
            init_second = np.asarray([0.75 + idx], dtype=DATA_DTYPE)
            out.append(
                SimpleNamespace(
                    order=order,
                    src=src,
                    init_first=init_first,
                    init_second=init_second,
                    expected=_expected_prepared_chain(ref_grg, order, src, init_first, init_second),
                )
            )
    return out


def _assert_budget_accounting(layout, *, chosen_budget: int, fixed_bytes: int, block_bytes: int) -> None:
    assert sum(item.nbytes for item in layout.budget_items) == layout.bytes_total
    assert layout.required_budget_for_full_residency == int(fixed_bytes + 3 * block_bytes)
    assert layout.bytes_total <= int(chosen_budget) <= layout.required_budget_for_full_residency


def _assert_all_streamed_no_residency(layout, *, requested_ring_buffer_size: int, expected_slot_count: int, fixed_bytes: int, block_bytes: int) -> None:
    assert layout.allow_residency is False
    assert layout.requested_ring_buffer_size == int(requested_ring_buffer_size)
    assert layout.allocated_ring_buffer_size == int(expected_slot_count)
    assert _owner_keys(layout, resident=True) == ()
    assert _owner_keys(layout, resident=False) == _THREE_BLOCK_KEYS
    assert layout.bytes_by_category["resident_sparse"] == 0
    resident_items = [item for item in layout.budget_items if item.kind == "resident_sparse"]
    slot_items = [item for item in layout.budget_items if item.kind == "ring_slot"]
    assert resident_items == []
    assert len(slot_items) == int(expected_slot_count)
    assert layout.bytes_total == int(fixed_bytes + expected_slot_count * block_bytes)
    assert layout.required_budget_for_full_residency == int(fixed_bytes + 3 * block_bytes)


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


def _run_overlap_case(tmp_path, order, requested_ring_buffer_size, monkeypatch, *, n: int, bandwidth: int, runtime_k: int) -> None:
    clear_torch_state()
    artifact = write_overlap_band_artifact(tmp_path, f"triton-overlap-{n}-{bandwidth}", n=n, bandwidth=bandwidth)
    budget_bytes = triton_ring_thresholds(
        artifact,
        requirements=_requirements(runtime_k),
        total_vram_bytes=prepare_triton_stream_stress_case().total_vram_bytes,
        max_ring_buffer_size=2,
    )[int(requested_ring_buffer_size) - 1]
    layout = _build_layout(
        artifact,
        runtime_k=runtime_k,
        ring_buffer_size=requested_ring_buffer_size,
        budget_bytes=budget_bytes,
    )
    ref_layout = build_reference_layout([artifact], requirements=_requirements(runtime_k))
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
            primary = rng.choice(np.array([-1.0, 1.0], dtype=DATA_DTYPE), size=(int(runtime_k), size))
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


def test_triton_owned_device_bytes_do_not_exceed_tight_budget(primary_artifact):
    clear_torch_state()
    base = build_triton_layout([primary_artifact], requirements=_requirements(2), vram_budget_bytes=1_000_000_000_000)
    layout = build_triton_layout([primary_artifact], requirements=_requirements(2), vram_budget_bytes=base.bytes_total)
    assert layout.bytes_total == base.bytes_total
    with TritonRuntime(layout) as runtime:
        assert _triton_owned_device_nbytes(runtime) <= layout.vram_budget_bytes
    clear_torch_state()


@pytest.mark.parametrize("requested_ring_buffer_size", [1, 2, 3], ids=["ring1", "ring2", "ring3"])
def test_triton_allow_residency_false_forces_all_streamed_layout(triton_small_stream_artifact, requested_ring_buffer_size):
    fixed_bytes, block_bytes, owner_block_count = _full_budget_components(triton_small_stream_artifact, runtime_k=2)
    assert owner_block_count == 3
    budget_bytes = fixed_bytes + 3 * block_bytes
    layout = _build_layout(
        triton_small_stream_artifact,
        runtime_k=2,
        ring_buffer_size=requested_ring_buffer_size,
        budget_bytes=budget_bytes,
        allow_residency=False,
    )
    _assert_all_streamed_no_residency(
        layout,
        requested_ring_buffer_size=requested_ring_buffer_size,
        expected_slot_count=requested_ring_buffer_size,
        fixed_bytes=fixed_bytes,
        block_bytes=block_bytes,
    )
    if requested_ring_buffer_size < 3:
        assert layout.bytes_total < layout.required_budget_for_full_residency
    else:
        assert layout.bytes_total == layout.required_budget_for_full_residency


def test_triton_allow_residency_false_rejects_ring_zero(triton_small_stream_artifact):
    with pytest.raises(ValueError, match="ring_buffer_size"):
        _build_layout(
            triton_small_stream_artifact,
            runtime_k=1,
            ring_buffer_size=0,
            budget_bytes=1_000_000_000_000,
            allow_residency=False,
        )


def test_triton_allow_residency_false_warns_when_requested_ring_exceeds_streamed_blocks(triton_small_stream_artifact):
    fixed_bytes, block_bytes, owner_block_count = _full_budget_components(triton_small_stream_artifact, runtime_k=2)
    assert owner_block_count == 3
    budget_bytes = fixed_bytes + 3 * block_bytes
    with pytest.warns(RuntimeWarning, match="requested ring_buffer_size"):
        layout = _build_layout(
            triton_small_stream_artifact,
            runtime_k=2,
            ring_buffer_size=4,
            budget_bytes=budget_bytes,
            allow_residency=False,
        )
    _assert_all_streamed_no_residency(
        layout,
        requested_ring_buffer_size=4,
        expected_slot_count=3,
        fixed_bytes=fixed_bytes,
        block_bytes=block_bytes,
    )


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
@pytest.mark.parametrize("runtime_k", [2], ids=["k2"])
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


@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
@pytest.mark.parametrize("requested_ring_buffer_size", [1, 2, 3], ids=["ring1", "ring2", "ring3"])
def test_triton_small_three_block_exactness_allow_residency_false(order, runtime_k, requested_ring_buffer_size, gpu_small_stream_case, triton_small_stream_artifact):
    fixed_bytes, block_bytes, owner_block_count = _full_budget_components(triton_small_stream_artifact, runtime_k=runtime_k)
    assert owner_block_count == 3
    budget_bytes = fixed_bytes + requested_ring_buffer_size * block_bytes
    clear_torch_state()
    layout = _build_layout(
        triton_small_stream_artifact,
        runtime_k=runtime_k,
        ring_buffer_size=requested_ring_buffer_size,
        budget_bytes=budget_bytes,
        allow_residency=False,
    )
    _assert_all_streamed_no_residency(
        layout,
        requested_ring_buffer_size=requested_ring_buffer_size,
        expected_slot_count=requested_ring_buffer_size,
        fixed_bytes=fixed_bytes,
        block_bytes=block_bytes,
    )
    with TritonRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        for run_idx, direction in enumerate(order):
            seed = 63_000 + 1_000 * runtime_k + 100 * run_idx + 10 * requested_ring_buffer_size
            rng = np.random.default_rng(seed)
            primary = rng.choice(np.array([-1.0, 1.0], dtype=DATA_DTYPE), size=(int(runtime_k), gpu_small_stream_case.n))
            if direction == "up":
                expected = expected_up(primary.T, shifts=gpu_small_stream_case.shifts, bandwidth=gpu_small_stream_case.bandwidth).T
            else:
                expected = expected_down(primary.T, shifts=gpu_small_stream_case.shifts, bandwidth=gpu_small_stream_case.bandwidth).T
            np.testing.assert_array_equal(grg.matmul(primary, direction), expected)
    clear_torch_state()


@pytest.mark.stress
@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("runtime_k", [2], ids=["k2"])
@pytest.mark.parametrize("mode", THREE_BLOCK_EXACTNESS_MODES, ids=[mode.name for mode in THREE_BLOCK_EXACTNESS_MODES])
def test_triton_large_three_block_exactness(order, runtime_k, mode, triton_stream_stress_case, triton_stream_stress_artifact):
    clear_torch_state()
    _run_exactness_case(triton_stream_stress_artifact, triton_stream_stress_case, runtime_k=runtime_k, mode=mode, order=order)
    clear_torch_state()


@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("requested_ring_buffer_size", [1, 2], ids=["ring1", "ring2"])
def test_triton_small_stream_copy_overlaps_compute(tmp_path, order, requested_ring_buffer_size, monkeypatch):
    _run_overlap_case(tmp_path, order, requested_ring_buffer_size, monkeypatch, n=64, bandwidth=8, runtime_k=1)


@pytest.mark.stress
@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("requested_ring_buffer_size", [1, 2], ids=["ring1", "ring2"])
def test_triton_stream_copy_overlaps_compute(tmp_path, order, requested_ring_buffer_size, monkeypatch):
    _run_overlap_case(tmp_path, order, requested_ring_buffer_size, monkeypatch, n=4096, bandwidth=64, runtime_k=2)


def test_triton_prepare_cuda_graph_capture_resident(primary_artifact):
    with torch.cuda.device(0):
        capture_stream = torch.cuda.Stream()
    layout = build_triton_layout([primary_artifact], requirements=_requirements(2), stream=capture_stream)
    ref_layout = build_reference_layout([primary_artifact], requirements=_requirements(2))
    with ReferenceRuntime(ref_layout) as ref_runtime:
        (ref_grg,) = ref_runtime.grgs
        rng = np.random.default_rng(98_101)
        order = ("down", "up")
        src_np = rng.standard_normal((2, _source_cols(ref_grg, order[0])), dtype=DATA_DTYPE)
        init_first_np = rng.standard_normal((2,), dtype=DATA_DTYPE)
        init_second_np = rng.standard_normal((2,), dtype=DATA_DTYPE)
        expected = _expected_prepared_chain(ref_grg, order, src_np, init_first_np, init_second_np)
    with TritonRuntime(layout) as runtime, ExitStack() as stack:
        (grg,) = runtime.grgs
        first, second = _enter_prepared_chain(stack, grg, order, k=2)
        src = torch.from_numpy(src_np).to(device=first.input.device)
        init_first = torch.from_numpy(init_first_np).to(device=first.init_vector.device)
        init_second = torch.from_numpy(init_second_np).to(device=second.init_vector.device)
        graph = torch.cuda.CUDAGraph()
        # All capture-sensitive setup must already be complete once the prepared ops are entered.
        with torch.cuda.graph(graph, stream=capture_stream):
            _capture_prepared_chain(first, second, src, init_first, init_second)
        graph.replay()
        torch.cuda.synchronize(runtime.device)
        actual = second.output.cpu().numpy().copy()
    np.testing.assert_allclose(actual, expected, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])


def test_triton_prepare_cuda_graph_capture_streamed(triton_small_stream_artifact):
    fixed_bytes, block_bytes, owner_block_count = _full_budget_components(triton_small_stream_artifact, runtime_k=1)
    assert owner_block_count == 3
    with torch.cuda.device(0):
        capture_stream = torch.cuda.Stream()
    layout = _build_layout(
        triton_small_stream_artifact,
        runtime_k=1,
        ring_buffer_size=2,
        budget_bytes=fixed_bytes + 2 * block_bytes,
        allow_residency=False,
        stream=capture_stream,
    )
    ref_layout = build_reference_layout([triton_small_stream_artifact], requirements=_requirements(1))
    with ReferenceRuntime(ref_layout) as ref_runtime:
        (ref_grg,) = ref_runtime.grgs
        order = ("down", "up")
        src_np = np.arange(_source_cols(ref_grg, order[0]), dtype=DATA_DTYPE).reshape(1, -1)
        init_first_np = np.asarray([0.5], dtype=DATA_DTYPE)
        init_second_np = np.asarray([1.5], dtype=DATA_DTYPE)
        expected = _expected_prepared_chain(ref_grg, order, src_np, init_first_np, init_second_np)
    with TritonRuntime(layout) as runtime, ExitStack() as stack:
        (grg,) = runtime.grgs
        first, second = _enter_prepared_chain(stack, grg, order, k=1)
        src = torch.from_numpy(src_np).to(device=first.input.device)
        init_first = torch.from_numpy(init_first_np).to(device=first.init_vector.device)
        init_second = torch.from_numpy(init_second_np).to(device=second.init_vector.device)
        graph = torch.cuda.CUDAGraph()
        # All capture-sensitive setup must already be complete once the prepared ops are entered.
        with torch.cuda.graph(graph, stream=capture_stream):
            _capture_prepared_chain(first, second, src, init_first, init_second)
        graph.replay()
        torch.cuda.synchronize(runtime.device)
        actual = second.output.cpu().numpy().copy()
    np.testing.assert_allclose(actual, expected, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])


@pytest.mark.parametrize("streamed", [False, True], ids=["resident", "streamed"])
def test_triton_cuda_graph_capture_multi_grg_single_graph_mixed_heights(tmp_path, streamed):
    artifacts = _mixed_height_graph_artifacts(tmp_path)
    cases = _multi_grg_graph_cases(artifacts)
    with torch.cuda.device(0):
        capture_stream = torch.cuda.Stream()
    layout = _build_multi_grg_graph_layout(artifacts, streamed=streamed, stream=capture_stream)
    with TritonRuntime(layout) as runtime, ExitStack() as stack:
        chains = [
            _enter_prepared_chain(stack, grg, case.order, k=1)
            for grg, case in zip(runtime.grgs, cases, strict=True)
        ]
        srcs = [
            torch.from_numpy(case.src).to(device=chain[0].input.device)
            for chain, case in zip(chains, cases, strict=True)
        ]
        init_first = [
            torch.from_numpy(case.init_first).to(device=chain[0].init_vector.device)
            for chain, case in zip(chains, cases, strict=True)
        ]
        init_second = [
            torch.from_numpy(case.init_second).to(device=chain[1].init_vector.device)
            for chain, case in zip(chains, cases, strict=True)
        ]
        results = [torch.empty_like(chain[1].output) for chain in chains]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            for chain, src, first_init, second_init, result in zip(chains, srcs, init_first, init_second, results, strict=True):
                _capture_prepared_chain(chain[0], chain[1], src, first_init, second_init, result)
        graph.replay()
        torch.cuda.synchronize(runtime.device)
        actual = [result.cpu().numpy().copy() for result in results]
    for case, value in zip(cases, actual, strict=True):
        np.testing.assert_allclose(value, case.expected, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])


@pytest.mark.parametrize("streamed", [False, True], ids=["resident", "streamed"])
def test_triton_cuda_graph_capture_multi_grg_sequential_mixed_heights(tmp_path, streamed):
    artifacts = _mixed_height_graph_artifacts(tmp_path)
    cases = _multi_grg_graph_cases(artifacts)
    with torch.cuda.device(0):
        capture_stream = torch.cuda.Stream()
    layout = _build_multi_grg_graph_layout(artifacts, streamed=streamed, stream=capture_stream)
    with TritonRuntime(layout) as runtime, ExitStack() as stack:
        chains = [
            _enter_prepared_chain(stack, grg, case.order, k=1)
            for grg, case in zip(runtime.grgs, cases, strict=True)
        ]
        srcs = [
            torch.from_numpy(case.src).to(device=chain[0].input.device)
            for chain, case in zip(chains, cases, strict=True)
        ]
        init_first = [
            torch.from_numpy(case.init_first).to(device=chain[0].init_vector.device)
            for chain, case in zip(chains, cases, strict=True)
        ]
        init_second = [
            torch.from_numpy(case.init_second).to(device=chain[1].init_vector.device)
            for chain, case in zip(chains, cases, strict=True)
        ]
        captures = []
        for chain, src, first_init, second_init in zip(chains, srcs, init_first, init_second, strict=True):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                _capture_prepared_chain(chain[0], chain[1], src, first_init, second_init)
            captures.append((graph, chain[1]))
        actual = []
        for graph, final in captures:
            graph.replay()
            torch.cuda.synchronize(runtime.device)
            actual.append(final.output.cpu().numpy().copy())
    for case, value in zip(cases, actual, strict=True):
        np.testing.assert_allclose(value, case.expected, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])
