from __future__ import annotations

from contextlib import ExitStack
import functools
from types import SimpleNamespace

import numpy as np
import pygrgl
import pytest
import scipy.sparse as sp

from pygrgl_spmv.backends.base import materialize_sparse_block
from pygrgl_spmv.backends.cusparse import CusparsePlan, CusparsePlanPair, CusparseRuntime
from pygrgl_spmv.backends.cusparse.backend import _block_plan
from pygrgl_spmv.backends.cusparse.ffi import CudaVmmDriver
from pygrgl_spmv.backends.reference import ReferenceRuntime
from pygrgl_spmv.backends.types import Direction, SparseFormat
from pygrgl_spmv.grg.artifact import scan_grg_spmv
from pygrgl_spmv.tests.conftest import DATA_DTYPE, tol
from pygrgl_spmv.tests.runtime._runtime_builders import (
    build_cusparse_layout,
    build_reference_layout,
    full_requirements,
)
from pygrgl_spmv.tests.runtime._streaming_cases import (
    THREE_BLOCK_EXACTNESS_MODES,
    THREE_BLOCK_TRANSITION_MODES,
    ThreeBlockMode,
    _equal_block_budget_components,
    clear_cupy_state,
    cusparse_ring_thresholds,
    expected_down,
    expected_up,
    prepare_cusparse_stream_stress_case,
    three_block_mode_budget_bytes,
    write_overlap_band_artifact,
    write_three_level_band_artifact,
)

cp = pytest.importorskip("cupy")

pytestmark = [pytest.mark.gpu, pytest.mark.cusparse]

_MODE_BY_NAME = {mode.name: mode for mode in THREE_BLOCK_TRANSITION_MODES}
_THREE_BLOCK_KEYS = ((1, 0), (2, 0), (2, 1))
_MIXED_HEIGHT_GRAPH_ORDERS = (("up", "down"), ("down", "up"))


def _requirements(runtime_k: int):
    return full_requirements(max_k_up=int(runtime_k), max_k_down=int(runtime_k))


def _assert_cusparse_reset(runtime: CusparseRuntime) -> None:
    assert runtime._cslib is None
    assert runtime._spmm_lib_by_stream_ptr == {}
    assert runtime._shared_ones is None
    assert runtime._alpha is None
    assert runtime._beta_zero is None
    assert runtime._beta_one is None
    assert runtime._caller_stream is None
    assert runtime._root_stream is None
    assert runtime._caller_to_root_event is None
    assert runtime._root_to_caller_event is None
    assert runtime._level_streams == []
    assert runtime._slot_copy_streams == []
    assert runtime._scratch_streams_up == []
    assert runtime._scratch_streams_down == []
    assert runtime._slot_struct0 == []
    assert runtime._slot_struct1 == []
    assert runtime._up_level_bufs == []
    assert runtime._down_level_bufs == []
    assert runtime._up_src_bufs is None
    assert runtime._down_src_bufs is None
    assert runtime._up_scratch == []
    assert runtime._down_scratch == []
    assert runtime._ext_main_up == []
    assert runtime._ext_main_down == []
    assert runtime._ext_scratch_up == []
    assert runtime._ext_scratch_down == []
    assert runtime._artifacts == ()
    assert runtime._grgs == ()
    assert runtime.stream is None
    assert not runtime._entered
    assert not runtime._active_call


def _assert_runtime_matches_reference(grg, primary_grg, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
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


def _cupy_device_nbytes(value, seen: set[int]) -> int:
    if value is None or not hasattr(value, "data"):
        return 0
    nbytes = int(value.nbytes)
    if nbytes == 0:
        return 0
    ptr = int(value.data.ptr)
    if ptr in seen:
        return 0
    seen.add(ptr)
    return nbytes


def _cupy_nested_nbytes(values, seen: set[int]) -> int:
    if values is None:
        return 0
    total = 0
    for value in values:
        if isinstance(value, (list, tuple)):
            total += _cupy_nested_nbytes(value, seen)
        else:
            total += _cupy_device_nbytes(value, seen)
    return total


def _shared_ones_nbytes(shared_ones, seen: set[int]) -> int:
    if shared_ones is None or int(shared_ones.physical_nbytes) == 0:
        return 0
    if shared_ones._materialized is not None:
        return _cupy_device_nbytes(shared_ones._materialized, seen)
    ptr = int(shared_ones.ptr)
    if ptr in seen:
        return 0
    seen.add(ptr)
    return int(shared_ones.physical_nbytes)


def _cusparse_owned_device_nbytes(runtime: CusparseRuntime) -> int:
    seen: set[int] = set()
    total = _shared_ones_nbytes(runtime._shared_ones, seen)
    for value in (runtime._alpha, runtime._beta_zero, runtime._beta_one, runtime._io0, runtime._io1, runtime._aux):
        total += _cupy_device_nbytes(value, seen)
    for values in (
        runtime._slot_struct0,
        runtime._slot_struct1,
        runtime._up_level_bufs,
        runtime._down_level_bufs,
        runtime._up_src_bufs,
        runtime._down_src_bufs,
        runtime._up_scratch,
        runtime._down_scratch,
        runtime._ext_main_up,
        runtime._ext_main_down,
        runtime._ext_scratch_up,
        runtime._ext_scratch_down,
    ):
        total += _cupy_nested_nbytes(values, seen)
    for artifact in runtime._artifacts:
        for selector in (artifact.mut_selector, artifact.miss_selector):
            total += _cupy_nested_nbytes(selector.rows_by_level, seen)
            total += _cupy_nested_nbytes(selector.cols_by_level, seen)
        for value in (
            artifact.node_perm,
            artifact.sample_to_individual,
            artifact.xtx_bias,
            artifact.init_vector_up_bias,
            artifact.init_vector_down_bias,
            artifact.init_xtx_up_bias,
            artifact.init_xtx_down_bias,
        ):
            total += _cupy_device_nbytes(value, seen)
        for ops_by_level in (artifact.up_ops, artifact.down_ops):
            for ops in ops_by_level:
                for op in ops:
                    total += _cupy_device_nbytes(op.block.struct0, seen)
                    total += _cupy_device_nbytes(op.block.struct1, seen)
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
    return build_cusparse_layout(
        [artifact],
        requirements=_requirements(runtime_k),
        ring_buffer_size=int(ring_buffer_size),
        vram_budget_bytes=int(budget_bytes),
        allow_residency=bool(allow_residency),
        stream=stream,
    )


def _full_budget_components(artifact, *, runtime_k: int) -> tuple[int, int, int]:
    layout = build_cusparse_layout(
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


def _prepared_dense_map(runtime: CusparseRuntime, direction: str):
    artifact = runtime._artifacts[0]
    return artifact.up_dense_by_k if direction == "up" else artifact.down_dense_by_k


def _prepared_direction_enum(direction: str):
    return pygrgl.TraversalDirection.UP if direction == "up" else pygrgl.TraversalDirection.DOWN


def _prepared_input_cols(grg, direction: str) -> int:
    return int(grg.num_samples if direction == "up" else grg.num_mutations)


def _copy_prepared_input(torch, op, values: np.ndarray) -> None:
    op.input.copy_(torch.from_numpy(values).to(device=op.input.device))


@pytest.mark.parametrize("direction", ["up", "down"])
def test_cusparse_prepared_same_direction_mixed_widths(primary_artifact, primary_grg, monkeypatch, direction):
    torch = pytest.importorskip("torch")
    layout = build_cusparse_layout([primary_artifact], requirements=_requirements(4))
    rng = np.random.default_rng(120_401 if direction == "up" else 120_402)
    x1_a = rng.standard_normal((1, _prepared_input_cols(primary_grg, direction)), dtype=DATA_DTYPE)
    x4 = rng.standard_normal((4, _prepared_input_cols(primary_grg, direction)), dtype=DATA_DTYPE)
    x1_b = rng.standard_normal((1, _prepared_input_cols(primary_grg, direction)), dtype=DATA_DTYPE)
    direction_enum = _prepared_direction_enum(direction)
    expected = [
        np.asarray(pygrgl.matmul(primary_grg, x1_a, direction_enum)),
        np.asarray(pygrgl.matmul(primary_grg, x4, direction_enum)),
        np.asarray(pygrgl.matmul(primary_grg, x1_b, direction_enum)),
    ]
    seen = []
    with CusparseRuntime(layout) as runtime, ExitStack() as stack:
        (grg,) = runtime.grgs
        op1 = stack.enter_context(grg.prepare_matmul_cuda(direction=direction, k=1))
        op4 = stack.enter_context(grg.prepare_matmul_cuda(direction=direction, k=4))
        dense_map = _prepared_dense_map(runtime, direction)
        assert set(dense_map) == {1, 4}
        assert dense_map[1].state is op1._dense_state
        assert dense_map[4].state is op4._dense_state

        original = runtime._enqueue_wavefront

        def record_dense(artifact, direction_arg, dense):
            seen.append(dense)
            return original(artifact, direction_arg, dense)

        monkeypatch.setattr(runtime, "_enqueue_wavefront", record_dense)
        actual = []
        for op, values in ((op1, x1_a), (op4, x4), (op1, x1_b)):
            _copy_prepared_input(torch, op, values)
            op()
            actual.append(op.output.cpu().numpy().copy())
        assert seen == [op1._dense_state, op4._dense_state, op1._dense_state]
    assert dense_map == {}
    for value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, expected_value, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])


@pytest.mark.parametrize("direction", ["up", "down"])
def test_cusparse_prepared_same_width_refcount(primary_artifact, primary_grg, direction):
    torch = pytest.importorskip("torch")
    layout = build_cusparse_layout([primary_artifact], requirements=_requirements(4))
    rng = np.random.default_rng(120_501 if direction == "up" else 120_502)
    x = rng.standard_normal((2, _prepared_input_cols(primary_grg, direction)), dtype=DATA_DTYPE)
    expected = np.asarray(pygrgl.matmul(primary_grg, x, _prepared_direction_enum(direction)))
    with CusparseRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        with grg.prepare_matmul_cuda(direction=direction, k=2) as outer:
            with grg.prepare_matmul_cuda(direction=direction, k=2) as inner:
                dense_map = _prepared_dense_map(runtime, direction)
                assert set(dense_map) == {2}
                assert outer._dense_state is inner._dense_state
                assert dense_map[2].state is outer._dense_state
                assert dense_map[2].refs == 2
            dense_map = _prepared_dense_map(runtime, direction)
            assert set(dense_map) == {2}
            assert dense_map[2].refs == 1
            _copy_prepared_input(torch, outer, x)
            outer()
            actual = outer.output.cpu().numpy().copy()
    assert dense_map == {}
    np.testing.assert_allclose(actual, expected, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])


def _mixed_height_graph_artifacts(tmp_path) -> tuple[object, object]:
    tall = write_overlap_band_artifact(tmp_path, "tall-h4", n=16, bandwidth=4)
    short = write_three_level_band_artifact(tmp_path, "short-h3", n=16, bandwidth=4)
    assert scan_grg_spmv(tall).num_levels == 4
    assert scan_grg_spmv(short).num_levels == 3
    return tall, short


def _build_multi_grg_graph_layout(artifacts, *, streamed: bool, stream):
    layout = build_cusparse_layout(
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
    with CusparseRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        for run_idx, direction in enumerate(order):
            seed = 41_000 + 1_000 * runtime_k + 100 * run_idx + sum(ord(ch) for ch in mode.name)
            rng = np.random.default_rng(seed)
            primary = rng.choice(np.array([-1.0, 1.0], dtype=DATA_DTYPE), size=(int(runtime_k), case.n))
            if direction == "up":
                expected = expected_up(primary.T, shifts=case.shifts, bandwidth=case.bandwidth).T
            else:
                expected = expected_down(primary.T, shifts=case.shifts, bandwidth=case.bandwidth).T
            actual = grg.matmul(primary, direction)
            np.testing.assert_array_equal(actual, expected)


@functools.cache
def _spin_kernel():
    module = cp.RawModule(
        code=r"""
        extern "C" __global__ void spin(unsigned long long iters) {
          unsigned long long start = clock64();
          while (clock64() - start < iters) {}
        }
        """,
    )
    return module.get_function("spin")


def _warm_spin_kernel() -> None:
    _spin_kernel()((1,), (1,), (1_000_000,))
    cp.cuda.runtime.deviceSynchronize()


def _run_overlap_case(tmp_path, order, requested_ring_buffer_size, monkeypatch, *, n: int, bandwidth: int, runtime_k: int) -> None:
    clear_cupy_state()
    _warm_spin_kernel()
    artifact = write_overlap_band_artifact(tmp_path, f"cusparse-overlap-{n}-{bandwidth}", n=n, bandwidth=bandwidth)
    budget_bytes = cusparse_ring_thresholds(
        artifact,
        requirements=_requirements(runtime_k),
        total_vram_bytes=prepare_cusparse_stream_stress_case().total_vram_bytes,
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
    original_launch = CusparseRuntime._launch_spmm

    def _wrapped_launch(self, artifact, direction, op, dst_desc, beta_ptr, src_desc, ext) -> None:
        stream = cp.cuda.get_current_stream()
        if state["active"]:
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record(stream)
            original_launch(self, artifact, direction, op, dst_desc, beta_ptr, src_desc, ext)
            with stream:
                _spin_kernel()((1,), (1,), (10_000_000,))
            end.record(stream)
            state["compute"].append((start, end))
            return
        original_launch(self, artifact, direction, op, dst_desc, beta_ptr, src_desc, ext)

    def _wrapped_copy(self, copy_done, compute_done, dst_level: int, op_idx: int, op) -> None:
        if op.slot is None or op.host0 is None or op.host1 is None:
            return
        stream = self._slot_copy_streams[op.slot]
        with stream:
            if op.prev_in_slot is not None:
                prev_dst, prev_idx = op.prev_in_slot
                stream.wait_event(compute_done[prev_dst][prev_idx])
            start = None
            end = None
            if state["active"]:
                start = cp.cuda.Event()
                end = cp.cuda.Event()
                start.record(stream)
            cp.cuda.runtime.memcpyAsync(
                self._slot_struct0[op.slot].data.ptr,
                int(np.asarray(op.host0).ctypes.data),
                int(op.host0.nbytes),
                cp.cuda.runtime.memcpyHostToDevice,
                stream.ptr,
            )
            cp.cuda.runtime.memcpyAsync(
                self._slot_struct1[op.slot].data.ptr,
                int(np.asarray(op.host1).ctypes.data),
                int(op.host1.nbytes),
                cp.cuda.runtime.memcpyHostToDevice,
                stream.ptr,
            )
            if state["active"]:
                assert start is not None and end is not None
                end.record(stream)
                state["copy"].append((start, end))
            copy_done[dst_level][op_idx].record(stream)

    monkeypatch.setattr(CusparseRuntime, "_launch_spmm", _wrapped_launch)
    monkeypatch.setattr(CusparseRuntime, "_copy_to_slot", _wrapped_copy)

    def _has_overlap() -> bool:
        for compute_start, compute_end in state["compute"]:
            if cp.cuda.get_elapsed_time(compute_start, compute_end) <= 0.0:
                continue
            for copy_start, copy_end in state["copy"]:
                if cp.cuda.get_elapsed_time(copy_start, copy_end) <= 0.0:
                    continue
                if cp.cuda.get_elapsed_time(copy_start, compute_end) > 0.0 and cp.cuda.get_elapsed_time(compute_start, copy_end) > 0.0:
                    return True
        return False

    with ReferenceRuntime(ref_layout) as ref_runtime, CusparseRuntime(layout) as runtime:
        (ref_grg,) = ref_runtime.grgs
        (grg,) = runtime.grgs
        for run_idx, direction in enumerate(order):
            size = ref_grg.num_samples if direction == "up" else ref_grg.num_mutations
            rng = np.random.default_rng(51_000 + 1_000 * requested_ring_buffer_size + 10 * run_idx + (0 if direction == "up" else 1))
            primary = rng.choice(np.array([-1.0, 1.0], dtype=DATA_DTYPE), size=(int(runtime_k), size))
            expected = ref_grg.matmul(primary, direction)
            actual = grg.matmul(primary, direction)
            np.testing.assert_array_equal(actual, expected)
            cp.cuda.runtime.deviceSynchronize()
            if run_idx == 0:
                if int(requested_ring_buffer_size) == 1:
                    assert not _has_overlap()
                else:
                    assert _has_overlap()
                state["active"] = False
    clear_cupy_state()


@pytest.fixture(scope="module")
def cusparse_stream_stress_case():
    return prepare_cusparse_stream_stress_case()


@pytest.fixture(scope="module")
def gpu_small_stream_case():
    return SimpleNamespace(n=64, bandwidth=8, shifts=(0, 8, 16))


@pytest.fixture(scope="module")
def cusparse_small_stream_artifact(artifact_cache_dir, gpu_small_stream_case):
    return write_three_level_band_artifact(
        artifact_cache_dir,
        f"gpu-small-stream-n{gpu_small_stream_case.n}",
        n=gpu_small_stream_case.n,
        bandwidth=gpu_small_stream_case.bandwidth,
        shifts=gpu_small_stream_case.shifts,
    )


@pytest.fixture(scope="module")
def cusparse_stream_stress_artifact(artifact_cache_dir, cusparse_stream_stress_case):
    return write_three_level_band_artifact(
        artifact_cache_dir,
        f"gpu-stream-stress-n{cusparse_stream_stress_case.n}",
        n=cusparse_stream_stress_case.n,
        bandwidth=cusparse_stream_stress_case.bandwidth,
        shifts=cusparse_stream_stress_case.shifts,
    )


def test_cusparse_runtime_exposes_requested_stream_and_stream_ptr(primary_artifact):
    with cp.cuda.Device(0):
        requested = cp.cuda.Stream(non_blocking=True)
    layout = build_cusparse_layout([primary_artifact], stream=requested)
    with CusparseRuntime(layout) as runtime:
        assert int(runtime.device.id) == int(layout.device)
        assert runtime.stream is not None
        assert runtime.stream_ptr == layout.stream_ptr == int(requested.ptr)
        assert int(runtime.stream.ptr) == int(requested.ptr)
        assert len(runtime.grgs) == 1


def test_cusparse_runtime_reusable_across_context_cycles(primary_artifact, primary_grg):
    layout = build_cusparse_layout([primary_artifact], requirements=_requirements(2))
    runtime = CusparseRuntime(layout)
    with runtime:
        first_handle = runtime._cslib
        assert first_handle is not None
        (grg,) = runtime.grgs
        _assert_runtime_matches_reference(grg, primary_grg, seed=9101)
    _assert_cusparse_reset(runtime)
    with runtime:
        second_handle = runtime._cslib
        assert second_handle is not None
        assert second_handle is not first_handle
        (grg,) = runtime.grgs
        _assert_runtime_matches_reference(grg, primary_grg, seed=9102)
    _assert_cusparse_reset(runtime)


def test_cusparse_runtime_exit_releases_owned_state(primary_artifact):
    layout = build_cusparse_layout([primary_artifact], requirements=_requirements(2))
    runtime = CusparseRuntime(layout)
    with runtime:
        assert runtime._cslib is not None
        assert runtime._artifacts
        assert runtime._grgs
    _assert_cusparse_reset(runtime)


def test_cusparse_runtime_enter_failure_releases_owned_state(primary_artifact, monkeypatch):
    layout = build_cusparse_layout([primary_artifact, primary_artifact], requirements=_requirements(1))
    runtime = CusparseRuntime(layout)
    original = CusparseRuntime._build_artifact
    calls = {"count": 0}

    def _boom_after_first(self, artifact_layout, state):
        calls["count"] += 1
        if calls["count"] == 1:
            return original(self, artifact_layout, state)
        raise RuntimeError("boom")

    monkeypatch.setattr(CusparseRuntime, "_build_artifact", _boom_after_first)
    with pytest.raises(RuntimeError, match="boom"):
        runtime.__enter__()
    _assert_cusparse_reset(runtime)


def test_cusparse_caller_root_scope_uses_explicit_handoff_streams(primary_artifact):
    with cp.cuda.Device(0):
        requested = cp.cuda.Stream(non_blocking=True)
    layout = build_cusparse_layout([primary_artifact], stream=requested, requirements=_requirements(1))
    runtime = CusparseRuntime(layout)
    log: list[tuple[object, ...]] = []

    class _SpyStream:
        def __init__(self, name: str):
            self.name = name

        def __enter__(self):
            log.append(("enter_stream", self.name))
            return self

        def __exit__(self, exc_type, exc, tb):
            log.append(("exit_stream", self.name))

        def wait_event(self, event) -> None:
            log.append(("wait_event", self.name, event.name))

    class _SpyEvent:
        def __init__(self, name: str):
            self.name = name

        def record(self, stream) -> None:
            log.append(("record", self.name, stream.name))

    runtime._caller_stream = _SpyStream("caller")
    runtime._root_stream = _SpyStream("root")
    runtime._caller_to_root_event = _SpyEvent("caller_to_root")
    runtime._root_to_caller_event = _SpyEvent("root_to_caller")
    ambient = cp.cuda.Stream(non_blocking=True)
    with ambient:
        ambient_ptr = int(cp.cuda.get_current_stream().ptr)
        log.append(("ambient_current_ptr", ambient_ptr))
        with runtime._caller_root_scope():
            log.append(("body_current_ptr", int(cp.cuda.get_current_stream().ptr)))
    assert log == [
        ("ambient_current_ptr", ambient_ptr),
        ("enter_stream", "caller"),
        ("record", "caller_to_root", "caller"),
        ("exit_stream", "caller"),
        ("wait_event", "root", "caller_to_root"),
        ("body_current_ptr", ambient_ptr),
        ("enter_stream", "root"),
        ("record", "root_to_caller", "root"),
        ("exit_stream", "root"),
        ("wait_event", "caller", "root_to_caller"),
    ]


def test_cusparse_distinct_external_streams_use_distinct_runtime_handoffs(primary_artifact, primary_grg):
    ref_layout = build_reference_layout([primary_artifact], requirements=_requirements(2))
    with cp.cuda.Device(0):
        requested_a = cp.cuda.Stream(non_blocking=True)
        requested_b = cp.cuda.Stream(non_blocking=True)
    layout_a = build_cusparse_layout([primary_artifact], stream=requested_a, requirements=_requirements(2))
    layout_b = build_cusparse_layout([primary_artifact], stream=requested_b, requirements=_requirements(2))
    with ReferenceRuntime(ref_layout) as ref_runtime, CusparseRuntime(layout_a) as runtime_a, CusparseRuntime(layout_b) as runtime_b:
        assert runtime_a.stream_ptr == int(requested_a.ptr)
        assert runtime_b.stream_ptr == int(requested_b.ptr)
        assert runtime_a.stream_ptr != runtime_b.stream_ptr
        (ref_grg,) = ref_runtime.grgs
        (grg_a,) = runtime_a.grgs
        (grg_b,) = runtime_b.grgs
        rng = np.random.default_rng(9103)
        x_up = rng.standard_normal((2, primary_grg.num_samples), dtype=DATA_DTYPE)
        x_down = rng.standard_normal((2, primary_grg.num_mutations), dtype=DATA_DTYPE)
        expected_up = ref_grg.matmul(x_up, "up")
        expected_down = ref_grg.matmul(x_down, "down")
        np.testing.assert_allclose(grg_a.matmul(x_up, "up"), expected_up, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])
        np.testing.assert_allclose(grg_b.matmul(x_up, "up"), expected_up, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])
        np.testing.assert_allclose(grg_a.matmul(x_down, "down"), expected_down, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])
        np.testing.assert_allclose(grg_b.matmul(x_down, "down"), expected_down, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])


def test_cusparse_executes_under_current_device_switch(primary_artifact, primary_grg):
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs at least two CUDA devices")
    layout = build_cusparse_layout([primary_artifact], device=0, requirements=_requirements(2))
    with cp.cuda.Device(1):
        with CusparseRuntime(layout) as runtime:
            (grg,) = runtime.grgs
            rng = np.random.default_rng(7301)
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


def test_cusparse_executes_on_foreign_torch_stream(primary_artifact, primary_grg):
    torch = pytest.importorskip("torch")
    with torch.cuda.device(0):
        requested = torch.cuda.Stream()
    layout = build_cusparse_layout([primary_artifact], device=0, stream=requested, requirements=_requirements(2))
    with CusparseRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(7302)
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


def test_cusparse_csc_block_plan_uses_separate_struct_dtypes_for_large_row_bound():
    huge = int(np.iinfo(np.int32).max) + 2
    fake = SimpleNamespace(dst_level=1, src_level=0, shape=(huge, 1), nnz=1)
    plan = CusparsePlan(store="N", fmt="CSC", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT")
    block = _block_plan(0, Direction.UP, False, fake, plan)
    assert block.stored_shape == (huge, 1)
    assert block.struct0_dtype == np.dtype(np.int32)
    assert block.struct1_dtype == np.dtype(np.int64)


def test_cusparse_store_t_coo_materialization_is_row_sorted():
    block = sp.csr_matrix(
        (
            np.ones(4, dtype=np.bool_),
            np.array([1, 2, 0, 2], dtype=np.int32),
            np.array([0, 2, 4], dtype=np.int32),
        ),
        shape=(2, 3),
    )
    direct = block.T.tocoo()
    assert not np.all(np.asarray(direct.row)[1:] >= np.asarray(direct.row)[:-1])
    stored = materialize_sparse_block(block, store="T", fmt="COO")
    rows = np.asarray(stored.row)
    cols = np.asarray(stored.col)
    assert np.all(rows[1:] >= rows[:-1])
    assert np.all((rows[1:] > rows[:-1]) | ((rows[1:] == rows[:-1]) & (cols[1:] >= cols[:-1])))
    if hasattr(stored, "has_canonical_format"):
        assert bool(stored.has_canonical_format)


def test_cusparse_small_int64_artifact_compacts_slot_dtypes(tmp_path):
    artifact = write_three_level_band_artifact(tmp_path, "tiny-cusparse-i64", n=4, bandwidth=1, struct_dtype=np.int64)
    scan = scan_grg_spmv(artifact)
    assert all(block.indices_dtype == np.dtype(np.int64) and block.indptr_dtype == np.dtype(np.int64) for block in scan.blocks)
    full = build_cusparse_layout([artifact], ring_buffer_size=0, vram_budget_bytes=1_000_000_000, requirements=_requirements(2))
    block_bytes = min(block.nbytes for a in full.artifacts for block in (*a.blocks_up, *a.blocks_down))
    streamed = build_cusparse_layout(
        [artifact],
        ring_buffer_size=1,
        vram_budget_bytes=full.required_budget_for_full_residency - block_bytes,
        requirements=_requirements(2),
    )
    assert streamed.allocated_ring_buffer_size == 1
    assert streamed.slot_plans[0].struct0_dtype == np.dtype(np.int32)
    assert streamed.slot_plans[0].struct1_dtype == np.dtype(np.int32)
    with CusparseRuntime(streamed) as runtime:
        assert runtime._slot_struct0[0].dtype == np.int32
        assert runtime._slot_struct1[0].dtype == np.int32


def test_cusparse_owned_device_bytes_do_not_exceed_tight_budget(primary_artifact):
    clear_cupy_state()
    base = build_cusparse_layout([primary_artifact], requirements=_requirements(2), vram_budget_bytes=1_000_000_000_000)
    layout = build_cusparse_layout([primary_artifact], requirements=_requirements(2), vram_budget_bytes=base.bytes_total)
    assert layout.bytes_total == base.bytes_total
    with CusparseRuntime(layout) as runtime:
        assert _cusparse_owned_device_nbytes(runtime) <= layout.vram_budget_bytes
    clear_cupy_state()


@pytest.mark.parametrize("requested_ring_buffer_size", [1, 2, 3], ids=["ring1", "ring2", "ring3"])
def test_cusparse_allow_residency_false_forces_all_streamed_layout(cusparse_small_stream_artifact, requested_ring_buffer_size):
    fixed_bytes, block_bytes, owner_block_count = _full_budget_components(cusparse_small_stream_artifact, runtime_k=2)
    assert owner_block_count == 3
    budget_bytes = fixed_bytes + 3 * block_bytes
    layout = _build_layout(
        cusparse_small_stream_artifact,
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


def test_cusparse_allow_residency_false_rejects_ring_zero(cusparse_small_stream_artifact):
    with pytest.raises(ValueError, match="ring_buffer_size"):
        _build_layout(
            cusparse_small_stream_artifact,
            runtime_k=1,
            ring_buffer_size=0,
            budget_bytes=1_000_000_000_000,
            allow_residency=False,
        )


def test_cusparse_allow_residency_false_warns_when_requested_ring_exceeds_streamed_blocks(cusparse_small_stream_artifact):
    fixed_bytes, block_bytes, owner_block_count = _full_budget_components(cusparse_small_stream_artifact, runtime_k=2)
    assert owner_block_count == 3
    budget_bytes = fixed_bytes + 3 * block_bytes
    with pytest.warns(RuntimeWarning, match="requested ring_buffer_size"):
        layout = _build_layout(
            cusparse_small_stream_artifact,
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


@pytest.mark.parametrize(
    "plan_up",
    [
        pytest.param(CusparsePlan(store="N", fmt="CSR", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"), id="direct-row"),
        pytest.param(CusparsePlan(store="N", fmt="CSR", op_a="N", op_b="T", order_b="COL", order_c="ROW", algo="DEFAULT"), id="reinterpret"),
        pytest.param(CusparsePlan(store="N", fmt="CSR", op_a="N", op_b="N", order_b="COL", order_c="ROW", algo="DEFAULT"), id="repack"),
    ],
)
def test_cusparse_explicit_dense_view_plans_up(primary_artifact, primary_grg, plan_up):
    pair = CusparsePlanPair(plan_up=plan_up, plan_down=None)
    layout = build_cusparse_layout([primary_artifact], pair=pair, requirements=_requirements(4))
    with CusparseRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(841)
        x = rng.standard_normal((4, primary_grg.num_samples), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, "up"), np.asarray(pygrgl.matmul(primary_grg, x, pygrgl.TraversalDirection.UP)), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "plan_down",
    [
        pytest.param(CusparsePlan(store="T", fmt="CSC", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"), id="direct-row"),
        pytest.param(CusparsePlan(store="T", fmt="CSC", op_a="N", op_b="T", order_b="COL", order_c="ROW", algo="DEFAULT"), id="reinterpret"),
        pytest.param(CusparsePlan(store="T", fmt="CSC", op_a="N", op_b="N", order_b="COL", order_c="ROW", algo="DEFAULT"), id="repack"),
    ],
)
def test_cusparse_explicit_dense_view_plans_down(primary_artifact, primary_grg, plan_down):
    pair = CusparsePlanPair(plan_up=None, plan_down=plan_down)
    layout = build_cusparse_layout([primary_artifact], pair=pair, requirements=_requirements(4))
    with CusparseRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(842)
        x = rng.standard_normal((4, primary_grg.num_mutations), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, "down"), np.asarray(pygrgl.matmul(primary_grg, x, pygrgl.TraversalDirection.DOWN)), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("mode", "allocation_granularity", "expected_mode"),
    [
        pytest.param("unsupported", 0, "materialized", id="unsupported"),
        pytest.param("savings", 64, "vmm", id="vmm"),
        pytest.param("no-savings", 1 << 30, "materialized", id="no-savings"),
    ],
)
def test_cusparse_shared_ones_plan_modes(primary_artifact, monkeypatch, mode, allocation_granularity, expected_mode):
    monkeypatch.setattr(CudaVmmDriver, "current_context", lambda self: 1)
    monkeypatch.setattr(CudaVmmDriver, "vmm_supported", lambda self, device_id: mode != "unsupported")
    monkeypatch.setattr(CudaVmmDriver, "allocation_granularity", lambda self, device_id, *, recommended: allocation_granularity)
    layout = build_cusparse_layout([primary_artifact], requirements=_requirements(2))
    assert layout.shared_ones.mode == expected_mode
    if expected_mode == "vmm":
        assert layout.shared_ones.physical_bytes < layout.shared_ones.logical_bytes
    else:
        assert layout.shared_ones.physical_bytes == layout.shared_ones.logical_bytes


@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
def test_cusparse_scratch_enabled_up_matches_reference(primary_artifact, primary_grg, runtime_k):
    pair = CusparsePlanPair(
        plan_up=CusparsePlan(store="N", fmt="CSR", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT", scratch="1"),
        plan_down=None,
    )
    layout = build_cusparse_layout([primary_artifact], pair=pair, requirements=_requirements(runtime_k))
    with CusparseRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(5011 + runtime_k)
        x = rng.standard_normal((runtime_k, primary_grg.num_samples), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, "up"), np.asarray(pygrgl.matmul(primary_grg, x, pygrgl.TraversalDirection.UP)), atol=atol, rtol=rtol)


@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
def test_cusparse_scratch_enabled_down_matches_reference(primary_artifact, primary_grg, runtime_k):
    pair = CusparsePlanPair(
        plan_up=None,
        plan_down=CusparsePlan(store="T", fmt="CSC", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT", scratch="0"),
    )
    layout = build_cusparse_layout([primary_artifact], pair=pair, requirements=_requirements(runtime_k))
    with CusparseRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(6011 + runtime_k)
        x = rng.standard_normal((runtime_k, primary_grg.num_mutations), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, "down"), np.asarray(pygrgl.matmul(primary_grg, x, pygrgl.TraversalDirection.DOWN)), atol=atol, rtol=rtol)


def test_cusparse_f_order_input(primary_artifact, primary_grg):
    layout = build_cusparse_layout([primary_artifact], requirements=_requirements(4))
    with CusparseRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(9005)
        x = np.asfortranarray(rng.standard_normal((4, primary_grg.num_samples), dtype=DATA_DTYPE))
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, "up"), np.asarray(pygrgl.matmul(primary_grg, x, pygrgl.TraversalDirection.UP)), atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt", [SparseFormat.CSR, SparseFormat.CSC, SparseFormat.COO], ids=["csr", "csc", "coo"])
def test_cusparse_primary_grg_exact_binary(primary_artifact, primary_grg, fmt):
    if fmt == SparseFormat.COO:
        up_plan = CusparsePlan(store="N", fmt="COO", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT")
        down_plan = CusparsePlan(store="T", fmt="COO", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT")
    else:
        up_plan = CusparsePlan(store="N", fmt=fmt, op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT")
        down_plan = CusparsePlan(store="T", fmt=(SparseFormat.CSC if fmt == SparseFormat.CSR else SparseFormat.CSR), op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT")
    pair = CusparsePlanPair(plan_up=up_plan, plan_down=down_plan)
    layout = build_cusparse_layout([primary_artifact], pair=pair, requirements=_requirements(4))
    with CusparseRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(8001)
        x_up = rng.choice(np.array([-1.0, 1.0], dtype=DATA_DTYPE), size=(4, primary_grg.num_samples))
        x_down = rng.choice(np.array([-1.0, 1.0], dtype=DATA_DTYPE), size=(4, primary_grg.num_mutations))
        np.testing.assert_array_equal(grg.matmul(x_up, "up"), np.asarray(pygrgl.matmul(primary_grg, x_up, pygrgl.TraversalDirection.UP)))
        np.testing.assert_array_equal(grg.matmul(x_down, "down"), np.asarray(pygrgl.matmul(primary_grg, x_down, pygrgl.TraversalDirection.DOWN)))


@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
@pytest.mark.parametrize("mode", THREE_BLOCK_TRANSITION_MODES, ids=[mode.name for mode in THREE_BLOCK_TRANSITION_MODES])
def test_cusparse_small_three_block_planner_transitions(cusparse_small_stream_artifact, runtime_k, mode):
    _assert_planner_transition(cusparse_small_stream_artifact, runtime_k=runtime_k, mode=mode)


@pytest.mark.stress
@pytest.mark.parametrize("runtime_k", [2], ids=["k2"])
@pytest.mark.parametrize("mode", THREE_BLOCK_TRANSITION_MODES, ids=[mode.name for mode in THREE_BLOCK_TRANSITION_MODES])
def test_cusparse_large_three_block_planner_transitions(cusparse_stream_stress_artifact, runtime_k, mode):
    _assert_planner_transition(cusparse_stream_stress_artifact, runtime_k=runtime_k, mode=mode)


@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
@pytest.mark.parametrize("mode", THREE_BLOCK_EXACTNESS_MODES, ids=[mode.name for mode in THREE_BLOCK_EXACTNESS_MODES])
def test_cusparse_small_three_block_exactness(order, runtime_k, mode, gpu_small_stream_case, cusparse_small_stream_artifact):
    clear_cupy_state()
    _run_exactness_case(cusparse_small_stream_artifact, gpu_small_stream_case, runtime_k=runtime_k, mode=mode, order=order)
    clear_cupy_state()


@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
@pytest.mark.parametrize("requested_ring_buffer_size", [1, 2, 3], ids=["ring1", "ring2", "ring3"])
def test_cusparse_small_three_block_exactness_allow_residency_false(order, runtime_k, requested_ring_buffer_size, gpu_small_stream_case, cusparse_small_stream_artifact):
    fixed_bytes, block_bytes, owner_block_count = _full_budget_components(cusparse_small_stream_artifact, runtime_k=runtime_k)
    assert owner_block_count == 3
    budget_bytes = fixed_bytes + requested_ring_buffer_size * block_bytes
    clear_cupy_state()
    layout = _build_layout(
        cusparse_small_stream_artifact,
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
    with CusparseRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        for run_idx, direction in enumerate(order):
            seed = 43_000 + 1_000 * runtime_k + 100 * run_idx + 10 * requested_ring_buffer_size
            rng = np.random.default_rng(seed)
            primary = rng.choice(np.array([-1.0, 1.0], dtype=DATA_DTYPE), size=(int(runtime_k), gpu_small_stream_case.n))
            if direction == "up":
                expected = expected_up(primary.T, shifts=gpu_small_stream_case.shifts, bandwidth=gpu_small_stream_case.bandwidth).T
            else:
                expected = expected_down(primary.T, shifts=gpu_small_stream_case.shifts, bandwidth=gpu_small_stream_case.bandwidth).T
            np.testing.assert_array_equal(grg.matmul(primary, direction), expected)
    clear_cupy_state()


@pytest.mark.stress
@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("runtime_k", [2], ids=["k2"])
@pytest.mark.parametrize("mode", THREE_BLOCK_EXACTNESS_MODES, ids=[mode.name for mode in THREE_BLOCK_EXACTNESS_MODES])
def test_cusparse_large_three_block_exactness(order, runtime_k, mode, cusparse_stream_stress_case, cusparse_stream_stress_artifact):
    clear_cupy_state()
    _run_exactness_case(cusparse_stream_stress_artifact, cusparse_stream_stress_case, runtime_k=runtime_k, mode=mode, order=order)
    clear_cupy_state()


@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("requested_ring_buffer_size", [1, 2], ids=["ring1", "ring2"])
def test_cusparse_small_stream_copy_overlaps_compute(tmp_path, order, requested_ring_buffer_size, monkeypatch):
    _run_overlap_case(tmp_path, order, requested_ring_buffer_size, monkeypatch, n=64, bandwidth=8, runtime_k=1)


@pytest.mark.stress
@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("requested_ring_buffer_size", [1, 2], ids=["ring1", "ring2"])
def test_cusparse_stream_copy_overlaps_compute(tmp_path, order, requested_ring_buffer_size, monkeypatch):
    _run_overlap_case(tmp_path, order, requested_ring_buffer_size, monkeypatch, n=4096, bandwidth=64, runtime_k=2)


def test_cusparse_prepare_cuda_graph_capture_resident(primary_artifact):
    torch = pytest.importorskip("torch")
    with torch.cuda.device(0):
        capture_stream = torch.cuda.Stream()
    layout = build_cusparse_layout([primary_artifact], requirements=_requirements(2), stream=capture_stream)
    ref_layout = build_reference_layout([primary_artifact], requirements=_requirements(2))
    with ReferenceRuntime(ref_layout) as ref_runtime:
        (ref_grg,) = ref_runtime.grgs
        rng = np.random.default_rng(98_201)
        order = ("down", "up")
        src_np = rng.standard_normal((2, _source_cols(ref_grg, order[0])), dtype=DATA_DTYPE)
        init_first_np = rng.standard_normal((2,), dtype=DATA_DTYPE)
        init_second_np = rng.standard_normal((2,), dtype=DATA_DTYPE)
        expected = _expected_prepared_chain(ref_grg, order, src_np, init_first_np, init_second_np)
    with CusparseRuntime(layout) as runtime, ExitStack() as stack:
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
        cp.cuda.runtime.deviceSynchronize()
        actual = second.output.cpu().numpy().copy()
    np.testing.assert_allclose(actual, expected, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])


def test_cusparse_prepare_cuda_graph_capture_streamed(cusparse_small_stream_artifact):
    torch = pytest.importorskip("torch")
    fixed_bytes, block_bytes, owner_block_count = _full_budget_components(cusparse_small_stream_artifact, runtime_k=1)
    assert owner_block_count == 3
    with torch.cuda.device(0):
        capture_stream = torch.cuda.Stream()
    layout = _build_layout(
        cusparse_small_stream_artifact,
        runtime_k=1,
        ring_buffer_size=2,
        budget_bytes=fixed_bytes + 2 * block_bytes,
        allow_residency=False,
        stream=capture_stream,
    )
    ref_layout = build_reference_layout([cusparse_small_stream_artifact], requirements=_requirements(1))
    with ReferenceRuntime(ref_layout) as ref_runtime:
        (ref_grg,) = ref_runtime.grgs
        order = ("down", "up")
        src_np = np.arange(_source_cols(ref_grg, order[0]), dtype=DATA_DTYPE).reshape(1, -1)
        init_first_np = np.asarray([0.5], dtype=DATA_DTYPE)
        init_second_np = np.asarray([1.5], dtype=DATA_DTYPE)
        expected = _expected_prepared_chain(ref_grg, order, src_np, init_first_np, init_second_np)
    with CusparseRuntime(layout) as runtime, ExitStack() as stack:
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
        cp.cuda.runtime.deviceSynchronize()
        actual = second.output.cpu().numpy().copy()
    np.testing.assert_allclose(actual, expected, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])


@pytest.mark.parametrize("streamed", [False, True], ids=["resident", "streamed"])
def test_cusparse_cuda_graph_capture_multi_grg_single_graph_mixed_heights(tmp_path, streamed):
    torch = pytest.importorskip("torch")
    artifacts = _mixed_height_graph_artifacts(tmp_path)
    cases = _multi_grg_graph_cases(artifacts)
    with torch.cuda.device(0):
        capture_stream = torch.cuda.Stream()
    layout = _build_multi_grg_graph_layout(artifacts, streamed=streamed, stream=capture_stream)
    with CusparseRuntime(layout) as runtime, ExitStack() as stack:
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
        cp.cuda.runtime.deviceSynchronize()
        actual = [result.cpu().numpy().copy() for result in results]
    for case, value in zip(cases, actual, strict=True):
        np.testing.assert_allclose(value, case.expected, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])


@pytest.mark.parametrize("streamed", [False, True], ids=["resident", "streamed"])
def test_cusparse_cuda_graph_capture_multi_grg_sequential_mixed_heights(tmp_path, streamed):
    torch = pytest.importorskip("torch")
    artifacts = _mixed_height_graph_artifacts(tmp_path)
    cases = _multi_grg_graph_cases(artifacts)
    with torch.cuda.device(0):
        capture_stream = torch.cuda.Stream()
    layout = _build_multi_grg_graph_layout(artifacts, streamed=streamed, stream=capture_stream)
    with CusparseRuntime(layout) as runtime, ExitStack() as stack:
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
            cp.cuda.runtime.deviceSynchronize()
            actual.append(final.output.cpu().numpy().copy())
    for case, value in zip(cases, actual, strict=True):
        np.testing.assert_allclose(value, case.expected, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])
