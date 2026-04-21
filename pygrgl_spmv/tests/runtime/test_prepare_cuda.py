from __future__ import annotations

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv.backends.types import InitMode
from pygrgl_spmv.tests.conftest import DATA_DTYPE, HAS_MKL_RUNTIME, tol
from pygrgl_spmv.tests.runtime._runtime_builders import (
    build_layout_for_backend,
    full_requirements,
    runtime_cls_for_backend,
)

_GPU_BACKENDS = (
    pytest.param("triton", marks=[pytest.mark.gpu, pytest.mark.triton], id="triton"),
    pytest.param("cusparse", marks=[pytest.mark.gpu, pytest.mark.cusparse], id="cusparse"),
)
_DIRECTIONS = (
    pytest.param("up", pygrgl.TraversalDirection.UP, id="up"),
    pytest.param("down", pygrgl.TraversalDirection.DOWN, id="down"),
)
_INIT_MODES = (
    pytest.param(InitMode.NONE, id="init-none"),
    pytest.param(InitMode.VECTOR, id="init-vector"),
    pytest.param(InitMode.MATRIX, id="init-matrix"),
    pytest.param(InitMode.XTX, id="init-xtx"),
)
_REUSABLE_INIT_MODES = (
    pytest.param(InitMode.VECTOR, id="init-vector"),
    pytest.param(InitMode.MATRIX, id="init-matrix"),
)


def _input_cols(grg, *, direction: str, by_individual: bool) -> int:
    if direction == "up":
        return int(grg.num_individuals if by_individual else grg.num_samples)
    return int(grg.num_mutations)


def _prepare_init(rng: np.random.Generator, grg, *, init_mode: InitMode, k: int):
    match init_mode:
        case InitMode.NONE:
            return None
        case InitMode.VECTOR:
            return rng.standard_normal((k,), dtype=DATA_DTYPE)
        case InitMode.MATRIX:
            return rng.standard_normal((k, grg.num_nodes), dtype=DATA_DTYPE)
        case InitMode.XTX:
            return "xtx"
        case _:
            raise ValueError(f"unexpected init mode {init_mode!r}")


def _mode_seed(backend_name: str, *, direction: str, by_individual: bool, init_mode: InitMode) -> int:
    return (
        72_100
        + (1_000 if backend_name == "cusparse" else 0)
        + (100 if direction == "down" else 0)
        + (10 if by_individual else 0)
        + {InitMode.NONE: 0, InitMode.VECTOR: 1, InitMode.MATRIX: 2, InitMode.XTX: 3}[init_mode]
    )


def _run_prepared(
    runtime_cls,
    layout,
    *,
    x: np.ndarray,
    direction: str,
    init_mode: InitMode = InitMode.NONE,
    init=None,
    by_individual: bool = False,
    emit_all_nodes: bool = False,
    use_miss: bool = False,
    miss_input: np.ndarray | None = None,
):
    torch = pytest.importorskip("torch")
    with runtime_cls(layout) as runtime:
        (grg,) = runtime.grgs
        with grg.prepare_matmul_cuda(
            direction=direction,
            k=int(x.shape[0]),
            by_individual=by_individual,
            emit_all_nodes=emit_all_nodes,
            init_mode=init_mode,
            use_miss=use_miss,
        ) as op:
            op.input.copy_(torch.from_numpy(x).to(device=op.input.device))
            if miss_input is not None:
                op.miss_input.copy_(torch.from_numpy(miss_input).to(device=op.miss_input.device))
            if init_mode == InitMode.VECTOR:
                assert init is not None
                op.init_vector.copy_(torch.from_numpy(init).to(device=op.init_vector.device))
            elif init_mode == InitMode.MATRIX:
                assert init is not None
                op.init_matrix.copy_(torch.from_numpy(init).to(device=op.init_matrix.device))
            op()
            output = op.output.cpu().numpy().copy()
            miss_output = None
            if use_miss and direction == "up":
                miss_output = op.miss_output.cpu().numpy().copy()
            return output, miss_output


@pytest.mark.parametrize(
    "backend_name",
    (
        pytest.param("reference", id="reference"),
        pytest.param("mkl", id="mkl", marks=pytest.mark.mkl),
    ),
)
def test_cpu_backends_reject_prepare_matmul_cuda(primary_artifact, backend_name):
    if backend_name == "mkl" and not HAS_MKL_RUNTIME:
        pytest.skip("MKL runtime unavailable")
    layout = build_layout_for_backend(backend_name, [primary_artifact], requirements=full_requirements(max_k_up=2, max_k_down=2))
    runtime_cls = runtime_cls_for_backend(backend_name)
    with runtime_cls(layout) as runtime:
        (grg,) = runtime.grgs
        with pytest.raises(NotImplementedError):
            grg.prepare_matmul_cuda(direction="up", k=1)


@pytest.mark.parametrize("backend_name", _GPU_BACKENDS)
@pytest.mark.parametrize(("direction", "direction_enum"), _DIRECTIONS)
@pytest.mark.parametrize("by_individual", [False, True], ids=["by-node", "by-individual"])
@pytest.mark.parametrize("init_mode", _INIT_MODES)
def test_prepare_cuda_matches_pygrgl_modes(missing_artifact, missing_grg, backend_name, direction, direction_enum, by_individual, init_mode):
    if by_individual and missing_grg.num_individuals == missing_grg.num_samples:
        pytest.skip("fixture is not grouped by individuals")
    k = 2
    layout = build_layout_for_backend(backend_name, [missing_artifact], requirements=full_requirements(max_k_up=4, max_k_down=4))
    runtime_cls = runtime_cls_for_backend(backend_name)
    rng = np.random.default_rng(_mode_seed(backend_name, direction=direction, by_individual=by_individual, init_mode=init_mode))
    x = rng.standard_normal((k, _input_cols(missing_grg, direction=direction, by_individual=by_individual)), dtype=DATA_DTYPE)
    init = _prepare_init(rng, missing_grg, init_mode=init_mode, k=k)
    expected = np.asarray(pygrgl.matmul(missing_grg, x, direction_enum, by_individual=by_individual, init=init))
    actual, _ = _run_prepared(
        runtime_cls,
        layout,
        x=x,
        direction=direction,
        init_mode=init_mode,
        init=init,
        by_individual=by_individual,
    )
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("backend_name", _GPU_BACKENDS)
@pytest.mark.parametrize(("direction", "direction_enum"), _DIRECTIONS)
@pytest.mark.parametrize("init_mode", _REUSABLE_INIT_MODES)
def test_prepare_cuda_reuses_init_buffer(primary_artifact, primary_grg, backend_name, direction, direction_enum, init_mode):
    torch = pytest.importorskip("torch")
    k = 2
    layout = build_layout_for_backend(backend_name, [primary_artifact], requirements=full_requirements(max_k_up=4, max_k_down=4))
    runtime_cls = runtime_cls_for_backend(backend_name)
    rng = np.random.default_rng(_mode_seed(backend_name, direction=direction, by_individual=False, init_mode=init_mode) + 500)
    x0 = rng.standard_normal((k, _input_cols(primary_grg, direction=direction, by_individual=False)), dtype=DATA_DTYPE)
    x1 = rng.standard_normal(x0.shape, dtype=DATA_DTYPE)
    init = _prepare_init(rng, primary_grg, init_mode=init_mode, k=k)
    expected0 = np.asarray(pygrgl.matmul(primary_grg, x0, direction_enum, init=init))
    expected1 = np.asarray(pygrgl.matmul(primary_grg, x1, direction_enum, init=init))
    with runtime_cls(layout) as runtime:
        (grg,) = runtime.grgs
        with grg.prepare_matmul_cuda(direction=direction, k=k, init_mode=init_mode) as op:
            if init_mode == InitMode.VECTOR:
                op.init_vector.copy_(torch.from_numpy(init).to(device=op.init_vector.device))
            else:
                op.init_matrix.copy_(torch.from_numpy(init).to(device=op.init_matrix.device))
            op.input.copy_(torch.from_numpy(x0).to(device=op.input.device))
            op()
            actual0 = op.output.cpu().numpy().copy()
            op.input.copy_(torch.from_numpy(x1).to(device=op.input.device))
            op()
            actual1 = op.output.cpu().numpy().copy()
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(actual0, expected0, atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual1, expected1, atol=atol, rtol=rtol)


@pytest.mark.parametrize("backend_name", _GPU_BACKENDS)
def test_prepare_cuda_emit_all_nodes_by_individual(primary_artifact, primary_grg, backend_name):
    if primary_grg.num_individuals == primary_grg.num_samples:
        pytest.skip("fixture is not grouped by individuals")
    k = 2
    layout = build_layout_for_backend(backend_name, [primary_artifact], requirements=full_requirements(max_k_up=4, max_k_down=4))
    runtime_cls = runtime_cls_for_backend(backend_name)
    rng = np.random.default_rng(72_200)
    x = rng.standard_normal((k, primary_grg.num_individuals), dtype=DATA_DTYPE)
    expected = np.asarray(pygrgl.matmul(primary_grg, x, pygrgl.TraversalDirection.UP, by_individual=True, emit_all_nodes=True))
    actual, _ = _run_prepared(
        runtime_cls,
        layout,
        x=x,
        direction="up",
        by_individual=True,
        emit_all_nodes=True,
    )
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("backend_name", _GPU_BACKENDS)
def test_prepare_cuda_missing_up_matches_pygrgl(missing_artifact, missing_grg, backend_name):
    if missing_grg.num_individuals == missing_grg.num_samples:
        pytest.skip("fixture is not grouped by individuals")
    k = 2
    layout = build_layout_for_backend(backend_name, [missing_artifact], requirements=full_requirements(max_k_up=4, max_k_down=4))
    runtime_cls = runtime_cls_for_backend(backend_name)
    rng = np.random.default_rng(72_400)
    x = rng.standard_normal((k, missing_grg.num_individuals), dtype=DATA_DTYPE)
    miss_expected = np.zeros((k, missing_grg.num_mutations), dtype=DATA_DTYPE)
    expected = np.asarray(pygrgl.matmul(missing_grg, x, pygrgl.TraversalDirection.UP, by_individual=True, miss=miss_expected))
    actual, actual_miss = _run_prepared(
        runtime_cls,
        layout,
        x=x,
        direction="up",
        by_individual=True,
        use_miss=True,
    )
    atol, rtol = tol(DATA_DTYPE)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    assert actual_miss is not None
    np.testing.assert_allclose(actual_miss, miss_expected, atol=atol, rtol=rtol)
