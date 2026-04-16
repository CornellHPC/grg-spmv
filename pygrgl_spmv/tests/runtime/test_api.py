from __future__ import annotations

import numpy as np
import pytest

from pygrgl_spmv import ReferenceRuntime
from pygrgl_spmv.tests.runtime._runtime_builders import build_reference_layout, full_requirements


def test_reference_runtime_requires_enter_for_grgs(primary_artifact):
    runtime = ReferenceRuntime(build_reference_layout([primary_artifact]))
    with pytest.raises(RuntimeError, match="entered"):
        _ = runtime.grgs


def test_reference_runtime_exposes_ordered_grgs_and_cpu_null_stream(primary_artifact):
    layout = build_reference_layout([primary_artifact, primary_artifact])
    with ReferenceRuntime(layout) as runtime:
        assert runtime.device is None
        assert runtime.stream is None
        assert runtime.stream_ptr is None
        assert [grg.artifact_path for grg in runtime.grgs] == [primary_artifact, primary_artifact]


def test_runtime_requirements_fail_fast(primary_artifact):
    layout = build_reference_layout(
        [primary_artifact],
        requirements=full_requirements(
            max_k_up=1,
            max_k_down=1,
            need_down_miss_input=False,
            need_up_miss_output=False,
            need_init_vector=False,
            need_init_matrix=False,
            need_init_xtx=False,
        ),
    )
    with ReferenceRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        with pytest.raises(ValueError, match="max_k_up"):
            grg.matmul(np.ones((2, grg.num_samples), dtype=np.float64), "up")
        with pytest.raises(ValueError, match="UP miss output"):
            grg.matmul(
                np.ones((1, grg.num_samples), dtype=np.float64),
                "up",
                miss=np.zeros((1, grg.num_mutations), dtype=np.float64),
            )
        with pytest.raises(ValueError, match="init vector"):
            grg.matmul(
                np.ones((1, grg.num_samples), dtype=np.float64),
                "up",
                init=np.ones((1,), dtype=np.float64),
            )


def test_runtime_rejects_concurrent_calls(primary_artifact):
    with ReferenceRuntime(build_reference_layout([primary_artifact])) as runtime:
        (grg,) = runtime.grgs
        with runtime._call_scope():
            with pytest.raises(RuntimeError, match="concurrent"):
                grg.matmul(np.ones((1, grg.num_samples), dtype=np.float64), "up")


def test_runtime_reuses_owned_workspaces_across_calls(primary_artifact):
    layout = build_reference_layout([primary_artifact], requirements=full_requirements(max_k_up=4, max_k_down=4))
    with ReferenceRuntime(layout) as runtime:
        assert runtime._up_workspace is not None
        assert runtime._down_workspace is not None
        up_id = id(runtime._up_workspace)
        down_id = id(runtime._down_workspace)
        (grg,) = runtime.grgs
        rng = np.random.default_rng(41)
        _ = grg.matmul(rng.standard_normal((2, grg.num_samples), dtype=np.float64), "up")
        _ = grg.matmul(rng.standard_normal((2, grg.num_mutations), dtype=np.float64), "down")
        assert id(runtime._up_workspace) == up_id
        assert id(runtime._down_workspace) == down_id


def test_runtime_supports_multiple_grgs_from_one_runtime(primary_artifact):
    layout = build_reference_layout(
        [primary_artifact, primary_artifact],
        requirements=full_requirements(max_k_up=2, max_k_down=2),
    )
    with ReferenceRuntime(layout) as runtime:
        a, b = runtime.grgs
        rng = np.random.default_rng(99)
        x_up = rng.standard_normal((2, a.num_samples), dtype=np.float64)
        x_down = rng.standard_normal((2, b.num_mutations), dtype=np.float64)
        out_a = a.matmul(x_up, "up")
        out_b = b.matmul(x_down, "down")
        assert out_a.shape == (2, a.num_mutations)
        assert out_b.shape == (2, b.num_samples)
