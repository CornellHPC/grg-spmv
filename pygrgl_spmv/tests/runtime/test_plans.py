from __future__ import annotations

import ctypes

import pygrgl_spmv.backends._cuda_stream as cuda_stream_mod
import pytest

from pygrgl_spmv.backends._cuda_stream import parse_cuda_device, parse_cuda_stream
from pygrgl_spmv.backends.cusparse.plan import CusparsePlan, CusparsePlanPair, DenseOrder, Operation, SpMMAlgorithm
from pygrgl_spmv.backends.mkl import MklPlan, MklPlanPair
from pygrgl_spmv.backends.reference import ReferencePlan, ReferencePlanPair
from pygrgl_spmv.backends.triton.plan import TritonPlan, TritonPlanPair
from pygrgl_spmv.backends.types import SparseFormat, StoredMatrix


class _ProtocolStream:
    def __init__(self, token) -> None:
        self._token = token

    def __cuda_stream__(self):
        return self._token


class _FakeCudaRuntimeLib:
    def __init__(
        self,
        *,
        device_count: int = 2,
        device_count_status: int = 0,
        stream_devices: dict[int, int] | None = None,
        stream_statuses: dict[int, int] | None = None,
    ) -> None:
        self.device_count = int(device_count)
        self.device_count_status = int(device_count_status)
        self.stream_devices = {} if stream_devices is None else dict(stream_devices)
        self.stream_statuses = {} if stream_statuses is None else dict(stream_statuses)

    def cudaGetErrorString(self, status):
        code = int(getattr(status, "value", status))
        return f"cuda-status-{code}".encode("utf-8")

    def cudaGetDeviceCount(self, count_ptr):
        if self.device_count_status == 0:
            ctypes.cast(count_ptr, ctypes.POINTER(ctypes.c_int))[0] = self.device_count
        return self.device_count_status

    def cudaStreamGetDevice(self, stream, device_ptr):
        ptr = int(0 if getattr(stream, "value", None) is None else stream.value)
        status = int(self.stream_statuses.get(ptr, 0))
        if status == 0:
            ctypes.cast(device_ptr, ctypes.POINTER(ctypes.c_int))[0] = int(self.stream_devices.get(ptr, 0))
        return status


def _install_fake_cuda_runtime(monkeypatch, fake: _FakeCudaRuntimeLib) -> None:
    cuda_stream_mod._cuda_runtime.cache_clear()
    monkeypatch.setattr(cuda_stream_mod, "_load_cuda_runtime_library", lambda: fake)


def test_parse_cuda_stream_accepts_raw_null_handle():
    ptr, owner = parse_cuda_stream(0)
    assert ptr == 0
    assert owner is None


def test_parse_cuda_stream_accepts_protocol_pair():
    stream = _ProtocolStream((0, 1234))
    ptr, owner = parse_cuda_stream(stream)
    assert ptr == 1234
    assert owner is stream


def test_parse_cuda_device_accepts_visible_ordinal(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch, _FakeCudaRuntimeLib(device_count=3))
    try:
        assert parse_cuda_device(2) == 2
        assert cuda_stream_mod._cuda_device_count() == 3
    finally:
        cuda_stream_mod._cuda_runtime.cache_clear()


@pytest.mark.parametrize(
    ("device", "match"),
    [
        pytest.param(True, "non-bool int ordinal", id="bool"),
        pytest.param("0", "non-bool int ordinal", id="string"),
        pytest.param(-1, "non-negative", id="negative"),
        pytest.param(3, "out of range", id="too-large"),
    ],
)
def test_parse_cuda_device_rejects_invalid_inputs(monkeypatch, device, match):
    _install_fake_cuda_runtime(monkeypatch, _FakeCudaRuntimeLib(device_count=3))
    try:
        with pytest.raises((TypeError, ValueError), match=match):
            parse_cuda_device(device)
    finally:
        cuda_stream_mod._cuda_runtime.cache_clear()


def test_parse_cuda_device_raises_on_runtime_query_failure(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch, _FakeCudaRuntimeLib(device_count_status=17))
    try:
        with pytest.raises(RuntimeError, match="cudaGetDeviceCount"):
            parse_cuda_device(0)
    finally:
        cuda_stream_mod._cuda_runtime.cache_clear()


def test_cuda_stream_device_uses_runtime_query(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch, _FakeCudaRuntimeLib(stream_devices={1234: 1}))
    try:
        assert cuda_stream_mod._cuda_stream_device(1234) == 1
    finally:
        cuda_stream_mod._cuda_runtime.cache_clear()


def test_cuda_stream_device_rejects_null_handle():
    with pytest.raises(ValueError, match="non-null"):
        cuda_stream_mod._cuda_stream_device(0)


def test_cuda_stream_device_raises_on_runtime_query_failure(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch, _FakeCudaRuntimeLib(stream_statuses={1234: 17}))
    try:
        with pytest.raises(RuntimeError, match="cudaStreamGetDevice"):
            cuda_stream_mod._cuda_stream_device(1234)
    finally:
        cuda_stream_mod._cuda_runtime.cache_clear()


@pytest.mark.parametrize(
    ("stream", "match"),
    [
        pytest.param(True, "non-bool int handle", id="bool"),
        pytest.param(-1, "non-negative", id="negative-int"),
        pytest.param(object(), "__cuda_stream__", id="plain-object"),
        pytest.param(_ProtocolStream((1, 5)), "version 1", id="bad-version"),
        pytest.param(_ProtocolStream(("x", 5)), "invalid literal|base 10", id="bad-version-type"),
    ],
)
def test_parse_cuda_stream_rejects_invalid_inputs(stream, match):
    with pytest.raises((TypeError, ValueError), match=match):
        parse_cuda_stream(stream)


def test_reference_plan_storage_sharing_and_pair_validation():
    up = ReferencePlan(store="N", fmt="CSR")
    down_same = ReferencePlan(store="N", fmt="CSR")
    down_transpose = ReferencePlan(store="T", fmt="CSC")
    assert up.can_share_storage_with(down_same)
    assert up.can_share_storage_with(down_transpose)
    with pytest.raises(ValueError, match="at least one"):
        ReferencePlanPair(plan_up=None, plan_down=None)


def test_mkl_plan_from_dict_and_storage_sharing():
    up = MklPlan.from_dict({"store": "N", "fmt": "CSR", "n_threads": 4, "optimize": False})
    down_same = MklPlan.from_dict({"store": "N", "fmt": "CSR", "n_threads": 4})
    down_transpose = MklPlan.from_dict({"store": "T", "fmt": "CSC", "n_threads": 4})
    assert not up.optimize
    assert up.can_share_storage_with(down_same)
    assert up.can_share_storage_with(down_transpose)
    with pytest.raises(ValueError, match="unknown MklPlan field"):
        MklPlan.from_dict({"store": "N", "fmt": "CSR", "n_thread": 4})
    with pytest.raises(ValueError, match="at least one"):
        MklPlanPair(plan_up=None, plan_down=None)


def test_triton_plan_scratch_normalization():
    plan = TritonPlan.from_dict({"store": "N", "fmt": "CSR", "scratch": "3|1|2"})
    assert plan.store == StoredMatrix.N
    assert plan.fmt == SparseFormat.CSR
    assert plan.scratch == "1|2|3"


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        pytest.param({"store": "N", "fmt": "COO"}, "CSR/CSC", id="coo-format"),
        pytest.param({"store": "N", "fmt": "CSR", "scratch": "1||2"}, "invalid Triton scratch", id="bad-scratch-empty"),
        pytest.param({"store": "N", "fmt": "CSR", "scratch": "1|1"}, "duplicate Triton scratch", id="bad-scratch-dup"),
    ],
)
def test_triton_plan_rejects_invalid_values(raw, match):
    with pytest.raises(ValueError, match=match):
        TritonPlan.from_dict(raw)


def test_triton_plan_pair_validates_store_direction():
    with pytest.raises(ValueError, match="plan_up expects store=N"):
        TritonPlanPair.from_dicts({"store": "T", "fmt": "CSC"}, None)
    with pytest.raises(ValueError, match="plan_down expects store=T"):
        TritonPlanPair.from_dicts(None, {"store": "N", "fmt": "CSR"})


def test_cusparse_plan_parse_and_properties():
    plan = CusparsePlan.from_dict(
        {
            "store": "N",
            "fmt": "CSR",
            "opA": "N",
            "opB": "N",
            "orderB": "ROW",
            "orderC": "ROW",
            "algo": "DEFAULT",
            "scratch": "3|1|2",
        }
    )
    assert plan.store == StoredMatrix.N
    assert plan.fmt == SparseFormat.CSR
    assert plan.op_a == Operation.N
    assert plan.op_b == Operation.N
    assert plan.order_b == DenseOrder.ROW
    assert plan.order_c == DenseOrder.ROW
    assert plan.algo == SpMMAlgorithm.DEFAULT
    assert plan.supported
    assert str(getattr(plan.direction, "name", plan.direction)).lower() == "up"
    assert plan.scratch == "1|2|3"


@pytest.mark.parametrize(
    ("mapping", "match"),
    [
        pytest.param(
            {
                "store": "N",
                "fmt": "CSR",
                "opA": "N",
                "opB": "N",
                "orderB": "ROW",
                "orderC": "ROW",
                "algo": "DEFAULT",
                "scratch": "1||2",
            },
            "invalid cuSPARSE scratch",
            id="bad-scratch",
        ),
        pytest.param(
            {
                "store": "N",
                "fmt": "CSC",
                "opA": "N",
                "opB": "N",
                "orderB": "ROW",
                "orderC": "ROW",
                "algo": "CSR_ALG3",
            },
            "unsupported cuSPARSE plan",
            id="bad-format-algo",
        ),
        pytest.param(
            {
                "store": "N",
                "fmt": "CSR",
                "opA": "N",
                "opB": "N",
                "orderB": "ROW",
                "orderC": "ROW",
                "algo": "DEFAULT",
                "extra": True,
            },
            "unknown CusparsePlan field",
            id="unknown-field",
        ),
    ],
)
def test_cusparse_plan_rejects_invalid_values(mapping, match):
    with pytest.raises(ValueError, match=match):
        CusparsePlan.from_dict(mapping)


def test_cusparse_plan_storage_sharing_rules():
    up = CusparsePlan(store="N", fmt="CSR", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT")
    down = CusparsePlan(store="T", fmt="CSC", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT")
    assert up.can_share_storage_with(down)

    up_coo = CusparsePlan(store="N", fmt="COO", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT")
    down_coo = CusparsePlan(store="T", fmt="COO", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT")
    assert not up_coo.can_share_storage_with(down_coo)


def test_cusparse_plan_pair_rejects_direction_mismatch():
    with pytest.raises(ValueError, match="plan_up expects a UP plan"):
        CusparsePlanPair(
            plan_up=CusparsePlan(store="T", fmt="CSC", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"),
            plan_down=None,
        )
    with pytest.raises(ValueError, match="plan_down expects a DOWN plan"):
        CusparsePlanPair(
            plan_up=None,
            plan_down=CusparsePlan(store="N", fmt="CSR", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT"),
        )
