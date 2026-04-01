"""Plan-object tests for cuSPARSE and MKL backends."""

from __future__ import annotations

import ctypes

import pygrgl_spmv.backends._cuda_stream as cuda_stream_mod
import pytest

from pygrgl_spmv.backends._cuda_stream import parse_cuda_device, parse_cuda_stream
from pygrgl_spmv.backends import ReferenceBackend, ReferencePlanPair
from pygrgl_spmv.backends.mkl import MklBackend, MklPlan, MklPlanPair
from pygrgl_spmv.backends.types import SparseFormat, StoredMatrix
from scripts.bench.configs import parse_plan_pair_literal


def test_mkl_plan_string_literal():
    plan = MklPlan.from_dict({"store": "N", "fmt": "CSR", "n_threads": 4, "k_hint": None})
    assert str(plan) == "[k_hint=none,store=N,fmt=CSR,n_threads=4]"


def test_reference_plan_factory():
    plan = ReferenceBackend.plan(fmt="CSR", store="N", k_hint=4)
    assert plan.fmt == SparseFormat.CSR
    assert plan.store == StoredMatrix.N
    assert plan.k_hint == 4


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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(None, None, id="none"),
        pytest.param("none", None, id="string-none"),
        pytest.param(" None ", None, id="string-none-spaced"),
        pytest.param(1, 1, id="int"),
        pytest.param("4", 4, id="string-int"),
    ],
)
def test_shared_k_hint_parsing(raw, expected):
    reference = ReferenceBackend.plan(fmt="CSR", store="N", k_hint=raw)
    mkl = MklPlan.from_dict({"store": "N", "fmt": "CSR", "n_threads": 4, "k_hint": raw})
    assert reference.k_hint == expected
    assert mkl.k_hint == expected


@pytest.mark.parametrize("raw", [0, -1, "0", "-3"])
def test_shared_k_hint_rejects_non_positive_values(raw):
    with pytest.raises(ValueError, match="k_hint must be positive or none"):
        ReferenceBackend.plan(fmt="CSR", store="N", k_hint=raw)
    with pytest.raises(ValueError, match="k_hint must be positive or none"):
        MklPlan.from_dict({"store": "N", "fmt": "CSR", "n_threads": 4, "k_hint": raw})


def test_mkl_plan_from_dict_and_storage_sharing():
    up = MklPlan.from_dict({"store": "N", "fmt": "CSR", "n_threads": 4, "k_hint": None})
    down_same = MklPlan.from_dict({"store": "N", "fmt": "CSR", "n_threads": 4, "k_hint": None})
    down_transpose = MklPlan.from_dict({"store": "T", "fmt": "CSC", "n_threads": 4, "k_hint": None})
    assert up.can_share_storage_with(down_same)
    assert up.can_share_storage_with(down_transpose)


def test_mkl_plan_from_dict_rejects_unknown_dict_fields():
    with pytest.raises(ValueError, match=r"Unknown MklPlan field\(s\): \['n_thread'\]"):
        MklPlan.from_dict({"store": "N", "fmt": "CSR", "n_thread": 4})

    with pytest.raises(ValueError, match=r"Unknown MklPlan field\(s\): \['fmt_up', 'old_key'\]"):
        MklPlan.from_dict({"store": "N", "fmt": "CSR", "fmt_up": "CSC", "old_key": 1})


def test_mkl_plan_pair_rejects_both_missing():
    with pytest.raises(ValueError, match="At least one of plan_up/plan_down"):
        MklPlanPair(plan_up=None, plan_down=None)


def test_mkl_backend_constructs_from_pair():
    backend = MklBackend(
        pair=MklPlanPair.from_dicts(
            {"store": "N", "fmt": "CSR", "n_threads": 1, "k_hint": None},
            {"store": "T", "fmt": "CSC", "n_threads": 1, "k_hint": None},
        ),
        log_level="WARNING",
    )
    assert backend._plan_up is not None
    assert backend._plan_down is not None


def test_parse_plan_pair_literal_allows_empty_sides():
    left, right = parse_plan_pair_literal("[][k_hint=none,store=T,fmt=CSC,n_threads=1]")
    assert left is None
    assert right == {"k_hint": "none", "store": "T", "fmt": "CSC", "n_threads": "1"}

    left, right = parse_plan_pair_literal("[k_hint=none,store=N,fmt=CSR,n_threads=1][]")
    assert left == {"k_hint": "none", "store": "N", "fmt": "CSR", "n_threads": "1"}
    assert right is None

    with pytest.raises(ValueError, match="both sides cannot be empty"):
        parse_plan_pair_literal("[][]")


def test_triton_plan_parse_and_storage_sharing():
    from pygrgl_spmv.backends.triton import TritonPlan

    up = TritonPlan.from_dict({"k_hint": 1, "store": "N", "fmt": "CSR"})
    down = TritonPlan.from_dict({"k_hint": 1, "store": "T", "fmt": "CSC"})
    literal = str(up)
    assert literal == "[k_hint=1,store=N,fmt=CSR,scratch=none]"
    assert up.k_hint == 1
    assert up.store == StoredMatrix.N
    assert up.fmt == SparseFormat.CSR
    assert up.scratch == "none"
    assert up.can_share_storage_with(down)


def test_triton_plan_accepts_none_hint():
    from pygrgl_spmv.backends.triton import TritonPlan

    plan = TritonPlan.from_dict({"k_hint": None, "store": "N", "fmt": "CSR"})
    assert plan.k_hint is None
    assert str(plan) == "[k_hint=none,store=N,fmt=CSR,scratch=none]"


def test_triton_plan_scratch_normalization():
    from pygrgl_spmv.backends.triton import TritonPlan

    plan = TritonPlan.from_dict({"k_hint": 1, "store": "N", "fmt": "CSR", "scratch": "3|1|2"})
    assert plan is not None
    assert plan.scratch == "1|2|3"
    assert str(plan) == "[k_hint=1,store=N,fmt=CSR,scratch=1|2|3]"


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        pytest.param({"k_hint": 4, "store": "N", "fmt": "CSR"}, "k_hint=none or 1", id="wrong-hint"),
        pytest.param({"k_hint": 1, "store": "N", "fmt": "COO"}, "CSR/CSC", id="coo-format"),
        pytest.param({"k_hint": 1, "store": "N", "fmt": "CSR", "algo": "naive"}, "Unknown TritonPlan field", id="unknown-field"),
        pytest.param({"k_hint": 1, "store": "N", "fmt": "CSR", "scratch": "1||2"}, "Invalid Triton scratch", id="bad-scratch-empty"),
        pytest.param({"k_hint": 1, "store": "N", "fmt": "CSR", "scratch": "1|1"}, "Duplicate Triton scratch level", id="bad-scratch-dup"),
    ],
)
def test_triton_plan_rejects_invalid_values(raw, match):
    from pygrgl_spmv.backends.triton import TritonPlan

    with pytest.raises(ValueError, match=match):
        TritonPlan.from_dict(raw)


def test_triton_plan_pair_rejects_both_missing():
    from pygrgl_spmv.backends.triton import TritonPlanPair

    with pytest.raises(ValueError, match="At least one of plan_up/plan_down"):
        TritonPlanPair(plan_up=None, plan_down=None)


def test_triton_plan_pair_validates_store_direction():
    from pygrgl_spmv.backends.triton import TritonPlanPair

    with pytest.raises(ValueError, match="plan_up expects store=N"):
        TritonPlanPair.from_dicts({"k_hint": 1, "store": "T", "fmt": "CSC"}, None)
    with pytest.raises(ValueError, match="plan_down expects store=T"):
        TritonPlanPair.from_dicts(None, {"k_hint": 1, "store": "N", "fmt": "CSR"})


def test_cusparse_plan_parse_and_properties():
    from pygrgl_spmv.backends.cusparse import CusparsePlan, DenseOrder, Operation, SpMMAlgorithm

    literal = "[k_hint=none,store=N,fmt=CSR,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT,scratch=none]"
    plan = CusparsePlan.from_literal(literal)
    assert str(plan) == literal
    assert plan.k_hint is None
    assert plan.store == StoredMatrix.N
    assert plan.fmt == SparseFormat.CSR
    assert plan.op_a == Operation.N
    assert plan.op_b == Operation.N
    assert plan.order_b == DenseOrder.ROW
    assert plan.order_c == DenseOrder.ROW
    assert plan.algo == SpMMAlgorithm.DEFAULT
    assert plan.supported
    assert not plan.deterministic
    assert plan.need_buffer
    assert plan.need_preprocess
    assert plan.scratch == "none"


def test_cusparse_plan_scratch_normalization():
    from pygrgl_spmv.backends.cusparse import CusparsePlan

    plan = CusparsePlan.from_dict(
        {
            "k_hint": 4,
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
    assert plan.scratch == "1|2|3"
    assert str(plan) == "[k_hint=4,store=N,fmt=CSR,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT,scratch=1|2|3]"


def test_cusparse_plan_scratch_all_round_trip():
    from pygrgl_spmv.backends.cusparse import CusparsePlan

    literal = "[k_hint=none,store=N,fmt=CSR,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT,scratch=all]"
    plan = CusparsePlan.from_literal(literal)
    assert str(plan) == literal
    assert plan.scratch == "all"


@pytest.mark.parametrize(
    ("scratch", "match"),
    [
        pytest.param("1||2", "Invalid cuSPARSE scratch", id="empty-piece"),
        pytest.param("1|1", "Duplicate cuSPARSE scratch level", id="duplicate"),
        pytest.param("-1", "must be non-negative", id="negative"),
        pytest.param("*", "does not support wildcard/negation", id="wildcard"),
        pytest.param("!1", "does not support wildcard/negation", id="negation"),
    ],
)
def test_cusparse_plan_rejects_invalid_scratch(scratch, match):
    from pygrgl_spmv.backends.cusparse import CusparsePlan

    with pytest.raises(ValueError, match=match):
        CusparsePlan.from_dict(
            {
                "k_hint": None,
                "store": "N",
                "fmt": "CSR",
                "opA": "N",
                "opB": "N",
                "orderB": "ROW",
                "orderC": "ROW",
                "algo": "DEFAULT",
                "scratch": scratch,
            }
        )


def test_cusparse_plan_string_round_trip_rejects_cuda_field():
    from pygrgl_spmv.backends.cusparse import CusparsePlan

    literal = "[k_hint=2,store=T,fmt=CSC,opA=N,opB=T,orderB=COL,orderC=ROW,algo=DEFAULT,scratch=none]"
    plan = CusparsePlan.from_literal(literal)
    assert str(plan) == literal
    assert CusparsePlan.from_literal(str(plan)) == plan

    with pytest.raises(ValueError, match="Unknown plan field"):
        CusparsePlan.from_literal(
            "[cuda=12.9.0,k_hint=2,store=T,fmt=CSC,opA=N,opB=T,orderB=COL,orderC=ROW,algo=DEFAULT,scratch=none]"
        )


def test_cusparse_plan_construction_is_runtime_pure(monkeypatch):
    from pygrgl_spmv.backends.cusparse import CusparsePlan, is_valid_combo
    from pygrgl_spmv.backends.cusparse import plan as cusparse_plan

    def fail_runtime_probe():
        raise AssertionError("_runtime_cuda_version should not be called")

    monkeypatch.setattr(cusparse_plan, "_runtime_cuda_version", fail_runtime_probe)

    plan = CusparsePlan.from_literal(
        "[k_hint=none,store=N,fmt=CSR,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT,scratch=none]"
    )
    plans = CusparsePlan.expand_literal(
        "[k_hint=1,store=*,fmt=CSR,opA=*,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT,scratch=none]"
    )

    assert plan.supported
    assert plans
    assert is_valid_combo("csr", False, "default")
    assert not is_valid_combo("csc", False, "csr_alg3")


def test_cusparse_plan_expand_literal_and_storage_sharing():
    from pygrgl_spmv.backends.cusparse import CusparsePlan

    plans = CusparsePlan.expand_literal(
        "[k_hint=1,store=*,fmt=CSR,opA=*,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT,scratch=none]"
    )
    assert plans

    up = CusparsePlan.from_literal(
        "[k_hint=none,store=N,fmt=CSR,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT,scratch=none]"
    )
    down = CusparsePlan.from_literal(
        "[k_hint=none,store=N,fmt=CSR,opA=T,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT,scratch=none]"
    )
    assert up.can_share_storage_with(down)

    transpose_down = CusparsePlan.from_literal(
        "[k_hint=none,store=T,fmt=CSC,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT,scratch=none]"
    )
    assert up.can_share_storage_with(transpose_down)


def test_cusparse_plan_coo_transpose_storage_does_not_share():
    from pygrgl_spmv.backends.cusparse import CusparsePlan

    up = CusparsePlan.from_literal(
        "[k_hint=none,store=N,fmt=COO,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT,scratch=none]"
    )
    down = CusparsePlan.from_literal(
        "[k_hint=none,store=T,fmt=COO,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT,scratch=none]"
    )
    assert not up.can_share_storage_with(down)


def test_cusparse_plan_negation_expansion():
    from pygrgl_spmv.backends.cusparse import CusparsePlan, SparseFormat

    plans = CusparsePlan.expand_literal(
        "[k_hint=1,store=*,fmt=!COO,opA=*,opB=N,orderB=ROW,orderC=ROW,algo=*,scratch=none]"
    )
    assert plans
    assert all(plan.fmt in {SparseFormat.CSR, SparseFormat.CSC} for plan in plans)

    plans = CusparsePlan.expand_literal(
        "[k_hint=1,store=*,fmt=!CSR!CSC,opA=*,opB=N,orderB=ROW,orderC=ROW,algo=*,scratch=none]"
    )
    assert plans
    assert all(plan.fmt == SparseFormat.COO for plan in plans)


def test_cusparse_plan_csr_alg3_support_is_rejected_at_plan_layer():
    from pygrgl_spmv.backends.cusparse import CusparsePlan, is_valid_combo

    plan = CusparsePlan.from_literal(
        "[k_hint=none,store=N,fmt=CSC,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=CSR_ALG3,scratch=none]"
    )
    assert not plan.supported
    assert not is_valid_combo("csc", False, "csr_alg3")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param(
            {
                "pair": {
                    "plan_up": {
                        "k_hint": None,
                        "store": "T",
                        "fmt": "CSC",
                        "opA": "N",
                        "opB": "N",
                        "orderB": "ROW",
                        "orderC": "ROW",
                        "algo": "DEFAULT",
                    },
                    "plan_down": None,
                },
                "log_level": "WARNING",
            },
            r"cuSPARSE plan_up expects a UP plan, got DOWN",
            id="plan-up-rejects-down-plan",
        ),
        pytest.param(
            {
                "pair": {
                    "plan_up": None,
                    "plan_down": {
                        "k_hint": None,
                        "store": "N",
                        "fmt": "CSR",
                        "opA": "N",
                        "opB": "N",
                        "orderB": "ROW",
                        "orderC": "ROW",
                        "algo": "DEFAULT",
                    },
                },
                "log_level": "WARNING",
            },
            r"cuSPARSE plan_down expects a DOWN plan, got UP",
            id="plan-down-rejects-up-plan",
        ),
    ],
)
def test_cusparse_backend_rejects_slot_direction_mismatch(kwargs, match):
    from pygrgl_spmv.backends.cusparse import CusparseBackend, CusparsePlanPair

    with pytest.raises(ValueError, match=match):
        pair_dict = kwargs.pop("pair")
        CusparseBackend(
            device=0,
            stream=0,
            pair=CusparsePlanPair.from_dicts(pair_dict["plan_up"], pair_dict["plan_down"]),
            **kwargs,
        )


@pytest.mark.parametrize("runtime_version", [(12, 8, 0), (12, 9, 1)])
def test_cusparse_plan_accepts_cuda12_non_doc_versions(monkeypatch, runtime_version):
    from pygrgl_spmv.backends.cusparse import CusparsePlan
    from pygrgl_spmv.backends.cusparse import plan as cusparse_plan

    cusparse_plan._runtime_cuda_version.cache_clear()
    monkeypatch.setattr(cusparse_plan, "_probe_cuda_version", lambda: runtime_version)
    try:
        plan = CusparsePlan.from_literal(
            "[k_hint=none,store=N,fmt=CSR,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT,scratch=none]"
        )
        assert plan.cuda_version == runtime_version
    finally:
        cusparse_plan._runtime_cuda_version.cache_clear()


@pytest.mark.parametrize("runtime_version", [(11, 8, 0), (13, 0, 0)])
def test_cusparse_plan_rejects_non_cuda12_runtime(monkeypatch, runtime_version):
    from pygrgl_spmv.backends.cusparse import CusparsePlan
    from pygrgl_spmv.backends.cusparse import plan as cusparse_plan

    cusparse_plan._runtime_cuda_version.cache_clear()
    monkeypatch.setattr(cusparse_plan, "_probe_cuda_version", lambda: runtime_version)
    try:
        plan = CusparsePlan.from_literal(
            "[k_hint=none,store=N,fmt=CSR,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT,scratch=none]"
        )
        with pytest.raises(ValueError, match="CUDA 12.x runtime"):
            _ = plan.cuda_version
    finally:
        cusparse_plan._runtime_cuda_version.cache_clear()
