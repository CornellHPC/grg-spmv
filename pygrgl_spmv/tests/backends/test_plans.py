"""Plan-object tests for cuSPARSE and MKL backends."""

from __future__ import annotations

import pytest

from pygrgl_spmv.backends import ReferenceBackend
from pygrgl_spmv.backends.mkl import MklPlan
from pygrgl_spmv.backends.types import SparseFormat, StoredMatrix
from scripts.bench.configs import parse_plan_pair_literal


def test_mkl_plan_string_literal():
    plan = MklPlan.from_any({"store": "N", "fmt": "CSR", "n_threads": 4, "k_hint": None})
    assert str(plan) == "[k_hint=none,store=N,fmt=CSR,n_threads=4]"


def test_reference_plan_factory():
    plan = ReferenceBackend.plan(fmt="CSR", store="N", k_hint=4)
    assert plan.fmt == SparseFormat.CSR
    assert plan.store == StoredMatrix.N
    assert plan.k_hint == 4


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
    mkl = MklPlan.from_any({"store": "N", "fmt": "CSR", "n_threads": 4, "k_hint": raw})
    assert reference.k_hint == expected
    assert mkl.k_hint == expected


@pytest.mark.parametrize("raw", [0, -1, "0", "-3"])
def test_shared_k_hint_rejects_non_positive_values(raw):
    with pytest.raises(ValueError, match="k_hint must be positive or none"):
        ReferenceBackend.plan(fmt="CSR", store="N", k_hint=raw)
    with pytest.raises(ValueError, match="k_hint must be positive or none"):
        MklPlan.from_any({"store": "N", "fmt": "CSR", "n_threads": 4, "k_hint": raw})


def test_mkl_plan_from_any_and_storage_sharing():
    up = MklPlan.from_any({"store": "N", "fmt": "CSR", "n_threads": 4, "k_hint": None})
    down_same = MklPlan.from_any({"store": "N", "fmt": "CSR", "n_threads": 4, "k_hint": None})
    down_transpose = MklPlan.from_any({"store": "T", "fmt": "CSC", "n_threads": 4, "k_hint": None})
    assert up.can_share_storage_with(down_same)
    assert up.can_share_storage_with(down_transpose)


def test_mkl_plan_from_any_rejects_unknown_dict_fields():
    with pytest.raises(ValueError, match=r"Unknown MklPlan field\(s\): \['n_thread'\]"):
        MklPlan.from_any({"store": "N", "fmt": "CSR", "n_thread": 4})

    with pytest.raises(ValueError, match=r"Unknown MklPlan field\(s\): \['fmt_up', 'old_key'\]"):
        MklPlan.from_any({"store": "N", "fmt": "CSR", "fmt_up": "CSC", "old_key": 1})


def test_parse_plan_pair_literal_allows_empty_sides():
    left, right = parse_plan_pair_literal("[][k_hint=none,store=T,fmt=CSC,n_threads=1]")
    assert left is None
    assert right == {"k_hint": "none", "store": "T", "fmt": "CSC", "n_threads": "1"}

    left, right = parse_plan_pair_literal("[k_hint=none,store=N,fmt=CSR,n_threads=1][]")
    assert left == {"k_hint": "none", "store": "N", "fmt": "CSR", "n_threads": "1"}
    assert right is None

    with pytest.raises(ValueError, match="both sides cannot be empty"):
        parse_plan_pair_literal("[][]")


def test_plan_from_any_accepts_none():
    assert MklPlan.from_any(None) is None

    from pygrgl_spmv.backends.cusparse import CusparsePlan

    assert CusparsePlan.from_any(None) is None


def test_cusparse_plan_parse_and_properties():
    from pygrgl_spmv.backends.cusparse import CusparsePlan, DenseOrder, Operation, SpMMAlgorithm

    literal = "[k_hint=none,store=N,fmt=CSR,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT]"
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


def test_cusparse_plan_string_round_trip_rejects_cuda_field():
    from pygrgl_spmv.backends.cusparse import CusparsePlan

    literal = "[k_hint=2,store=T,fmt=CSC,opA=N,opB=T,orderB=COL,orderC=ROW,algo=DEFAULT]"
    plan = CusparsePlan.from_literal(literal)
    assert str(plan) == literal
    assert CusparsePlan.from_literal(str(plan)) == plan

    with pytest.raises(ValueError, match="Unknown plan field"):
        CusparsePlan.from_literal(
            "[cuda=12.9.0,k_hint=2,store=T,fmt=CSC,opA=N,opB=T,orderB=COL,orderC=ROW,algo=DEFAULT]"
        )


def test_cusparse_plan_construction_is_runtime_pure(monkeypatch):
    from pygrgl_spmv.backends.cusparse import CusparsePlan, is_valid_combo
    from pygrgl_spmv.backends.cusparse import plan as cusparse_plan

    def fail_runtime_probe():
        raise AssertionError("_runtime_cuda_version should not be called")

    monkeypatch.setattr(cusparse_plan, "_runtime_cuda_version", fail_runtime_probe)

    plan = CusparsePlan.from_literal(
        "[k_hint=none,store=N,fmt=CSR,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT]"
    )
    plans = CusparsePlan.expand_literal(
        "[k_hint=1,store=*,fmt=CSR,opA=*,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT]"
    )

    assert plan.supported
    assert plans
    assert is_valid_combo("csr", False, "default")
    assert not is_valid_combo("csc", False, "csr_alg3")


def test_cusparse_plan_expand_literal_and_storage_sharing():
    from pygrgl_spmv.backends.cusparse import CusparsePlan

    plans = CusparsePlan.expand_literal(
        "[k_hint=1,store=*,fmt=CSR,opA=*,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT]"
    )
    assert plans

    up = CusparsePlan.from_literal(
        "[k_hint=none,store=N,fmt=CSR,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT]"
    )
    down = CusparsePlan.from_literal(
        "[k_hint=none,store=N,fmt=CSR,opA=T,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT]"
    )
    assert up.can_share_storage_with(down)


def test_cusparse_plan_coo_transpose_storage_does_not_share():
    from pygrgl_spmv.backends.cusparse import CusparsePlan

    up = CusparsePlan.from_literal(
        "[k_hint=none,store=N,fmt=COO,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT]"
    )
    down = CusparsePlan.from_literal(
        "[k_hint=none,store=T,fmt=COO,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT]"
    )
    assert not up.can_share_storage_with(down)


def test_cusparse_plan_negation_expansion():
    from pygrgl_spmv.backends.cusparse import CusparsePlan, SparseFormat

    plans = CusparsePlan.expand_literal(
        "[k_hint=1,store=*,fmt=!COO,opA=*,opB=N,orderB=ROW,orderC=ROW,algo=*]"
    )
    assert plans
    assert all(plan.fmt in {SparseFormat.CSR, SparseFormat.CSC} for plan in plans)

    plans = CusparsePlan.expand_literal(
        "[k_hint=1,store=*,fmt=!CSR!CSC,opA=*,opB=N,orderB=ROW,orderC=ROW,algo=*]"
    )
    assert plans
    assert all(plan.fmt == SparseFormat.COO for plan in plans)


def test_cusparse_plan_csr_alg3_runtime_subset():
    from pygrgl_spmv.backends.cusparse import CusparsePlan, is_valid_combo

    plan = CusparsePlan.from_literal(
        "[k_hint=none,store=N,fmt=CSC,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=CSR_ALG3]"
    )
    assert plan.supported
    assert not is_valid_combo("csc", False, "csr_alg3")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param(
            {
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
                "log_level": "WARNING",
            },
            r"cuSPARSE plan_up expects a UP plan, got DOWN",
            id="plan-up-rejects-down-plan",
        ),
        pytest.param(
            {
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
                "log_level": "WARNING",
            },
            r"cuSPARSE plan_down expects a DOWN plan, got UP",
            id="plan-down-rejects-up-plan",
        ),
    ],
)
def test_cusparse_backend_rejects_slot_direction_mismatch(kwargs, match):
    from pygrgl_spmv.backends.cusparse import CusparseBackend

    with pytest.raises(ValueError, match=match):
        CusparseBackend(**kwargs)


@pytest.mark.parametrize("runtime_version", [(12, 8, 0), (12, 9, 1)])
def test_cusparse_plan_accepts_cuda12_non_doc_versions(monkeypatch, runtime_version):
    from pygrgl_spmv.backends.cusparse import CusparsePlan
    from pygrgl_spmv.backends.cusparse import plan as cusparse_plan

    cusparse_plan._runtime_cuda_version.cache_clear()
    monkeypatch.setattr(cusparse_plan, "_probe_cuda_version", lambda: runtime_version)
    try:
        plan = CusparsePlan.from_literal(
            "[k_hint=none,store=N,fmt=CSR,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT]"
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
            "[k_hint=none,store=N,fmt=CSR,opA=N,opB=N,orderB=ROW,orderC=ROW,algo=DEFAULT]"
        )
        with pytest.raises(ValueError, match="CUDA 12.x runtime"):
            _ = plan.cuda_version
    finally:
        cusparse_plan._runtime_cuda_version.cache_clear()
