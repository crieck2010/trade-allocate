"""End-to-end pipeline tests on synthetic data."""

from datetime import datetime, timezone

import pytest

from trade_allocate import adapters, demo as demo_mod, pipeline
from trade_allocate.evidence import EvidenceRefused
from trade_allocate.portfolio import PortfolioGateFailed


def _inputs(**overrides):
    streams_in = demo_mod.synthetic_streams()
    inputs = {}
    for sid, series in streams_in.items():
        ev = demo_mod.synthetic_evidence(sid)
        kw = dict(overrides.get(sid, {}))
        if "evidence" in kw:
            ev = kw.pop("evidence")
        inputs[sid] = adapters.strategy_input(sid, series, ev, **kw)
    return inputs


NOW_KW = {"evidence_kwargs": {"now": datetime(2026, 9, 26, tzinfo=timezone.utc)}}


def test_pipeline_full_run():
    res = pipeline.run(_inputs(), method="risk_parity", **NOW_KW)
    assert abs(sum(res["weights"]) - 1.0) < 1e-9
    assert res["turnover_action"] == "initial"
    assert res["deallocated"] == []
    assert res["portfolio_gates_passed"]
    assert set(res["pm_payload"]) >= {"schema_version", "consumer", "weights"}
    assert res["pm_payload"]["consumer"] == "trade-agents/pm"
    assert res["risk_payload"]["consumer"] == "trade-risk"
    assert res["paper_payload"]["consumer"] == "trade-paper/approvals"


def test_pipeline_all_methods():
    for method in ("equal", "risk_parity", "hrp"):
        res = pipeline.run(_inputs(), method=method, **NOW_KW)
        assert abs(sum(res["weights"]) - 1.0) < 1e-9, method


def test_pipeline_refuses_missing_evidence():
    inputs = _inputs()
    inputs["B"] = adapters.strategy_input("B", inputs["B"]["returns"], None)
    with pytest.raises(EvidenceRefused):
        pipeline.run(inputs, **NOW_KW)


def test_pipeline_refuses_failed_verdict():
    ev = demo_mod.synthetic_evidence("C")
    ev["verdict"] = "FAIL"
    inputs = _inputs(C={"evidence": ev})
    with pytest.raises(EvidenceRefused):
        pipeline.run(inputs, **NOW_KW)


def test_pipeline_deallocation_flows_through():
    inputs = _inputs(A={"lifecycle_state": "RETIRED"})
    res = pipeline.run(inputs, **NOW_KW)
    assert res["deallocated"] == ["A"]
    assert res["weights_by_id"]["A"] == 0.0
    # Forced: with an existing mix and a huge tolerance, the damper would
    # hold — but a deallocation is a risk event and must apply anyway.
    res2 = pipeline.run(inputs, current_weights=[0.25, 0.25, 0.25, 0.25],
                        tolerance=1.0, **NOW_KW)
    assert res2["deallocated"] == ["A"]
    assert res2["weights_by_id"]["A"] == 0.0
    assert res2["turnover_action"] == "forced"


def test_pipeline_turnover_damping():
    inputs = _inputs()
    first = pipeline.run(inputs, **NOW_KW)
    second = pipeline.run(inputs, current_weights=first["weights"],
                          tolerance=0.5, **NOW_KW)
    assert second["turnover_action"] == "held_within_tolerance"
    assert second["weights"] == first["weights"]


def test_pipeline_strict_gates_raise():
    inputs = _inputs()
    with pytest.raises(PortfolioGateFailed):
        pipeline.run(inputs, strict_portfolio_gates=True,
                     min_portfolio_sharpe=999.0, **NOW_KW)


def test_pipeline_preset():
    res = pipeline.run_preset(_inputs(), "conservative", **NOW_KW)
    assert res["method"] == "hrp"
    assert max(res["weights"]) <= 0.35 + 1e-9


def test_pipeline_json_serializable():
    import json
    res = pipeline.run(_inputs(), **NOW_KW)
    json.dumps(res, default=str)
