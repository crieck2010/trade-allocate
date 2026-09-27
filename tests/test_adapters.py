"""Adapter contract tests (schema versions, shapes, no fabrication)."""

import pytest

from trade_allocate.adapters import (
    IN_SCHEMA_VERSION,
    OUT_SCHEMA_VERSION,
    from_backtest_run,
    from_overfit_report,
    health_from_track_record,
    lifecycle_states,
    strategy_input,
    to_paper_mix,
    to_pm_mix,
    to_risk_limits,
)


def test_strategy_input_shape():
    inp = strategy_input("A", {"2024-01-01": 0.01}, {"verdict": "PASS"})
    assert inp["schema_version"] == IN_SCHEMA_VERSION
    assert inp["strategy_id"] == "A"
    assert inp["lifecycle_state"] is None


def test_from_backtest_run_returns():
    run = {"strategy_id": "X", "returns": {"2024-01-01": 0.01}}
    inp = from_backtest_run(run)
    assert inp["strategy_id"] == "X"
    assert inp["evidence"] is None  # never fabricated


def test_from_backtest_run_equity_curve():
    run = {"strategy_id": "X",
           "equity_curve": [("2024-01-01", 100.0), ("2024-01-02", 101.0),
                            ("2024-01-03", 100.5)]}
    inp = from_backtest_run(run)
    rets = inp["returns"]
    assert abs(rets[0][1] - 0.01) < 1e-12
    assert abs(rets[1][1] - (100.5 / 101.0 - 1.0)) < 1e-12


def test_from_backtest_run_missing_series():
    with pytest.raises(ValueError):
        from_backtest_run({"strategy_id": "X"})


def test_from_overfit_report():
    report = {"verdict": "PASS", "evaluated_at": "2026-09-20T00:00:00+00:00",
              "gates": [{"name": "g1", "value": 1.2, "threshold": 1.0,
                         "passed": True}],
              "evaluator": "trade-overfit", "n_trials": 7}
    ev = from_overfit_report(report, "TREND-VT")
    assert ev["schema_version"] == 1
    assert ev["verdict"] == "PASS"
    assert ev["strategy_id"] == "TREND-VT"
    assert ev["gates"][0]["passed"] is True


def test_health_from_track_record_passthrough():
    events = [
        {"strategy_id": "A", "recent": {"sharpe": 1.1, "max_drawdown": -0.05}},
        {"strategy_id": "B", "recent": {"sharpe": "high"}},  # non-numeric dropped
        {"no_strategy": True},
    ]
    out = health_from_track_record(events)
    assert out == {"A": {"sharpe": 1.1, "max_drawdown": -0.05}}


def test_lifecycle_states_normalizes():
    assert lifecycle_states({"A": "retired", "B": " Live "}) == {
        "A": "RETIRED", "B": "LIVE"}
    with pytest.raises(ValueError):
        lifecycle_states("nope")


def _result():
    return {"method": "risk_parity", "weights": [0.6, 0.4], "ids": ["A", "B"],
            "deallocated": [], "evidence_ages_days": {"A": 3.0},
            "portfolio_gates": {"passed": True, "metrics": {"portfolio_vol": 0.12}},
            "turnover_action": "initial", "rho_star": 0.05, "notes": []}


def test_out_payloads():
    for fn, consumer in ((to_pm_mix, "trade-agents/pm"),
                         (to_risk_limits, "trade-risk"),
                         (to_paper_mix, "trade-paper/approvals")):
        p = fn(_result())
        assert p["schema_version"] == OUT_SCHEMA_VERSION
        assert p["consumer"] == consumer
        assert p["weights"] == [0.6, 0.4]
    risk = to_risk_limits(_result())
    assert risk["max_single_weight"] == 0.6
    assert risk["portfolio_vol"] == 0.12
    pm = to_pm_mix(_result())
    assert pm["turnover_action"] == "initial"
