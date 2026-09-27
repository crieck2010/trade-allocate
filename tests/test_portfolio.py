"""Portfolio-level gate tests."""

import math

import pytest

from trade_allocate.portfolio import (
    PortfolioGateFailed,
    diversification_ratio,
    evaluate_portfolio,
    mix_returns,
    require_portfolio_gates,
)


def _panel():
    # Two mildly-correlated positive-drift streams, 120 obs.
    import random
    rng = random.Random(11)
    ids = ["A", "B"]
    matrix = []
    for t in range(120):
        f = rng.gauss(0, 1)
        a = 0.002 + 0.01 * (0.5 * f + 0.866 * rng.gauss(0, 1))
        b = 0.002 + 0.01 * (0.5 * f + 0.866 * rng.gauss(0, 1))
        matrix.append([a, b])
    return {"ids": ids, "matrix": matrix, "n_obs": 120, "n_strategies": 2}


def test_diversification_ratio_formula():
    # Equal weight, equal vol, uncorrelated -> DR = sqrt(2).
    # Portfolio variance = 0.25*0.01 + 0.25*0.01 = 0.005.
    dr = diversification_ratio([0.5, 0.5], [0.1, 0.1], math.sqrt(0.005))
    assert abs(dr - math.sqrt(2)) < 1e-9


def test_diversification_ratio_floor():
    # Single strategy -> no diversification.
    assert diversification_ratio([1.0], [0.1], 0.1) == 1.0


def test_mix_returns():
    panel = {"ids": ["A", "B"], "matrix": [[0.1, 0.2], [0.3, 0.4]],
             "n_obs": 2, "n_strategies": 2}
    got = mix_returns(panel, [0.5, 0.5])
    assert got == pytest.approx([0.15, 0.35])


def test_gates_pass_on_good_mix():
    panel = _panel()
    res = evaluate_portfolio(panel, [0.5, 0.5], min_sharpe=0.5,
                             max_drawdown_limit=-0.5,
                             min_diversification_ratio=1.0)
    assert res["passed"]
    assert all(g["passed"] for g in res["gates"])
    assert res["metrics"]["diversification_ratio"] > 1.0


def test_gates_fail_on_concentrated_mix():
    panel = _panel()
    res = evaluate_portfolio(panel, [1.0, 0.0], min_sharpe=0.5,
                             max_drawdown_limit=-0.5,
                             min_diversification_ratio=1.2)
    assert not res["passed"]
    failed = [g["gate"] for g in res["gates"] if not g["passed"]]
    assert "diversification_ratio" in failed


def test_require_raises():
    panel = _panel()
    with pytest.raises(PortfolioGateFailed) as e:
        require_portfolio_gates(panel, [1.0, 0.0], min_sharpe=999.0)
    assert any(f["gate"] == "portfolio_sharpe" for f in e.value.failures)


def test_gate_names_stable():
    panel = _panel()
    res = evaluate_portfolio(panel, [0.5, 0.5])
    assert [g["gate"] for g in res["gates"]] == [
        "portfolio_sharpe", "portfolio_max_drawdown", "diversification_ratio"]
