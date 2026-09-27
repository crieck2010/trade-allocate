"""Allocation method tests: equal weight, risk parity convergence, HRP."""

import pytest

from trade_allocate.allocate import (
    METHODS,
    allocate,
    equal_weights,
    hrp_weights,
    portfolio_variance,
    risk_contributions,
    risk_parity_weights,
)


def test_equal_weights():
    w = equal_weights(["A", "B", "C", "D"])
    assert w == [0.25] * 4
    with pytest.raises(ValueError):
        equal_weights([])


def test_risk_parity_diagonal_known_answer():
    # Uncorrelated assets: risk parity = inverse-volatility weights.
    # vols 1, 2, 3 -> weights propto 1, 1/2, 1/3.
    cov = [[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]]
    res = risk_parity_weights(cov)
    assert res["converged"]
    total = 1 + 0.5 + 1 / 3
    expected = [1 / total, 0.5 / total, (1 / 3) / total]
    for got, exp in zip(res["weights"], expected):
        assert abs(got - exp) < 1e-6
    assert res["max_rc_deviation"] < 1e-6


def test_risk_parity_equal_contributions_correlated():
    cov = [[1.0, 0.5, 0.2],
           [0.5, 1.0, 0.3],
           [0.2, 0.3, 1.0]]
    res = risk_parity_weights(cov)
    assert res["converged"]
    assert res["max_rc_deviation"] < 1e-6
    w = res["weights"]
    assert abs(sum(w) - 1.0) < 1e-12
    assert all(v > 0 for v in w)
    # Equal risk contributions, directly:
    rc = risk_contributions(cov, w)
    var = portfolio_variance(cov, w)
    for r in rc:
        assert abs(r - var / 3) / (var / 3) < 1e-6


def test_risk_parity_single_strategy():
    res = risk_parity_weights([[2.0]])
    assert res["weights"] == [1.0]


def test_risk_parity_two_asset_known_answer():
    # Two uncorrelated assets, vols 1 and 2: w = [2/3, 1/3].
    cov = [[1.0, 0.0], [0.0, 4.0]]
    res = risk_parity_weights(cov)
    assert abs(res["weights"][0] - 2 / 3) < 1e-6
    assert abs(res["weights"][1] - 1 / 3) < 1e-6


def test_hrp_block_structure():
    # Two blocks of highly-correlated assets, low between-block corr.
    # HRP should balance the blocks ~50/50 and split within blocks.
    hi, lo = 0.9, 0.1
    cov = [[1.0, hi, lo, lo],
           [hi, 1.0, lo, lo],
           [lo, lo, 1.0, hi],
           [lo, lo, hi, 1.0]]
    corr = [row[:] for row in cov]
    res = hrp_weights(cov, corr)
    w = res["weights"]
    assert abs(sum(w) - 1.0) < 1e-12
    assert all(v > 0 for v in w)
    assert abs((w[0] + w[1]) - 0.5) < 0.05  # blocks balanced
    assert abs(w[0] - w[1]) < 0.05          # symmetric within block
    assert abs(w[2] - w[3]) < 0.05


def test_hrp_single_strategy():
    res = hrp_weights([[1.0]], [[1.0]])
    assert res["weights"] == [1.0]


def test_hrp_uncorrelated_matches_intuition():
    # Uncorrelated equal-vol: HRP should be ~equal weight.
    cov = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    res = hrp_weights(cov, [row[:] for row in cov])
    for v in res["weights"]:
        assert abs(v - 1 / 3) < 1e-9


def test_allocate_dispatcher():
    cov = [[1.0, 0.2], [0.2, 1.0]]
    corr = [row[:] for row in cov]
    for method in METHODS:
        out = allocate(cov, corr, ["A", "B"], method=method)
        assert out["method"] == method
        assert out["ids"] == ["A", "B"]
        assert abs(sum(out["weights"]) - 1.0) < 1e-9
    with pytest.raises(ValueError):
        allocate(cov, corr, ["A", "B"], method="magic")


def test_allocate_empty():
    with pytest.raises(ValueError):
        allocate([], [], [], method="equal")
