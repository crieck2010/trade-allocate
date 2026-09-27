"""Turnover damping and rebalance schedule tests."""

import pytest

from trade_allocate.turnover import damp_turnover, should_rebalance


def test_initial():
    out = damp_turnover(None, [0.5, 0.5])
    assert out["action"] == "initial"
    assert out["weights"] == [0.5, 0.5]


def test_held_within_tolerance():
    out = damp_turnover([0.5, 0.5], [0.52, 0.48], tol=0.05)
    assert out["action"] == "held_within_tolerance"
    assert out["weights"] == [0.5, 0.5]
    assert abs(out["max_drift"] - 0.02) < 1e-12


def test_rebalanced_beyond_tolerance():
    out = damp_turnover([0.5, 0.5], [0.6, 0.4], tol=0.05)
    assert out["action"] == "rebalanced"
    assert out["weights"] == [0.6, 0.4]


def test_forced_always_applies():
    out = damp_turnover([0.5, 0.5], [0.51, 0.49], tol=0.05, force=True)
    assert out["action"] == "forced"
    assert out["weights"] == [0.51, 0.49]


def test_zero_tolerance_always_rebalances():
    out = damp_turnover([0.5, 0.5], [0.5001, 0.4999], tol=0.0)
    assert out["action"] == "rebalanced"


def test_length_mismatch():
    with pytest.raises(ValueError):
        damp_turnover([0.5, 0.5], [0.3, 0.3, 0.4])


def test_monthly_schedule():
    assert should_rebalance("2026-08-31", "2026-09-01", "monthly")
    assert not should_rebalance("2026-09-01", "2026-09-15", "monthly")
    assert not should_rebalance("2026-09-15", "2026-09-01", "monthly")


def test_quarterly_schedule():
    assert should_rebalance("2026-03-31", "2026-04-01", "quarterly")
    assert not should_rebalance("2026-04-01", "2026-05-01", "quarterly")
    assert should_rebalance("2026-06-30", "2026-07-01", "quarterly")


def test_weekly_schedule():
    assert should_rebalance("2026-09-01", "2026-09-08", "weekly")
    assert not should_rebalance("2026-09-01", "2026-09-07", "weekly")


def test_unknown_frequency():
    with pytest.raises(ValueError):
        should_rebalance("2026-09-01", "2026-10-01", "fortnightly")
