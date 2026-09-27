"""Monitoring / deallocation tests."""

from trade_allocate.monitor import check_strategy, deallocate


def test_healthy_keeps_weight():
    out = deallocate([0.5, 0.5], ["A", "B"],
                     recent={"A": {"sharpe": 1.2, "max_drawdown": -0.05},
                             "B": {"sharpe": 0.8, "max_drawdown": -0.08}})
    assert out["weights"] == [0.5, 0.5]
    assert out["deallocated"] == []
    assert not out["forced"]


def test_retired_zeroed_with_reason():
    out = deallocate([0.5, 0.5], ["A", "B"], lifecycle={"A": "RETIRED"})
    assert out["weights"] == [0.0, 1.0]
    assert out["deallocated"] == ["A"]
    assert out["forced"]
    reasons = out["checks"][0]["reasons"]
    assert any("retired" in r for r in reasons)


def test_paused_zeroed():
    out = deallocate([0.5, 0.5], ["A", "B"], lifecycle={"B": "paused"})
    assert out["weights"] == [1.0, 0.0]
    assert out["deallocated"] == ["B"]


def test_recent_decay_breaches():
    out = deallocate([0.4, 0.6], ["A", "B"],
                     recent={"A": {"sharpe": -0.5, "max_drawdown": -0.05}})
    assert out["deallocated"] == ["A"]
    assert out["weights"] == [0.0, 1.0]


def test_recent_drawdown_breach():
    out = deallocate([0.4, 0.6], ["A", "B"],
                     recent={"B": {"sharpe": 0.5, "max_drawdown": -0.35}},
                     max_recent_drawdown=-0.20)
    assert out["deallocated"] == ["B"]


def test_unmonitored_not_a_breach():
    out = deallocate([0.5, 0.5], ["A", "B"])  # no monitoring data at all
    assert out["deallocated"] == []
    assert out["checks"][0]["status"] == "unmonitored"


def test_never_reallocates_into_breacher():
    # Even if the target wanted 90% in A, a breached A gets 0.
    out = deallocate([0.9, 0.1], ["A", "B"], lifecycle={"A": "RETIRED"})
    assert out["weights"][0] == 0.0
    assert out["weights"][1] == 1.0


def test_all_breached_flag():
    out = deallocate([0.5, 0.5], ["A", "B"],
                     lifecycle={"A": "RETIRED", "B": "RETIRED"})
    assert out["all_breached"]
    assert out["weights"] == [0.0, 0.0]


def test_check_strategy_statuses():
    assert check_strategy("A")["status"] == "unmonitored"
    ok = check_strategy("A", recent={"sharpe": 1.0})
    assert ok["status"] == "healthy" and not ok["breached"]
    bad = check_strategy("A", recent={"sharpe": -1.0})
    assert bad["breached"] and bad["status"] == "breached"


def test_malformed_recent_recorded_not_crash():
    out = check_strategy("A", recent="nonsense")
    assert out["breached"]
    assert "recent_malformed" in out["reasons"]
