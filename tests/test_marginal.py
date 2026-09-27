"""Tests for the marginal-diversification helper (Occam's Desk phase 3)."""

import math
import random

import pytest

from trade_allocate import (
    DEFAULT_EPSILON,
    DEFAULT_RHO_MAX,
    MIN_OVERLAP_DAYS,
    marginal_contribution,
)


def series(n=260, mean=0.0008, vol=0.01, seed=1):
    rng = random.Random(seed)
    rets = [rng.gauss(mean, vol) for _ in range(n)]
    dates = [f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}#{i:04d}" for i in range(n)]
    return {"dates": dates, "returns": rets}


def test_diversifier_is_admitted():
    book = {"base": series(seed=1, mean=0.0008)}
    # uncorrelated cousin-free edge with its own drift
    cand = series(seed=999, mean=0.0008)
    rep = marginal_contribution(book, "div", cand)
    assert rep["n_overlap"] == 260
    assert rep["method"] == "risk_parity"
    assert rep["max_correlation"] < 0.6
    assert rep["delta_sharpe"] > DEFAULT_EPSILON
    assert rep["admitted"] is True
    assert rep["reasons"] == []
    assert math.isclose(sum(rep["weights_with"]), 1.0, rel_tol=1e-9)
    assert rep["ids_with"] == ["base", "div"]


def test_correlated_cousin_is_rejected():
    rng = random.Random(1234)  # independent stream: true ~0.999 cousin
    base = series(seed=7, mean=0.0008)
    # cousin: same signal plus a whisper of noise -> rho ~= 0.99
    cousin_rets = [r + rng.gauss(0, 0.0005) for r in base["returns"]]
    cousin = {"dates": base["dates"], "returns": cousin_rets}
    rep = marginal_contribution({"base": base}, "cousin", cousin)
    assert rep["max_correlation"] > 0.9
    assert rep["admitted"] is False
    assert any("rho_max" in r for r in rep["reasons"])
    assert rep["correlations"]["base"] == pytest.approx(
        rep["max_correlation"])


def test_useless_candidate_rejected_on_epsilon():
    book = {"base": series(seed=1, mean=0.0008)}
    noise = series(seed=555, mean=0.0, vol=0.01)  # no edge, no correlation
    rep = marginal_contribution(book, "noise", noise)
    assert rep["max_correlation"] < DEFAULT_RHO_MAX
    assert not rep["delta_sharpe"] > DEFAULT_EPSILON
    assert rep["admitted"] is False
    assert any("epsilon" in r for r in rep["reasons"])


def test_insufficient_overlap_fails_closed():
    book = {"base": series(n=260, seed=1)}
    short = series(n=50, seed=2)
    rep = marginal_contribution(book, "short", short)
    assert rep["n_overlap"] == 50
    assert rep["n_overlap"] < MIN_OVERLAP_DAYS
    assert rep["admitted"] is False
    assert rep["delta_sharpe"] is None
    assert any("overlap" in r for r in rep["reasons"])


def test_empty_book_fails_closed():
    rep = marginal_contribution({}, "x", series(seed=1))
    assert rep["admitted"] is False
    assert rep["delta_sharpe"] is None
    assert any("empty book" in r for r in rep["reasons"])


def test_candidate_already_in_book_rejected():
    book = {"base": series(seed=1)}
    rep = marginal_contribution(book, "base", series(seed=2))
    assert rep["admitted"] is False
    assert any("already in the book" in r for r in rep["reasons"])


def test_all_input_shapes_accepted():
    s = series(seed=1, mean=0.0008)
    as_map = dict(zip(s["dates"], s["returns"]))
    as_pairs = list(zip(s["dates"], s["returns"]))
    cand = series(seed=999, mean=0.0008)
    r1 = marginal_contribution({"b": s}, "c", cand)
    r2 = marginal_contribution({"b": as_map}, "c", cand)
    r3 = marginal_contribution({"b": as_pairs}, "c", cand)
    assert r1["admitted"] and r2["admitted"] and r3["admitted"]
    assert r1["delta_sharpe"] == pytest.approx(r2["delta_sharpe"])
    assert r1["delta_sharpe"] == pytest.approx(r3["delta_sharpe"])


def test_equal_method_recorded_and_usable():
    book = {"a": series(seed=1, mean=0.0008),
            "b": series(seed=2, mean=0.0008)}
    cand = series(seed=999, mean=0.0008)
    rep = marginal_contribution(book, "c", cand, method="equal")
    assert rep["method"] == "equal"
    assert rep["admitted"] is True
    assert rep["weights_with"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_custom_policy_thresholds():
    book = {"base": series(seed=1, mean=0.0008)}
    cand = series(seed=999, mean=0.0008)
    # an impossibly strict epsilon rejects even a real diversifier
    rep = marginal_contribution(book, "div", cand, epsilon=5.0)
    assert rep["admitted"] is False
    assert any("epsilon" in r for r in rep["reasons"])
    # and a lax rho_max admits a cousin on correlation (epsilon still bites)
    rng = random.Random(1234)  # independent stream
    base = series(seed=7, mean=0.0008)
    cousin = {"dates": base["dates"],
              "returns": [r + rng.gauss(0, 0.0005) for r in base["returns"]]}
    rep2 = marginal_contribution({"base": base}, "cousin", cousin,
                                 rho_max=0.999)
    assert not any("rho_max" in r for r in rep2["reasons"])


def test_multi_member_book():
    book = {"a": series(seed=1, mean=0.0008),
            "b": series(seed=2, mean=0.0006)}
    cand = series(seed=999, mean=0.0008)
    rep = marginal_contribution(book, "c", cand)
    assert rep["n_book"] == 2
    assert set(rep["correlations"]) == {"a", "b"}
    assert rep["admitted"] is True
