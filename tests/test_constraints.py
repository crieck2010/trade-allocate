"""Constraint tests: caps, dust, long-only, sum-to-one."""

import pytest

from trade_allocate.constraints import apply_constraints


def test_basic_passthrough():
    out = apply_constraints([0.25, 0.25, 0.25, 0.25], max_weight=0.5)
    assert out["weights"] == [0.25] * 4
    assert out["capped"] == [] and out["dusted"] == []
    assert not out["degenerate"]


def test_cap_water_filling():
    out = apply_constraints([0.7, 0.2, 0.1], max_weight=0.5)
    w = out["weights"]
    assert max(w) <= 0.5 + 1e-12
    assert abs(sum(w) - 1.0) < 1e-12
    assert out["capped"] == [0]
    # Overflow 0.2 redistributed proportionally over [0.2, 0.1]:
    assert abs(w[1] - (0.2 + 0.2 * 2 / 3)) < 1e-9
    assert abs(w[2] - (0.1 + 0.2 * 1 / 3)) < 1e-9


def test_cap_repeated_capping_converges():
    # Capping one strategy pushes another over the cap -> second round.
    out = apply_constraints([0.6, 0.35, 0.05], max_weight=0.4)
    w = out["weights"]
    assert max(w) <= 0.4 + 1e-12
    assert abs(sum(w) - 1.0) < 1e-12
    assert out["n_iter"] >= 2


def test_dust_zeroed_and_renormalized():
    out = apply_constraints([0.9999995, 0.0000005], max_weight=1.0, dust=1e-6)
    w = out["weights"]
    assert w[1] == 0.0
    assert w[0] == 1.0
    assert out["dusted"] == [1]


def test_negative_clipped():
    out = apply_constraints([0.6, -0.1, 0.5], max_weight=1.0)
    assert out["weights"][1] == 0.0
    assert abs(sum(out["weights"]) - 1.0) < 1e-12


def test_degenerate_all_zero():
    out = apply_constraints([0.0, 0.0, 0.0])
    assert out["weights"] == [0.0, 0.0, 0.0]
    assert out["degenerate"]


def test_invalid_args():
    with pytest.raises(ValueError):
        apply_constraints([0.5, 0.5], max_weight=0.0)
    with pytest.raises(ValueError):
        apply_constraints([0.5, 0.5], max_weight=1.5)
    with pytest.raises(ValueError):
        apply_constraints([0.5, 0.5], dust=-1.0)
    with pytest.raises(ValueError):
        apply_constraints([])
