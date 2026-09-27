"""Demo regression tests: seeded, deterministic, diversification shown."""

from trade_allocate import demo as demo_mod


def test_demo_deterministic():
    r1 = demo_mod.run_demo()
    r2 = demo_mod.run_demo()
    assert r1["weights"] == r2["weights"]
    assert r1["correlation"] == r2["correlation"]


def test_demo_recovers_known_correlation():
    r = demo_mod.run_demo()
    # A and C share factor 1 with loading 0.8 -> rho ~ 0.64.
    assert abs(r["correlation"]["A"]["C"] - 0.64) < 0.08
    # B and D are idiosyncratic -> ~0.
    assert abs(r["correlation"]["B"]["D"]) < 0.15
    assert abs(r["correlation"]["A"]["B"]) < 0.15


def test_demo_diversification_benefit():
    r = demo_mod.run_demo()
    assert r["portfolio_sharpe"] > r["mean_individual_sharpe"]
    assert r["diversification_ratio"] > 1.0


def test_demo_uncorrelated_beats_correlated():
    c = demo_mod.run_demo()["contrast"]
    assert (c["uncorrelated_AB"]["diversification_ratio"]
            > c["correlated_AC"]["diversification_ratio"])


def test_demo_synthetic_labeled():
    assert demo_mod.run_demo()["synthetic"] is True


def test_demo_all_presets():
    for preset in ("conservative", "balanced", "aggressive"):
        r = demo_mod.run_demo(preset_name=preset)
        assert abs(sum(r["weights"].values()) - 1.0) < 1e-9
