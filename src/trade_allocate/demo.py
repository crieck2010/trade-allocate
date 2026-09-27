"""Seeded demo: diversification, demonstrated numerically.

Four synthetic daily strategy streams (seed 42) with a *known*
correlation structure built from a two-factor model:

- A "trend-ish":      vol 1.0%/day, loads on factor 1
- C "trend-ish twin": vol 1.0%/day, loads on factor 1  -> corr(A, C) ~ 0.64
- B "carry-ish":      vol 1.0%/day, idiosyncratic      -> ~uncorrelated
- D "arb-ish":        vol 0.6%/day, idiosyncratic      -> ~uncorrelated

The demo allocates by each method and shows:

1. portfolio Sharpe  >  average individual Sharpe (diversification works),
2. the uncorrelated subset {A, B} earns a higher diversification ratio
   than the correlated pair {A, C} (correlation is the enemy).

Evidence is synthetic but structurally valid (verdict PASS) — labeled
``synthetic`` everywhere so it can never be mistaken for real
gate-pass evidence.
"""

from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta, timezone

from . import adapters, allocate as allocate_mod, pipeline
from . import portfolio as portfolio_mod, presets, streams
from .correlation import estimate_correlation

SEED = 42
N_OBS = 500


def _gaussians(rng: random.Random, n: int) -> list[float]:
    """Box-Muller normals from a seeded RNG."""
    out = []
    while len(out) < n:
        u1 = max(rng.random(), 1e-12)
        u2 = rng.random()
        r = math.sqrt(-2.0 * math.log(u1))
        out.append(r * math.cos(2.0 * math.pi * u2))
        if len(out) < n:
            out.append(r * math.sin(2.0 * math.pi * u2))
    return out[:n]


def synthetic_streams(seed: int = SEED, n_obs: int = N_OBS) -> dict[str, list[tuple[str, float]]]:
    """Four synthetic strategy return streams with known structure."""
    rng = random.Random(seed)
    f1 = _gaussians(rng, n_obs)
    e = {k: _gaussians(rng, n_obs) for k in ("A", "B", "C", "D")}
    load = 0.8
    idio = math.sqrt(1.0 - load * load)
    spec = {
        "A": (0.00080, 0.010, lambda t: load * f1[t] + idio * e["A"][t]),
        "B": (0.00070, 0.010, lambda t: e["B"][t]),
        "C": (0.00060, 0.010, lambda t: load * f1[t] + idio * e["C"][t]),
        "D": (0.00040, 0.006, lambda t: e["D"][t]),
    }
    start = date(2024, 1, 1)
    out: dict[str, list[tuple[str, float]]] = {}
    for sid, (mu, vol, gen) in spec.items():
        out[sid] = [((start + timedelta(days=t)).isoformat(), mu + vol * gen(t))
                    for t in range(n_obs)]
    return out


def synthetic_evidence(strategy_id: str) -> dict:
    """Structurally-valid but synthetic PASS evidence (demo only)."""
    return {
        "schema_version": 1,
        "strategy_id": strategy_id,
        "verdict": "PASS",
        "evaluated_at": "2026-09-20T12:00:00+00:00",
        "gates": [
            {"name": "deflated_sharpe", "value": 1.10, "threshold": 0.95, "passed": True},
            {"name": "oos_sharpe", "value": 1.05, "threshold": 1.0, "passed": True},
            {"name": "max_drawdown", "value": -0.10, "threshold": -0.15, "passed": True},
            {"name": "worst_regime_sharpe", "value": 0.20, "threshold": 0.0, "passed": True},
            {"name": "beats_benchmark_net", "value": 0.03, "threshold": 0.0, "passed": True},
        ],
        "evaluator": "trade-overfit (synthetic demo)",
        "evidence_ref": "synthetic — demo only, not real gate evidence",
        "n_trials": 1,
    }


def run_demo(seed: int = SEED, n_obs: int = N_OBS,
             preset_name: str = "balanced") -> dict:
    """Run the full pipeline on synthetic data. Deterministic."""
    streams_in = synthetic_streams(seed, n_obs)
    inputs = {sid: adapters.strategy_input(sid, series, synthetic_evidence(sid))
              for sid, series in streams_in.items()}

    # The demo pins "now" so evidence age is deterministic.
    res = pipeline.run_preset(
        inputs, preset_name,
        evidence_kwargs={"now": datetime(2026, 9, 26, tzinfo=timezone.utc)})

    panel = streams.align_streams({sid: inp["returns"] for sid, inp in inputs.items()})
    indiv_sharpes = {sid: streams.annualized_sharpe(streams.stream_returns(panel, sid))
                     for sid in panel["ids"]}
    port = res["portfolio_gates"]

    # Contrast: correlated pair {A,C} vs uncorrelated pair {A,B}.
    def subset(ids):
        sub = streams.align_streams({sid: streams_in[sid] for sid in ids})
        sub_est = estimate_correlation(sub)
        w = allocate_mod.allocate(sub_est["cov"], sub_est["corr"], sub["ids"],
                                  method="risk_parity")["weights"]
        vols = []
        for k in range(len(w)):
            col = [row[k] for row in sub["matrix"]]
            m = sum(col) / len(col)
            vols.append(math.sqrt(sum((v - m) ** 2 for v in col) / max(len(col) - 1, 1)))
        port_var = sum(w[i] * w[j] * _cov2(sub, i, j)
                       for i in range(len(w)) for j in range(len(w)))
        dr = portfolio_mod.diversification_ratio(w, vols, math.sqrt(max(port_var, 0)))
        return {"weights": w, "diversification_ratio": dr,
                "sharpe": streams.annualized_sharpe(portfolio_mod.mix_returns(sub, w))}

    contrast = {"correlated_AC": subset(["A", "C"]), "uncorrelated_AB": subset(["A", "B"])}

    return {
        "seed": seed, "n_obs": n_obs, "preset": preset_name,
        "method": res["method"],
        "ids": res["ids"],
        "weights": res["weights_by_id"],
        "individual_sharpes": indiv_sharpes,
        "mean_individual_sharpe": sum(indiv_sharpes.values()) / len(indiv_sharpes),
        "portfolio_sharpe": port["metrics"]["sharpe"],
        "portfolio_max_drawdown": port["metrics"]["max_drawdown"],
        "diversification_ratio": port["metrics"]["diversification_ratio"],
        "rho_star": res["rho_star"],
        "correlation": {sid: dict(zip(res["ids"], row))
                        for sid, row in zip(res["ids"], res["correlation"])},
        "portfolio_gates": port["gates"],
        "portfolio_gates_passed": port["passed"],
        "deallocated": res["deallocated"],
        "turnover_action": res["turnover_action"],
        "contrast": contrast,
        "synthetic": True,
    }


def _cov2(panel: dict, i: int, j: int) -> float:
    ci = [row[i] for row in panel["matrix"]]
    cj = [row[j] for row in panel["matrix"]]
    mi = sum(ci) / len(ci)
    mj = sum(cj) / len(cj)
    return sum((a - mi) * (b - mj) for a, b in zip(ci, cj)) / max(len(ci) - 1, 1)


def print_demo(result: dict) -> None:
    """Human-readable demo summary."""
    print("trade-allocate demo (seeded synthetic data — not real evidence)")
    print(f"preset: {result['preset']}  method: {result['method']}  "
          f"rho* (shrinkage): {result['rho_star']:.3f}")
    print("\nweights:")
    for sid, w in result["weights"].items():
        print(f"  {sid}: {w:.4f}   (individual Sharpe {result['individual_sharpes'][sid]:+.2f})")
    print(f"\nmean individual Sharpe : {result['mean_individual_sharpe']:+.3f}")
    print(f"portfolio Sharpe       : {result['portfolio_sharpe']:+.3f}")
    print(f"portfolio max drawdown : {result['portfolio_max_drawdown']:.2%}")
    print(f"diversification ratio  : {result['diversification_ratio']:.3f}")
    print("\ncorrelation matrix:")
    ids = result["ids"]
    print("      " + "".join(f"{s:>8}" for s in ids))
    for sid in ids:
        print(f"  {sid}  " + "".join(f"{result['correlation'][sid][o]:+8.2f}" for o in ids))
    c = result["contrast"]
    print("\ncontrast — correlated pair {A,C} vs uncorrelated pair {A,B}:")
    print(f"  correlated   DR={c['correlated_AC']['diversification_ratio']:.3f} "
          f"Sharpe={c['correlated_AC']['sharpe']:+.2f}")
    print(f"  uncorrelated DR={c['uncorrelated_AB']['diversification_ratio']:.3f} "
          f"Sharpe={c['uncorrelated_AB']['sharpe']:+.2f}")
    print("\nportfolio gates:",
          "PASS" if result["portfolio_gates_passed"] else "FAIL")
    for g in result["portfolio_gates"]:
        print(f"  {g['gate']}: {g['value']:.3f} vs {g['threshold']} "
              f"({'PASS' if g['passed'] else 'FAIL'})")
