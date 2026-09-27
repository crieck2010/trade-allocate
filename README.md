# trade-allocate

The **strategy allocator** for the trade-suite: the portfolio brain the
system was missing. It allocates capital *across validated strategies*
by their co-movement — equal weight, risk parity, or hierarchical risk
parity.

The regime arbiter (`trade-regime`) sizes the whole book up or down.
This engine decides the **mix within the book**: how much of the
book's capital each strategy earns, from how the strategies co-move.
Two uncorrelated 0.5-Sharpe strategies combine to ~0.7; four get you
to ~1.0. Diversification across strategies is the most reliable free
lunch in quant finance — this engine harvests it, mechanically.

**The hard rule (machine-enforced, not documentation):** the allocator
ONLY allocates to gate-passing strategies. Every input strategy must
carry overfit-desk PASS evidence (verdict + timestamp + which gates).
Strategies without it are **refused with a non-zero exit and a
reason** — never silently included, never down-weighted as a
compromise. Individual gates are never relaxed here; the allocator
cannot create edge, it can only combine real, validated edges.

It sits in the decision loop after validation and before paper:
overfit-desk PASS → allocator → `trade-agents` PM (strategy-mix
sizing) → `trade-risk` (portfolio constraints) → `trade-paper`
(approvals show the mix).

## Install

```bash
pip install trade-allocate
# or from source
git clone https://github.com/crieck2010/trade-allocate
cd trade-allocate
pip install -e .
```

Requires Python 3.10+. No third-party dependencies — not even NumPy.
Every algorithm (OAS shrinkage, risk-parity CCD, HRP) is implemented in
the standard library, on purpose: the engine must be auditable and
packaging-light.

## Quick start

```python
from trade_allocate import adapters, pipeline

# One input per strategy: return series + gate-pass evidence.
inputs = {
    "TREND-VT": adapters.strategy_input(
        "TREND-VT",
        {"2024-01-01": 0.002, "2024-01-02": -0.001, ...},  # or [(ts, ret), ...]
        evidence={  # schema_version 1, from the overfit desk
            "schema_version": 1,
            "strategy_id": "TREND-VT",
            "verdict": "PASS",
            "evaluated_at": "2026-09-20T12:00:00+00:00",
            "gates": [{"name": "deflated_sharpe", "value": 1.1,
                       "threshold": 0.95, "passed": True}, ...],
        },
        lifecycle_state="LIVE",            # optional, from trade-lifecycle
        recent={"sharpe": 0.9, "max_drawdown": -0.06},  # optional monitoring
    ),
    ...
}

result = pipeline.run_preset(inputs, "balanced")
print(result["weights_by_id"])      # {'TREND-VT': 0.42, ...}
print(result["turnover_action"])    # initial | rebalanced | held_within_tolerance | forced
print(result["portfolio_gates_passed"])

# Handoff payloads for the siblings (schema_version 1, JSON-serializable):
result["pm_payload"]     # -> trade-agents PM
result["risk_payload"]   # -> trade-risk
result["paper_payload"]  # -> trade-paper approvals
```

The pipeline runs eight stages in a fixed order — evidence →
align → correlate → allocate → monitor → constrain → damp →
portfolio gates — and the result records every stage (deallocations
with reasons, capped/dusted indices, turnover action, shrinkage
intensity, gate verdicts).

CLI:

```bash
trade-allocate demo                                   # seeded diversification demo
trade-allocate demo --preset conservative --json
trade-allocate allocate --streams s.json --evidence e.json --method risk_parity
trade-allocate allocate --streams s.json --evidence e.json --preset balanced --strict
trade-allocate check --evidence e.json                # exit 2 if any evidence refused
trade-allocate presets                                # postures + rationale
trade-allocate methods                                # when each method fits
```

Refusal is loud: missing/invalid/non-PASS/stale evidence aborts the
whole run (`REFUSED: ...`, exit 2). `--strict` additionally fails
(exit 3) when the final mix fails the portfolio-level gates.

See `examples/allocate_example.py` for the full walkthrough.

## The maths

**What you learn.** How to split capital across strategies so the
*book* is worth more than the sum of its parts. The allocator turns a
panel of validated strategy return streams into one weight vector —
plus the proof that the mix diversifies (diversification ratio), the
proof that it clears the bar as a portfolio (portfolio Sharpe,
drawdown), and a refusal whenever the inputs don't qualify.

**Why it matters.** A single strategy with Sharpe 0.8 is good; four
uncorrelated strategies with Sharpe 0.5 each are a portfolio with
Sharpe ~1.0. Nobody has to find the one magic strategy if the machine
combines several real-but-modest edges without letting estimation
error or a dying strategy poison the mix. That is the multi-strat pod
model, mechanized: gates at entry, correlation-aware sizing, kill the
losers fast, damp the churn.

**The maths.**

- *Shrinkage (OAS).* Strategy panels are small-N: a handful of
  strategies, a few hundred observations. Raw sample correlations at
  small N are noise — two unrelated strategies can print |ρ| > 0.5 by
  chance. We shrink the sample covariance S toward the structured
  target μI (μ = tr(S)/p, "uncorrelated, equal-variance") using OAS
  (Chen–Ledoit–Wolf 2010), the closed-form Gaussian-oracle refinement
  of Ledoit–Wolf (2004):
  `ρ* = min(1, max(0, num/den))`,
  `num = (1−2/p)·tr(S²) + tr(S)²`,
  `den = (T+1−2/p)·(tr(S²) − tr(S)²/p)`,
  `S* = (1−ρ*)·S + ρ*·μI`.
  Lots of data → ρ* → 0 (trust the sample); scarce data → ρ* → 1
  (fall back to the target). The shrunk covariance is converted to a
  correlation matrix for the allocators.
- *Equal weight.* `wᵢ = 1/N`. The honest benchmark. No estimation, no
  optimization, no pretense — use it when histories are too short for
  correlation estimates to mean anything, and require the fancier
  methods to beat it before trusting them.
- *Risk parity (cyclical coordinate descent, Spinu 2013).* Equal risk
  contribution: each strategy contributes 1/N of portfolio variance,
  `RCᵢ = wᵢ(Σw)ᵢ = σ²/N`. Each sweep minimizes the pairwise
  RC-gap coordinate-by-coordinate — the first-order condition in one
  coordinate is a cubic, solved exactly (Cardano) with the
  non-negative root kept — then renormalizes to sum 1. Converges from
  any positive start; the engine reports `max_rc_deviation`, the worst
  relative gap to the 1/N target, and the tests assert it directly.
  Needs no expected returns (which we don't trust) — only the
  covariance we shrunk.
- *HRP (Lopez de Prado 2016).* Single-linkage clustering on the
  correlation distance `d = √(½(1−ρ))`, quasi-diagonal seriation of
  the leaves, then recursive bisection: each split allocates between
  halves by inverse cluster variance
  (`α = 1 − Var(left)/(Var(left)+Var(right))`). Never inverts the
  covariance matrix — the whole point is robustness to estimation
  error. Fits the suite's clustering theme (`trade-pairs`,
  regime work).
- *Constraints.* Long-only (negatives clipped), per-strategy cap via
  iterative water-filling (capped strategies are excluded from
  redistribution so the iteration terminates), dust threshold
  (weights below ε → 0, mass redistributed), exact renormalization.
  All-zero (everything deallocated) stays all-zero — the engine does
  not invent an allocation; the portfolio gates fail loudly instead.
- *Turnover damping.* A new target replaces the current mix only if
  `max|target − current| ≥ tol` (default 0.05); otherwise the mix is
  held as `held_within_tolerance`. Same philosophy as the regime
  arbiter's hysteresis: small wobbles are estimation noise, and
  rebalancing on noise pays spread to stand still. Deallocations
  bypass damping (`forced`) — risk control is never damped. Optional
  weekly/monthly/quarterly schedule gating.
- *Deallocation.* Any strategy whose `trade-lifecycle` state is
  `RETIRED`/`PAUSED`, or whose trailing window shows Sharpe <
  threshold / drawdown beyond threshold, is zeroed with a recorded
  reason and survivors renormalized. There is no code path that
  allocates *into* a breaching strategy. Missing monitoring data is
  `unmonitored`, not a breach — absence of evidence isn't evidence
  of failure, but it is recorded.
- *Portfolio gates.* The mix itself must pass before promotion toward
  paper: portfolio Sharpe > 1.0 (the same elite bar as the strategy
  gate), portfolio max drawdown > −0.15, and diversification ratio
  `DR = Σwᵢσᵢ/σₚ ≥ 1.10` (policy, documented as such). DR = 1 means
  no diversification benefit was harvested at all.
- *Marginal contribution* (`marginal.marginal_contribution`): for a
  candidate against the current book,
  `Δ = Sharpe(book ∪ candidate) − Sharpe(book)`, both legs weighted by
  the allocator's own machinery (default risk parity over the
  OAS-shrunk covariance — the same `allocate()` the book uses), on the
  inner-joined daily panel. `max_correlation` is the largest Pearson
  correlation between the candidate and any book member on aligned
  daily strategy returns. Admission needs `Δ > ε` (default 0.05),
  `maxρ < ρ_max` (default 0.6), and ≥ 126 overlapping trading days for
  every candidate/book pair — insufficient overlap fails closed
  (`admitted=False`, reason recorded). The engine measures; the
  admission policy (ε, ρ_max, the complexity budget) lives with
  `trade-agents` (Occam's Desk §4).

**Honest limitations.** Correlation estimates from short strategy
histories are fragile — shrinkage dampens the noise, it does not
cure it; with 60 observations no estimator saves you. The allocator
cannot create edge: it harvests diversification from real, validated
edges, and garbage strategies in means a garbage mix out (the hard
rule keeps them out at entry). Regime shifts break historical
correlations — the monitor watches for decay, but correlation
breakdown is the risk this engine is most exposed to. OAS assumes
near-Gaussian returns; strategy returns are heavy-tailed, so the
shrinkage intensity is approximate. No validated strategies exist in
the suite yet, so the demo is synthetic by necessity.

## Interop

Lazy adapters, plain data, zero sibling imports. Full handoff
contracts (with schema versions) in `docs/ARCHITECTURE.md`.

| Direction | Sibling | Contract |
|---|---|---|
| In | trade-backtest | `from_backtest_run` — run output → strategy input (evidence NOT fabricated) |
| In | trade-overfit | `from_overfit_report` — validation report → gate-pass evidence |
| In | trade-agents | `health_from_track_record` — ledger events → monitoring input |
| In | trade-lifecycle | `lifecycle_states` — `{id: state}` mapping |
| Out | trade-agents PM | `to_pm_mix` — weights + method + evidence ages + deallocations |
| Out | trade-risk | `to_risk_limits` — implied caps + portfolio vol |
| Out | trade-paper | `to_paper_mix` — mix as approval display context |

## License

MIT — see LICENSE.
