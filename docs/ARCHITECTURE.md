# ARCHITECTURE — trade-allocate v0.1.0

## Position in the suite

```
trade-backtest / trade-overfit (validation, PASS verdicts)
        │  strategy return streams + gate-pass evidence
        ▼
┌──────────────────────────────┐
│        trade-allocate        │   this engine: the portfolio brain
│  evidence → align → correlate│
│  → allocate → monitor →      │
│  constrain → damp → gates    │
└──────────────────────────────┘
   │ mix + evidence ages + deallocations (schema_version 1)
   ├─► trade-agents PM      (strategy-mix sizing)
   ├─► trade-risk           (portfolio constraints)
   └─► trade-paper          (approvals show the mix)
```

The regime arbiter (`trade-regime`) sizes the **whole book**; this
engine decides the **mix within the book**. `trade-risk` keeps final
say on sizing — the allocator proposes, risk disposes.

## Module map

| Module | Responsibility | Key entry points |
|---|---|---|
| `evidence.py` | **Hard rule.** Validates gate-pass evidence; refuses loudly. | `validate_evidence`, `validate_all`, `EvidenceRefused` |
| `streams.py` | Return-stream inner-join alignment; metric helpers. | `align_streams`, `annualized_sharpe`, `max_drawdown` |
| `correlation.py` | OAS shrinkage → shrunk covariance → correlation. | `oas_shrinkage`, `cov_to_corr`, `estimate_correlation` |
| `allocate.py` | Three allocation methods + dispatcher. | `allocate`, `risk_parity_weights`, `hrp_weights`, `equal_weights` |
| `constraints.py` | Caps (water-filling), dust, long-only, sum-to-one. | `apply_constraints` |
| `turnover.py` | Damping + rebalance schedule. | `damp_turnover`, `should_rebalance` |
| `monitor.py` | Deallocation on breach (lifecycle / decay). | `deallocate`, `check_strategy` |
| `portfolio.py` | Portfolio-level gates on the final mix. | `evaluate_portfolio`, `require_portfolio_gates`, `diversification_ratio` |
| `pipeline.py` | Fixed 8-stage orchestration. | `run`, `run_preset` |
| `adapters.py` | Lazy in/out handoff contracts (schema_version 1). | `from_backtest_run`, `from_overfit_report`, `to_pm_mix`, `to_risk_limits`, `to_paper_mix` |
| `presets.py` | Named postures with rationale. | `preset_kwargs`, `preset_rationale` |
| `demo.py` | Seeded synthetic diversification demo. | `run_demo`, `print_demo`, `synthetic_streams` |
| `cli.py` | `demo`/`allocate`/`check`/`presets`/`methods`. | `main` |
| `licensing.py` | Suite license-key/update-check hooks. | `check_license`, `check_update` |

## Data flow (the one path)

`pipeline.run()` executes exactly this order; stages communicate
through plain dicts, never shared mutable state:

1. `evidence.validate_all` — any refusal raises `EvidenceRefused`,
   aborting everything. No partial allocation, ever.
2. `streams.align_streams` — inner join; `< 30` common timestamps or
   zero-variance series raise `StreamError`.
3. `correlation.estimate_correlation` — OAS shrunk corr + `rho_star`.
4. `allocate.allocate` — unconstrained target weights.
5. `monitor.deallocate` — breachers → 0, survivors renormalized;
   sets `forced=True` when anything was deallocated.
6. `constraints.apply_constraints` — caps, dust, sum-to-one.
7. `turnover.damp_turnover` — hold vs rebalance (bypassed when forced).
8. `portfolio.evaluate_portfolio` (or `require_*` when strict) —
   the final mix judged as a portfolio.

## Invariants (test-enforced)

- Weights are long-only and sum to 1, except the degenerate
  all-breached case which stays all-zero by design.
- Evidence refusal is total: one bad strategy kills the run.
- Deallocation can only remove capital, never direct it into a
  breacher (zeroing happens after allocation; no inverse path exists).
- Turnover damping never absorbs a deallocation (`forced`).
- Every public result is JSON-serializable (a test dumps the pipeline
  result through `json.dumps`).

## Scaling notes

- Correlation estimation is O(T·p²) with p = #strategies (small);
  the CCD risk-parity solver is O(iter·p³)-ish in the worst case but
  p is tiny (strategies, not assets) — the demo's 4-strategy solve is
  milliseconds. No NumPy needed at any scale this engine targets.
- The engine is stateless except for caller-held `current_weights`;
  hysteresis-like memory lives with the caller (same pattern as
  `trade-regime`'s `--state` CLI flag).
- Thread-safety: all functions are pure (inputs in, new dicts out).

## Handoff contracts

All contracts are plain data with `schema_version: 1`. Inbound
adapters never fabricate: `from_backtest_run` sets `evidence=None`
(a backtest is not gate evidence — the hard rule will refuse it
until the overfit desk signs off). Outbound payloads
(`to_pm_mix` / `to_risk_limits` / `to_paper_mix`) carry weights,
method, deallocations with reasons, evidence ages, and the
portfolio-gate verdict so downstream consumers — and the human
approver — can see *why* the book is shaped this way.
