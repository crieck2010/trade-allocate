# Changelog

## v0.2.0 — 2026-09-27

**Occam's Desk phase 3 (allocator side).** Exposes the
marginal-diversification measurement that `trade-agents` uses for
marginal PM ranking (master spec:
`trade-agents/docs/design/OCCAMS_DESK.md` §4):

- `src/trade_allocate/marginal.py` — `marginal_contribution(book_series,
  candidate_id, candidate_series, method="risk_parity", ...)`:
  `Δ = Sharpe(book ∪ candidate) − Sharpe(book)`, both legs weighted by
  the allocator's own correlation-aware machinery (default risk parity
  over the OAS-shrunk covariance — the same `allocate()` the book
  uses), plus `max_correlation` (Pearson on aligned daily strategy
  returns — the machine's cousinship test). Accepts
  `{"dates","returns"}`, `{timestamp: return}`, or `[(timestamp,
  return)]` streams. Fail-closed: < 126 overlapping trading days, empty
  book, or degenerate streams return `admitted=False` with the reason
  recorded — never a scored guess. The engine measures; the admission
  policy (ε, ρ_max, complexity budget) stays with `trade-agents`.
- `docs/METHODOLOGY.md` §9: why marginal ranking (standalone Sharpe
  re-selects cousins), why correlation on returns not labels, why 126
  days fail-closed, why the engine doesn't judge — with a cross-link to
  the master spec instead of duplicating it.
- README § The maths: the Δ / correlation / overlap formulas.

Tests: 10 new (`tests/test_marginal.py`) — diversifier admitted, cousin
rejected on correlation, useless candidate rejected on ε, insufficient
overlap fails closed, empty book fails closed, all input shapes, equal
weighting, custom thresholds, multi-member book.

## v0.1.0 — 2026-09-26

Initial release: the strategy allocator — the portfolio brain of the
trade-suite. Allocates capital across validated strategies by their
co-movement (the regime arbiter sizes the whole book; this engine
decides the mix within the book).

**Hard rule (machine-enforced):** only gate-passing strategies may be
allocated to. Every input strategy must carry overfit-desk PASS
evidence (`evidence.py`: verdict + timestamp + gates, schema_version
1); missing/invalid/non-PASS/stale evidence raises `EvidenceRefused`
and the CLI exits 2 — never silently included, never down-weighted.

**Engine:**
- `streams.py` — return-stream inner-join alignment (any frequency),
  zero-variance/NaN guards, Sharpe/maxDD helpers
- `correlation.py` — OAS (Chen–Ledoit–Wolf 2010) shrinkage toward μI
  in pure stdlib, then correlation; `rho_star` reported
- `allocate.py` — three methods: `equal` (1/N baseline), `risk_parity`
  (equal risk contribution via cyclical coordinate descent, Spinu
  2013 — exact cubic solve per coordinate, convergence reported as
  max RC deviation), `hrp` (single-linkage on correlation distance +
  recursive bisection, Lopez de Prado 2016 — never inverts covariance)
- `constraints.py` — long-only, per-strategy cap (terminating
  water-filling), dust threshold, exact sum-to-one; all-zero stays
  all-zero by design
- `turnover.py` — tolerance-band damping (anti-churn, same philosophy
  as the regime arbiter's hysteresis) + weekly/monthly/quarterly
  schedule gating; deallocations bypass damping (`forced`)
- `monitor.py` — deallocation on `trade-lifecycle` RETIRED/PAUSED or
  trailing-window decay, with recorded reasons; structurally cannot
  allocate into a breacher
- `portfolio.py` — portfolio-level gates on the final mix: Sharpe >
  1.0, max drawdown > −0.15, diversification ratio ≥ 1.10
- `pipeline.py` — fixed 8-stage orchestration, JSON-serializable
  result, adapter payloads attached
- `adapters.py` — lazy in/out contracts (schema_version 1), zero
  sibling imports; `from_backtest_run` never fabricates evidence
- `presets.py` — conservative (HRP) / balanced (risk parity) /
  aggressive (equal weight) with documented rationale
- `demo.py` — seeded 4-strategy synthetic demo with known correlation
  structure; demonstrates portfolio Sharpe (1.50) > mean individual
  Sharpe (0.88) and uncorrelated-pair DR (1.43) > correlated-pair DR
  (1.09)
- CLI: `demo` / `allocate` / `check` / `presets` / `methods`;
  `--strict` fails (exit 3) on portfolio-gate failure
- `licensing.py` — suite license-key/update-check hooks

**Tests:** 107 passing. Includes risk-parity known-answer cases
(diagonal → inverse-vol weights; 3-asset correlated → equal risk
contributions within 1e-6), HRP block-structure balance, shrinkage
beating sample covariance on synthetic truth, cap/dust/deallocation/
damping coverage, and CLI refusal exit codes.

**Stated limitations:** correlation estimates from short histories are
fragile (shrinkage damps, doesn't cure); the allocator can't create
edge — it harvests diversification from validated edges; regime
shifts break historical correlations; OAS assumes near-Gaussian
returns; no validated strategies exist yet so the demo is synthetic.
