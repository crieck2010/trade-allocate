# Changelog

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
