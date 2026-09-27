# METHODOLOGY — trade-allocate v0.1.0

How the engine reasons, and the judgment calls baked into it. This is
the companion to the README's `## The maths` (which states the
formulas); this document states the *why*.

## 1. Why allocate across strategies at all

The suite validates strategies one at a time through the overfit
desk. But a book of individually-validated strategies is not
automatically a good portfolio: it can be concentrated (three trend
strategies that all die in the same chop), drawdown-prone, or simply
undiversified. The allocator exists because **the portfolio is a
decision, not an accident** — and because diversification across
uncorrelated edges is the most reliable return-free-lunch in the
business: N uncorrelated Sharpe-S strategies combine to ~√N·S.

## 2. Why these three methods

- **Equal weight** is the null hypothesis. Every quantitative method
  must justify its complexity against 1/N — a famous result
  (DeMiguel, Garlappi & Uppal 2009) is that 1/N is embarrassingly
  hard to beat out-of-sample. We keep it as a first-class method,
  not a fallback.
- **Risk parity** is the default because it needs no expected
  returns. Mean-variance optimization would ask us to estimate
  expected strategy returns — the single noisiest input in finance —
  and then maximize error. Risk parity budgets *risk*, which we can
  estimate (with shrinkage), and ignores return, which we can't.
- **HRP** is the conservative choice because it never inverts a
  covariance matrix. Inverting a noisy small-N covariance amplifies
  estimation error; HRP's clustering + bisection sidesteps inversion
  entirely.

## 3. Why OAS shrinkage

With p strategies and T observations, the sample covariance has
O(p²) parameters estimated from T·p data points. When T ≈ p — or
worse — the sample correlation matrix is mostly noise. OAS shrinks
toward μI ("strategies are uncorrelated with average variance"),
with intensity chosen by the data: the noisier the sample relative
to the target, the harder the shrink. It is honest about its
assumption (Gaussianity) and we state the limitation: heavy-tailed
strategy returns make the intensity approximate. Shrinkage is
damping, not a cure.

## 4. Why the hard rule is a refusal, not a weight

A softer design would down-weight unevidenced strategies. We refuse
them instead, for two reasons. First, a weight is a *position*, and
a position in an unvalidated strategy is indistinguishable from
hope. Second, silent inclusion is how discipline rots: today's
"small weight for the promising one" is tomorrow's book. The refusal
is loud (non-zero exit, named reason) so the failure is always
visible to the human gate.

## 5. Why turnover damping

Rebalancing is not free (spread, fees, operational risk) and target
weights jitter with estimation noise. The damper's tolerance band
(5 pts default) absorbs the jitter; only sustained moves reallocate.
This mirrors the regime arbiter's hysteresis deliberately — the
suite's shared philosophy is that *small wobbles are noise until
proven otherwise*. Deallocations bypass damping because risk control
answers to a different clock than efficiency.

## 6. Why deallocation is structural

Monitoring zeroes breachers *after* the allocator runs, so there is
structurally no path that increases a breaching strategy's weight.
We chose post-allocation zeroing over pre-allocation exclusion
because the target mix is still informative (it shows what the book
*would* be), while the final mix is what gets traded. The recorded
reasons make the invalidation auditable.

## 7. Why portfolio-level gates

> **2026-09-27 note.** The suite adopted the two-tier validation framework
> (`trade-strategies/docs/validation/GATES.md`). The "individual gates" this
> section refers to are the **Tier-1 entry filter** (OOS Sharpe > 0.3,
> maxDD shallower than −25%, DSR > 0.8, beat benchmark net, Sortino > 0.75);
> the portfolio-level bar is the **Tier-2 promotion set** (portfolio Sharpe
> > 1.0, portfolio maxDD < 15%, portfolio DSR ≥ 0.95 with `n_trials` counting
> *all* strategy-selection trials, Sortino ≥ 1.5, Calmar ≥ 2.0, DR ≥ 1.10).
> The hard rule is unchanged: entry requires machine-readable PASS evidence
> against Tier-1, and this engine will not relax individual gates under any
> configuration. Terminology: strategies are *validated* / *invalidated* /
> *discarded*; "kill" is retired from this lane's vocabulary.

Individual gates certify strategies; portfolio gates certify the
*combination*. A mix can pass every strategy gate and still be
untradeable — e.g. three correlated trend strategies with a joint
drawdown of 30%. The diversification-ratio gate (≥ 1.10) is the
novel one: it demands the mix actually harvest diversification,
not merely claim it. Its threshold is policy, stated as policy.

## 8. What this engine will not do

- It will not invent expected returns or "views" (no Black-Litterman;
  views are opinions with algebra).
- It will not short strategies (long-only by design; shorting a
  strategy is a different product).
- It will not relax individual gates, under any configuration.
- It will not allocate without evidence, even at weight ~0.
