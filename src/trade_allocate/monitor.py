"""Ongoing monitoring: deallocation on breach.

Entry is governed by :mod:`trade_allocate.evidence` (the hard rule).
This module governs *tenure*: a strategy that was valid when allocated
can still die. Three breach signals, any one of which zeroes the
strategy with a recorded reason:

1. **Lifecycle.** ``trade-lifecycle`` state ``RETIRED`` (terminal) —
   weight goes to 0 immediately. ``PAUSED`` also zeroes (a paused
   strategy must not receive capital); it may return on resume.
2. **Recent-window decay.** Trailing-window Sharpe below
   ``min_recent_sharpe`` (default 0.0) or trailing max drawdown worse
   than ``max_recent_drawdown`` (default -0.20). These are deliberately
   looser than the entry gates — monitoring watches for death, the
   gates screened for life.
3. **Evidence expiry.** (Handled at entry by evidence.py; recorded
   here if a re-check is requested.)

Deallocation is structural, not advisory: breachers are zeroed *after*
the allocation method runs, and survivors are renormalized. Capital is
never reallocated *into* a breaching strategy — there is no code path
for that. If every strategy breaches, weights go all-zero and the
portfolio gates fail loudly downstream; the engine does not invent an
allocation.

Missing monitoring data is not a breach: a strategy with no health
input is ``"unmonitored"`` and keeps its weight. Absence of evidence
is not evidence of failure — but it is recorded, so callers can see
the blind spot.
"""

from __future__ import annotations

TERMINAL_STATES = ("RETIRED",)
ZEROING_STATES = ("RETIRED", "PAUSED")


def check_strategy(strategy_id: str,
                   *,
                   lifecycle_state: str | None = None,
                   recent: dict | None = None,
                   min_recent_sharpe: float = 0.0,
                   max_recent_drawdown: float = -0.20) -> dict:
    """Check one strategy. Returns ``{"breached": bool, "reasons": [...],
    "status": ...}``."""
    reasons: list[str] = []
    state = str(lifecycle_state).strip().upper() if lifecycle_state else None
    if state in ZEROING_STATES:
        reasons.append(f"lifecycle_{state.lower()}")
    if recent is not None:
        if not isinstance(recent, dict):
            reasons.append("recent_malformed")
        else:
            sharpe = recent.get("sharpe")
            if isinstance(sharpe, (int, float)) and sharpe < min_recent_sharpe:
                reasons.append(f"recent_sharpe_{sharpe:.2f}_below_{min_recent_sharpe}")
            dd = recent.get("max_drawdown")
            if isinstance(dd, (int, float)) and dd < max_recent_drawdown:
                reasons.append(f"recent_drawdown_{dd:.2%}_beyond_{max_recent_drawdown:.0%}")
    status = "breached" if reasons else ("unmonitored" if recent is None and state is None
                                         else "healthy")
    return {"strategy_id": strategy_id, "breached": bool(reasons),
            "reasons": reasons, "status": status}


def deallocate(weights: list[float],
               ids: list[str],
               *,
               lifecycle: dict[str, str] | None = None,
               recent: dict[str, dict] | None = None,
               min_recent_sharpe: float = 0.0,
               max_recent_drawdown: float = -0.20) -> dict:
    """Zero breachers, renormalize survivors.

    Returns ``{"weights", "ids", "checks", "deallocated": [...],
    "all_breached": bool, "forced": bool}``. ``forced`` is True when
    any deallocation happened — the turnover damper must not absorb
    a risk event (pass it as ``force=True``).
    """
    lifecycle = lifecycle or {}
    recent = recent or {}
    checks = [check_strategy(sid,
                             lifecycle_state=lifecycle.get(sid),
                             recent=recent.get(sid),
                             min_recent_sharpe=min_recent_sharpe,
                             max_recent_drawdown=max_recent_drawdown)
              for sid in ids]
    breached = {c["strategy_id"] for c in checks if c["breached"]}
    w = [0.0 if sid in breached else max(0.0, float(v))
         for sid, v in zip(ids, weights)]
    s = sum(w)
    all_breached = s <= 0
    if not all_breached:
        w = [v / s for v in w]
    return {"weights": w, "ids": list(ids), "checks": checks,
            "deallocated": sorted(breached), "all_breached": all_breached,
            "forced": bool(breached)}
