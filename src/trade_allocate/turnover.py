"""Turnover control: damping against churn.

Same philosophy as the regime arbiter's hysteresis: small wobbles in
estimated optimal weights are estimation noise, not information.
Rebalancing on noise pays spread and fees to stand still.

Two mechanisms, usable together:

1. **Tolerance band.** A new target mix replaces the current mix only
   if ``max |target_i - current_i| >= tol`` (default 0.05). Otherwise
   the current mix is held and the decision is recorded as
   ``"held_within_tolerance"``.
2. **Schedule.** ``should_rebalance`` answers whether a calendar date
   is a rebalance date for ``"monthly"`` / ``"quarterly"`` /
   ``"weekly"`` cadences, so the pipeline can skip allocation work
   entirely off-schedule. The engine itself is schedule-agnostic: it
   just compares dates.

Forced events bypass damping: a deallocation (monitor breach) always
takes effect immediately — risk control is not damped.
"""

from __future__ import annotations

from datetime import date, datetime


def _to_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    raise ValueError(f"cannot interpret {value!r} as a date")


def should_rebalance(last_date, current_date, frequency: str = "monthly") -> bool:
    """True when ``current_date`` is a rebalance date after ``last_date``."""
    last = _to_date(last_date)
    cur = _to_date(current_date)
    if cur <= last:
        return False
    if frequency == "weekly":
        return (cur - last).days >= 7
    if frequency == "monthly":
        return (cur.year, cur.month) != (last.year, last.month)
    if frequency == "quarterly":
        return (cur.year, (cur.month - 1) // 3) != (last.year, (last.month - 1) // 3)
    raise ValueError(f"unknown frequency {frequency!r}; use weekly/monthly/quarterly")


def damp_turnover(current: list[float] | None,
                  target: list[float],
                  *,
                  tol: float = 0.05,
                  force: bool = False) -> dict:
    """Decide whether to move from ``current`` to ``target`` weights.

    Returns ``{"weights", "action", "max_drift"}`` where action is one
    of ``"initial"`` (no current mix), ``"rebalanced"``,
    ``"held_within_tolerance"``, or ``"forced"`` (deallocation event —
    always applied, never damped).
    """
    if tol < 0:
        raise ValueError("tol must be >= 0")
    if current is None:
        return {"weights": list(target), "action": "initial", "max_drift": None}
    if len(current) != len(target):
        raise ValueError("current and target must have the same length")
    if force:
        return {"weights": list(target), "action": "forced",
                "max_drift": max(abs(t - c) for t, c in zip(target, current))}
    drift = max(abs(t - c) for t, c in zip(target, current))
    if drift < tol:
        return {"weights": list(current), "action": "held_within_tolerance",
                "max_drift": drift}
    return {"weights": list(target), "action": "rebalanced", "max_drift": drift}
