"""Lazy adapters: handoff contracts with the sibling engines.

Zero sibling imports — every adapter consumes and produces plain
dicts/lists. Each contract carries a ``schema_version`` so consumers
can pin what they read.

IN (what this engine accepts)
----------------------------
- ``strategy_input(strategy_id, returns, evidence, ...)`` — one
  strategy's return series plus its gate-pass evidence. ``returns`` is
  ``{timestamp_iso: ret}`` or ``[(ts, ret), ...]``.
- ``from_backtest_run(run)`` — shape a ``trade-backtest`` run output
  into a strategy input (expects ``run["strategy_id"]`` and either
  ``run["returns"]`` as above or ``run["equity_curve"]`` as
  ``[(ts, equity), ...]``, converted to simple returns).
- ``from_overfit_report(report)`` — pull gate-pass evidence out of a
  ``trade-overfit`` validation report (expects ``report["verdict"]``,
  ``report["gates"]``, ``report["evaluated_at"]`` /
  ``report["evaluated"]``).
- ``health_from_track_record(events)`` — shape ``trade-agents``
  track-record ledger events into the ``recent`` monitoring input
  (expects per-strategy ``{"sharpe", "max_drawdown"}`` entries;
  passes through anything shaped that way, ignores the rest).
- ``lifecycle_states(mapping)`` — ``{strategy_id: state}`` from
  ``trade-lifecycle``; states are uppercased, unknown ids pass
  through untouched.

OUT (what this engine emits)
---------------------------
- ``to_pm_mix(result)`` — ``schema_version: 1`` payload for the
  ``trade-agents`` PM: strategy weights, method, evidence ages,
  deallocations, and the portfolio-gate verdict.
- ``to_risk_limits(result)`` — ``schema_version: 1`` for
  ``trade-risk``: per-strategy weight caps implied by the mix plus
  the portfolio volatility estimate.
- ``to_paper_mix(result)`` — ``schema_version: 1`` for
  ``trade-paper`` approvals: the mix as display context (approvals
  show *why* the book is shaped this way).
"""

from __future__ import annotations

IN_SCHEMA_VERSION = 1
OUT_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# in
# ---------------------------------------------------------------------------

def strategy_input(strategy_id: str,
                   returns,
                   evidence: dict,
                   *,
                   lifecycle_state: str | None = None,
                   recent: dict | None = None) -> dict:
    """Bundle one strategy's series + evidence + monitoring into an input."""
    return {"schema_version": IN_SCHEMA_VERSION,
            "strategy_id": str(strategy_id),
            "returns": returns,
            "evidence": evidence,
            "lifecycle_state": lifecycle_state,
            "recent": recent}


def from_backtest_run(run: dict) -> dict:
    """Shape a trade-backtest run output into a strategy input (sans evidence).

    Evidence must come from the overfit desk — a backtest run is not
    gate evidence, and this adapter refuses to pretend otherwise: the
    returned input carries ``evidence: None`` and the pipeline's hard
    rule will refuse it until real evidence is attached.
    """
    if not isinstance(run, dict):
        raise ValueError("run must be a mapping")
    sid = run.get("strategy_id") or run.get("name")
    if not sid:
        raise ValueError("run has no strategy_id/name")
    returns = run.get("returns")
    if returns is None and run.get("equity_curve"):
        curve = run["equity_curve"]
        pairs = list(curve.items()) if isinstance(curve, dict) else list(curve)
        returns = [(pairs[i][0], pairs[i][1] / pairs[i - 1][1] - 1.0)
                   for i in range(1, len(pairs))]
    if returns is None:
        raise ValueError(f"run for {sid!r} has neither returns nor equity_curve")
    return strategy_input(sid, returns, evidence=None)


def from_overfit_report(report: dict, strategy_id: str) -> dict:
    """Extract gate-pass evidence from a trade-overfit validation report."""
    if not isinstance(report, dict):
        raise ValueError("report must be a mapping")
    evaluated_at = (report.get("evaluated_at") or report.get("evaluated")
                    or report.get("timestamp"))
    gates = report.get("gates") or []
    norm_gates = []
    for g in gates:
        if isinstance(g, dict):
            norm_gates.append({"name": g.get("name") or g.get("gate"),
                               "value": g.get("value"),
                               "threshold": g.get("threshold"),
                               "passed": bool(g.get("passed"))})
    return {"schema_version": 1,
            "strategy_id": str(strategy_id),
            "verdict": report.get("verdict"),
            "evaluated_at": evaluated_at,
            "gates": norm_gates,
            "evaluator": report.get("evaluator", "trade-overfit"),
            "evidence_ref": report.get("evidence_ref") or report.get("ref"),
            "n_trials": report.get("n_trials")}


def health_from_track_record(events: list[dict]) -> dict[str, dict]:
    """Shape track-record ledger events into ``recent`` monitoring input.

    Passes through per-strategy ``{"sharpe", "max_drawdown"}`` payloads;
    anything else is ignored (not fabricated).
    """
    out: dict[str, dict] = {}
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        sid = ev.get("strategy_id")
        payload = ev.get("recent") or ev.get("health")
        if sid and isinstance(payload, dict):
            shaped = {}
            if isinstance(payload.get("sharpe"), (int, float)):
                shaped["sharpe"] = float(payload["sharpe"])
            if isinstance(payload.get("max_drawdown"), (int, float)):
                shaped["max_drawdown"] = float(payload["max_drawdown"])
            if shaped:
                out[str(sid)] = shaped
    return out


def lifecycle_states(mapping: dict) -> dict[str, str]:
    """Normalize a trade-lifecycle state mapping."""
    if not isinstance(mapping, dict):
        raise ValueError("mapping must be a dict")
    return {str(k): str(v).strip().upper() for k, v in mapping.items()}


# ---------------------------------------------------------------------------
# out
# ---------------------------------------------------------------------------

def _base_out(result: dict) -> dict:
    return {"schema_version": OUT_SCHEMA_VERSION,
            "generated_by": "trade-allocate",
            "method": result.get("method"),
            "weights": result.get("weights"),
            "ids": result.get("ids"),
            "deallocated": result.get("deallocated", []),
            "evidence_ages_days": result.get("evidence_ages_days", {}),
            "portfolio_gates": result.get("portfolio_gates"),
            "notes": result.get("notes", [])}


def to_pm_mix(result: dict) -> dict:
    """Payload for the trade-agents PM: the strategy mix to size by."""
    payload = _base_out(result)
    payload["consumer"] = "trade-agents/pm"
    payload["turnover_action"] = result.get("turnover_action")
    return payload


def to_risk_limits(result: dict) -> dict:
    """Payload for trade-risk: implied per-strategy caps + portfolio vol."""
    payload = _base_out(result)
    payload["consumer"] = "trade-risk"
    metrics = (result.get("portfolio_gates") or {}).get("metrics", {})
    payload["portfolio_vol"] = metrics.get("portfolio_vol")
    payload["max_single_weight"] = max(result.get("weights", [0.0]) or [0.0])
    return payload


def to_paper_mix(result: dict) -> dict:
    """Payload for trade-paper approvals: display context for the mix."""
    payload = _base_out(result)
    payload["consumer"] = "trade-paper/approvals"
    payload["rho_star"] = result.get("rho_star")
    return payload
