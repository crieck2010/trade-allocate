"""The allocation pipeline: evidence -> mix, in the mandated order.

Order of operations (each stage's output feeds the next; nothing is
skipped, nothing is reordered):

1. **evidence** — every strategy's gate-pass evidence validated; any
   refusal aborts the whole run (hard rule).
2. **align** — return streams inner-joined on timestamps.
3. **correlate** — OAS-shrunk strategy×strategy correlation.
4. **allocate** — unconstrained target mix (equal / risk parity / HRP).
5. **monitor** — breachers (retired/paused/decayed) zeroed, survivors
   renormalized. Sets ``force=True`` for the damper when anything was
   deallocated: risk events are never damped.
6. **constrain** — caps, dust, long-only, sum-to-one.
7. **damp** — turnover damping against the current mix (bypassed when
   forced).
8. **portfolio gates** — the final mix judged as a portfolio.

The result dict is JSON-serializable and carries everything the
adapters need for the PM / risk / paper handoffs.
"""

from __future__ import annotations

from . import adapters, allocate as allocate_mod, constraints
from . import evidence as evidence_mod, monitor, portfolio as portfolio_mod
from . import streams, turnover
from .correlation import estimate_correlation


def run(strategy_inputs: dict[str, dict],
        *,
        method: str = "risk_parity",
        max_weight: float = 0.5,
        dust: float = 1e-6,
        tolerance: float = 0.05,
        current_weights: list[float] | None = None,
        force_rebalance: bool = False,
        lifecycle: dict[str, str] | None = None,
        recent: dict[str, dict] | None = None,
        min_recent_sharpe: float = 0.0,
        max_recent_drawdown: float = -0.20,
        min_portfolio_sharpe: float = 1.0,
        max_portfolio_drawdown: float = -0.15,
        min_diversification_ratio: float = 1.10,
        periods_per_year: int = 252,
        evidence_kwargs: dict | None = None,
        strict_portfolio_gates: bool = False) -> dict:
    """Run the full pipeline. Raises on evidence refusal or gate failure
    (when ``strict_portfolio_gates``)."""
    if not strategy_inputs:
        raise ValueError("no strategy inputs")
    ids = [str(s) for s in strategy_inputs]

    # 1. hard rule
    normed = evidence_mod.validate_all(
        {sid: strategy_inputs[sid].get("evidence") for sid in ids},
        **(evidence_kwargs or {}))

    # 2. align
    panel = streams.align_streams(
        {sid: strategy_inputs[sid]["returns"] for sid in ids})

    # 3. correlate
    est = estimate_correlation(panel)

    # 4. allocate
    target = allocate_mod.allocate(est["cov"], est["corr"], panel["ids"],
                                  method=method)

    # 5. monitor (inputs may carry their own lifecycle/recent; explicit
    #    args override)
    lc = dict(lifecycle or {})
    rc = dict(recent or {})
    for sid in ids:
        inp = strategy_inputs[sid]
        lc.setdefault(sid, inp.get("lifecycle_state"))
        if inp.get("recent") is not None:
            rc.setdefault(sid, inp.get("recent"))
    mon = monitor.deallocate(target["weights"], panel["ids"],
                             lifecycle=lc, recent=rc,
                             min_recent_sharpe=min_recent_sharpe,
                             max_recent_drawdown=max_recent_drawdown)

    # 6. constrain
    constrained = constraints.apply_constraints(mon["weights"],
                                                max_weight=max_weight,
                                                dust=dust)

    # 7. damp (forced when deallocation happened)
    damped = turnover.damp_turnover(current_weights, constrained["weights"],
                                    tol=tolerance,
                                    force=force_rebalance or mon["forced"])

    # 8. portfolio gates on the FINAL mix
    if strict_portfolio_gates:
        port = portfolio_mod.require_portfolio_gates(
            panel, damped["weights"], periods_per_year=periods_per_year,
            min_sharpe=min_portfolio_sharpe, max_drawdown_limit=max_portfolio_drawdown,
            min_diversification_ratio=min_diversification_ratio)
    else:
        port = portfolio_mod.evaluate_portfolio(
            panel, damped["weights"], periods_per_year=periods_per_year,
            min_sharpe=min_portfolio_sharpe, max_drawdown_limit=max_portfolio_drawdown,
            min_diversification_ratio=min_diversification_ratio)

    notes: list[str] = []
    if constrained.get("degenerate"):
        notes.append("all weights zero after deallocation/constraints; "
                     "no allocation proposed")
    if not port["passed"]:
        notes.append("portfolio gates FAILED — mix not promotable to paper")

    result = {
        "schema_version": 1,
        "ids": panel["ids"],
        "method": method,
        "weights": damped["weights"],
        "weights_by_id": dict(zip(panel["ids"], damped["weights"])),
        "target_weights": constrained["weights"],
        "turnover_action": damped["action"],
        "max_drift": damped["max_drift"],
        "deallocated": mon["deallocated"],
        "monitor_checks": mon["checks"],
        "capped": constrained["capped"],
        "dusted": constrained["dusted"],
        "rho_star": est["rho_star"],
        "correlation": [list(row) for row in est["corr"]],
        "n_obs": panel["n_obs"],
        "portfolio_gates": port,
        "portfolio_gates_passed": port["passed"],
        "evidence_ages_days": {sid: normed[sid]["age_days"] for sid in ids},
        "allocation_detail": target["detail"],
        "notes": notes,
    }
    result["pm_payload"] = adapters.to_pm_mix(result)
    result["risk_payload"] = adapters.to_risk_limits(result)
    result["paper_payload"] = adapters.to_paper_mix(result)
    return result


def run_preset(strategy_inputs: dict[str, dict],
               preset_name: str = "balanced",
               **overrides) -> dict:
    """Run the pipeline with a named preset (overrides allowed)."""
    from .presets import preset_kwargs
    kw = preset_kwargs(preset_name)
    kw.update(overrides)
    method = kw.pop("method")
    tolerance = kw.pop("tolerance")
    return run(strategy_inputs, method=method, tolerance=tolerance, **kw)
