"""Presets: named postures with documented rationale (not just numbers)."""

from __future__ import annotations

PRESETS = {
    "conservative": {
        "method": "hrp",
        "max_weight": 0.35,
        "dust": 1e-4,
        "tolerance": 0.03,
        "min_recent_sharpe": 0.25,
        "max_recent_drawdown": -0.15,
        "min_portfolio_sharpe": 1.0,
        "max_portfolio_drawdown": -0.12,
        "min_diversification_ratio": 1.15,
        "rationale": (
            "HRP never inverts the covariance matrix, so it is the least "
            "likely to blow up on estimation error. Tight cap (0.35) and "
            "high diversification-ratio bar (1.15) force the mix to earn "
            "its diversification; strict monitoring (recent Sharpe < 0.25 "
            "deallocates) kills decay early. Rebalances rarely (3pt "
            "tolerance). For capital you cannot afford to lose."
        ),
    },
    "balanced": {
        "method": "risk_parity",
        "max_weight": 0.50,
        "dust": 1e-6,
        "tolerance": 0.05,
        "min_recent_sharpe": 0.0,
        "max_recent_drawdown": -0.20,
        "min_portfolio_sharpe": 1.0,
        "max_portfolio_drawdown": -0.15,
        "min_diversification_ratio": 1.10,
        "rationale": (
            "Risk parity budgets variance equally — the natural middle "
            "ground between naive 1/N and full mean-variance optimization "
            "(which would need expected returns we do not trust). The 0.5 "
            "cap prevents any single strategy dominating; monitoring "
            "deallocates only on genuine decay (negative recent Sharpe). "
            "The default posture."
        ),
    },
    "aggressive": {
        "method": "equal",
        "max_weight": 1.00,
        "dust": 0.0,
        "tolerance": 0.10,
        "min_recent_sharpe": -0.50,
        "max_recent_drawdown": -0.30,
        "min_portfolio_sharpe": 0.75,
        "max_portfolio_drawdown": -0.20,
        "min_diversification_ratio": 1.05,
        "rationale": (
            "Equal weight: no estimation, no optimization, no pretense. "
            "Useful as the honest benchmark the fancier methods must beat, "
            "or when strategy histories are too short for correlation "
            "estimates to mean anything. Looser bars throughout — "
            "aggressive means accepting more estimation risk, not "
            "ignoring the hard rule (gate-pass evidence is still "
            "mandatory)."
        ),
    },
}


def preset_kwargs(name: str) -> dict:
    """Return the allocator kwargs for a named preset (without rationale)."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return {k: v for k, v in PRESETS[name].items() if k != "rationale"}


def preset_rationale(name: str) -> str:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]["rationale"]
