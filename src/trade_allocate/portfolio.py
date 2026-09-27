"""Portfolio-level gates: the mix itself must earn promotion.

Individual gates stay as-is — the allocator never relaxes them. But a
panel of individually-validated strategies can still combine into an
untradeable portfolio (concentrated, drawdown-prone, undiversified),
so the allocated mix faces its own thresholds before it may be
promoted toward paper:

- **Portfolio Sharpe** (annualized, from the allocated mix's return
  series): ``> min_portfolio_sharpe`` (default 1.0 — the same elite
  bar as the strategy gate).
- **Portfolio max drawdown**: ``> max_portfolio_drawdown``
  (default -0.15 — the same bar as the strategy gate).
- **Diversification ratio**: ``DR = sum(w_i sigma_i) / sigma_p``,
  ``>= min_diversification_ratio`` (default 1.10). DR = 1 means no
  diversification benefit at all; the threshold demands the mix
  actually harvests some. This is a *policy* choice, documented as
  such — not science.

A mix failing any gate is refused for promotion with named reasons
(:class:`PortfolioGateFailed`). It is not "fixed" by re-weighting
behind the caller's back: the gates judge the mix the allocator
proposed. If you want a different mix, change the inputs.
"""

from __future__ import annotations

import math

from .streams import annualized_sharpe, max_drawdown

DEFAULT_MIN_PORTFOLIO_SHARPE = 1.0
DEFAULT_MAX_PORTFOLIO_DRAWDOWN = -0.15
DEFAULT_MIN_DIVERSIFICATION_RATIO = 1.10


class PortfolioGateFailed(Exception):
    """Raised when the allocated mix fails the portfolio-level gates."""

    def __init__(self, failures: list[dict]) -> None:
        self.failures = failures
        names = ", ".join(f["gate"] for f in failures)
        super().__init__(f"portfolio gates failed: {names}")


def diversification_ratio(weights: list[float], vols: list[float],
                          portfolio_vol: float) -> float:
    """DR = sum(w_i sigma_i) / sigma_p. Always >= 1 for long-only."""
    if portfolio_vol <= 0:
        return 1.0
    return sum(w * v for w, v in zip(weights, vols)) / portfolio_vol


def mix_returns(panel: dict, weights: list[float]) -> list[float]:
    """Return series of the allocated mix over the aligned panel."""
    return [sum(row[k] * weights[k] for k in range(len(weights)))
            for row in panel["matrix"]]


def evaluate_portfolio(panel: dict,
                       weights: list[float],
                       *,
                       periods_per_year: int = 252,
                       min_sharpe: float = DEFAULT_MIN_PORTFOLIO_SHARPE,
                       max_drawdown_limit: float = DEFAULT_MAX_PORTFOLIO_DRAWDOWN,
                       min_diversification_ratio: float = DEFAULT_MIN_DIVERSIFICATION_RATIO,
                       vols: list[float] | None = None) -> dict:
    """Compute portfolio metrics and judge the gates.

    Returns ``{"metrics": {...}, "gates": [...], "passed": bool}``.
    Raises :class:`PortfolioGateFailed` when ``strict=True`` and any
    gate fails.
    """
    returns = mix_returns(panel, weights)
    sharpe = annualized_sharpe(returns, periods_per_year)
    dd = max_drawdown(returns)
    if vols is None:
        vols = []
        for k in range(len(weights)):
            col = [row[k] for row in panel["matrix"]]
            mean = sum(col) / len(col)
            var = sum((v - mean) ** 2 for v in col) / max(len(col) - 1, 1)
            vols.append(math.sqrt(max(var, 0.0)))
    port_var = sum(
        weights[i] * weights[j] * _cov(panel, i, j)
        for i in range(len(weights)) for j in range(len(weights))
    )
    port_vol = math.sqrt(max(port_var, 0.0))
    dr = diversification_ratio(weights, vols, port_vol)

    gates = [
        {"gate": "portfolio_sharpe", "value": sharpe,
         "threshold": min_sharpe, "passed": sharpe > min_sharpe},
        {"gate": "portfolio_max_drawdown", "value": dd,
         "threshold": max_drawdown_limit, "passed": dd > max_drawdown_limit},
        {"gate": "diversification_ratio", "value": dr,
         "threshold": min_diversification_ratio,
         "passed": dr >= min_diversification_ratio},
    ]
    return {"metrics": {"sharpe": sharpe, "max_drawdown": dd,
                        "diversification_ratio": dr,
                        "portfolio_vol": port_vol,
                        "n_obs": len(returns)},
            "gates": gates,
            "passed": all(g["passed"] for g in gates)}


def _cov(panel: dict, i: int, j: int) -> float:
    ci = [row[i] for row in panel["matrix"]]
    cj = [row[j] for row in panel["matrix"]]
    mi = sum(ci) / len(ci)
    mj = sum(cj) / len(cj)
    return sum((a - mi) * (b - mj) for a, b in zip(ci, cj)) / max(len(ci) - 1, 1)


def require_portfolio_gates(*args, **kwargs) -> dict:
    """Like :func:`evaluate_portfolio` but raises on any failure."""
    result = evaluate_portfolio(*args, **kwargs)
    failures = [g for g in result["gates"] if not g["passed"]]
    if failures:
        raise PortfolioGateFailed(failures)
    return result
