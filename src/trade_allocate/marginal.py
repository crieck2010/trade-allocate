"""Marginal-diversification analysis — Occam's Desk, section 4.

``trade-agents`` owns the admission *policy* (the marginal-Sharpe bar
``epsilon``, the correlation cap ``rho_max``, the complexity budget);
this engine owns the *measurement*. Given the allocated book's return
streams and one candidate's stream, :func:`marginal_contribution`
reports how much the candidate would improve the book — and how
correlated it is with what's already there.

Definitions (all on the inner-joined daily panel):

- ``sharpe_without``: annualized Sharpe of the current book, weighted by
  the allocator's own weighting (default risk parity over the
  OAS-shrunk covariance — the same machinery as :func:`allocate`).
- ``sharpe_with``: annualized Sharpe of book + candidate, same weighting.
- ``delta_sharpe = sharpe_with - sharpe_without``: the candidate's
  marginal contribution. Standalone Sharpe never enters the verdict —
  a brilliant loner that duplicates the book scores ~0 here, by design.
- ``max_correlation``: the largest Pearson correlation between the
  candidate and any book member, on the aligned daily returns. This is
  the machine's cousinship test: two strategies can have different
  symbols, different strategy names, even different theses, and still
  be the same trade.

Overlap is fail-closed: every candidate/book pair needs at least
``min_overlap`` (default 126) common trading days, or the measurement
is refused (``admitted=False`` with the reason recorded). Short
histories produce flattering, meaningless correlations — the engine
says so instead of scoring them.

Plain data in, plain data out; stdlib only.
"""

from __future__ import annotations

import math

from .allocate import allocate
from .correlation import cov_to_corr, oas_shrinkage
from .portfolio import mix_returns
from .streams import StreamError, align_streams, annualized_sharpe

__all__ = [
    "MIN_OVERLAP_DAYS",
    "DEFAULT_EPSILON",
    "DEFAULT_RHO_MAX",
    "marginal_contribution",
]

MIN_OVERLAP_DAYS = 126  # six trading months of common history
DEFAULT_EPSILON = 0.05  # Occam's Desk §4: minimum marginal Sharpe gain
DEFAULT_RHO_MAX = 0.6   # Occam's Desk §4: maximum book correlation


def _to_pairs(series) -> list[tuple[str, float]]:
    """Accept ``{"dates","returns"}`` plus everything ``align_streams`` takes."""
    if isinstance(series, dict) and "dates" in series and "returns" in series:
        dates, rets = series["dates"], series["returns"]
        if len(dates) != len(rets):
            raise StreamError(
                f"dates/returns length mismatch: {len(dates)} != {len(rets)}")
        return list(zip([str(d) for d in dates], [float(r) for r in rets]))
    return series


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0
    return max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))


def _weighted_sharpe(panel: dict, ids: list[str], *,
                     method: str, periods_per_year: int) -> tuple[float, list[float]]:
    """Annualized Sharpe of an ids-subset, weighted by the allocator."""
    cols = [panel["ids"].index(i) for i in ids]
    matrix = [[row[k] for k in cols] for row in panel["matrix"]]
    if len(ids) == 1:
        weights = [1.0]
    else:
        shrunk = oas_shrinkage(matrix)["cov"]
        corr = cov_to_corr(shrunk)
        weights = allocate(shrunk, corr, ids, method=method)["weights"]
    mixed = mix_returns({"ids": ids, "matrix": matrix}, weights)
    return annualized_sharpe(mixed, periods_per_year), weights


def marginal_contribution(book_series: dict[str, object],
                          candidate_id: str,
                          candidate_series: object,
                          *,
                          method: str = "risk_parity",
                          min_overlap: int = MIN_OVERLAP_DAYS,
                          epsilon: float = DEFAULT_EPSILON,
                          rho_max: float = DEFAULT_RHO_MAX,
                          periods_per_year: int = 252) -> dict:
    """Measure a candidate's marginal diversification value vs a book.

    ``book_series`` maps strategy id -> return stream; a stream is
    ``{"dates": [...], "returns": [...]}``, ``{timestamp: return}``, or
    ``[(timestamp, return)]``. ``candidate_series`` is one such stream.

    Returns a plain-data report with ``sharpe_without``,
    ``sharpe_with``, ``delta_sharpe``, ``max_correlation``,
    ``correlations`` (per book member), the weighting ``method`` used,
    ``n_overlap``, and the admission verdict ``admitted`` with
    ``reasons``. ``admitted`` is True only when overlap is sufficient,
    ``delta_sharpe > epsilon``, and ``max_correlation < rho_max``.
    An empty book, insufficient overlap, or degenerate streams fail
    closed (``admitted=False``; the reason says why).
    """
    book_ids = [str(s) for s in book_series]
    report: dict = {
        "candidate_id": str(candidate_id),
        "n_book": len(book_ids),
        "method": method,
        "epsilon": epsilon,
        "rho_max": rho_max,
        "n_overlap": 0,
        "sharpe_without": None,
        "sharpe_with": None,
        "delta_sharpe": None,
        "max_correlation": None,
        "correlations": {},
        "weights_with": [],
        "ids_with": [],
        "admitted": False,
        "reasons": [],
    }
    if not book_ids:
        report["reasons"].append(
            "empty book: marginal contribution is undefined; the caller "
            "ranks the first strategy by standalone evidence")
        return report
    if str(candidate_id) in book_ids:
        report["reasons"].append(
            f"candidate {candidate_id!r} is already in the book")
        return report

    streams = {sid: _to_pairs(s) for sid, s in book_series.items()}
    streams[str(candidate_id)] = _to_pairs(candidate_series)
    try:
        panel = align_streams(streams, min_common=1)
    except StreamError as exc:
        report["reasons"].append(f"stream alignment failed: {exc}")
        return report

    n_overlap = panel["n_obs"]
    report["n_overlap"] = n_overlap
    if n_overlap < min_overlap:
        report["reasons"].append(
            f"insufficient overlap: {n_overlap} common trading days "
            f"< {min_overlap} required; measurement refused (fail closed)")
        return report

    try:
        sharpe_without, _ = _weighted_sharpe(
            panel, book_ids, method=method, periods_per_year=periods_per_year)
        ids_with = book_ids + [str(candidate_id)]
        sharpe_with, weights_with = _weighted_sharpe(
            panel, ids_with, method=method, periods_per_year=periods_per_year)
    except (StreamError, ValueError, ArithmeticError) as exc:
        report["reasons"].append(f"weighting failed: {exc}")
        return report

    cand_col = [row[panel["ids"].index(str(candidate_id))]
                for row in panel["matrix"]]
    correlations = {}
    for sid in book_ids:
        col = [row[panel["ids"].index(sid)] for row in panel["matrix"]]
        correlations[sid] = _pearson(cand_col, col)
    max_corr = max(correlations.values())

    delta = sharpe_with - sharpe_without
    report.update({
        "sharpe_without": sharpe_without,
        "sharpe_with": sharpe_with,
        "delta_sharpe": delta,
        "max_correlation": max_corr,
        "correlations": correlations,
        "weights_with": weights_with,
        "ids_with": ids_with,
    })

    reasons = []
    if not delta > epsilon:
        reasons.append(
            f"marginal Sharpe gain {delta:+.3f} does not clear epsilon "
            f"{epsilon:.2f}")
    if not max_corr < rho_max:
        worst = max(correlations, key=correlations.get)
        reasons.append(
            f"max book correlation {max_corr:.2f} (vs {worst!r}) breaches "
            f"rho_max {rho_max:.2f}")
    report["reasons"].extend(reasons)
    report["admitted"] = not reasons
    return report
