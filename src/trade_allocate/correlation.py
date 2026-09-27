"""Covariance/correlation estimation with shrinkage.

Strategy panels are small-N problems: a handful of strategies, a few
hundred observations if you're lucky. Raw sample correlations from
small panels are dangerously noisy — two unrelated strategies can show
|rho| > 0.5 by pure chance. We shrink toward a structured target.

We implement OAS (Oracle Approximating Shrinkage, Chen–Ledoit–Wolf
2010): the closed-form Gaussian-oracle refinement of Ledoit–Wolf
(2004). Given the T×p demeaned return matrix X:

    S      = X'X / T                      (sample covariance, MLE form)
    mu     = tr(S) / p                    (average variance)
    rho*   = min(1, max(0, num / den))     with
             num = (1 - 2/p) tr(S^2) + tr(S)^2
             den = (T + 1 - 2/p) (tr(S^2) - tr(S)^2 / p)
    S*     = (1 - rho*) S + rho* mu I      (shrunk covariance)

The shrunk covariance is converted to a correlation matrix. rho* = 0
recovers the sample estimator (lots of data, trust it); rho* -> 1 when
data is scarce (distrust the sample, fall back to "uncorrelated,
equal-variance").

Honest limitation, stated up front: OAS is derived under Gaussianity.
Strategy returns are heavy-tailed, so the intensity is mis-estimated
to some degree. Shrinkage still beats raw sample correlation on
mean-squared error in practice — but it does not *cure* fragile
estimates from short histories. If your strategies have 60
observations, no estimator saves you; get more history.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# tiny stdlib linear algebra (p is small: strategies, not assets)
# ---------------------------------------------------------------------------

def _transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*a)]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    bt = _transpose(b)
    return [[sum(x * y for x, y in zip(ra, cb)) for cb in bt] for ra in a]


def _trace(a: list[list[float]]) -> float:
    return sum(a[i][i] for i in range(len(a)))


def _frobenius2(a: list[list[float]]) -> float:
    return sum(v * v for row in a for v in row)


def _demean_columns(x: list[list[float]]) -> list[list[float]]:
    t = len(x)
    p = len(x[0])
    means = [sum(row[j] for row in x) / t for j in range(p)]
    return [[row[j] - means[j] for j in range(p)] for row in x]


# ---------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------

def sample_covariance(matrix: list[list[float]]) -> list[list[float]]:
    """MLE sample covariance (1/T normalization, demeaned)."""
    t = len(matrix)
    if t < 2:
        raise ValueError("need at least 2 observations")
    xc = _demean_columns(matrix)
    xt = _transpose(xc)
    return [[v / t for v in row] for row in _matmul(xt, xc)]


def oas_shrinkage(matrix: list[list[float]]) -> dict:
    """OAS-shrunk covariance of a T×p return matrix.

    Returns ``{"cov": S*, "rho_star": rho*, "target": mu}`` where S* is
    the shrunk covariance, rho* the shrinkage intensity in [0, 1], and
    mu the structured-target variance (tr(S)/p).
    """
    t = len(matrix)
    p = len(matrix[0])
    if t < 2:
        raise ValueError("need at least 2 observations")
    if p < 1:
        raise ValueError("need at least 1 series")
    s = sample_covariance(matrix)
    if p == 1:
        return {"cov": s, "rho_star": 0.0, "target": s[0][0]}

    mu = _trace(s) / p
    s2 = _matmul(s, s)
    tr_s2 = _trace(s2)
    tr_s = _trace(s)
    num = (1.0 - 2.0 / p) * tr_s2 + tr_s * tr_s
    den = (t + 1.0 - 2.0 / p) * (tr_s2 - tr_s * tr_s / p)
    if den <= 0:
        rho_star = 1.0  # degenerate (e.g. constant target): shrink fully
    else:
        rho_star = min(1.0, max(0.0, num / den))
    shrunk = [[(1.0 - rho_star) * s[i][j] + (rho_star * mu if i == j else 0.0)
               for j in range(p)] for i in range(p)]
    return {"cov": shrunk, "rho_star": rho_star, "target": mu}


def cov_to_corr(cov: list[list[float]]) -> list[list[float]]:
    """Covariance -> correlation matrix (unit diagonal)."""
    p = len(cov)
    vols = [math.sqrt(max(cov[i][i], 0.0)) for i in range(p)]
    corr = []
    for i in range(p):
        row = []
        for j in range(p):
            denom = vols[i] * vols[j]
            row.append(cov[i][j] / denom if denom > 0 else (1.0 if i == j else 0.0))
        corr.append(row)
    for i in range(p):
        corr[i][i] = 1.0
    return corr


def estimate_correlation(panel: dict) -> dict:
    """Shrunk strategy×strategy correlation from an aligned panel.

    Returns ``{"ids", "corr", "cov", "rho_star", "n_obs"}``.
    """
    matrix = panel["matrix"]
    ids = panel["ids"]
    res = oas_shrinkage(matrix)
    corr = cov_to_corr(res["cov"])
    return {"ids": list(ids), "corr": corr, "cov": res["cov"],
            "rho_star": res["rho_star"], "target_variance": res["target"],
            "n_obs": panel["n_obs"]}


def correlation_distance(corr: list[list[float]]) -> list[list[float]]:
    """Correlation -> distance matrix d = sqrt(0.5 * (1 - rho)).

    The HRP clustering metric (Lopez de Prado 2016): 0 for perfect
    correlation, 1 for perfect anti-correlation.
    """
    p = len(corr)
    return [[math.sqrt(max(0.0, 0.5 * (1.0 - corr[i][j])))
             for j in range(p)] for i in range(p)]
