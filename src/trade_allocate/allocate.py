"""Allocation methods: equal weight, risk parity, HRP.

All three consume a covariance matrix and strategy ids, and return
long-only weights summing to one. Caps, dust, and deallocation live in
:mod:`trade_allocate.constraints` and :mod:`trade_allocate.monitor` —
this module computes the *unconstrained* target mix.

- ``equal`` — 1/N baseline. The honest benchmark every fancier method
  must beat to justify its complexity.
- ``risk_parity`` — equal risk contribution: each strategy contributes
  1/N of portfolio variance. Solved by cyclical coordinate descent
  (Spinu 2013): each sweep minimizes the pairwise risk-contribution
  gap coordinate-by-coordinate via an exact cubic solve, then
  renormalizes. Converges from any positive start; we test the
  equal-contribution property directly.
- ``hrp`` — hierarchical risk parity (Lopez de Prado 2016):
  single-linkage clustering on correlation distance, quasi-diagonal
  seriation, then recursive bisection allocating by inverse cluster
  variance. No matrix inversion anywhere — the point of HRP is that
  it never inverts a noisy covariance.
"""

from __future__ import annotations

import math

METHODS = ("equal", "risk_parity", "hrp")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _matvec(cov: list[list[float]], x: list[float]) -> list[float]:
    return [sum(cov[i][j] * x[j] for j in range(len(x))) for i in range(len(x))]


def portfolio_variance(cov: list[list[float]], w: list[float]) -> float:
    sw = _matvec(cov, w)
    return sum(w[i] * sw[i] for i in range(len(w)))


def risk_contributions(cov: list[list[float]], w: list[float]) -> list[float]:
    """RC_i = w_i (Sw)_i — each strategy's contribution to variance."""
    sw = _matvec(cov, w)
    return [w[i] * sw[i] for i in range(len(w))]


def _cbrt(v: float) -> float:
    return math.copysign(abs(v) ** (1.0 / 3.0), v)


def _quadratic_real_roots(a: float, b: float, c: float) -> list[float]:
    if abs(a) < 1e-300:
        if abs(b) < 1e-300:
            return []
        return [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return []
    if disc == 0:
        return [-b / (2.0 * a)]
    s = math.sqrt(disc)
    return sorted([(-b - s) / (2.0 * a), (-b + s) / (2.0 * a)])


def _cubic_real_roots(a: float, b: float, c: float, d: float) -> list[float]:
    """All real roots of a x^3 + b x^2 + c x + d = 0 (Cardano)."""
    if abs(a) < 1e-300:
        return _quadratic_real_roots(b, c, d)
    b, c, d = b / a, c / a, d / a
    # Depressed cubic t^3 + p t + q = 0 via x = t - b/3.
    p = c - b * b / 3.0
    q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    shift = -b / 3.0
    if disc > 1e-18:
        sq = math.sqrt(disc)
        return [_cbrt(-q / 2.0 + sq) + _cbrt(-q / 2.0 - sq) + shift]
    if disc < -1e-18:
        r = math.sqrt(-(p / 3.0) ** 3)
        theta = math.acos(max(-1.0, min(1.0, -q / (2.0 * r))))
        m = 2.0 * math.sqrt(-p / 3.0)
        return sorted(m * math.cos((theta + 2.0 * math.pi * k) / 3.0) + shift
                      for k in range(3))
    u = _cbrt(-q / 2.0)  # multiple root
    return sorted({2.0 * u + shift, -u + shift})


# ---------------------------------------------------------------------------
# equal weight
# ---------------------------------------------------------------------------

def equal_weights(ids: list[str]) -> list[float]:
    n = len(ids)
    if n < 1:
        raise ValueError("need at least one strategy")
    return [1.0 / n] * n


# ---------------------------------------------------------------------------
# risk parity — cyclical coordinate descent (Spinu 2013)
# ---------------------------------------------------------------------------

def _rp_coordinate_update(cov: list[list[float]], x: list[float], i: int) -> float:
    """Exact minimizer of the pairwise RC-gap objective in x_i >= 0.

    Objective (others fixed): g(x_i) = sum_{j != i} (A_j x_i^2 + B_j x_i
    + C_j)^2 with A_j = S_ii, B_j = c_i - x_j S_ij, C_j = -x_j d_j,
    c_i = sum_{k != i} S_ik x_k, d_j = sum_{k != i} S_jk x_k.
    g'(x_i) = 0 is cubic; we take the nonnegative root minimizing g
    (falling back to the current x_i if none improves).
    """
    p = len(x)
    a = cov[i][i]
    if a <= 0:
        return x[i]
    others = [j for j in range(p) if j != i]
    c_i = sum(cov[i][k] * x[k] for k in others)
    d = {j: sum(cov[j][k] * x[k] for k in others) for j in others}

    P = Q = R = S = 0.0
    for j in others:
        Aj = a
        Bj = c_i - x[j] * cov[i][j]
        Cj = -x[j] * d[j]
        P += 4.0 * Aj * Aj
        Q += 6.0 * Aj * Bj
        R += 2.0 * (Bj * Bj + 2.0 * Aj * Cj)
        S += 2.0 * Bj * Cj
    if abs(P) < 1e-300:
        return x[i]

    def g(v: float) -> float:
        total = 0.0
        for j in others:
            Aj = a
            Bj = c_i - x[j] * cov[i][j]
            Cj = -x[j] * d[j]
            t = Aj * v * v + Bj * v + Cj
            total += t * t
        return total

    best_v, best_g = x[i], g(x[i])
    for root in _cubic_real_roots(P, Q, R, S):
        if root >= 0.0:
            gv = g(root)
            if gv < best_g:
                best_g, best_v = gv, root
    return best_v


def risk_parity_weights(cov: list[list[float]],
                        *,
                        tol: float = 1e-10,
                        max_iter: int = 10000) -> dict:
    """Risk-parity weights via cyclical coordinate descent.

    Returns ``{"weights", "risk_contributions", "max_rc_deviation",
    "n_iter", "converged"}``. ``max_rc_deviation`` is the worst
    relative gap between a strategy's risk contribution and the
    1/N target — the property the tests assert directly.
    """
    p = len(cov)
    if p < 1:
        raise ValueError("need at least one strategy")
    if p == 1:
        return {"weights": [1.0], "risk_contributions": [cov[0][0]],
                "max_rc_deviation": 0.0, "n_iter": 0, "converged": True}
    x = [1.0 / p] * p
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        max_change = 0.0
        for i in range(p):
            new = _rp_coordinate_update(cov, x, i)
            max_change = max(max_change, abs(new - x[i]))
            x[i] = new
        s = sum(x)
        if s <= 0:
            raise ArithmeticError("risk parity collapsed to zero weights")
        x = [v / s for v in x]
        if max_change < tol:
            converged = True
            break
    rc = risk_contributions(cov, x)
    var = sum(rc)
    target = var / p if var > 0 else 0.0
    dev = max(abs(r - target) / target for r in rc) if target > 0 else 0.0
    return {"weights": x, "risk_contributions": rc,
            "max_rc_deviation": dev, "n_iter": n_iter, "converged": converged}


# ---------------------------------------------------------------------------
# HRP — single linkage + recursive bisection (Lopez de Prado 2016)
# ---------------------------------------------------------------------------

def _single_linkage_order(dist: list[list[float]]) -> list[int]:
    """Quasi-diagonal leaf order from single-linkage clustering."""
    p = len(dist)
    # clusters: id -> member list; cluster distance = min pairwise dist
    clusters: dict[int, list[int]] = {i: [i] for i in range(p)}
    children: dict[int, tuple[int, int]] = {}
    active = set(range(p))
    next_id = p

    def cdist(a: int, b: int) -> float:
        return min(dist[i][j] for i in clusters[a] for j in clusters[b])

    while len(active) > 1:
        best = None
        best_d = float("inf")
        act = sorted(active)
        for ii in range(len(act)):
            for jj in range(ii + 1, len(act)):
                d = cdist(act[ii], act[jj])
                if d < best_d:
                    best_d, best = d, (act[ii], act[jj])
        a, b = best  # type: ignore[misc]
        clusters[next_id] = clusters[a] + clusters[b]
        children[next_id] = (a, b)
        active.discard(a)
        active.discard(b)
        active.add(next_id)
        next_id += 1

    root = next(iter(active))

    def expand(cid: int) -> list[int]:
        if cid < p:
            return [cid]
        left, right = children[cid]
        return expand(left) + expand(right)

    return expand(root)


def _cluster_variance(cov: list[list[float]], items: list[int]) -> float:
    """Variance of the inverse-variance portfolio of a cluster."""
    iv = []
    for i in items:
        v = cov[i][i]
        iv.append(1.0 / v if v > 0 else 0.0)
    s = sum(iv)
    if s <= 0:
        return float("inf")
    w = [v / s for v in iv]
    var = 0.0
    for a_i, i in enumerate(items):
        for b_j, j in enumerate(items):
            var += w[a_i] * w[b_j] * cov[i][j]
    return var


def hrp_weights(cov: list[list[float]], corr: list[list[float]]) -> dict:
    """Hierarchical risk parity weights.

    Returns ``{"weights", "order", "merges"}`` where ``order`` is the
    quasi-diagonal leaf order used for bisection.
    """
    from .correlation import correlation_distance
    p = len(cov)
    if p < 1:
        raise ValueError("need at least one strategy")
    if p == 1:
        return {"weights": [1.0], "order": [0], "merges": []}
    order = _single_linkage_order(correlation_distance(corr))
    w = [1.0] * p
    queue = [order]
    while queue:
        items = queue.pop(0)
        if len(items) == 1:
            continue
        half = len(items) // 2
        left, right = items[:half], items[half:]
        var_l = _cluster_variance(cov, left)
        var_r = _cluster_variance(cov, right)
        denom = var_l + var_r
        alpha = 1.0 - var_l / denom if denom > 0 else 0.5
        for i in left:
            w[i] *= alpha
        for i in right:
            w[i] *= (1.0 - alpha)
        queue.append(left)
        queue.append(right)
    s = sum(w)
    w = [v / s for v in w]
    return {"weights": w, "order": order, "merges": []}


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

def allocate(cov: list[list[float]],
             corr: list[list[float]],
             ids: list[str],
             *,
             method: str = "risk_parity",
             **kwargs) -> dict:
    """Compute unconstrained target weights.

    Returns ``{"ids", "method", "weights", "detail"}``. Raises
    ``ValueError`` on unknown method.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if method == "equal":
        weights = equal_weights(ids)
        detail: dict = {"note": "1/N baseline"}
    elif method == "risk_parity":
        rp = risk_parity_weights(cov, **kwargs)
        weights = rp["weights"]
        detail = {"n_iter": rp["n_iter"], "converged": rp["converged"],
                  "max_rc_deviation": rp["max_rc_deviation"],
                  "risk_contributions": rp["risk_contributions"]}
    else:  # hrp
        hrp = hrp_weights(cov, corr)
        weights = hrp["weights"]
        detail = {"order": [ids[i] for i in hrp["order"]]}
    return {"ids": list(ids), "method": method, "weights": weights,
            "detail": detail}
