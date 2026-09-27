"""Shrinkage tests: OAS intensity, structure, and the shrinkage property."""

import math
import random

from trade_allocate.correlation import (
    _matmul,
    correlation_distance,
    cov_to_corr,
    estimate_correlation,
    oas_shrinkage,
    sample_covariance,
)


def _gauss(rng, n):
    out = []
    while len(out) < n:
        u1, u2 = max(rng.random(), 1e-12), rng.random()
        r = math.sqrt(-2.0 * math.log(u1))
        out += [r * math.cos(2 * math.pi * u2), r * math.sin(2 * math.pi * u2)]
    return out[:n]


def _synthetic_panel(true_cov, t, seed=7):
    """Gaussian data with known true covariance (via Cholesky)."""
    rng = random.Random(seed)
    p = len(true_cov)
    # Cholesky
    L = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(true_cov[i][i] - s, 1e-12))
            else:
                L[i][j] = (true_cov[i][j] - s) / L[j][j]
    z = [_gauss(rng, t) for _ in range(p)]  # p x t
    # x_t = L z_t  -> t x p
    return [[sum(L[i][j] * z[j][tt] for j in range(p)) for i in range(p)]
            for tt in range(t)]


def _frobenius(a, b):
    return math.sqrt(sum((a[i][j] - b[i][j]) ** 2
                         for i in range(len(a)) for j in range(len(a))))


def test_rho_star_in_unit_interval():
    rng = random.Random(3)
    x = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(100)]
    res = oas_shrinkage(x)
    assert 0.0 <= res["rho_star"] <= 1.0


def test_shrinkage_beats_sample_on_synthetic_truth():
    # Small T, moderate p: the regime where shrinkage earns its keep.
    true_cov = [[1.0, 0.6, 0.2, 0.0, 0.1],
                [0.6, 1.0, 0.3, 0.1, 0.0],
                [0.2, 0.3, 1.0, 0.0, 0.2],
                [0.0, 0.1, 0.0, 1.0, 0.4],
                [0.1, 0.0, 0.2, 0.4, 1.0]]
    x = _synthetic_panel(true_cov, t=40, seed=7)
    sample = sample_covariance(x)
    shrunk = oas_shrinkage(x)["cov"]
    assert _frobenius(shrunk, true_cov) < _frobenius(sample, true_cov)


def test_shrinkage_vanishes_with_big_data():
    true_cov = [[1.0, 0.5], [0.5, 1.0]]
    x = _synthetic_panel(true_cov, t=20000, seed=11)
    res = oas_shrinkage(x)
    assert res["rho_star"] < 0.05  # lots of data -> trust the sample


def test_correlation_structure():
    x = _synthetic_panel([[1.0, 0.8], [0.8, 1.0]], t=500, seed=5)
    panel = {"ids": ["A", "B"], "matrix": x, "n_obs": 500, "n_strategies": 2}
    est = estimate_correlation(panel)
    c = est["corr"]
    assert c[0][0] == 1.0 and c[1][1] == 1.0
    assert abs(c[0][1] - c[1][0]) < 1e-12  # symmetric
    assert 0.6 < c[0][1] < 0.95  # recovers the designed correlation


def test_cov_to_corr_unit_diagonal():
    c = cov_to_corr([[4.0, 1.0], [1.0, 9.0]])
    assert c[0][0] == 1.0 and c[1][1] == 1.0
    assert abs(c[0][1] - 1.0 / 6.0) < 1e-12


def test_correlation_distance_bounds():
    d = correlation_distance([[1.0, 1.0], [1.0, 1.0]])
    assert d[0][1] == 0.0  # perfect correlation -> zero distance
    d = correlation_distance([[1.0, -1.0], [-1.0, 1.0]])
    assert d[0][1] == 1.0  # perfect anti-correlation -> max distance
    d = correlation_distance([[1.0, 0.0], [0.0, 1.0]])
    assert abs(d[0][1] - math.sqrt(0.5)) < 1e-12


def test_matmul_sanity():
    assert _matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]
