"""Strategy allocator: the portfolio brain of the trade-suite.

Allocates capital across *validated* strategies by their co-movement —
equal weight, risk parity, or hierarchical risk parity — with a
machine-enforced hard rule: only gate-passing strategies may be
allocated to. Strategies without overfit-desk PASS evidence are
refused, never silently included.

Plain data in, plain data out. Zero sibling imports.
"""

from __future__ import annotations

from .marginal import (
    DEFAULT_EPSILON,
    DEFAULT_RHO_MAX,
    MIN_OVERLAP_DAYS,
    marginal_contribution,
)

__version__ = "0.2.0"
__all__ = [
    "__version__",
    # Occam's Desk phase 3: marginal-diversification measurement (v0.2.0)
    "marginal_contribution",
    "MIN_OVERLAP_DAYS",
    "DEFAULT_EPSILON",
    "DEFAULT_RHO_MAX",
]
