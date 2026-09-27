"""Strategy allocator: the portfolio brain of the trade-suite.

Allocates capital across *validated* strategies by their co-movement —
equal weight, risk parity, or hierarchical risk parity — with a
machine-enforced hard rule: only gate-passing strategies may be
allocated to. Strategies without overfit-desk PASS evidence are
refused, never silently included.

Plain data in, plain data out. Zero sibling imports.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
