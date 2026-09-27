"""Weight constraints: caps, dust, long-only, sum-to-one.

Applied *after* the allocation method produces its target mix and
*after* monitor deallocation zeroes breachers. Order of operations in
the pipeline is: evidence -> allocate -> monitor (zero breachers) ->
constrain -> turnover damping -> portfolio gates.
"""

from __future__ import annotations


def apply_constraints(weights: list[float],
                      *,
                      max_weight: float = 0.5,
                      dust: float = 1e-6) -> dict:
    """Enforce long-only, per-strategy cap, dust threshold, sum-to-one.

    - negatives are clipped to 0 (methods here are long-only anyway;
      the clip is a defensive invariant),
    - weights above ``max_weight`` are capped and the overflow is
      redistributed proportionally among uncapped strategies
      (iterative water-filling, so repeated capping converges),
    - weights below ``dust`` go to 0 and their mass is redistributed
      proportionally,
    - final renormalization to sum exactly 1.

    Returns ``{"weights", "capped": [...ids hit cap...],
    "dusted": [...], "n_iter"}`` — ids are positional indices here;
    the pipeline maps them back to strategy ids.
    """
    if not 0 < max_weight <= 1.0:
        raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
    if dust < 0:
        raise ValueError(f"dust must be >= 0, got {dust}")
    w = [max(0.0, float(v)) for v in weights]
    n = len(w)
    if n == 0:
        raise ValueError("no weights to constrain")
    total = sum(w)
    if total <= 0:
        # Degenerate (e.g. everything deallocated): stay all-zero and let
        # the portfolio gates fail loudly downstream. Do NOT invent weights.
        return {"weights": [0.0] * n, "capped": [], "dusted": list(range(n)),
                "n_iter": 0, "degenerate": True}
    w = [v / total for v in w]

    capped: set[int] = set()
    # Water-filling: cap, redistribute overflow among the *uncapped*,
    # repeat until no breach. Each pass caps at least one new strategy,
    # so this terminates in at most n passes.
    n_iter = 0
    for n_iter in range(1, 2 * n + 1):
        over = [i for i in range(n) if i not in capped and w[i] > max_weight]
        if not over:
            break
        for i in over:
            capped.add(i)
        overflow = sum(w[i] - max_weight for i in over)
        for i in over:
            w[i] = max_weight
        under = [i for i in range(n) if i not in capped]
        under_total = sum(w[i] for i in under)
        if under_total > 0:
            for i in under:
                w[i] += overflow * (w[i] / under_total)
        else:
            # Every strategy is capped and mass remains: the cap cannot
            # be satisfied. Fall back to equal split (documented edge).
            w = [1.0 / n] * n
            break

    dusted = [i for i in range(n) if 0.0 < w[i] < dust]
    for i in dusted:
        w[i] = 0.0
    # Exact zeros (including deallocated) are not "dusted".
    dusted = [i for i in dusted]
    s = sum(w)
    if s > 0:
        w = [v / s for v in w]
    return {"weights": w, "capped": sorted(capped), "dusted": dusted,
            "n_iter": n_iter, "degenerate": False}
