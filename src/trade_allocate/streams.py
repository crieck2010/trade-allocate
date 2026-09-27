"""Return-stream alignment: plain-data inner join across strategies.

Inputs are ``{strategy_id: series}`` where a series is either a mapping
``{timestamp_iso: return}`` or a list of ``(timestamp_iso, return)``
pairs. Any frequency works; the panel is inner-joined on the timestamp
strings, exactly like trade-macro's series alignment. No look-ahead is
possible: alignment is purely mechanical.

Returns are simple period returns (not prices, not log — document your
convention upstream; mixing conventions across strategies is a caller
error this module cannot detect).
"""

from __future__ import annotations


class StreamError(Exception):
    """Raised when return streams cannot be aligned into a panel."""


def _to_pairs(series) -> list[tuple[str, float]]:
    if isinstance(series, dict):
        items = list(series.items())
    elif isinstance(series, (list, tuple)):
        items = list(series)
    else:
        raise StreamError(f"series must be a mapping or a list of pairs, got {type(series).__name__}")
    pairs: list[tuple[str, float]] = []
    seen: set[str] = set()
    for item in items:
        try:
            ts, ret = item
        except (TypeError, ValueError):
            raise StreamError(f"series entry {item!r} is not a (timestamp, return) pair")
        if not isinstance(ts, str) or not ts:
            raise StreamError(f"timestamp must be a non-empty string, got {ts!r}")
        try:
            ret_f = float(ret)
        except (TypeError, ValueError):
            raise StreamError(f"return for {ts!r} is not numeric: {ret!r}")
        if ret_f != ret_f or ret_f in (float("inf"), float("-inf")):
            raise StreamError(f"return for {ts!r} is NaN or infinite")
        if ts in seen:
            raise StreamError(f"duplicate timestamp {ts!r} in series")
        seen.add(ts)
        pairs.append((ts, ret_f))
    return pairs


def align_streams(streams: dict[str, object],
                  *,
                  min_common: int = 30) -> dict:
    """Inner-join strategy return streams on timestamps.

    Returns ``{"timestamps": [...], "ids": [...], "matrix": [[...], ...]}``
    where ``matrix[t][k]`` is the return of strategy ``ids[k]`` at
    ``timestamps[t]``. Timestamps sorted ascending (ISO-8601 strings sort
    lexicographically only if they share a format — callers should use
    a consistent format; we sort and document).

    Raises :class:`StreamError` when fewer than two strategies are
    supplied, a series is empty, the intersection has fewer than
    ``min_common`` timestamps, or any series has zero variance
    (a constant return stream carries no information and would make
    correlation undefined).
    """
    if not isinstance(streams, dict) or len(streams) < 1:
        raise StreamError("need at least one strategy return stream")
    ids = [str(s) for s in streams]
    if len(set(ids)) != len(ids):
        raise StreamError("duplicate strategy ids")
    parsed: dict[str, dict[str, float]] = {}
    for raw_sid in streams:
        sid = str(raw_sid)
        parsed[sid] = dict(_to_pairs(streams[raw_sid]))
    for sid, series in parsed.items():
        if not series:
            raise StreamError(f"strategy {sid!r} has an empty return series")

    common = set.intersection(*(set(s) for s in parsed.values()))
    if len(common) < min_common:
        raise StreamError(
            f"only {len(common)} common timestamps (need >= {min_common}); "
            "strategies must overlap in time"
        )
    timestamps = sorted(common)
    matrix = [[parsed[sid][ts] for sid in ids] for ts in timestamps]

    # Zero-variance guard: correlation is undefined for a constant stream.
    for k, sid in enumerate(ids):
        col = [row[k] for row in matrix]
        mean = sum(col) / len(col)
        if all(abs(v - mean) < 1e-15 for v in col):
            raise StreamError(
                f"strategy {sid!r} has zero variance over the common window; "
                "correlation is undefined"
            )
    return {"timestamps": timestamps, "ids": ids, "matrix": matrix,
            "n_obs": len(timestamps), "n_strategies": len(ids)}


def stream_returns(panel: dict, strategy_id: str) -> list[float]:
    """Extract one strategy's aligned return column from a panel."""
    k = panel["ids"].index(strategy_id)
    return [row[k] for row in panel["matrix"]]


def annualized_sharpe(returns: list[float], periods_per_year: int = 252) -> float:
    """Annualized Sharpe of a return series (risk-free = 0)."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    if var <= 0:
        return 0.0
    return mean / (var ** 0.5) * (periods_per_year ** 0.5)


def max_drawdown(returns: list[float]) -> float:
    """Max drawdown of a cumulative return series (<= 0)."""
    peak = 1.0
    equity = 1.0
    worst = 0.0
    for r in returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        dd = equity / peak - 1.0
        worst = min(worst, dd)
    return worst
