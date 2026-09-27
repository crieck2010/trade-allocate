"""Stream alignment tests."""

import pytest

from trade_allocate.streams import (
    StreamError,
    align_streams,
    annualized_sharpe,
    max_drawdown,
    stream_returns,
)


def _series(start_day, n, fn):
    return [(f"2024-01-{d:02d}", fn(d)) for d in range(start_day, start_day + n)]


def test_inner_join():
    panel = align_streams({
        "A": _series(1, 40, lambda d: 0.01 * d),
        "B": _series(11, 40, lambda d: -0.005 * d),  # overlaps days 11..40
    }, min_common=30)
    assert panel["ids"] == ["A", "B"]
    assert panel["n_obs"] == 30
    assert panel["timestamps"][0] == "2024-01-11"
    assert len(panel["matrix"][0]) == 2


def test_dict_and_list_inputs_mix():
    panel = align_streams({
        "A": {f"2024-01-{d:02d}": 0.01 * (d % 3) for d in range(1, 35)},
        "B": [(f"2024-01-{d:02d}", -0.01 * (d % 5)) for d in range(1, 35)],
    })
    assert panel["n_obs"] == 34


def test_refuse_insufficient_overlap():
    with pytest.raises(StreamError):
        align_streams({
            "A": _series(1, 40, lambda d: 0.01),
            "B": _series(1, 40, lambda d: 0.02),
        }, min_common=100)


def test_refuse_zero_variance():
    with pytest.raises(StreamError) as e:
        align_streams({
            "A": _series(1, 40, lambda d: 0.01 * d),
            "B": _series(1, 40, lambda d: 0.0),  # constant
        })
    assert "zero variance" in str(e.value)


def test_refuse_nan_and_inf():
    with pytest.raises(StreamError):
        align_streams({"A": [("2024-01-01", float("nan"))] * 40,
                       "B": _series(1, 40, lambda d: 0.01 * d)})
    with pytest.raises(StreamError):
        align_streams({"A": [("2024-01-01", float("inf"))] * 40,
                       "B": _series(1, 40, lambda d: 0.01 * d)})


def test_refuse_duplicate_timestamp():
    with pytest.raises(StreamError):
        align_streams({"A": [("2024-01-01", 0.01), ("2024-01-01", 0.02)],
                       "B": _series(1, 40, lambda d: 0.01 * d)})


def test_refuse_empty_series():
    with pytest.raises(StreamError):
        align_streams({"A": [], "B": _series(1, 40, lambda d: 0.01)})


def test_stream_returns_and_metrics():
    panel = align_streams({
        "A": [(f"2024-01-{d:02d}", 0.001 + (0.0001 if d % 2 else -0.0001))
              for d in range(1, 60)],
        "B": [(f"2024-01-{d:02d}", -0.001 if d % 2 else 0.003) for d in range(1, 60)],
    })
    a = stream_returns(panel, "A")
    assert len(a) == 59
    assert annualized_sharpe(a) > 0
    b = stream_returns(panel, "B")
    assert annualized_sharpe(b) > 0
    assert max_drawdown([0.1, -0.5, 0.1]) < 0
    assert max_drawdown([0.1, 0.1]) == 0.0
