"""Hard-rule tests: gate-pass evidence validation and refusal."""

from datetime import datetime, timedelta, timezone

import pytest

from trade_allocate.evidence import (
    DEFAULT_MAX_EVIDENCE_AGE_DAYS,
    EVIDENCE_SCHEMA_VERSION,
    EvidenceRefused,
    validate_all,
    validate_evidence,
)

NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)


def good_evidence(sid="TREND-VT", age_days=10, verdict="PASS"):
    return {
        "schema_version": 1,
        "strategy_id": sid,
        "verdict": verdict,
        "evaluated_at": (NOW - timedelta(days=age_days)).isoformat(),
        "gates": [
            {"name": "deflated_sharpe", "value": 1.1, "threshold": 0.95, "passed": True},
            {"name": "oos_sharpe", "value": 1.05, "threshold": 1.0, "passed": True},
        ],
        "evaluator": "trade-overfit",
        "evidence_ref": "commit abc",
        "n_trials": 7,
    }


def test_valid_evidence_normalizes():
    out = validate_evidence(good_evidence(), now=NOW)
    assert out["verdict"] == "PASS"
    assert out["strategy_id"] == "TREND-VT"
    assert out["n_gates"] == 2
    assert out["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert 9.9 < out["age_days"] < 10.1


def test_refuse_not_a_mapping():
    with pytest.raises(EvidenceRefused) as e:
        validate_evidence(None, now=NOW)
    assert e.value.reason == "not_a_mapping"


def test_refuse_bad_schema():
    ev = good_evidence()
    ev["schema_version"] = 2
    with pytest.raises(EvidenceRefused) as e:
        validate_evidence(ev, now=NOW)
    assert e.value.reason == "bad_schema"


def test_refuse_missing_strategy_id():
    ev = good_evidence()
    del ev["strategy_id"]
    with pytest.raises(EvidenceRefused) as e:
        validate_evidence(ev, now=NOW)
    assert e.value.reason == "bad_strategy_id"


def test_refuse_verdict_not_pass():
    for verdict in ("FAIL", "PENDING", "pass", None):
        with pytest.raises(EvidenceRefused) as e:
            validate_evidence(good_evidence(verdict=verdict), now=NOW)
        assert e.value.reason == "verdict_not_pass"


def test_refuse_gate_not_passed():
    ev = good_evidence()
    ev["gates"][1]["passed"] = False
    with pytest.raises(EvidenceRefused) as e:
        validate_evidence(ev, now=NOW)
    assert e.value.reason == "gate_not_passed"


def test_refuse_empty_gates():
    ev = good_evidence()
    ev["gates"] = []
    with pytest.raises(EvidenceRefused) as e:
        validate_evidence(ev, now=NOW)
    assert e.value.reason == "gates_missing"


def test_refuse_stale_evidence():
    ev = good_evidence(age_days=DEFAULT_MAX_EVIDENCE_AGE_DAYS + 1)
    with pytest.raises(EvidenceRefused) as e:
        validate_evidence(ev, now=NOW)
    assert e.value.reason == "evidence_stale"


def test_refuse_future_timestamp():
    ev = good_evidence()
    ev["evaluated_at"] = (NOW + timedelta(days=2)).isoformat()
    with pytest.raises(EvidenceRefused) as e:
        validate_evidence(ev, now=NOW)
    assert e.value.reason == "timestamp_in_future"


def test_refuse_bad_timestamp():
    ev = good_evidence()
    ev["evaluated_at"] = "not-a-date"
    with pytest.raises(EvidenceRefused) as e:
        validate_evidence(ev, now=NOW)
    assert e.value.reason == "bad_timestamp"


def test_validate_all_first_refusal_aborts():
    with pytest.raises(EvidenceRefused):
        validate_all({"A": good_evidence("A"), "B": good_evidence("B", verdict="FAIL")},
                     now=NOW)


def test_validate_all_strategy_id_mismatch():
    with pytest.raises(EvidenceRefused) as e:
        validate_all({"WRONG": good_evidence("RIGHT")}, now=NOW)
    assert e.value.reason == "strategy_id_mismatch"


def test_validate_all_empty():
    with pytest.raises(EvidenceRefused) as e:
        validate_all({}, now=NOW)
    assert e.value.reason == "no_evidence"


def test_validate_all_ok():
    out = validate_all({"A": good_evidence("A"), "B": good_evidence("B")}, now=NOW)
    assert set(out) == {"A", "B"}
