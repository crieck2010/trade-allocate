"""Gate-pass evidence: the machine-enforced hard rule.

The allocator ONLY allocates to strategies that carry overfit-desk PASS
evidence. This module validates that evidence. Anything missing,
malformed, non-PASS, or stale is refused with :class:`EvidenceRefused` —
never silently included, never down-weighted as a compromise.

Evidence contract (``schema_version: 1``)::

    {
        "schema_version": 1,
        "strategy_id": "TREND-VT",
        "verdict": "PASS",                 # exact, case-sensitive
        "evaluated_at": "2026-09-20T14:00:00+00:00",   # ISO-8601
        "gates": [                          # every gate must show passed=true
            {"name": "deflated_sharpe", "value": 1.2,
             "threshold": 0.95, "passed": True},
            ...
        ],
        "evaluator": "trade-overfit",       # informational
        "evidence_ref": "commit 42ecab2 / docs/validation/...",
        "n_trials": 7                       # informational (DSR honesty)
    }

Staleness is a refusal, not a warning: a PASS from two years ago is not
evidence the strategy still has edge. Ongoing health is the job of
:mod:`trade_allocate.monitor`; entry is the job of this module.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

EVIDENCE_SCHEMA_VERSION = 1
DEFAULT_MAX_EVIDENCE_AGE_DAYS = 365


class EvidenceRefused(Exception):
    """Raised when gate-pass evidence is missing, invalid, or stale.

    Carries a machine-readable ``reason`` code plus a human sentence.
    The CLI converts this into a non-zero exit; it never becomes a
    silent exclusion.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"evidence refused ({reason})" + (f": {detail}" if detail else ""))


def _parse_ts(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceRefused("bad_timestamp", "evaluated_at must be a non-empty ISO-8601 string")
    text = value.strip()
    try:
        # fromisoformat handles offsets; tolerate a trailing "Z".
        ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise EvidenceRefused("bad_timestamp", f"evaluated_at not parseable: {value!r}")
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def validate_evidence(evidence: dict | None,
                      *,
                      max_age_days: int = DEFAULT_MAX_EVIDENCE_AGE_DAYS,
                      now: datetime | None = None) -> dict:
    """Validate one strategy's gate-pass evidence.

    Returns a normalized evidence dict on success. Raises
    :class:`EvidenceRefused` on any failure — missing, malformed,
    non-PASS verdict, any gate not passed, or stale.
    """
    if not isinstance(evidence, dict):
        raise EvidenceRefused("not_a_mapping", "evidence must be a JSON object")
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceRefused(
            "bad_schema",
            f"schema_version must be {EVIDENCE_SCHEMA_VERSION}, "
            f"got {evidence.get('schema_version')!r}",
        )
    strategy_id = evidence.get("strategy_id")
    if not isinstance(strategy_id, str) or not strategy_id.strip():
        raise EvidenceRefused("bad_strategy_id", "strategy_id must be a non-empty string")

    verdict = evidence.get("verdict")
    if verdict != "PASS":
        raise EvidenceRefused(
            "verdict_not_pass",
            f"strategy {strategy_id.strip()!r} verdict is {verdict!r}, not 'PASS'",
        )

    evaluated_at = _parse_ts(evidence.get("evaluated_at"))
    now = now or datetime.now(timezone.utc)
    if evaluated_at > now + timedelta(seconds=60):
        raise EvidenceRefused("timestamp_in_future",
                              f"evaluated_at {evidence.get('evaluated_at')!r} is in the future")
    age_days = (now - evaluated_at).total_seconds() / 86400.0
    if age_days > max_age_days:
        raise EvidenceRefused(
            "evidence_stale",
            f"PASS evidence for {strategy_id.strip()!r} is {age_days:.0f} days old "
            f"(max {max_age_days}); re-validate through the overfit desk",
        )

    gates = evidence.get("gates")
    if not isinstance(gates, list) or not gates:
        raise EvidenceRefused("gates_missing", "gates must be a non-empty list")
    failed = []
    for g in gates:
        if not isinstance(g, dict) or g.get("passed") is not True:
            failed.append(g.get("name", "?") if isinstance(g, dict) else "?")
    if failed:
        raise EvidenceRefused(
            "gate_not_passed",
            f"strategy {strategy_id.strip()!r} has non-passing gates: {failed}",
        )

    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "strategy_id": strategy_id.strip(),
        "verdict": "PASS",
        "evaluated_at": evaluated_at.isoformat(),
        "age_days": age_days,
        "gates": [{"name": g.get("name"), "value": g.get("value"),
                   "threshold": g.get("threshold"), "passed": True} for g in gates],
        "n_gates": len(gates),
        "evaluator": evidence.get("evaluator"),
        "evidence_ref": evidence.get("evidence_ref"),
        "n_trials": evidence.get("n_trials"),
    }


def validate_all(evidence_by_strategy: dict[str, dict] | None,
                 **kwargs) -> dict[str, dict]:
    """Validate evidence for every strategy in the panel.

    Returns ``{strategy_id: normalized_evidence}``. The FIRST refusal
    aborts the whole allocation — a panel with one unevidenced strategy
    is not allocated at all. Refusal is loud, by design.
    """
    if not isinstance(evidence_by_strategy, dict) or not evidence_by_strategy:
        raise EvidenceRefused("no_evidence", "no gate-pass evidence supplied")
    out: dict[str, dict] = {}
    for sid, ev in evidence_by_strategy.items():
        norm = validate_evidence(ev, **kwargs)
        # The evidence's own strategy_id wins over the dict key if they differ;
        # a mismatch is itself suspicious — refuse rather than guess.
        if norm["strategy_id"] != str(sid).strip():
            raise EvidenceRefused(
                "strategy_id_mismatch",
                f"dict key {sid!r} != evidence strategy_id {norm['strategy_id']!r}",
            )
        out[str(sid).strip()] = norm
    return out
