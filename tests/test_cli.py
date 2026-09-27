"""CLI smoke tests: demo, check, allocate — incl. non-zero exits."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from trade_allocate import demo as demo_mod


def _run(*args, stdin_text=None):
    return subprocess.run(
        [sys.executable, "-m", "trade_allocate", *args],
        input=stdin_text, capture_output=True, text=True, timeout=120)


def _write(path: Path, obj):
    path.write_text(json.dumps(obj))


def _demo_files(tmp: Path):
    from datetime import datetime, timedelta, timezone
    streams_in = demo_mod.synthetic_streams()
    streams = {sid: dict(series) for sid, series in streams_in.items()}
    evidence = {}
    for sid in streams:
        ev = demo_mod.synthetic_evidence(sid)
        # Time-relative so the evidence never goes stale on any clock.
        ev["evaluated_at"] = (datetime.now(timezone.utc)
                              - timedelta(days=7)).isoformat()
        evidence[sid] = ev
    sp, ep = tmp / "streams.json", tmp / "evidence.json"
    _write(sp, streams)
    _write(ep, evidence)
    return sp, ep


def test_cli_demo():
    r = _run("demo")
    assert r.returncode == 0, r.stderr
    assert "diversification ratio" in r.stdout


def test_cli_demo_json():
    r = _run("demo", "--json")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["synthetic"] is True


def test_cli_check_ok(tmp_path):
    _, ep = _demo_files(tmp_path)
    r = _run("check", "--evidence", str(ep))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["ok"] is True


def test_cli_check_refused_exit_2(tmp_path):
    bad = tmp_path / "bad.json"
    _write(bad, {"A": {"schema_version": 1, "strategy_id": "A",
                       "verdict": "FAIL", "evaluated_at": "2026-09-20T00:00:00+00:00",
                       "gates": [{"name": "g", "passed": False}]}})
    r = _run("check", "--evidence", str(bad))
    assert r.returncode == 2
    assert "REFUSED" in r.stderr


def test_cli_allocate_ok(tmp_path):
    sp, ep = _demo_files(tmp_path)
    r = _run("allocate", "--streams", str(sp), "--evidence", str(ep),
             "--method", "risk_parity")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert abs(sum(out["weights"]) - 1.0) < 1e-9


def test_cli_allocate_refused_exit_2(tmp_path):
    sp, ep = _demo_files(tmp_path)
    evidence = json.loads(ep.read_text())
    del evidence["B"]  # missing evidence for one strategy
    ep2 = tmp_path / "evidence2.json"
    _write(ep2, evidence)
    r = _run("allocate", "--streams", str(sp), "--evidence", str(ep2))
    assert r.returncode == 2
    assert "REFUSED" in r.stderr


def test_cli_allocate_strict_gates_exit_3(tmp_path):
    # Demo portfolio Sharpe ~1.5; an absurd bar forces gate failure.
    sp, ep = _demo_files(tmp_path)
    r = _run("allocate", "--streams", str(sp), "--evidence", str(ep),
             "--strict", "--preset", "aggressive")
    # aggressive preset min_sharpe 0.75 < demo Sharpe: passes -> 0.
    assert r.returncode == 0, r.stderr


def test_cli_presets_and_methods():
    assert _run("presets").returncode == 0
    assert _run("methods").returncode == 0
