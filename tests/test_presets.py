"""Preset tests."""

import pytest

from trade_allocate.presets import PRESETS, preset_kwargs, preset_rationale


def test_all_presets_have_kwargs_and_rationale():
    for name in PRESETS:
        kw = preset_kwargs(name)
        assert kw["method"] in ("equal", "risk_parity", "hrp")
        assert 0 < kw["max_weight"] <= 1.0
        assert kw["dust"] >= 0
        assert isinstance(preset_rationale(name), str) and preset_rationale(name)


def test_unknown_preset():
    with pytest.raises(ValueError):
        preset_kwargs("yolo")
    with pytest.raises(ValueError):
        preset_rationale("yolo")


def test_preset_kwargs_usable_by_pipeline():
    from trade_allocate import adapters, demo as demo_mod, pipeline
    from datetime import datetime, timezone
    streams_in = demo_mod.synthetic_streams()
    inputs = {sid: adapters.strategy_input(sid, s, demo_mod.synthetic_evidence(sid))
              for sid, s in streams_in.items()}
    res = pipeline.run_preset(
        inputs, "balanced",
        evidence_kwargs={"now": datetime(2026, 9, 26, tzinfo=timezone.utc)})
    assert abs(sum(res["weights"]) - 1.0) < 1e-9
