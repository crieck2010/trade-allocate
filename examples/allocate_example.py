"""Full walkthrough: build inputs, run the pipeline, read the result.

Run:  python examples/allocate_example.py
"""

from datetime import datetime, timedelta, timezone

from trade_allocate import adapters, demo, pipeline, presets

NOW = datetime.now(timezone.utc)


def fresh_evidence(sid: str) -> dict:
    ev = demo.synthetic_evidence(sid)
    ev["evaluated_at"] = (NOW - timedelta(days=7)).isoformat()
    return ev


def main() -> None:
    # 1. Strategy inputs: return streams + gate-pass evidence.
    streams_in = demo.synthetic_streams()
    inputs = {
        sid: adapters.strategy_input(sid, series, fresh_evidence(sid))
        for sid, series in streams_in.items()
    }

    # 2. Run the balanced preset (risk parity, 0.5 cap, 5pt tolerance).
    print("preset rationale:", presets.preset_rationale("balanced"), "\n")
    res = pipeline.run_preset(inputs, "balanced")

    # 3. Read the mix.
    print("weights:")
    for sid, w in res["weights_by_id"].items():
        print(f"  {sid}: {w:.4f}")
    print("turnover action:", res["turnover_action"])
    print("shrinkage rho*:", round(res["rho_star"], 4))
    print("portfolio gates passed:", res["portfolio_gates_passed"])
    for g in res["portfolio_gates"]["gates"]:
        print(f"  {g['gate']}: {g['value']:.3f} vs {g['threshold']} "
              f"({'PASS' if g['passed'] else 'FAIL'})")

    # 4. Hand the mix downstream.
    print("\nPM payload consumer:", res["pm_payload"]["consumer"])
    print("risk payload max single weight:",
          round(res["risk_payload"]["max_single_weight"], 4))

    # 5. The hard rule, demonstrated: drop one strategy's evidence.
    bad = dict(inputs)
    bad["B"] = adapters.strategy_input("B", bad["B"]["returns"], None)
    try:
        pipeline.run_preset(bad, "balanced")
    except Exception as exc:  # EvidenceRefused
        print("\nmissing evidence ->", exc)


if __name__ == "__main__":
    main()
