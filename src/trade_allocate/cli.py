"""CLI: trade-allocate demo | allocate | check | presets | methods."""

from __future__ import annotations

import argparse
import json
import sys

from . import adapters, demo as demo_mod, licensing, pipeline, presets as presets_mod
from .evidence import EvidenceRefused, validate_all
from .portfolio import PortfolioGateFailed


def _load_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def cmd_demo(args: argparse.Namespace) -> int:
    result = demo_mod.run_demo(preset_name=args.preset)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        demo_mod.print_demo(result)
    return 0


def cmd_allocate(args: argparse.Namespace) -> int:
    streams = _load_json(args.streams)
    evidence = _load_json(args.evidence)
    inputs = {}
    for sid, series in streams.items():
        ev = (evidence or {}).get(sid)
        inputs[sid] = adapters.strategy_input(sid, series, ev)
    kwargs: dict = {}
    if args.preset:
        kw = presets_mod.preset_kwargs(args.preset)
        method = kw.pop("method")
        kwargs.update(kw)
    else:
        method = args.method
        kwargs.update(max_weight=args.max_weight, dust=args.dust,
                      tolerance=args.tolerance)
    if args.current:
        kwargs["current_weights"] = _load_json(args.current)
    if args.lifecycle:
        kwargs["lifecycle"] = adapters.lifecycle_states(_load_json(args.lifecycle))
    if args.recent:
        kwargs["recent"] = _load_json(args.recent)
    kwargs["strict_portfolio_gates"] = args.strict
    try:
        result = pipeline.run(inputs, method=method, **kwargs)
    except EvidenceRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except PortfolioGateFailed as exc:
        print(f"PORTFOLIO GATES FAILED: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    evidence = _load_json(args.evidence)
    try:
        normed = validate_all(evidence)
    except EvidenceRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "strategies": sorted(normed),
                      "ages_days": {s: round(n["age_days"], 1)
                                    for s, n in normed.items()}},
                     indent=2))
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    for name in sorted(presets_mod.PRESETS):
        print(f"== {name} ==")
        kw = presets_mod.preset_kwargs(name)
        print("  " + json.dumps(kw, default=str))
        print(f"  rationale: {presets_mod.preset_rationale(name)}\n")
    return 0


def cmd_methods(args: argparse.Namespace) -> int:
    print("""equal        1/N baseline. No estimation, no optimization. The honest
             benchmark; use when histories are too short to estimate
             correlation, or to keep fancier methods honest.
risk_parity  Equal risk contribution via cyclical coordinate descent.
             Each strategy contributes 1/N of portfolio variance. The
             default: balances without needing expected returns.
hrp          Hierarchical risk parity: single-linkage clustering on
             correlation distance + recursive bisection by inverse
             cluster variance. Never inverts the covariance matrix —
             most robust to estimation error; pairs with the
             conservative preset.""")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trade-allocate",
                                description="Allocate capital across validated "
                                            "strategies by co-movement.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="Seeded synthetic diversification demo.")
    d.add_argument("--preset", default="balanced",
                   choices=sorted(presets_mod.PRESETS))
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_demo)

    a = sub.add_parser("allocate", help="Run the allocation pipeline.")
    a.add_argument("--streams", required=True,
                   help="JSON: {strategy_id: {ts: ret} or [(ts, ret), ...]}")
    a.add_argument("--evidence", required=True,
                   help="JSON: {strategy_id: gate-pass evidence}")
    a.add_argument("--method", default="risk_parity",
                   choices=("equal", "risk_parity", "hrp"))
    a.add_argument("--preset", choices=sorted(presets_mod.PRESETS),
                   help="Use a preset instead of manual knobs.")
    a.add_argument("--max-weight", type=float, default=0.5)
    a.add_argument("--dust", type=float, default=1e-6)
    a.add_argument("--tolerance", type=float, default=0.05)
    a.add_argument("--current",
                   help="JSON list of current weights (turnover damping).")
    a.add_argument("--lifecycle", help="JSON: {strategy_id: lifecycle state}")
    a.add_argument("--recent", help="JSON: {strategy_id: {sharpe, max_drawdown}}")
    a.add_argument("--strict", action="store_true",
                   help="Fail (exit 3) when portfolio gates fail.")
    a.set_defaults(func=cmd_allocate)

    c = sub.add_parser("check", help="Validate gate-pass evidence (exit 2 if refused).")
    c.add_argument("--evidence", required=True)
    c.set_defaults(func=cmd_check)

    pr = sub.add_parser("presets", help="List presets and rationale.")
    pr.set_defaults(func=cmd_presets)

    m = sub.add_parser("methods", help="Describe the allocation methods.")
    m.set_defaults(func=cmd_methods)

    p.add_argument("--version", action="version", version="trade-allocate 0.1.0")
    return p


def main(argv: list[str] | None = None) -> int:
    licensing.check_license()  # hook: community tier always valid in v0.1.x
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
