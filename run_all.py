"""Orchestrator: run all 5 predictors, write results.json, print summary table."""
from __future__ import annotations
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from predictors import (p1_enso_overlay, p2_enso_lag,
                        p3_hurricane_premium, p4_storm_surprise,
                        p5_winter_enso)


def run_all():
    print("Running P1 ENSO Overlay...");          p1 = p1_enso_overlay.run()
    print("Running P2 ENSO Lag-13...");           p2 = p2_enso_lag.run()
    print("Running P3 Hurricane Premium...");     p3 = p3_hurricane_premium.run()
    print("Running P4 Storm Surprise...");        p4 = p4_storm_surprise.run()
    print("Running P5 Winter ENSO...");           p5 = p5_winter_enso.run()

    rows = []
    def emit(key, name, asset, r):
        rows.append({
            "key":      key,
            "name":     name,
            "asset":    asset,
            "n_obs":    r.get("n_obs", 0),
            "n_active": r.get("n_active", 0),
            "sharpe":   r.get("sharpe_ann", 0.0),
            "t_stat":   r.get("t_stat", 0.0),
            "hit":      r.get("hit_rate", 0.0),
            "max_dd":   r.get("max_dd", 0.0),
            "cagr":     r.get("cagr", None),
            "calmar":   r.get("calmar", None),
            "verdict":  r.get("verdict", "?"),
            "start":    r.get("sample_start", ""),
            "end":      r.get("sample_end", ""),
        })

    for asset, r in p1.items(): emit(f"P1.{asset}", f"P1 ENSO Overlay",       asset, r)
    for asset, r in p2.items(): emit(f"P2.{asset}", f"P2 ENSO Lag-13",        asset, r)
    for variant, r in p3.items(): emit(f"P3.{variant}", f"P3 Hurricane {variant}", "NG", r)
    for asset, r in p4.items(): emit(f"P4.{asset}", f"P4 Storm Surprise",     "NG (Sep)", r)
    for asset, r in p5.items(): emit(f"P5.{asset}", f"P5 Winter ENSO",        "NG (Winter)", r)

    summary = {"predictors": rows}
    with open(ROOT / "results" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print(f"{'Predictor':<28} {'Asset':<12} {'N':>4} {'Act':>4} "
          f"{'Sharpe':>8} {'t-stat':>8} {'Hit%':>6} {'MaxDD':>8}  Verdict")
    print("-" * 100)
    for r in rows:
        print(f"{r['name']:<28} {r['asset']:<12} {r['n_obs']:>4} {r['n_active']:>4} "
              f"{r['sharpe']:+8.2f} {r['t_stat']:+8.2f} "
              f"{r['hit']*100:5.1f}% {r['max_dd']*100:7.1f}%  {r['verdict']}")
    return summary


if __name__ == "__main__":
    run_all()
