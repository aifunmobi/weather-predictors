"""P1 — ENSO state overlay.

Hypothesis
----------
The ENSO state at month-end (computed from the NOAA ONI anomaly) carries
information about the following month's return in weather-sensitive commodities.
We test the cross-commodity mapping documented in the prior research note:

  El Niño  → bullish OJ, sugar, soybeans (S. America);
              bearish NG (mild US winter), Colombian coffee
  La Niña  → bullish NG (cold US winter), Colombian coffee;
              bearish sugar, soybeans

Mapping used in the test (per asset's directional bias):
  NG  : long if La Niña,  short if El Niño,  flat else
  KC  : long if La Niña,  short if El Niño,  flat else  (Colombian-driven)
  CC  : long if El Niño,  short if La Niña,  flat else  (drought stress)
  OJ  : long if El Niño,  flat else                       (polar vortex risk)

Method
------
* Monthly frequency.
* Signal at month-end t → position over month t+1 (no lookahead).
* Threshold: |anom| >= 0.5 (NOAA's own ENSO definition).
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from lib.data import load_oni, monthly_returns
from lib.backtest import run_backtest, summary_row


def build_signal(oni: pd.DataFrame, asset: str,
                 threshold: float = 0.5) -> pd.Series:
    """Build a monthly long/short/flat signal for `asset` from ONI anomaly."""
    a = oni["anom"].copy()
    a.index = a.index.to_period("M").to_timestamp("M")  # month-end normalize
    if asset in ("NG", "KC"):
        sig = pd.Series(0, index=a.index, dtype=float)
        sig[a <= -threshold] = +1   # La Niña → long
        sig[a >=  threshold] = -1   # El Niño → short
    elif asset == "CC":
        sig = pd.Series(0, index=a.index, dtype=float)
        sig[a >=  threshold] = +1   # El Niño → long
        sig[a <= -threshold] = -1   # La Niña → short
    elif asset == "OJ":
        sig = pd.Series(0, index=a.index, dtype=float)
        sig[a >=  threshold] = +1   # El Niño → long
        # No symmetric La Niña short (OJ freeze risk is asymmetric)
    else:
        raise ValueError(f"unmapped asset {asset!r}")
    return sig


def run(threshold: float = 0.5, save: bool = True) -> dict:
    oni = load_oni()
    out = {}
    series_out = {}

    for asset in ("NG", "KC", "CC", "OJ"):
        ret = monthly_returns(asset)
        ret.index = ret.index.to_period("M").to_timestamp("M")
        sig = build_signal(oni, asset, threshold=threshold)
        res, df = run_backtest(
            name=f"P1 ENSO Overlay ({asset})",
            underlying=asset, signal=sig, returns=ret, freq="monthly",
        )
        out[asset] = res.as_dict()
        series_out[asset] = df

    if save:
        with open(ROOT / "results" / "p1_enso_overlay.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        for asset, df in series_out.items():
            df.to_csv(ROOT / "results" / f"p1_enso_overlay_{asset}.csv")

    return out


if __name__ == "__main__":
    out = run()
    print(f"{'Asset':<6} {'Sharpe':>8} {'t-stat':>8} {'Hit %':>8} {'MaxDD':>9}  N={len(out)}")
    print("-" * 60)
    for asset, r in out.items():
        print(f"{asset:<6} {r['sharpe_ann']:+8.2f} {r['t_stat']:+8.2f} "
              f"{r['hit_rate']*100:7.1f}  {r['max_dd']*100:8.1f}%  "
              f"verdict={r['verdict']}")
