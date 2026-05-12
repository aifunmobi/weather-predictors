"""P2 — ENSO 13-month lag → Arabica coffee.

Hypothesis (academic claim)
---------------------------
ENSO state precedes Arabica coffee prices by 13–15 months. The mechanism is
flowering-stage weather impact on Brazilian/Colombian production that affects
harvest delivery roughly a year later.

We test the directional claim with the cleanest specification:
  signal_t = +1 if ONI[t - 13] anomaly <= -0.5   (La Niña 13 months ago → bullish)
  signal_t = -1 if ONI[t - 13] anomaly >= +0.5   (El Niño 13 months ago → bearish)
  signal_t =  0 else

Method
------
* Monthly frequency, KC=F (coffee Arabica front-month).
* Lag length sweep tested separately in p2_sensitivity.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from lib.data import load_oni, monthly_returns
from lib.backtest import run_backtest


def build_signal(oni: pd.DataFrame, lag_months: int = 13,
                 threshold: float = 0.5) -> pd.Series:
    a = oni["anom"].copy()
    a.index = a.index.to_period("M").to_timestamp("M")
    lagged = a.shift(lag_months)
    sig = pd.Series(0.0, index=lagged.index)
    sig[lagged <= -threshold] = +1
    sig[lagged >=  threshold] = -1
    return sig


def run(lag_months: int = 13, threshold: float = 0.5, save: bool = True) -> dict:
    oni = load_oni()
    ret = monthly_returns("KC")
    ret.index = ret.index.to_period("M").to_timestamp("M")
    sig = build_signal(oni, lag_months=lag_months, threshold=threshold)
    res, df = run_backtest(
        name=f"P2 ENSO Lag-{lag_months} (KC)",
        underlying="KC", signal=sig, returns=ret, freq="monthly",
    )
    out = {"KC": res.as_dict()}
    if save:
        with open(ROOT / "results" / "p2_enso_lag.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        df.to_csv(ROOT / "results" / "p2_enso_lag_KC.csv")
    return out


def sensitivity(threshold: float = 0.5) -> pd.DataFrame:
    """Sweep lag from 1 to 24 months."""
    oni = load_oni()
    ret = monthly_returns("KC")
    ret.index = ret.index.to_period("M").to_timestamp("M")
    rows = []
    for lag in range(1, 25):
        sig = build_signal(oni, lag_months=lag, threshold=threshold)
        res, _ = run_backtest(f"lag={lag}", "KC", sig, ret, "monthly")
        rows.append({"lag": lag, "sharpe": res.sharpe_ann, "t_stat": res.t_stat,
                     "hit_rate": res.hit_rate, "n_active": res.n_active})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = run()
    r = out["KC"]
    print(f"P2 ENSO Lag-13 (KC): Sharpe {r['sharpe_ann']:+.2f}  t {r['t_stat']:+.2f}  "
          f"Hit {r['hit_rate']*100:.1f}%  MaxDD {r['max_dd']*100:.1f}%  "
          f"verdict={r['verdict']}")
    print("\nLag sensitivity sweep (1-24 months):")
    sens = sensitivity()
    sens.to_csv(ROOT / "results" / "p2_lag_sensitivity.csv", index=False)
    print(sens.to_string(index=False))
