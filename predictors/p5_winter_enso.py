"""P5 — Winter heating ENSO → NG Dec–Feb directional.

Hypothesis
----------
La Niña → colder than normal US winter → more heating-degree-days →
bullish NG over the heating season (Dec, Jan, Feb).
El Niño → milder winter → bearish.

Decision date: end of October each year (when CPC Winter Outlook is published
and the ONI O-N-D average can be approximated).
Position held: Nov–Feb log return of NG (4-month exposure).

Method
------
* Annual frequency (one obs per heating season).
* Threshold: |Oct ONI anomaly| >= 0.5.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from lib.data import load_oni, load_futures


def winter_return(symbol: str) -> pd.Series:
    """Nov→Feb log return for `symbol`, indexed by the entry-year (Oct-31)."""
    daily = load_futures(symbol)
    daily.index = pd.to_datetime(daily.index)
    monthly = daily["close"].resample("ME").last()
    rows = []
    for ts in monthly.index:
        if ts.month != 10:
            continue
        y = ts.year
        try:
            oct_px = monthly.loc[ts]
            feb_ts = pd.Timestamp(year=y+1, month=2, day=1) + pd.offsets.MonthEnd(0)
            if feb_ts not in monthly.index:
                continue
            feb_px = monthly.loc[feb_ts]
            r = float(np.log(feb_px / oct_px))
            rows.append((ts, r))
        except KeyError:
            continue
    return pd.Series({ts: r for ts, r in rows}).sort_index()


def run(threshold: float = 0.5, save: bool = True) -> dict:
    oni = load_oni()
    a = oni["anom"]
    a.index = a.index.to_period("M").to_timestamp("M")
    # Find each Oct ONI anom value
    oct_anom = a[a.index.month == 10]

    ret = winter_return("NG")
    common = oct_anom.index.intersection(ret.index)
    a_aligned = oct_anom.reindex(common)
    r_aligned = ret.reindex(common)

    sig = pd.Series(0.0, index=common)
    sig[a_aligned <= -threshold] = +1   # La Niña → long
    sig[a_aligned >=  threshold] = -1   # El Niño → short

    df = pd.DataFrame({"oni_oct": a_aligned, "signal": sig,
                       "winter_ret": r_aligned})
    df["strat_ret"] = df["signal"] * df["winter_ret"]
    n = len(df)
    active = (df["signal"] != 0).sum()
    if active == 0:
        out = {"NG": {"name": "P5 Winter ENSO (NG)", "n_obs": n,
                      "n_active": 0, "sharpe_ann": 0.0, "t_stat": 0.0,
                      "hit_rate": 0.0, "max_dd": 0.0, "verdict": "INSUFFICIENT"}}
        return out

    mean_ret = float(df["strat_ret"].mean())
    vol = float(df["strat_ret"].std(ddof=1))
    sharpe_ann = mean_ret / vol if vol > 0 else 0.0   # 1 obs/year
    t_stat = mean_ret / (vol / np.sqrt(n)) if vol > 0 else 0.0
    hit = float((df["strat_ret"][df["signal"] != 0] > 0).mean())

    eq = np.exp(df["strat_ret"].cumsum())
    dd = float((eq / eq.cummax() - 1).min())

    out = {
        "NG": {
            "name": "P5 Winter ENSO (NG)",
            "underlying": "NG", "freq": "annual",
            "n_obs": n, "n_active": int(active),
            "mean_ret": mean_ret, "vol_ret": vol,
            "sharpe_ann": float(sharpe_ann),
            "t_stat": float(t_stat),
            "hit_rate": hit, "max_dd": dd,
            "total_log_ret": float(df["strat_ret"].sum()),
            "sample_start": str(df.index[0].date()),
            "sample_end": str(df.index[-1].date()),
            "verdict": ("REAL" if abs(t_stat) >= 2 and abs(sharpe_ann) >= 0.3
                        else "WEAK" if abs(t_stat) >= 1 else "NOISE"),
        }
    }
    if save:
        with open(ROOT / "results" / "p5_winter_enso.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        df.to_csv(ROOT / "results" / "p5_winter_enso_NG.csv")
    return out


if __name__ == "__main__":
    out = run()
    r = out["NG"]
    print(f"P5 Winter ENSO (NG): Sharpe {r['sharpe_ann']:+.2f}  "
          f"t {r['t_stat']:+.2f}  Hit {r['hit_rate']*100:.1f}%  "
          f"MaxDD {r['max_dd']*100:.1f}%  N_active={r['n_active']}  "
          f"verdict={r['verdict']}")
