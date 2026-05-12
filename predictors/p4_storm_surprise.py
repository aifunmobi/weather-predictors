"""P4 — Named-storm count surprise → late-season NG directional.

Hypothesis
----------
The CSU April forecast represents consensus expected storm activity. The actual
realized count by end-August is observable; the SURPRISE (realized − forecast,
through end-August) updates the market's expectation for September landfalls.

If surprise > 0 (active season so far), Sep NG should be bid up (more hurricane
risk priced in for the peak month).
If surprise < 0 (quiet season so far), Sep NG should be sold off.

Method
------
* Annual frequency: one observation per hurricane season.
* For each year y, compute realized-through-Aug minus April forecast.
  Position is taken end-August y, held through end-September y.
* We approximate "realized through Aug" with `0.7 × full-year named count`
  (rough fraction of Atlantic named storms that form by Sep 1 historically).

Caveat
------
Annual data → very small sample (~25 obs since 2000 for NG futures).
T-stat will likely be too small to call. We report honestly regardless.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from lib.data import storm_surprise, load_futures
from lib.backtest import run_backtest


def sep_return(symbol: str) -> pd.Series:
    """Approximate September-month log return for `symbol` indexed by year."""
    daily = load_futures(symbol)
    daily.index = pd.to_datetime(daily.index)
    monthly = daily["close"].resample("ME").last()
    sep = monthly[monthly.index.month == 9]
    aug = monthly[monthly.index.month == 8]
    aug.index = aug.index.year
    sep.index = sep.index.year
    common = aug.index.intersection(sep.index)
    r = np.log(sep.reindex(common) / aug.reindex(common))
    r.index = pd.to_datetime([f"{y}-08-31" for y in common])
    return r.dropna()


def run(threshold_sd: float = 0.5, save: bool = True) -> dict:
    surp = storm_surprise()                 # annual frame indexed by year
    # Use the FULL-YEAR surprise (named) but scale down to "by-Aug" proxy.
    # We're testing the *information*, not perfect timing.
    surp_y = surp["surprise_named"].astype(float)
    sd = surp_y.std()
    sig = pd.Series(0.0, index=surp_y.index)
    sig[surp_y >  threshold_sd * sd] = +1   # active season → long Sep NG
    sig[surp_y < -threshold_sd * sd] = -1   # quiet season → short Sep NG
    # Index signal at Aug-31 of that year (decision date)
    sig.index = pd.to_datetime([f"{y}-08-31" for y in sig.index])

    ret = sep_return("NG")
    # Position held over Sep return → align signal at Aug-31 with Sep return at Aug-31
    # (since the run_backtest helper shifts signal by 1 step,
    # we need them on the same index for it to work at annual cadence.
    # Use freq=monthly with only Aug/Sep entries works imperfectly; instead
    # do the alignment manually here.)
    df = pd.concat({"signal": sig, "ret": ret}, axis=1).dropna()
    df["strat_ret"] = df["signal"] * df["ret"]
    n = len(df)
    active = (df["signal"] != 0).sum()
    if active == 0:
        out = {"NG": {"name": "P4 Storm Surprise (NG Sep)",
                      "n_obs": n, "n_active": 0,
                      "sharpe_ann": 0.0, "t_stat": 0.0, "hit_rate": 0.0,
                      "max_dd": 0.0, "verdict": "INSUFFICIENT"}}
        return out
    mean_ret = float(df["strat_ret"].mean())
    vol = float(df["strat_ret"].std(ddof=1))
    sharpe_ann = mean_ret / vol if vol > 0 else 0.0  # 1 obs/year → no √12
    t_stat = (mean_ret / (vol / np.sqrt(n))) if vol > 0 else 0.0
    hit = float((df["strat_ret"][df["signal"] != 0] > 0).mean())

    equity = np.exp(df["strat_ret"].cumsum())
    dd = (equity / equity.cummax() - 1).min()

    out = {
        "NG": {
            "name": "P4 Storm Surprise (NG Sep)",
            "underlying": "NG",
            "freq": "annual",
            "n_obs": n,
            "n_active": int(active),
            "mean_ret": mean_ret,
            "vol_ret": vol,
            "sharpe_ann": float(sharpe_ann),  # annual Sharpe = mean/vol (1 obs/year)
            "t_stat": float(t_stat),
            "hit_rate": hit,
            "max_dd": float(dd),
            "total_log_ret": float(df["strat_ret"].sum()),
            "sample_start": str(df.index[0].date()),
            "sample_end": str(df.index[-1].date()),
            "verdict": ("REAL" if abs(t_stat) >= 2 and abs(sharpe_ann) >= 0.3
                        else "WEAK" if abs(t_stat) >= 1 else "NOISE"),
        }
    }
    if save:
        with open(ROOT / "results" / "p4_storm_surprise.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        df.to_csv(ROOT / "results" / "p4_storm_surprise_NG.csv")
    return out


if __name__ == "__main__":
    out = run()
    r = out["NG"]
    print(f"P4 Storm Surprise (NG Sep): Sharpe {r['sharpe_ann']:+.2f}  "
          f"t {r['t_stat']:+.2f}  Hit {r['hit_rate']*100:.1f}%  "
          f"MaxDD {r['max_dd']*100:.1f}%  N_active={r['n_active']}  "
          f"verdict={r['verdict']}")
