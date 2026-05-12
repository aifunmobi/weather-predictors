"""P3 — Hurricane pre-season natural gas premium.

Hypothesis
----------
NG front-month carries a hurricane-risk premium during the
Atlantic season (June–November). Specifically:
  - May/June: premium builds (forward-looking risk pricing)
  - Aug/Sep:  peak risk window
  - Oct/Nov:  premium decays

If this is real, going LONG NG from late-May through peak (end of August)
should outperform a flat baseline. We test two variants:

  Variant A — "pre-peak":  long  end-May → end-Aug,    flat rest of year
  Variant B — "post-peak": short end-Aug → end-Oct,    flat rest of year

Method
------
* Monthly frequency, NG=F.
* No lookahead; positions taken at month-end t for return realized in t+1.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from lib.data import monthly_returns
from lib.backtest import run_backtest


def build_signal_pre_peak(returns: pd.Series) -> pd.Series:
    """Long NG from end-May (signal-set at month-end May → holds Jun, Jul, Aug).

    Holds for 3 months (Jun, Jul, Aug); flat otherwise.
    """
    sig = pd.Series(0.0, index=returns.index)
    # Signal value at month-end m determines position for month m+1.
    # We want to be long during Jun, Jul, Aug → set signal at end of May, Jun, Jul.
    months_to_set = {5, 6, 7}
    for ts in sig.index:
        if ts.month in months_to_set:
            sig.loc[ts] = +1.0
    return sig


def build_signal_post_peak(returns: pd.Series) -> pd.Series:
    """Short NG from end-Aug (holds Sep, Oct); flat else."""
    sig = pd.Series(0.0, index=returns.index)
    months_to_set = {8, 9}
    for ts in sig.index:
        if ts.month in months_to_set:
            sig.loc[ts] = -1.0
    return sig


def run(save: bool = True) -> dict:
    ret = monthly_returns("NG")
    ret.index = ret.index.to_period("M").to_timestamp("M")
    out = {}
    for label, builder in [("pre_peak", build_signal_pre_peak),
                           ("post_peak", build_signal_post_peak)]:
        sig = builder(ret)
        res, df = run_backtest(
            name=f"P3 Hurricane {label} (NG)",
            underlying="NG", signal=sig, returns=ret, freq="monthly",
        )
        out[label] = res.as_dict()
        if save:
            df.to_csv(ROOT / "results" / f"p3_hurricane_{label}_NG.csv")
    if save:
        with open(ROOT / "results" / "p3_hurricane_premium.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
    return out


if __name__ == "__main__":
    out = run()
    print(f"{'Variant':<14} {'Sharpe':>8} {'t-stat':>8} {'Hit %':>8} {'MaxDD':>9}  Verdict")
    print("-" * 70)
    for k, r in out.items():
        print(f"{k:<14} {r['sharpe_ann']:+8.2f} {r['t_stat']:+8.2f} "
              f"{r['hit_rate']*100:7.1f}  {r['max_dd']*100:8.1f}%  {r['verdict']}")
