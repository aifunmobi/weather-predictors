"""Backtest harness for weather predictors.

Treats a backtest as: signal (target position per period) × forward returns.
Produces Sharpe, t-stat, hit rate, max DD, and full equity curve.

Conventions
-----------
* Position values are in {-1, 0, +1} (or fractional for sized signals).
* Position at time t is held over period [t, t+1] — i.e. no lookahead.
* Returns are LOG returns of the underlying.
* Sharpe is annualized assuming 12 obs/year (monthly) or 252 (daily).
* Honest reporting: always include sample size and t-stat. If t-stat < 2,
  flag as "not statistically distinguishable from zero".
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BacktestResult:
    name: str
    underlying: str
    freq: str                # "monthly" or "daily"
    n_obs: int
    n_active: int            # periods with non-zero position
    pct_active: float
    mean_ret: float          # per-period mean strategy return
    vol_ret: float           # per-period stdev
    sharpe_ann: float        # annualized Sharpe
    t_stat: float            # mean / (vol / sqrt(n))
    p_value: float
    hit_rate: float          # fraction of active periods with positive return
    cagr: float
    total_log_ret: float
    max_dd: float
    calmar: float            # cagr / |max_dd|
    long_frac: float
    short_frac: float
    sample_start: str
    sample_end: str

    @property
    def verdict(self) -> str:
        if self.n_active < 12:
            return "INSUFFICIENT"
        if abs(self.t_stat) < 1.0:
            return "NOISE"
        if abs(self.t_stat) < 2.0:
            return "WEAK"
        if abs(self.sharpe_ann) < 0.3:
            return "WEAK"
        return "REAL"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict
        return d


def run_backtest(name: str,
                 underlying: str,
                 signal: pd.Series,
                 returns: pd.Series,
                 freq: str = "monthly") -> tuple[BacktestResult, pd.DataFrame]:
    """Run a single predictor backtest.

    Parameters
    ----------
    signal : pd.Series
        Position to hold over the NEXT period, indexed by period-end timestamps.
        +1 long, -1 short, 0 flat (or fractional).
    returns : pd.Series
        Log-returns of underlying at same frequency as signal.
    freq : "monthly" | "daily"

    Returns
    -------
    BacktestResult, equity_df
        equity_df has columns: signal, ret, strat_ret, cum_ret, dd
    """
    if freq == "monthly":
        ann = 12
    elif freq == "daily":
        ann = 252
    else:
        raise ValueError(f"Unknown freq {freq!r}")

    # Align on common index. Crucially: shift signal forward by 1 so that the
    # signal observed at time t is applied to the return realized at t+1.
    df = pd.concat({"signal": signal, "ret": returns}, axis=1).dropna(subset=["ret"])
    df["signal"] = df["signal"].ffill().fillna(0)
    df["pos"] = df["signal"].shift(1).fillna(0)
    df["strat_ret"] = df["pos"] * df["ret"]

    sr = df["strat_ret"]
    active = df["pos"] != 0
    n_obs = int(len(sr))
    n_active = int(active.sum())

    if n_active == 0:
        empty = BacktestResult(
            name=name, underlying=underlying, freq=freq,
            n_obs=n_obs, n_active=0, pct_active=0.0,
            mean_ret=0.0, vol_ret=0.0, sharpe_ann=0.0,
            t_stat=0.0, p_value=1.0, hit_rate=0.0,
            cagr=0.0, total_log_ret=0.0, max_dd=0.0, calmar=0.0,
            long_frac=0.0, short_frac=0.0,
            sample_start=str(df.index[0].date()) if len(df) else "",
            sample_end=str(df.index[-1].date()) if len(df) else "",
        )
        return empty, df

    mean_ret = float(sr.mean())
    vol_ret = float(sr.std(ddof=1))
    sharpe = (mean_ret / vol_ret) * np.sqrt(ann) if vol_ret > 0 else 0.0
    t_stat = (mean_ret / (vol_ret / np.sqrt(n_obs))) if vol_ret > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat)))) if vol_ret > 0 else 1.0

    # Hit rate on ACTIVE periods only
    active_ret = sr[active]
    hit_rate = float((active_ret > 0).mean()) if len(active_ret) > 0 else 0.0

    # Cumulative log return + equity curve + drawdown
    df["cum_ret"] = sr.cumsum()
    equity = np.exp(df["cum_ret"])
    peak = equity.cummax()
    df["dd"] = equity / peak - 1.0
    max_dd = float(df["dd"].min())

    total_log = float(sr.sum())
    n_years = n_obs / ann
    cagr = float(np.expm1(total_log / n_years)) if n_years > 0 else 0.0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    long_frac = float((df["pos"] > 0).mean())
    short_frac = float((df["pos"] < 0).mean())

    res = BacktestResult(
        name=name, underlying=underlying, freq=freq,
        n_obs=n_obs, n_active=n_active, pct_active=n_active / n_obs,
        mean_ret=mean_ret, vol_ret=vol_ret, sharpe_ann=sharpe,
        t_stat=t_stat, p_value=p_value, hit_rate=hit_rate,
        cagr=cagr, total_log_ret=total_log, max_dd=max_dd, calmar=calmar,
        long_frac=long_frac, short_frac=short_frac,
        sample_start=str(df.index[0].date()),
        sample_end=str(df.index[-1].date()),
    )
    return res, df


def summary_row(res: BacktestResult) -> dict:
    """Compact dict for table reporting."""
    return {
        "Predictor": res.name,
        "Asset": res.underlying,
        "N": res.n_obs,
        "Active": res.n_active,
        "Sharpe": f"{res.sharpe_ann:+.2f}",
        "t-stat": f"{res.t_stat:+.2f}",
        "Hit %": f"{res.hit_rate * 100:.1f}",
        "CAGR": f"{res.cagr * 100:+.1f}%",
        "MaxDD": f"{res.max_dd * 100:.1f}%",
        "Calmar": f"{res.calmar:+.2f}" if abs(res.calmar) < 100 else "—",
        "Period": f"{res.sample_start} → {res.sample_end}",
        "Verdict": res.verdict,
    }
