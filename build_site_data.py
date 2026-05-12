"""Bundle backtest results into a single JSON for the interactive site."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT / "site" / "data.json"
OUT.parent.mkdir(exist_ok=True)


def load_curve(csv_path: Path) -> list[dict]:
    """Load a backtest CSV → compact list of {d,e,dd} (date, equity, drawdown)."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    # Equity = cum_ret if it exists else cumsum of strat_ret
    if "cum_ret" in df.columns:
        eq = np.exp(df["cum_ret"])
    else:
        eq = np.exp(df["strat_ret"].cumsum())
    if "dd" in df.columns:
        dd = df["dd"]
    else:
        dd = eq / eq.cummax() - 1
    return [
        {"d": ts.strftime("%Y-%m-%d"), "e": round(float(e), 4), "dd": round(float(d), 4)}
        for ts, e, d in zip(df.index, eq.values, dd.values)
    ]


def load_buyhold(csv_path: Path) -> list[dict]:
    """Buy-and-hold equity from a predictor CSV (uses raw 'ret' column)."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    eq = np.exp(df["ret"].cumsum())
    dd = eq / eq.cummax() - 1
    return [
        {"d": ts.strftime("%Y-%m-%d"), "e": round(float(e), 4), "dd": round(float(d), 4)}
        for ts, e, d in zip(df.index, eq.values, dd.values)
    ]


def main():
    summary = json.loads((RES / "summary.json").read_text())

    # Per-predictor equity curves
    curves = {}
    curves["P1.NG"]     = load_curve(RES / "p1_enso_overlay_NG.csv")
    curves["P1.KC"]     = load_curve(RES / "p1_enso_overlay_KC.csv")
    curves["P1.CC"]     = load_curve(RES / "p1_enso_overlay_CC.csv")
    curves["P1.OJ"]     = load_curve(RES / "p1_enso_overlay_OJ.csv")
    curves["P2.KC"]     = load_curve(RES / "p2_enso_lag_KC.csv")
    curves["P3.pre"]    = load_curve(RES / "p3_hurricane_pre_peak_NG.csv")
    curves["P3.post"]   = load_curve(RES / "p3_hurricane_post_peak_NG.csv")

    # Inverted versions of the two real signals (for the equity overlay panel)
    def invert(curve):
        # Recompute from CSV: invert the strategy return
        return curve  # placeholder — we'll do it differently

    p2_df = pd.read_csv(RES / "p2_enso_lag_KC.csv", index_col=0, parse_dates=True)
    p2_inv_eq = np.exp((-p2_df["strat_ret"]).cumsum())
    p2_inv_dd = p2_inv_eq / p2_inv_eq.cummax() - 1
    curves["P2.KC.inv"] = [
        {"d": ts.strftime("%Y-%m-%d"), "e": round(float(e), 4), "dd": round(float(d), 4)}
        for ts, e, d in zip(p2_df.index, p2_inv_eq.values, p2_inv_dd.values)
    ]

    p3_df = pd.read_csv(RES / "p3_hurricane_post_peak_NG.csv", index_col=0, parse_dates=True)
    p3_inv_eq = np.exp((-p3_df["strat_ret"]).cumsum())
    p3_inv_dd = p3_inv_eq / p3_inv_eq.cummax() - 1
    curves["P3.post.inv"] = [
        {"d": ts.strftime("%Y-%m-%d"), "e": round(float(e), 4), "dd": round(float(d), 4)}
        for ts, e, d in zip(p3_df.index, p3_inv_eq.values, p3_inv_dd.values)
    ]

    # Buy-and-hold benchmarks
    bh = {}
    bh["NG"] = load_buyhold(RES / "p1_enso_overlay_NG.csv")
    bh["KC"] = load_buyhold(RES / "p1_enso_overlay_KC.csv")
    bh["CC"] = load_buyhold(RES / "p1_enso_overlay_CC.csv")
    bh["OJ"] = load_buyhold(RES / "p1_enso_overlay_OJ.csv")

    # P2 lag sensitivity sweep
    lag_df = pd.read_csv(RES / "p2_lag_sensitivity.csv")
    lag_sweep = [
        {"lag": int(r["lag"]), "sharpe": round(float(r["sharpe"]), 3),
         "t_stat": round(float(r["t_stat"]), 3),
         "hit_rate": round(float(r["hit_rate"]), 3),
         "n_active": int(r["n_active"])}
        for _, r in lag_df.iterrows()
    ]

    # ENSO timeline (ONI anomaly)
    oni_df = pd.read_csv(ROOT / "data" / "oni.csv", index_col=0, parse_dates=True)
    oni_df = oni_df[oni_df.index >= "1990-01-01"]
    oni_series = [
        {"d": ts.strftime("%Y-%m"), "a": round(float(a), 2)}
        for ts, a in zip(oni_df.index, oni_df["anom"].values)
    ]

    # NG seasonal pattern
    ng_df = pd.read_csv(RES / "p1_enso_overlay_NG.csv", index_col=0, parse_dates=True)
    months = list(range(1, 13))
    seasonal = []
    for m in months:
        mret = ng_df[ng_df.index.month == m]["ret"]
        seasonal.append({
            "month": m,
            "mean": round(float(mret.mean()) * 100, 2),
            "se":   round(float(mret.std() / np.sqrt(len(mret))) * 100, 2),
            "n":    int(len(mret)),
        })

    out = {
        "summary": summary,
        "curves": curves,
        "buyhold": bh,
        "lag_sweep": lag_sweep,
        "oni": oni_series,
        "ng_seasonal": seasonal,
        "meta": {
            "generated": pd.Timestamp.now().isoformat(),
            "sample_start": "2000-01-01",
            "sample_end": "2026-05-11",
        }
    }
    OUT.write_text(json.dumps(out))
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    print(f"  Curves: {len(curves)} series")
    print(f"  Lag sweep: {len(lag_sweep)} entries")
    print(f"  ONI: {len(oni_series)} months")


if __name__ == "__main__":
    main()
