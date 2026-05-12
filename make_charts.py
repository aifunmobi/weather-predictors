"""Generate publication-quality charts for the weather-predictors report."""
from __future__ import annotations
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "charts"
RESULTS = ROOT / "results"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY  = "#1a3a4a"
TEAL  = "#2a9d8f"
GOLD  = "#c8963e"
GREEN = "#3aa674"
AMBER = "#e0a93b"
RED   = "#c0392b"
GREY  = "#7f8c8d"
TEXT  = "#2c3e50"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": NAVY,
    "axes.labelsize": 10.5, "axes.labelcolor": TEXT,
    "axes.edgecolor": "#999999", "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.frameon": False, "legend.fontsize": 9.5,
    "grid.color": "#e0e0e0", "grid.linewidth": 0.5,
})


def load_summary():
    return json.loads((RESULTS / "summary.json").read_text())


# --------------------------------------------------------------------------
# Chart 1: Sharpe ratio bar chart with verdict color
# --------------------------------------------------------------------------
def chart_sharpe_bar():
    s = load_summary()
    rows = s["predictors"]
    rows = sorted(rows, key=lambda r: r["sharpe"])
    labels = [f'{r["name"].replace("P","P").split("(")[0].strip()} → {r["asset"]}'
              for r in rows]
    sharpes = [r["sharpe"] for r in rows]
    verdicts = [r["verdict"] for r in rows]
    color_map = {"REAL": GREEN, "WEAK": AMBER, "NOISE": RED, "INSUFFICIENT": GREY}
    colors = [color_map.get(v, GREY) for v in verdicts]

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    y = np.arange(len(labels))
    bars = ax.barh(y, sharpes, color=colors, height=0.65, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.axvline(0, color="#444", linewidth=0.7)
    ax.axvline(+0.3, color=GREEN, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.axvline(-0.3, color=GREEN, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlabel("Annualized Sharpe ratio (as-written; sign matters)")
    ax.set_title("Per-Predictor Sharpe — Academic Claim As-Written", pad=14)

    for bar, val, verd in zip(bars, sharpes, verdicts):
        x = bar.get_width()
        pad = 0.02 if x >= 0 else -0.02
        ha = "left" if x >= 0 else "right"
        ax.text(x + pad, bar.get_y() + bar.get_height()/2,
                f"{val:+.2f}", ha=ha, va="center", fontsize=9, fontweight="bold",
                color=TEXT)

    from matplotlib.patches import Patch
    legend = [Patch(color=GREEN, label="REAL (|t|≥2)"),
              Patch(color=AMBER, label="WEAK (1≤|t|<2)"),
              Patch(color=RED,   label="NOISE (|t|<1)")]
    ax.legend(handles=legend, loc="lower right", framealpha=1.0)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "01-sharpe-bar.png")
    plt.close()


# --------------------------------------------------------------------------
# Chart 2: t-stat | hit-rate scatter
# --------------------------------------------------------------------------
def chart_tstat_hitrate():
    s = load_summary()
    rows = s["predictors"]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    color_map = {"REAL": GREEN, "WEAK": AMBER, "NOISE": RED, "INSUFFICIENT": GREY}
    for r in rows:
        ax.scatter(r["t_stat"], r["hit"] * 100,
                   s=140, color=color_map.get(r["verdict"], GREY),
                   edgecolor=NAVY, linewidth=1.0, alpha=0.85, zorder=3)
        ax.annotate(f'{r["key"]}', (r["t_stat"], r["hit"] * 100),
                    xytext=(7, 6), textcoords="offset points",
                    fontsize=8.5, color=TEXT)
    ax.axvline(0, color="#444", linewidth=0.6)
    ax.axhline(50, color="#444", linewidth=0.6, linestyle=":")
    ax.axvspan(-2, 2, alpha=0.07, color=AMBER, zorder=0)
    ax.axvspan(-1, 1, alpha=0.12, color=RED, zorder=0)
    ax.set_xlabel("t-statistic (mean strategy return)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title("Significance × Hit Rate — All Predictor Variants", pad=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "02-tstat-hitrate.png")
    plt.close()


# --------------------------------------------------------------------------
# Chart 3: P2 ENSO lag sensitivity sweep
# --------------------------------------------------------------------------
def chart_p2_lag_sweep():
    df = pd.read_csv(RESULTS / "p2_lag_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(10, 5.0))
    colors = [GREEN if abs(t) >= 2 and s > 0
              else RED if abs(t) >= 2 and s < 0
              else AMBER if abs(t) >= 1 else GREY
              for s, t in zip(df["sharpe"], df["t_stat"])]
    bars = ax.bar(df["lag"], df["sharpe"], color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#444", linewidth=0.7)
    ax.axhspan(-0.3, 0.3, alpha=0.05, color=GREY)
    # Mark the academic-claim range
    ax.axvspan(13, 15, alpha=0.10, color=GOLD, zorder=0)
    ax.text(14, ax.get_ylim()[1] * 0.85,
            "Academic claim\n(13–15 mo)", ha="center", fontsize=9,
            color=GOLD, fontweight="bold")

    ax.set_xticks(df["lag"])
    ax.set_xlabel("ENSO anomaly lag (months)")
    ax.set_ylabel("Sharpe (long La Niña-lag → KC)")
    ax.set_title("P2 — ENSO Lag Sensitivity Sweep (KC Coffee, 2000–2026)",
                 pad=14)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "03-p2-lag-sweep.png")
    plt.close()


# --------------------------------------------------------------------------
# Chart 4: Equity curve — best surviving predictors
# --------------------------------------------------------------------------
def chart_best_equity():
    """Equity curves: P2 INVERTED (flip P2 KC) and P3 post_peak INVERTED."""
    fig, ax = plt.subplots(figsize=(10, 5.6))

    # P2 inverted (academic claim flipped)
    p2 = pd.read_csv(RESULTS / "p2_enso_lag_KC.csv", index_col=0, parse_dates=True)
    p2["inv_strat"] = -p2["strat_ret"]
    p2["inv_eq"] = np.exp(p2["inv_strat"].cumsum())
    ax.plot(p2.index, p2["inv_eq"], color=GREEN, linewidth=2.0,
            label="P2 inverted: short La Niña-lag KC, long El Niño-lag KC")

    # P3 post_peak inverted (long Sep-Oct)
    p3 = pd.read_csv(RESULTS / "p3_hurricane_post_peak_NG.csv", index_col=0, parse_dates=True)
    p3["inv_strat"] = -p3["strat_ret"]
    p3["inv_eq"] = np.exp(p3["inv_strat"].cumsum())
    ax.plot(p3.index, p3["inv_eq"], color=TEAL, linewidth=2.0,
            label="P3 inverted: long NG Sep-Oct (hurricane premium accrual)")

    # Buy-and-hold KC and NG as benchmarks
    p1_kc = pd.read_csv(RESULTS / "p1_enso_overlay_KC.csv", index_col=0, parse_dates=True)
    bh_kc_eq = np.exp(p1_kc["ret"].cumsum())
    ax.plot(p1_kc.index, bh_kc_eq, color=GOLD, linewidth=1.2, alpha=0.6,
            linestyle="--", label="Buy-and-hold KC (benchmark)")

    p1_ng = pd.read_csv(RESULTS / "p1_enso_overlay_NG.csv", index_col=0, parse_dates=True)
    bh_ng_eq = np.exp(p1_ng["ret"].cumsum())
    ax.plot(p1_ng.index, bh_ng_eq, color=GREY, linewidth=1.2, alpha=0.6,
            linestyle="--", label="Buy-and-hold NG (benchmark)")

    ax.set_yscale("log")
    ax.set_ylabel("Cumulative log equity (1.0 = start)")
    ax.set_xlabel("Year")
    ax.set_title("Equity Curves — Inverted Significant Signals vs Buy-and-Hold",
                 pad=14)
    ax.axhline(1.0, color="#444", linewidth=0.6)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "04-best-equity.png")
    plt.close()


# --------------------------------------------------------------------------
# Chart 5: ENSO state timeline (1990-now) with ONI
# --------------------------------------------------------------------------
def chart_enso_timeline():
    sys.path.insert(0, str(ROOT))
    from lib.data import load_oni
    oni = load_oni()
    oni = oni[oni.index >= "1990-01-01"]
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    a = oni["anom"]
    ax.fill_between(a.index, 0, a.where(a >= 0), color=RED, alpha=0.6,
                    label="El Niño (anom > 0)")
    ax.fill_between(a.index, 0, a.where(a < 0), color=TEAL, alpha=0.6,
                    label="La Niña (anom < 0)")
    ax.axhline(+0.5, color=GREY, linestyle=":", linewidth=0.7)
    ax.axhline(-0.5, color=GREY, linestyle=":", linewidth=0.7)
    ax.axhline(0, color="#444", linewidth=0.6)
    ax.set_ylabel("ONI anomaly (°C)")
    ax.set_xlabel("Year")
    ax.set_title("ENSO State 1990–2026 (NOAA ONI anomaly)", pad=14)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "05-enso-timeline.png")
    plt.close()


# --------------------------------------------------------------------------
# Chart 6: NG seasonal returns by month (bar)
# --------------------------------------------------------------------------
def chart_ng_seasonal():
    sys.path.insert(0, str(ROOT))
    from lib.data import monthly_returns
    ng = monthly_returns("NG")
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    avg = [ng[ng.index.month == m].mean() for m in range(1, 13)]
    std = [ng[ng.index.month == m].std() / np.sqrt(len(ng[ng.index.month == m]))
           for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = [GREEN if v > 0 else RED for v in avg]
    bars = ax.bar(months, [v * 100 for v in avg], color=colors,
                  yerr=[s * 100 for s in std], capsize=4,
                  edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="#444", linewidth=0.7)
    ax.axvspan(4.5, 10.5, alpha=0.07, color=AMBER, zorder=0)
    ax.text(7.5, max(v * 100 for v in avg) * 0.9,
            "Atlantic Hurricane Season (Jun–Nov)",
            ha="center", color=AMBER, fontweight="bold", fontsize=9.5)
    ax.set_ylabel("Mean monthly log return (%)")
    ax.set_title("NG Front-Month — Average Monthly Return by Calendar Month (2000–2026)",
                 pad=14)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "06-ng-seasonal.png")
    plt.close()


# --------------------------------------------------------------------------
# Chart 7: Storm surprise vs Sep NG return scatter
# --------------------------------------------------------------------------
def chart_storm_surprise_scatter():
    df = pd.read_csv(RESULTS / "p4_storm_surprise_NG.csv", index_col=0, parse_dates=True)
    # df has columns: signal, ret, strat_ret
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    # We want the raw surprise vs raw return. Reconstruct:
    from lib.data import storm_surprise
    surp_df = storm_surprise().reset_index()  # year becomes a column
    aligned = pd.DataFrame({
        "surprise": surp_df["surprise_named"].values,
        "year": surp_df["year"].values,
    })
    df2 = df.reset_index().rename(columns={"index": "ts"})
    df2["year"] = pd.to_datetime(df2["ts"]).dt.year
    merged = aligned.merge(df2[["year", "ret"]], on="year", how="inner")

    color = [GREEN if r > 0 else RED for r in merged["ret"]]
    ax.scatter(merged["surprise"], merged["ret"] * 100, s=80,
               color=color, edgecolor=NAVY, linewidth=0.8, alpha=0.85)
    ax.axhline(0, color="#444", linewidth=0.7)
    ax.axvline(0, color="#444", linewidth=0.7)

    # Fit line
    x = merged["surprise"].values.astype(float)
    y = (merged["ret"] * 100).values.astype(float)
    if len(x) > 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, m * xs + b, color=NAVY, linewidth=1.5, alpha=0.7,
                label=f"Linear fit: slope = {m:+.2f}%/storm")

    for _, row in merged.iterrows():
        ax.annotate(str(int(row["year"])), (row["surprise"], row["ret"] * 100),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=GREY)

    ax.set_xlabel("Storm count surprise (realized − CSU April forecast)")
    ax.set_ylabel("NG August → September log return (%)")
    ax.set_title("P4 — Storm Surprise vs Aug→Sep NG Return", pad=14)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "07-storm-scatter.png")
    plt.close()


# --------------------------------------------------------------------------
# Chart 8: Verdict distribution
# --------------------------------------------------------------------------
def chart_verdict_distribution():
    s = load_summary()
    rows = s["predictors"]
    verdicts = [r["verdict"] for r in rows]
    counts = {"REAL": 0, "WEAK": 0, "NOISE": 0, "INSUFFICIENT": 0}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1

    fig, ax = plt.subplots(figsize=(8, 4.4))
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [GREEN, AMBER, RED, GREY]
    bars = ax.bar(labels, values, color=colors, width=0.55,
                  edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(v), ha="center", fontsize=12, fontweight="bold", color=TEXT)
    ax.set_ylabel("Count of predictor variants")
    ax.set_title(f"Verdict Distribution Across {len(rows)} Predictor Variants",
                 pad=14)
    ax.set_ylim(0, max(values) + 1.2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "08-verdict-distribution.png")
    plt.close()


if __name__ == "__main__":
    chart_sharpe_bar()
    chart_tstat_hitrate()
    chart_p2_lag_sweep()
    chart_best_equity()
    chart_enso_timeline()
    chart_ng_seasonal()
    chart_storm_surprise_scatter()
    chart_verdict_distribution()
    print("Charts written to:", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" -", p.name)
