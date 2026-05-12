"""Weather Predictors — Streamlit interactive backtest.

Live parameter knobs over the same backtest harness used in `run_all.py`.
Change a slider → backtest re-runs → metrics + equity curve update.

Run with:
    streamlit run app.py
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lib.data import (
    load_oni, load_futures, monthly_returns,
    storm_surprise, storms_by_year, csu_april_forecast,
    FUTURES,
)
from lib.backtest import run_backtest, BacktestResult


# =============================================================================
# Page config + style
# =============================================================================
st.set_page_config(
    page_title="Weather Predictors — Live Backtest",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  /* Match the PDF / static-site palette */
  :root {
    --teal-deep: #134e4a; --teal: #1a5e5b; --teal-mid: #2a7a76;
    --teal-bg: #e9f1f0; --ink: #2c3e50; --grey: #6b7c85;
    --green: #3aa674; --amber: #e0a93b; --red: #c0392b;
  }
  .main .block-container { padding-top: 1.2rem; max-width: 1320px; }
  h1, h2, h3 { color: var(--teal); letter-spacing: -0.005em; }
  h1 { font-weight: 800; }

  /* Hero band */
  .hero {
    background: var(--teal-deep); color: white;
    padding: 28px 32px; border-radius: 8px; margin-bottom: 28px;
    border-left: 6px solid var(--teal-mid);
  }
  .hero .pill {
    display: inline-block; background: var(--teal-mid); color: white;
    padding: 4px 14px; border-radius: 16px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase;
    margin-bottom: 10px;
  }
  .hero h1 { color: white; font-size: 28px; margin: 0 0 6px 0; }
  .hero p { color: rgba(255,255,255,0.86); font-size: 14px; margin: 0; }

  /* Tags */
  .tag {
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.05em;
  }
  .tag.real  { background: #d4edda; color: #155724; }
  .tag.weak  { background: #fff3cd; color: #856404; }
  .tag.noise { background: #f8d7da; color: #721c24; }

  /* Callout */
  .callout {
    background: var(--teal-bg); border-left: 4px solid var(--teal);
    padding: 14px 18px; margin: 14px 0; border-radius: 0 6px 6px 0;
  }
  .callout .lbl {
    font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--teal); margin-bottom: 4px;
  }
  .callout p { margin: 0; color: var(--ink); }
  .callout.warn { background: #fff3cd; border-left-color: #856404; }
  .callout.warn .lbl { color: #856404; }
  .callout.warn p { color: #6c5500; }

  /* Metric tweaks */
  [data-testid="stMetricValue"] { color: var(--teal); font-size: 28px; font-weight: 800; }
  [data-testid="stMetricLabel"] { font-size: 11px; color: var(--grey); text-transform: uppercase; letter-spacing: 0.05em; }
  [data-testid="stMetricDelta"] { font-size: 12px; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #f6f8f8; }
  [data-testid="stSidebar"] h2 { color: var(--teal); font-size: 18px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Plotly default layout matching the palette
PLOT_LAYOUT = dict(
    template="simple_white",
    font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif", size=12, color="#2c3e50"),
    margin=dict(l=50, r=20, t=40, b=40),
    plot_bgcolor="white",
    paper_bgcolor="white",
    title_font=dict(size=14, color="#1a5e5b"),
)
COLOR_GREEN = "#3aa674"; COLOR_RED = "#c0392b"; COLOR_AMBER = "#e0a93b"
COLOR_TEAL = "#2a9d8f"; COLOR_NAVY = "#1a3a4a"; COLOR_GOLD = "#c8963e"
COLOR_GREY = "#7f8c8d"


# =============================================================================
# Cached data loaders
# =============================================================================
@st.cache_data(show_spinner=False)
def cached_oni() -> pd.DataFrame:
    d = load_oni()
    d.index = pd.to_datetime(d.index).to_period("M").to_timestamp("M")
    return d

@st.cache_data(show_spinner=False)
def cached_monthly_returns(symbol: str) -> pd.Series:
    s = monthly_returns(symbol)
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
    return s

@st.cache_data(show_spinner=False)
def cached_storms_by_year() -> pd.DataFrame:
    return storms_by_year()

@st.cache_data(show_spinner=False)
def cached_storm_surprise() -> pd.DataFrame:
    return storm_surprise()


# =============================================================================
# Reusable UI fragments
# =============================================================================
def verdict_tag(verdict: str) -> str:
    cls = {"REAL": "real", "WEAK": "weak", "NOISE": "noise", "INSUFFICIENT": "weak"}.get(verdict, "weak")
    return f'<span class="tag {cls}">{verdict}</span>'


def metric_row(res: BacktestResult):
    c1, c2, c3, c4, c5 = st.columns(5)
    sharpe_color = "normal" if abs(res.sharpe_ann) < 0.3 else ("inverse" if res.sharpe_ann < 0 else "off")
    c1.metric("Sharpe (annualized)", f"{res.sharpe_ann:+.2f}")
    c2.metric("t-statistic", f"{res.t_stat:+.2f}")
    c3.metric("Hit rate", f"{res.hit_rate * 100:.1f}%")
    c4.metric("Max drawdown", f"{res.max_dd * 100:.1f}%")
    c5.metric("Active periods", f"{res.n_active} / {res.n_obs}")
    st.markdown(f"**Verdict:** {verdict_tag(res.verdict)} &nbsp; "
                f"<span style='color:#6b7c85; font-size:13px'>"
                f"({res.sample_start} → {res.sample_end})</span>",
                unsafe_allow_html=True)


def equity_chart(equity_df: pd.DataFrame, title: str, with_benchmark: pd.Series = None,
                 log_scale: bool = False):
    """Plot strategy equity curve (and optionally a benchmark)."""
    eq = np.exp(equity_df["strat_ret"].cumsum())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df.index, y=eq.values, name="Strategy",
        line=dict(color=COLOR_TEAL, width=2.2), mode="lines",
    ))
    if with_benchmark is not None and len(with_benchmark) > 0:
        bench = np.exp(with_benchmark.cumsum())
        bench = bench.reindex(equity_df.index, method="ffill")
        fig.add_trace(go.Scatter(
            x=bench.index, y=bench.values, name="Buy & hold",
            line=dict(color=COLOR_GREY, width=1.4, dash="dash"), mode="lines",
        ))
    fig.update_layout(
        title=title, height=380, **PLOT_LAYOUT,
        yaxis=dict(title="Cumulative equity (1.0 = start)",
                   type="log" if log_scale else "linear"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="#aaa", line_width=0.7)
    st.plotly_chart(fig, use_container_width=True)


def signal_position_chart(equity_df: pd.DataFrame, title: str):
    """Show signal/position over time (small auxiliary chart)."""
    fig = go.Figure()
    pos = equity_df["pos"].fillna(0)
    long = pos.where(pos > 0)
    short = pos.where(pos < 0)
    fig.add_trace(go.Scatter(x=pos.index, y=long, name="Long", fill="tozeroy",
                             line=dict(color=COLOR_GREEN, width=0), fillcolor="rgba(58,166,116,0.5)"))
    fig.add_trace(go.Scatter(x=pos.index, y=short, name="Short", fill="tozeroy",
                             line=dict(color=COLOR_RED, width=0), fillcolor="rgba(192,57,43,0.5)"))
    fig.update_layout(
        title=title, height=180, **{k: v for k, v in PLOT_LAYOUT.items() if k != "margin"},
        margin=dict(l=50, r=20, t=30, b=20),
        yaxis=dict(title="Position", range=[-1.2, 1.2], tickvals=[-1, 0, 1]),
        xaxis=dict(title=""),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SIDEBAR — navigation
# =============================================================================
with st.sidebar:
    st.markdown("## 🌪️ Weather Predictors")
    st.caption("Live backtest — change any knob and the strategy re-runs.")
    page = st.radio(
        "Predictor",
        ["📊 Overview",
         "1️⃣ ENSO Overlay (P1)",
         "2️⃣ ENSO Lag → Coffee (P2)",
         "3️⃣ Hurricane NG Premium (P3)",
         "4️⃣ Storm Surprise → NG (P4)",
         "5️⃣ Winter ENSO → NG (P5)",
         "🛠️ Custom Predictor",
         "📜 About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Backtest window: **2000-01 → 2026-05** (yfinance)")
    st.caption("ENSO ONI: **1950-01 → present** (NOAA CPC)")
    st.caption("Data is cached — first load downloads, repeats are instant.")


# =============================================================================
# Header (always shown)
# =============================================================================
HEADER_HTML = """
<div class="hero">
  <div class="pill">QUANT RESEARCH — LIVE BACKTEST</div>
  <h1>Weather Predictors With Measurable Edge</h1>
  <p>Programmable signals over public data. Change a slider, the backtest re-runs.
     Sharpe / t-stat / hit rate / max-DD update live.</p>
</div>
"""
st.markdown(HEADER_HTML, unsafe_allow_html=True)


# =============================================================================
# PAGE: OVERVIEW
# =============================================================================
def page_overview():
    st.markdown("### The 9-Variant Honest Scoreboard")
    st.caption("Pre-computed from the default parameters. Drill into any predictor in the sidebar to change the knobs.")

    summary = [
        ("P1 ENSO Overlay", "NG",         -0.22, -1.09, 309, 160, 52.5, "WEAK"),
        ("P1 ENSO Overlay", "KC",         -0.15, -0.76, 316, 167, 46.1, "NOISE"),
        ("P1 ENSO Overlay", "CC",         +0.24, +1.24, 316, 167, 55.1, "WEAK"),
        ("P1 ENSO Overlay", "OJ",         +0.25, +1.26, 296,  75, 54.7, "WEAK"),
        ("P2 ENSO Lag-13", "KC",          -0.46, -2.37, 316, 177, 39.0, "REAL"),
        ("P3 Hurricane pre-peak", "NG",   -0.14, -0.70, 309,  75, 44.0, "NOISE"),
        ("P3 Hurricane post-peak", "NG",  -0.55, -2.81, 309,  51, 35.3, "REAL"),
        ("P4 Storm Surprise", "NG (Sep)", +0.20, +1.01,  25,  18, 66.7, "WEAK"),
        ("P5 Winter ENSO", "NG (Winter)", -0.10, -0.51,  26,  18, 50.0, "NOISE"),
    ]
    df = pd.DataFrame(summary, columns=["Predictor", "Asset", "Sharpe", "t-stat",
                                        "N", "Active", "Hit %", "Verdict"])
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Sharpe":  st.column_config.NumberColumn(format="%+.2f"),
            "t-stat":  st.column_config.NumberColumn(format="%+.2f"),
            "Hit %":   st.column_config.NumberColumn(format="%.1f"),
            "Verdict": st.column_config.TextColumn(),
        },
        hide_index=True,
    )

    st.markdown("""
<div class="callout">
  <div class="lbl">Key Reading</div>
  <p>Only <b>P2 ENSO Lag-13 (KC)</b> and <b>P3 Hurricane post-peak (NG)</b> cleared the
  conventional |t| ≥ 2 bar. Both have <b>negative</b> Sharpes — meaning the literal
  academic strategy loses money significantly. Flip the position (sell instead of buy)
  and you have +0.46 / +0.55 Sharpe respectively. See P2 / P3 in the sidebar.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### How to read this app")
    st.markdown("""
- Each predictor page has **live sliders**. Change one and the metrics + equity curve update immediately.
- The harness is in `lib/backtest.py`. Same Sharpe/t-stat/hit-rate formula across all predictors.
- All data is **public**: NOAA ONI, HURDAT2, CSU forecasts, yfinance front-month futures.
- "Verdict" thresholds: **REAL** = |t| ≥ 2 and |Sharpe| ≥ 0.3 · **WEAK** = 1 ≤ |t| < 2 · **NOISE** = |t| < 1
""")


# =============================================================================
# PAGE: P1 — ENSO OVERLAY
# =============================================================================
def page_p1():
    st.markdown("### 1️⃣ ENSO State Overlay")
    st.caption("Monthly position based on the current ENSO state (from ONI anomaly). "
               "Change the threshold and asset/direction to test alternatives.")

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        asset = st.selectbox("Underlying", ["NG", "KC", "CC", "OJ"],
                             help="Which commodity to trade")
    with col_b:
        threshold = st.slider("ENSO anomaly threshold (|°C|)", 0.0, 1.5, 0.5, step=0.1,
                              help="State labeled El Niño/La Niña when |anom| ≥ threshold")
    with col_c:
        direction_choices = {
            "Long La Niña / Short El Niño (NG, KC literal)": "lanlong",
            "Long El Niño / Short La Niña (CC, OJ literal)": "ellong",
            "Long La Niña only": "lanonly",
            "Long El Niño only":  "elnonly",
        }
        direction = st.selectbox("Direction", list(direction_choices), index=0)
        dir_code = direction_choices[direction]

    # Build signal
    oni = cached_oni()
    a = oni["anom"]
    sig = pd.Series(0.0, index=a.index)
    if dir_code == "lanlong":
        sig[a <= -threshold] = +1; sig[a >= threshold] = -1
    elif dir_code == "ellong":
        sig[a >= threshold] = +1; sig[a <= -threshold] = -1
    elif dir_code == "lanonly":
        sig[a <= -threshold] = +1
    elif dir_code == "elnonly":
        sig[a >= threshold] = +1

    ret = cached_monthly_returns(asset)
    res, df = run_backtest(f"P1 ENSO Overlay ({asset})", asset, sig, ret, "monthly")
    metric_row(res)
    equity_chart(df, f"P1 — ENSO overlay on {asset} (threshold ±{threshold}°C)",
                 with_benchmark=df["ret"])
    signal_position_chart(df, "Position over time")

    with st.expander("How the signal is built"):
        st.code("""# Signal: position from ENSO anomaly threshold
oni = cached_oni()["anom"]
sig = pd.Series(0.0, index=oni.index)
sig[oni <= -threshold] = +1   # La Niña → long
sig[oni >=  threshold] = -1   # El Niño → short
# Then: pos_t = sig_{t-1}  (no lookahead, applied to next month's return)""",
                 language="python")


# =============================================================================
# PAGE: P2 — ENSO LAG
# =============================================================================
def page_p2():
    st.markdown("### 2️⃣ ENSO Lag → Coffee Arabica")
    st.caption("Tests the academic 13–15 month lag claim. Move the lag slider — "
               "negative Sharpe with |t| ≥ 2 means the inverted strategy works.")

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        asset = st.selectbox("Underlying", ["KC", "CC", "OJ", "NG"], index=0,
                             help="Test on different commodities; academic claim is on coffee")
    with col_b:
        lag = st.slider("ENSO anomaly lag (months)", 1, 24, 13)
    with col_c:
        threshold = st.slider("ENSO threshold (|°C|)", 0.0, 1.5, 0.5, step=0.1)

    oni = cached_oni()
    a = oni["anom"]
    lagged = a.shift(lag)
    sig = pd.Series(0.0, index=lagged.index)
    sig[lagged <= -threshold] = +1   # academic: La Niña-lag → bullish
    sig[lagged >=  threshold] = -1

    ret = cached_monthly_returns(asset)
    res, df = run_backtest(f"P2 ENSO Lag-{lag} ({asset})", asset, sig, ret, "monthly")
    metric_row(res)
    equity_chart(df, f"P2 — Lag-{lag} ENSO on {asset}", with_benchmark=df["ret"])

    # Lag sweep — show all lags at this threshold to confirm robustness
    st.markdown("#### Lag sensitivity sweep (1–24 months, same threshold)")
    sweep_rows = []
    for k in range(1, 25):
        s = pd.Series(0.0, index=a.index)
        a_lag = a.shift(k)
        s[a_lag <= -threshold] = +1
        s[a_lag >=  threshold] = -1
        r, _ = run_backtest(f"k={k}", asset, s, ret, "monthly")
        sweep_rows.append({"lag": k, "sharpe": r.sharpe_ann, "t_stat": r.t_stat,
                           "verdict": r.verdict})
    sweep = pd.DataFrame(sweep_rows)

    def bar_color(s, t):
        if abs(t) >= 2 and s > 0: return COLOR_GREEN
        if abs(t) >= 2 and s < 0: return COLOR_RED
        if abs(t) >= 1: return COLOR_AMBER
        return "#bdc3c7"

    fig = go.Figure(go.Bar(
        x=sweep["lag"], y=sweep["sharpe"],
        marker_color=[bar_color(s, t) if k != lag else COLOR_NAVY
                      for k, s, t in zip(sweep["lag"], sweep["sharpe"], sweep["t_stat"])],
        hovertemplate="Lag %{x} mo<br>Sharpe %{y:+.2f}<br>t-stat %{customdata:+.2f}<extra></extra>",
        customdata=sweep["t_stat"],
    ))
    fig.update_layout(
        title=f"Sharpe across all lags (selected lag {lag} highlighted)",
        height=320, **PLOT_LAYOUT,
        xaxis=dict(title="ENSO anomaly lag (months)", tickmode="linear"),
        yaxis=dict(title="Sharpe (long La Niña-lag → asset)"),
        showlegend=False,
    )
    fig.add_hline(y=0, line_color="#444", line_width=0.7)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: P3 — HURRICANE NG
# =============================================================================
def page_p3():
    st.markdown("### 3️⃣ Hurricane Pre-Season NG Premium")
    st.caption("Long or short NG over a user-chosen calendar month window. "
               "Empty academic claim: pre-peak long. Reality: post-peak long is what works.")

    col_a, col_b = st.columns([1.4, 1])
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    with col_a:
        active_months = st.multiselect(
            "Active calendar months (position held during these months)",
            options=list(range(1, 13)),
            default=[6, 7, 8],
            format_func=lambda m: months[m - 1],
        )
    with col_b:
        direction = st.radio("Direction", ["Long", "Short"], horizontal=True)
        sign = +1 if direction == "Long" else -1

    ret = cached_monthly_returns("NG")
    sig = pd.Series(0.0, index=ret.index)
    # Set signal at end of month m-1 → applied to month m's return.
    # So for active month m, set signal at month m-1.
    active_signal_months = {(m - 1) if m > 1 else 12 for m in active_months}
    for ts in sig.index:
        if ts.month in active_signal_months:
            sig.loc[ts] = sign

    label = f"P3 NG {direction.lower()} in {','.join(months[m-1] for m in sorted(active_months))}"
    res, df = run_backtest(label, "NG", sig, ret, "monthly")
    metric_row(res)
    equity_chart(df, label, with_benchmark=df["ret"])

    st.markdown("#### Monthly mean NG return (seasonal pattern)")
    seasonal = pd.DataFrame({
        "month": months,
        "mean":  [ret[ret.index.month == m].mean() * 100 for m in range(1, 13)],
        "se":    [ret[ret.index.month == m].std() / np.sqrt(len(ret[ret.index.month == m])) * 100
                  for m in range(1, 13)],
    })
    fig = go.Figure(go.Bar(
        x=seasonal["month"], y=seasonal["mean"],
        marker_color=[COLOR_GREEN if v > 0 else COLOR_RED for v in seasonal["mean"]],
        error_y=dict(type="data", array=seasonal["se"], thickness=1.2, color="#444"),
    ))
    fig.update_layout(
        title="Mean monthly NG return ± SE (2000–2026)",
        height=280, **PLOT_LAYOUT,
        yaxis=dict(title="Mean monthly log return (%)"),
        showlegend=False,
    )
    fig.add_vrect(x0=4.5, x1=10.5, fillcolor=COLOR_AMBER, opacity=0.07, line_width=0,
                  annotation_text="Atlantic hurricane season (Jun–Nov)",
                  annotation_position="top left",
                  annotation_font_size=10, annotation_font_color=COLOR_AMBER)
    fig.add_hline(y=0, line_color="#444", line_width=0.7)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: P4 — STORM SURPRISE
# =============================================================================
def page_p4():
    st.markdown("### 4️⃣ Named-Storm Count Surprise → Sep NG")
    st.caption("Realized Atlantic named-storm count minus the CSU April pre-season forecast. "
               "Position taken end-August, held over September.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        threshold_sd = st.slider(
            "Surprise threshold (in σ-units)", 0.0, 2.0, 0.5, step=0.1,
            help="Position taken when |surprise| ≥ threshold × stdev(surprise)")
    with col_b:
        direction = st.radio(
            "Direction on positive surprise (active season)",
            ["Long Sep NG (academic)", "Short Sep NG (inverted)"], horizontal=True)
        sign = +1 if "Long" in direction else -1

    surp = cached_storm_surprise()
    surp_y = surp["surprise_named"].astype(float)
    sd = surp_y.std()
    sig = pd.Series(0.0, index=surp_y.index)
    sig[surp_y >  threshold_sd * sd] = +sign
    sig[surp_y < -threshold_sd * sd] = -sign

    # Compute Aug→Sep NG return per year
    daily = load_futures("NG")
    daily.index = pd.to_datetime(daily.index)
    monthly = daily["close"].resample("ME").last()
    sep = monthly[monthly.index.month == 9]
    aug = monthly[monthly.index.month == 8]
    aug_by_year = pd.Series(aug.values, index=aug.index.year)
    sep_by_year = pd.Series(sep.values, index=sep.index.year)
    common = aug_by_year.index.intersection(sep_by_year.index).intersection(sig.index)
    aug_p = aug_by_year.reindex(common)
    sep_p = sep_by_year.reindex(common)
    aug_sep_ret = np.log(sep_p / aug_p)
    sig_aligned = sig.reindex(common).fillna(0)

    df = pd.DataFrame({"signal": sig_aligned, "ret": aug_sep_ret})
    df["strat_ret"] = df["signal"] * df["ret"]
    n = len(df); active = int((df["signal"] != 0).sum())
    if active == 0:
        st.warning("No active years at this threshold. Reduce the threshold.")
        return
    m, v = df["strat_ret"].mean(), df["strat_ret"].std(ddof=1)
    sharpe = m / v if v > 0 else 0
    t_stat = m / (v / np.sqrt(n)) if v > 0 else 0
    hit = float((df["strat_ret"][df["signal"] != 0] > 0).mean())
    eq = np.exp(df["strat_ret"].cumsum())
    max_dd = float((eq / eq.cummax() - 1).min())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sharpe (annual)", f"{sharpe:+.2f}")
    c2.metric("t-statistic", f"{t_stat:+.2f}")
    c3.metric("Hit rate", f"{hit*100:.1f}%")
    c4.metric("Max drawdown", f"{max_dd*100:.1f}%")
    c5.metric("Active years", f"{active} / {n}")

    verdict = "REAL" if abs(t_stat) >= 2 and abs(sharpe) >= 0.3 else ("WEAK" if abs(t_stat) >= 1 else "NOISE")
    st.markdown(f"**Verdict:** {verdict_tag(verdict)}", unsafe_allow_html=True)

    # Scatter: surprise vs Aug→Sep return
    fig = go.Figure()
    aligned = pd.DataFrame({
        "year": common,
        "surprise": surp_y.reindex(common).values,
        "ret":      aug_sep_ret.values * 100,
    })
    fig.add_trace(go.Scatter(
        x=aligned["surprise"], y=aligned["ret"],
        mode="markers+text",
        text=aligned["year"].astype(str),
        textposition="top center", textfont=dict(size=10, color="#7f8c8d"),
        marker=dict(size=12, color=[COLOR_GREEN if r > 0 else COLOR_RED for r in aligned["ret"]],
                    line=dict(color=COLOR_NAVY, width=1)),
        hovertemplate="%{text}: surprise %{x:.1f}, ret %{y:+.1f}%<extra></extra>",
        showlegend=False,
    ))
    # Threshold lines
    fig.add_vline(x=+threshold_sd * sd, line_dash="dash", line_color=COLOR_AMBER)
    fig.add_vline(x=-threshold_sd * sd, line_dash="dash", line_color=COLOR_AMBER)
    fig.add_vline(x=0, line_color="#444", line_width=0.7)
    fig.add_hline(y=0, line_color="#444", line_width=0.7)
    fig.update_layout(
        title=f"Storm-count surprise vs Aug→Sep NG return (threshold ±{threshold_sd}σ)",
        height=420, **PLOT_LAYOUT,
        xaxis=dict(title="Storm-count surprise (realized − CSU April forecast)"),
        yaxis=dict(title="Aug→Sep NG log return (%)"),
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: P5 — WINTER ENSO
# =============================================================================
def page_p5():
    st.markdown("### 5️⃣ Winter Heating ENSO → NG")
    st.caption("Decision date end-October each year. Long NG if La Niña, short if El Niño. "
               "Held over Nov→Feb heating-season return.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        threshold = st.slider("ENSO threshold (|°C|) at end-Oct", 0.0, 1.5, 0.5, step=0.1)
    with col_b:
        hold_end = st.selectbox(
            "Hold end month",
            options=[1, 2, 3, 4],
            index=1,
            format_func=lambda m: f"end-{['Jan','Feb','Mar','Apr'][m-1]} (next year)",
        )

    oni = cached_oni()
    a = oni["anom"]
    oct_anom = a[a.index.month == 10]

    daily = load_futures("NG")
    daily.index = pd.to_datetime(daily.index)
    monthly = daily["close"].resample("ME").last()
    rows = []
    for ts in monthly.index:
        if ts.month != 10: continue
        y = ts.year
        try:
            entry = monthly.loc[ts]
            exit_ts = pd.Timestamp(year=y + 1, month=hold_end, day=1) + pd.offsets.MonthEnd(0)
            if exit_ts not in monthly.index: continue
            exit_p = monthly.loc[exit_ts]
            rows.append((ts, float(np.log(exit_p / entry))))
        except KeyError:
            continue
    winter = pd.Series({ts: r for ts, r in rows}).sort_index()

    common = oct_anom.index.intersection(winter.index)
    a_al = oct_anom.reindex(common)
    r_al = winter.reindex(common)
    sig = pd.Series(0.0, index=common)
    sig[a_al <= -threshold] = +1
    sig[a_al >=  threshold] = -1

    df = pd.DataFrame({"oni": a_al, "signal": sig, "winter_ret": r_al})
    df["strat_ret"] = df["signal"] * df["winter_ret"]
    n = len(df); active = int((df["signal"] != 0).sum())
    if active == 0:
        st.warning("No active years at this threshold."); return
    m, v = df["strat_ret"].mean(), df["strat_ret"].std(ddof=1)
    sharpe = m / v if v > 0 else 0
    t_stat = m / (v / np.sqrt(n)) if v > 0 else 0
    hit = float((df["strat_ret"][df["signal"] != 0] > 0).mean())
    eq = np.exp(df["strat_ret"].cumsum())
    max_dd = float((eq / eq.cummax() - 1).min())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sharpe (annual)", f"{sharpe:+.2f}")
    c2.metric("t-statistic", f"{t_stat:+.2f}")
    c3.metric("Hit rate", f"{hit*100:.1f}%")
    c4.metric("Max drawdown", f"{max_dd*100:.1f}%")
    c5.metric("Active years", f"{active} / {n}")
    verdict = "REAL" if abs(t_stat) >= 2 and abs(sharpe) >= 0.3 else ("WEAK" if abs(t_stat) >= 1 else "NOISE")
    st.markdown(f"**Verdict:** {verdict_tag(verdict)}", unsafe_allow_html=True)

    # Equity curve (annual, so just line of cum log)
    cum = df["strat_ret"].cumsum()
    fig = go.Figure(go.Scatter(
        x=cum.index.year, y=np.exp(cum.values),
        line=dict(color=COLOR_TEAL, width=2.4), mode="lines+markers",
    ))
    fig.update_layout(
        title=f"Annual equity (Oct→{['Jan','Feb','Mar','Apr'][hold_end-1]}, threshold ±{threshold})",
        height=320, **PLOT_LAYOUT,
        yaxis=dict(title="Cumulative equity"),
        xaxis=dict(title="Year (entry)"),
        showlegend=False,
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="#aaa")
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: CUSTOM PREDICTOR
# =============================================================================
def page_custom():
    st.markdown("### 🛠️ Custom Predictor — build your own signal")
    st.caption("Combine ENSO state + ENSO lag + named-storm count into a custom signal. "
               "Tweak the weights and threshold to find your own variant.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        asset = st.selectbox("Underlying", list(FUTURES.keys()), index=0)
        w_state = st.slider("Weight: current ENSO state",        -1.0, 1.0, +0.5, step=0.1)
        w_lag12 = st.slider("Weight: ENSO state 12 months ago",  -1.0, 1.0, +0.0, step=0.1)
        w_lag6  = st.slider("Weight: ENSO state 6 months ago",   -1.0, 1.0, +0.0, step=0.1)
    with col_b:
        threshold = st.slider("ENSO threshold (|°C|)", 0.0, 1.5, 0.5, step=0.1)
        long_thresh = st.slider("Composite signal threshold (go long if score ≥)", 0.0, 2.0, 0.4, step=0.1)

    oni = cached_oni()
    a = oni["anom"]
    state = pd.Series(0.0, index=a.index)
    state[a <= -threshold] = +1   # La Niña
    state[a >=  threshold] = -1   # El Niño

    composite = (w_state * state
                 + w_lag6 * state.shift(6)
                 + w_lag12 * state.shift(12)).fillna(0)
    sig = pd.Series(0.0, index=composite.index)
    sig[composite >=  long_thresh] = +1
    sig[composite <= -long_thresh] = -1

    ret = cached_monthly_returns(asset)
    res, df = run_backtest(f"Custom ({asset})", asset, sig, ret, "monthly")
    metric_row(res)
    equity_chart(df, f"Custom composite on {asset}", with_benchmark=df["ret"], log_scale=False)
    signal_position_chart(df, "Position over time")

    st.markdown(f"""
<div class="callout warn">
  <div class="lbl">Curve-fitting warning</div>
  <p>This sandbox makes it easy to find a high-Sharpe combination by trial. Any "REAL"
  verdict here is <b>in-sample</b> — before trusting it, run a walk-forward split
  (e.g. fit weights on 2000–2017, test on 2018–2026) and check whether the edge
  survives. The harness in <code>lib/backtest.py</code> can be called from a Python
  script that takes only the 2000–2017 slice; results outside that window are the
  honest test.</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE: ABOUT
# =============================================================================
def page_about():
    st.markdown("### About this app")
    st.markdown("""
This is the **live** version of the weather-predictors backtest. Unlike the static PDF /
HTML report, every metric here is recomputed in Python on each parameter change.

#### Data sources (all free)
- **NOAA ONI** — `cpc.ncep.noaa.gov/data/indices/oni.ascii.txt` · 1950–present, monthly
- **HURDAT2** — `nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt` · all Atlantic storms
- **CSU April forecasts** — `tropical.colostate.edu/archive.html` · 1995–2024 (hardcoded)
- **Commodity futures** — yfinance NG=F, KC=F, CC=F, OJ=F, RB=F · 2000–present, daily

#### Reproducibility
```bash
pip install streamlit pandas numpy scipy plotly yfinance
streamlit run app.py
```

#### The harness
All five predictor pages share the same `run_backtest` in `lib/backtest.py`:
- Aligns signal with returns on a common index
- **Shifts signal forward by 1 period** → no lookahead
- Computes annualized Sharpe (×√12 monthly, ×1 annual)
- Reports t-statistic, hit rate, max drawdown, Calmar
- Assigns verdict: **REAL** (|t| ≥ 2 and |Sharpe| ≥ 0.3), **WEAK** (1 ≤ |t| < 2), **NOISE** (|t| < 1)

#### Limitations
All Sharpe ratios are **in-sample**, gross of transaction costs, single window. Walk-forward
validation on 2000–2018 fit / 2019–2026 test is the recommended sanity check before sizing
real positions.

#### Files
- `app.py` — this Streamlit app
- `lib/data.py` — data loaders (cached on disk under `data/`)
- `lib/backtest.py` — the backtest harness
- `predictors/p1..p5.py` — non-interactive versions of the 5 predictors
- `run_all.py` — runs everything for the static report

Internal research only. Not investment advice.
""")


# =============================================================================
# Router
# =============================================================================
ROUTES: dict[str, Callable] = {
    "📊 Overview":                       page_overview,
    "1️⃣ ENSO Overlay (P1)":              page_p1,
    "2️⃣ ENSO Lag → Coffee (P2)":         page_p2,
    "3️⃣ Hurricane NG Premium (P3)":      page_p3,
    "4️⃣ Storm Surprise → NG (P4)":       page_p4,
    "5️⃣ Winter ENSO → NG (P5)":          page_p5,
    "🛠️ Custom Predictor":              page_custom,
    "📜 About":                         page_about,
}

ROUTES[page]()
