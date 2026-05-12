# Weather Predictors With Measurable Edge

Programmable weather-based commodity predictors, backtested on 25 years of free
public data. Five predictors across nine variants; **two cleared the conventional
|t| ≥ 2 statistical-significance bar — and both contradicted the published
academic direction**.

| | |
|---|---|
| **Predictors tested** | 5 (P1–P5), 9 variants |
| **Statistically significant** | 2 of 9 (P2 KC inverted, P3 NG post-peak inverted) |
| **Best Sharpe** | +0.55 (inverted P3 hurricane post-peak, NG) |
| **Sample** | 2000-01 → 2026-05, monthly + annual |
| **Data sources** | NOAA ONI, HURDAT2, CSU forecasts, yfinance — all free |

## Three ways to consume this

| Interface | Command | Live? |
|---|---|---|
| **Streamlit app** | `streamlit run app.py` | Yes — change knobs, backtest re-runs |
| **Static site** | `python3 -m http.server -d site 8000` | No — explore pre-computed results |
| **Python scripts** | `python3 run_all.py` | No — runs all 5 predictors and prints table |

## Quick start

```bash
git clone https://github.com/aifunmobi/weather-predictors.git
cd weather-predictors
pip install yfinance pandas numpy scipy matplotlib plotly streamlit

# Option 1 — Live Streamlit app (recommended)
streamlit run app.py

# Option 2 — Static interactive site
python3 -m http.server -d site 8000   # http://localhost:8000

# Option 3 — Just run all backtests and print results
python3 run_all.py
```

First run downloads ~5 MB of data (ONI + HURDAT2 + futures via yfinance) and
caches it in `data/`. Subsequent runs are instant.

## The 9-variant honest scoreboard

| Predictor | Asset | Sharpe | t-stat | N | Verdict |
|---|---|---|---|---|---|
| P1 ENSO Overlay | NG | −0.22 | −1.09 | 309 | WEAK |
| P1 ENSO Overlay | KC | −0.15 | −0.76 | 316 | NOISE |
| P1 ENSO Overlay | CC | +0.24 | +1.24 | 316 | WEAK |
| P1 ENSO Overlay | OJ | +0.25 | +1.26 | 296 | WEAK |
| **P2 ENSO Lag-13** | **KC** | **−0.46** | **−2.37** | **316** | **REAL** (inverted) |
| P3 Hurricane pre-peak | NG | −0.14 | −0.70 | 309 | NOISE |
| **P3 Hurricane post-peak** | **NG** | **−0.55** | **−2.81** | **309** | **REAL** (inverted) |
| P4 Storm Surprise | NG (Sep) | +0.20 | +1.01 | 25 | WEAK |
| P5 Winter ENSO | NG (Winter) | −0.10 | −0.51 | 26 | NOISE |

**Reading**: negative Sharpe with |t| ≥ 2 means the literal academic strategy
loses money significantly — the inverted strategy would win (+0.46 and +0.55
respectively).

## What the five predictors test

1. **P1 — ENSO overlay**: monthly position from current ENSO state, applied to
   NG, KC, CC, OJ per documented directional bias.
2. **P2 — ENSO 13-month lag**: tests the academic claim that ENSO state lagged
   13–15 months predicts Arabica coffee. Includes lag-1-to-24 sensitivity sweep.
3. **P3 — Hurricane NG premium**: tests two variants — long NG during the
   pre-peak hurricane window (Jun–Aug), short NG post-peak (Sep–Oct).
4. **P4 — Storm-count surprise**: realized Atlantic named-storm count minus
   CSU April forecast → September NG return.
5. **P5 — Winter heating ENSO**: ONI at end-October → NG Nov→Feb directional.

Each uses the same shared harness in [`lib/backtest.py`](lib/backtest.py) so
results are apples-to-apples comparable.

## Project structure

```
weather-predictors/
├── app.py                    # Streamlit live-knob app
├── run_all.py                # Run all 5 predictors → results/summary.json
├── make_charts.py            # Regenerate the 8 matplotlib charts
├── build_site_data.py        # Bundle results into site/data.json
├── report.html               # Polished PDF report (renderable with Chrome)
│
├── lib/
│   ├── data.py               # ONI, HURDAT2, yfinance, CSU loaders (cached)
│   └── backtest.py           # Shared harness — Sharpe, t-stat, hit, max-DD, verdict
│
├── predictors/
│   ├── p1_enso_overlay.py
│   ├── p2_enso_lag.py
│   ├── p3_hurricane_premium.py
│   ├── p4_storm_surprise.py
│   └── p5_winter_enso.py
│
├── site/                     # Static interactive site
│   ├── index.html
│   ├── data.js               # Bundled equity curves + lag sweep
│   └── README.md
│
└── charts/                   # PNGs for the PDF report
```

`data/` and `results/*.csv` are gitignored — regenerated on first run.

## Verdict thresholds

| Verdict | Criterion |
|---|---|
| **REAL** | \|t\| ≥ 2.0 **and** \|Sharpe\| ≥ 0.3 |
| **WEAK** | 1.0 ≤ \|t\| < 2.0 |
| **NOISE** | \|t\| < 1.0 |

## Honest limitations

All Sharpe ratios are **in-sample**, **gross of transaction costs**, and based
on a single 25-year window of yfinance continuous front-month futures. Before
sizing real positions:

- Walk-forward validation (e.g. fit on 2000–2018, test on 2019–2026)
- Re-run with properly back-adjusted continuous futures (not yfinance splice)
- Add a transaction-cost model (yfinance bid-ask not modeled)

The two REAL signals are statistically significant but in **the opposite
direction from the published academic literature**. That reversal could be a
genuine effect, an artifact of the modern post-2018 regime, or a data-dredging
artifact. The lag-sensitivity sweep on P2 (in the Streamlit app or the PDF
report) argues for "real" because the reversal is consistent across nearby
parameter values — but out-of-sample validation is the only honest test.

## Data sources

| Series | Source | Coverage |
|---|---|---|
| NOAA ONI (ENSO anomaly) | https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt | 1950–present, monthly |
| HURDAT2 (Atlantic storms) | https://www.nhc.noaa.gov/data/hurdat/ | 1851–2024 |
| CSU April forecasts | https://tropical.colostate.edu/archive.html | 1995–2024 |
| Commodity futures | yfinance — NG=F, KC=F, CC=F, OJ=F, RB=F | 2000–present, daily |

All free. No API keys, no paid feeds.

## License

MIT. Use at your own risk. Not investment advice.
