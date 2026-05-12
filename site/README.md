# Weather Predictors — Interactive Site

Static interactive site built from the backtest outputs of the weather-predictors
project. Same visual language as the institutional PDF report, plus working
interactivity: sortable scoreboard, toggleable equity curves, click-through
lag sensitivity, time-series charts.

## Files

| File | Role |
|------|------|
| `index.html` | Self-contained site (CSS + JS inlined, external Chart.js via CDN) |
| `data.js` | Bundled backtest results (`window.DATA = {…}`) |
| `data.json` | Same data in JSON form (kept for archival / re-bundling) |
| `README.md` | This file |

## Run it

Three ways, easiest first:

### 1. Local file (no server)
Open `index.html` directly in Chrome / Safari / Firefox. Works because
`data.js` is loaded as a script (no CORS issue).

```
open /Users/peter/Downloads/.super/weather-predictors/site/index.html
```

### 2. Local http server (recommended for sharing or screencasts)
```
cd /Users/peter/Downloads/.super/weather-predictors/site
python3 -m http.server 8000
# then visit http://localhost:8000
```

### 3. Deploy to any static host
Drop the four files into Vercel / Netlify / GitHub Pages / S3 / nginx.
Total payload is ~225 KB (212 KB is `data.js`, mostly equity-curve points).

## Interactive features

| Section | What you can do |
|---------|----------------|
| **Scoreboard** | Click column headers to sort by any metric. Click chips (REAL / WEAK / NOISE) to filter by verdict. |
| **Equity Explorer** | Toggle individual predictors on/off. Linear ↔ Log scale toggle. Hover for tooltips, end-period return shown in stats row. |
| **Lag Sweep** | Click any bar to drill into Sharpe / t-stat / hit rate at that lag. |
| **ENSO Backdrop** | ONI timeline (color-coded by phase) + NG monthly seasonality. Hover for exact values. |

## Updating the data

If the underlying backtest is re-run, rebuild `data.js` from the CSVs:

```bash
cd ..                          # back to weather-predictors/
python3 run_all.py             # re-run backtests, regenerates results/*.csv
python3 build_site_data.py     # rebuilds site/data.json
# Then turn JSON → JS:
python3 -c "import json; d = json.load(open('site/data.json')); open('site/data.js','w').write('window.DATA = ' + json.dumps(d) + ';')"
```

Or combine into one step in `build_site_data.py` if you do this often.

## Dependencies (CDN, loaded at runtime)

- Chart.js v4.4.0 — interactive charts
- chartjs-adapter-date-fns v3.0.0 — time-axis support for Chart.js

Both load from `jsdelivr.net`. If the host needs to work fully offline,
download these two files into the same directory and update the `<script
src>` attributes in `index.html`.

## Browser support

Tested in headless Chromium (Playwright). Should work in:
- Chrome / Edge / Brave (any recent version)
- Safari 15+
- Firefox 100+

No build step, no bundler, no framework — vanilla HTML/CSS/JS. The chart
library and the data are the only external moving parts.
