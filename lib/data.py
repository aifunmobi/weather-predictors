"""Data loading + caching for weather-predictors backtest.

Sources:
  - NOAA ONI: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
  - HURDAT2 (Atlantic tropical cyclones): NHC archive (best track)
  - Commodity futures: yfinance (continuous front-month)
  - CSU April hurricane forecasts: hardcoded (scraped from CSU archive)
"""
from __future__ import annotations
import os
import io
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data"
CACHE.mkdir(parents=True, exist_ok=True)

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
HURDAT_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt"

# CSU April pre-season Atlantic hurricane forecasts (from CSU archive at
# tropical.colostate.edu/archive.html). Columns: named_storms, hurricanes,
# major_hurricanes.
CSU_APRIL_FORECAST = {
    1995: (10, 6, 2), 1996: (11, 7, 3), 1997: (11, 7, 3), 1998: (10, 6, 2),
    1999: (14, 9, 4), 2000: (11, 7, 3), 2001: (10, 6, 2), 2002: (12, 7, 3),
    2003: (12, 8, 3), 2004: (14, 8, 3), 2005: (13, 7, 3), 2006: (17, 9, 5),
    2007: (17, 9, 5), 2008: (15, 8, 4), 2009: (12, 6, 2), 2010: (15, 8, 4),
    2011: (16, 9, 5), 2012: (10, 4, 2), 2013: (18, 9, 4), 2014: (9, 3, 1),
    2015: (7, 3, 1),  2016: (13, 6, 2), 2017: (11, 4, 2), 2018: (14, 7, 3),
    2019: (13, 5, 2), 2020: (16, 8, 4), 2021: (17, 8, 4), 2022: (19, 9, 4),
    2023: (13, 6, 2), 2024: (14, 7, 3),
}


# ---------------------------------------------------------------------------
# ONI (Oceanic Niño Index)
# ---------------------------------------------------------------------------

def load_oni(force: bool = False) -> pd.DataFrame:
    """Load monthly ONI series.

    Returns DataFrame indexed by month-end with columns:
      sst     — observed SST in Niño 3.4 region (deg C)
      anom    — anomaly vs 30-year climatology
      season  — 3-month rolling label (e.g. "DJF")
    """
    f = CACHE / "oni.csv"
    if force or not f.exists():
        import urllib.request
        with urllib.request.urlopen(ONI_URL, timeout=30) as r:
            raw = r.read().decode("utf-8")
        rows = []
        for line in raw.strip().splitlines()[1:]:  # skip header
            parts = line.split()
            if len(parts) != 4:
                continue
            season, year, sst, anom = parts
            try:
                year = int(year); sst = float(sst); anom = float(anom)
            except ValueError:
                continue
            # Map 3-month season label → center month
            month_map = {
                "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
            }
            month = month_map.get(season)
            if month is None:
                continue
            date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
            rows.append((date, sst, anom, season))
        df = pd.DataFrame(rows, columns=["date", "sst", "anom", "season"]).set_index("date")
        df.sort_index(inplace=True)
        df.to_csv(f)
    return pd.read_csv(f, index_col=0, parse_dates=True)


def enso_state(anom: float, threshold: float = 0.5) -> str:
    """ENSO state per NOAA convention: |anom| >= 0.5 sustained 5 months."""
    if anom >= threshold:
        return "el_nino"
    if anom <= -threshold:
        return "la_nina"
    return "neutral"


# ---------------------------------------------------------------------------
# HURDAT2 (Atlantic tropical cyclones)
# ---------------------------------------------------------------------------

def load_hurdat(force: bool = False) -> pd.DataFrame:
    """Load HURDAT2 storm tracks, parse to per-storm summary.

    Returns DataFrame with columns:
      id, name, year, max_wind_kt, became_named, became_hurricane, became_major
    """
    f = CACHE / "hurdat_storms.csv"
    if force or not f.exists():
        import urllib.request
        with urllib.request.urlopen(HURDAT_URL, timeout=60) as r:
            raw = r.read().decode("utf-8")
        storms = []
        cur = None
        for line in raw.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if parts[0].startswith("AL") and len(parts) >= 3:
                if cur is not None:
                    storms.append(cur)
                cur = {
                    "id": parts[0], "name": parts[1], "year": int(parts[0][4:8]),
                    "max_wind_kt": 0,
                }
            else:
                if cur is None or len(parts) < 7:
                    continue
                try:
                    w = int(parts[6])
                    if w > cur["max_wind_kt"]:
                        cur["max_wind_kt"] = w
                except ValueError:
                    pass
        if cur is not None:
            storms.append(cur)

        df = pd.DataFrame(storms)
        df["became_named"]     = df["max_wind_kt"] >= 34   # 39 mph
        df["became_hurricane"] = df["max_wind_kt"] >= 64   # 74 mph
        df["became_major"]     = df["max_wind_kt"] >= 96   # Cat 3+ (>= 111 mph)
        df.to_csv(f, index=False)
    return pd.read_csv(f)


def storms_by_year() -> pd.DataFrame:
    """Annual named/hurricane/major counts."""
    hd = load_hurdat()
    by_year = hd.groupby("year").agg(
        named=("became_named", "sum"),
        hurricanes=("became_hurricane", "sum"),
        majors=("became_major", "sum"),
    )
    return by_year


# ---------------------------------------------------------------------------
# Commodity futures prices (yfinance)
# ---------------------------------------------------------------------------

FUTURES = {
    "NG": "NG=F",   # Natural gas
    "KC": "KC=F",   # Coffee Arabica
    "CC": "CC=F",   # Cocoa
    "OJ": "OJ=F",   # Orange juice
    "RB": "RB=F",   # RBOB gasoline
}


def load_futures(symbol: str, force: bool = False) -> pd.DataFrame:
    """Load daily continuous front-month futures.

    Returns DataFrame indexed by date with columns: close, ret (log-return).
    """
    if symbol not in FUTURES:
        raise KeyError(f"Unknown future {symbol!r}; known: {list(FUTURES)}")
    f = CACHE / f"futures_{symbol}.csv"
    if force or not f.exists():
        import yfinance as yf
        d = yf.download(FUTURES[symbol], period="max",
                        progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d[["Close"]].rename(columns={"Close": "close"})
        d["ret"] = np.log(d["close"]).diff()
        d.to_csv(f)
    return pd.read_csv(f, index_col=0, parse_dates=True)


def monthly_returns(symbol: str) -> pd.Series:
    """Month-end log returns."""
    daily = load_futures(symbol)["close"]
    monthly = daily.resample("ME").last()
    return np.log(monthly).diff().dropna()


# ---------------------------------------------------------------------------
# CSU forecasts
# ---------------------------------------------------------------------------

def csu_april_forecast() -> pd.DataFrame:
    """CSU April pre-season Atlantic hurricane forecasts by year."""
    rows = []
    for year, (named, hurr, maj) in CSU_APRIL_FORECAST.items():
        rows.append({"year": year, "fcst_named": named,
                     "fcst_hurricanes": hurr, "fcst_majors": maj})
    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------------------
# Combined: forecast + realized storm surprise
# ---------------------------------------------------------------------------

def storm_surprise() -> pd.DataFrame:
    """Year-aligned forecast vs realized storms; surprise = realized − forecast."""
    fc = csu_april_forecast()
    actual = storms_by_year()
    df = fc.join(actual, how="inner")
    df["surprise_named"] = df["named"]      - df["fcst_named"]
    df["surprise_hurr"]  = df["hurricanes"] - df["fcst_hurricanes"]
    df["surprise_major"] = df["majors"]     - df["fcst_majors"]
    return df


if __name__ == "__main__":
    # Smoke test: download and summarize everything
    print("=== ONI ===")
    oni = load_oni()
    print(f"  {len(oni)} months | {oni.index[0].date()} → {oni.index[-1].date()}")
    print(f"  Latest: {oni.iloc[-1].to_dict()}")

    print("\n=== HURDAT2 storms by year (last 5 years) ===")
    print(storms_by_year().tail(5))

    print("\n=== Futures coverage ===")
    for sym in FUTURES:
        d = load_futures(sym)
        print(f"  {sym}: {len(d)} rows | {d.index[0].date()} → {d.index[-1].date()}")

    print("\n=== Storm surprise (last 5 years) ===")
    print(storm_surprise().tail())
