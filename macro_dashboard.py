"""
Macro Dashboard — Economic Regime Detector
============================================
Aggregates real macroeconomic data from the FRED API and automatically
classifies the current economic regime across four phases:
Expansion, Peak, Contraction, and Recovery.

Produces a full visual dashboard with live data, regime scoring,
yield curve analysis, and cycle-consistent allocation recommendations.

Author: Federico Bonessi | The Meridian Playbook
GitHub: github.com/federicobonessi

Requirements:
    pip install requests pandas matplotlib scipy numpy

Usage:
    # With FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html):
    python src/macro_dashboard.py --api-key YOUR_KEY

    # Demo mode (simulated data, no API key needed):
    python src/macro_dashboard.py --demo
"""

import argparse
import json
import os
import time
import warnings
from datetime import datetime, timedelta

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")

# ── PALETTE ──────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
GOLD     = "#c9a84c"
WHITE    = "#e6edf3"
GREY     = "#30363d"
MID      = "#8b949e"
RED      = "#f85149"
GREEN    = "#3fb950"
BLUE     = "#58a6ff"
ORANGE   = "#ffa657"
PURPLE   = "#a371f7"

PHASE_COLORS = {
    "Expansion":   GREEN,
    "Peak":        ORANGE,
    "Contraction": RED,
    "Recovery":    BLUE,
}

# ── FRED SERIES ──────────────────────────────────────────────────────────────
FRED_SERIES = {
    # Growth
    "GDPC1":    "Real GDP (Quarterly, SA)",
    "INDPRO":   "Industrial Production Index",
    "UNRATE":   "Unemployment Rate (%)",
    "ICSA":     "Initial Jobless Claims (Weekly)",
    # Inflation
    "CPIAUCSL": "CPI All Items (YoY %)",
    "PCEPI":    "PCE Price Index (YoY %)",
    "T5YIE":    "5Y Breakeven Inflation (%)",
    # Monetary / Credit
    "FEDFUNDS": "Federal Funds Rate (%)",
    "DGS10":    "10Y Treasury Yield (%)",
    "DGS2":     "2Y Treasury Yield (%)",
    "DGS3MO":   "3M Treasury Yield (%)",
    "BAMLH0A0HYM2": "HY Credit Spread (OAS, %)",
    "BAMLC0A0CM":   "IG Credit Spread (OAS, %)",
    # Leading indicators
    "M2SL":     "M2 Money Supply (SA, $bn)",
    "UMCSENT":  "U. Michigan Consumer Sentiment",
    "HOUST":    "Housing Starts (SAAR, thousands)",
    "PERMIT":   "Building Permits (SAAR, thousands)",
    "RETAILSALES": "Retail Sales (SA, $mn)",  # RSAFS
}

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


# ── FRED CLIENT ──────────────────────────────────────────────────────────────

def fetch_fred(series_id: str, api_key: str,
               start: str = "2018-01-01") -> pd.Series:
    """Fetch a single FRED series and return as a dated pd.Series."""
    params = {
        "series_id":       series_id,
        "api_key":         api_key,
        "file_type":       "json",
        "observation_start": start,
        "sort_order":      "asc",
    }
    try:
        import urllib.request, urllib.parse
        url = FRED_BASE + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        obs = data.get("observations", [])
        if not obs:
            return pd.Series(dtype=float)
        df = pd.DataFrame(obs)[["date", "value"]]
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna().set_index("date")["value"]
    except Exception as e:
        print(f"    [FRED] Could not fetch {series_id}: {e}")
        return pd.Series(dtype=float)


def fetch_all_fred(api_key: str) -> dict:
    """Fetch all required FRED series."""
    series_map = {
        "gdp":          "GDPC1",
        "indpro":       "INDPRO",
        "unrate":       "UNRATE",
        "claims":       "ICSA",
        "cpi":          "CPIAUCSL",
        "pce":          "PCEPI",
        "breakeven5y":  "T5YIE",
        "fedfunds":     "FEDFUNDS",
        "t10y":         "DGS10",
        "t2y":          "DGS2",
        "t3m":          "DGS3MO",
        "hy_spread":    "BAMLH0A0HYM2",
        "ig_spread":    "BAMLC0A0CM",
        "m2":           "M2SL",
        "sentiment":    "UMCSENT",
        "housing":      "HOUST",
        "permits":      "PERMIT",
        "retail":       "RSAFS",
    }
    out = {}
    print("  Fetching FRED data...")
    for key, sid in series_map.items():
        print(f"    {sid}...", end=" ", flush=True)
        s = fetch_fred(sid, api_key)
        out[key] = s
        print("ok" if len(s) > 0 else "empty")
        time.sleep(0.2)   # gentle rate limiting
    return out


# ── DEMO DATA ─────────────────────────────────────────────────────────────────

def generate_demo_data() -> dict:
    """
    Generate realistic simulated macro data for demo mode.
    Covers Jan 2018 – Mar 2026, spanning Expansion → Peak → COVID Contraction
    → Recovery → Re-expansion → 2022 tightening → current Recovery.
    """
    np.random.seed(42)
    dates_d  = pd.date_range("2018-01-01", "2026-03-01", freq="D")
    dates_m  = pd.date_range("2018-01-01", "2026-03-01", freq="MS")
    dates_q  = pd.date_range("2018-01-01", "2026-03-01", freq="QS")

    def ts(dates, values, noise=0.05):
        arr = np.array(values)
        n   = len(dates)
        # Interpolate base path
        x_base = np.linspace(0, 1, len(arr))
        x_new  = np.linspace(0, 1, n)
        base   = np.interp(x_new, x_base, arr)
        return pd.Series(base + np.random.normal(0, noise * abs(base).mean() + 0.01, n),
                         index=dates)

    # Macro narrative: 2018 late-expansion → 2020 crash → 2021 boom →
    #                  2022 peak/tighten → 2023 slowdown → 2024-25 recovery
    gdp_q = ts(dates_q,
        [19500,19800,20100,20300, 20500,20700,20900,21100,
         21300,21000,17500,18500, 19500,20500,21200,21800,
         22100,22400,22600,22700, 22500,22300,22100,22200,
         22400,22600,22800,23000, 23200,23400,23500,23600,23700],
        noise=0.01)

    indpro_m = ts(dates_m,
        [104,105,105,106,106,107,107,108,108,109,109,110,
         110,111,111,112,112,113,112,111,110,108,106,100,
          90, 92, 95, 98,100,102,103,104,105,106,107,108,
         108,109,110,110,111,111,112,113,114,115,116,117,
         117,118,118,119,119,118,117,116,115,115,115,116,
         116,117,117,118,118,119,119,120,120,121,121,122,
         122,123,124,124,125,125,126,126,127,127,128,128,
         128,129,129,130,130,131,131,132,132,133,133,134,],
        noise=0.3)

    unrate_m = ts(dates_m,
        [4.1,3.9,3.8,3.7,3.8,3.9,3.7,3.6,3.5,3.6,3.5,3.5,
         3.6,3.5,3.6,3.5,3.7,3.6,3.5,3.5,3.7,14.7,13.3,11.1,
         8.4,7.9,7.8,7.3,6.9,6.7,6.4,6.0,5.8,5.4,5.2,4.8,
         4.6,4.2,4.0,3.8,3.6,3.5,3.6,3.5,3.6,3.7,3.5,3.4,
         3.5,3.5,3.4,3.6,3.7,3.8,3.9,3.9,4.0,4.1,4.1,4.2,
         4.1,4.0,4.0,3.9,3.9,3.8,3.9,3.8,3.9,3.8,3.9,3.8,
         3.9,3.8,3.7,3.8,3.7,3.8,3.9,3.8,3.7,3.7,3.8,3.8,
         3.8,3.7,3.8,3.7,3.8,3.7,3.8,3.7,3.8,3.8,3.7,3.8,],
        noise=0.08)

    cpi_m = ts(dates_m,
        [2.1,2.2,2.4,2.5,2.8,2.9,2.9,2.7,2.3,2.2,2.2,1.9,
         2.5,2.3,1.9,2.0,1.8,1.6,1.8,1.7,2.3,0.3,0.1,1.2,
         1.3,1.4,1.6,2.6,5.0,5.4,5.3,6.2,6.8,7.0,7.5,7.9,
         8.3,8.5,8.3,8.2,7.7,7.1,6.5,5.0,4.9,4.0,3.7,3.4,
         3.1,3.0,3.1,3.2,3.4,3.5,3.3,3.0,2.9,2.7,2.7,2.6,
         2.6,2.7,2.8,2.7,2.7,2.6,2.6,2.5,2.6,2.5,2.6,2.5,
         2.5,2.6,2.5,2.6,2.5,2.5,2.6,2.5,2.6,2.5,2.5,2.6,
         2.5,2.6,2.5,2.5,2.6,2.5,2.6,2.5,2.5,2.6,2.5,2.5,],
        noise=0.05)

    fedfunds_m = ts(dates_m,
        [1.41,1.51,1.51,1.69,1.70,1.82,1.91,1.91,2.0,2.19,2.2,2.4,
         2.4,2.4,2.41,2.41,2.4,2.4,2.17,2.04,1.89,1.55,1.55,1.55,
         1.55,0.65,0.05,0.05,0.07,0.09,0.07,0.07,0.09,0.06,0.07,0.08,
         0.07,0.08,0.07,0.08,0.09,0.08,0.08,0.07,0.08,0.08,0.08,0.08,
         0.33,0.77,0.97,1.51,1.73,2.33,2.93,3.08,3.78,4.10,4.31,4.42,
         4.57,4.57,4.83,5.06,5.08,5.17,5.33,5.33,5.33,5.33,5.33,5.33,
         5.33,5.08,4.83,4.83,4.58,4.58,4.33,4.33,4.33,4.33,4.33,4.08,
         4.08,3.83,3.83,3.83,3.83,3.58,3.58,3.58,3.58,3.58,3.33,3.33,],
        noise=0.02)

    t10y_m = ts(dates_m,
        [2.58,2.87,2.74,2.95,2.98,2.89,2.97,2.86,3.05,3.14,3.13,2.83,
         2.71,2.65,2.41,2.58,2.39,2.06,2.02,1.92,1.88,0.73,0.70,0.89,
         0.91,1.05,1.45,1.59,1.59,1.45,1.52,1.62,1.48,1.45,1.48,1.52,
         1.63,1.73,2.14,2.39,2.89,2.98,3.01,3.13,3.45,3.83,3.97,3.88,
         3.97,4.06,3.84,4.01,3.96,3.81,3.65,3.97,4.57,4.69,4.80,4.69,
         4.33,3.96,4.21,4.59,4.68,4.49,4.25,4.19,4.30,4.50,4.46,4.43,
         4.28,4.25,4.15,4.20,4.25,4.20,4.18,4.15,4.12,4.10,4.08,4.05,
         4.02,3.98,3.95,3.92,3.88,3.85,3.82,3.78,3.75,3.72,3.68,3.65,],
        noise=0.05)

    t2y_m = ts(dates_m,
        [2.07,2.24,2.27,2.49,2.52,2.55,2.63,2.63,2.77,2.82,2.79,2.68,
         2.60,2.52,2.44,2.37,2.31,2.16,1.91,1.78,1.63,0.25,0.18,0.17,
         0.13,0.13,0.17,0.16,0.22,0.25,0.26,0.30,0.22,0.22,0.25,0.73,
         1.02,1.33,2.14,2.53,2.68,3.10,3.21,3.45,4.17,4.42,4.51,4.41,
         4.42,4.52,4.60,4.58,4.56,4.62,4.55,4.74,4.81,4.98,5.10,4.99,
         4.63,4.42,4.88,5.05,5.03,4.87,4.70,4.59,4.62,4.79,4.73,4.71,
         4.25,4.23,4.18,4.22,4.25,4.19,4.15,4.12,4.09,4.05,4.02,3.98,
         3.93,3.89,3.85,3.81,3.77,3.73,3.69,3.65,3.61,3.57,3.53,3.49,],
        noise=0.05)

    t3m_m = ts(dates_m,
        [1.45,1.65,1.73,1.87,1.91,1.93,2.01,2.04,2.14,2.22,2.25,2.35,
         2.38,2.40,2.41,2.42,2.43,2.18,1.97,1.82,1.68,0.14,0.11,0.09,
         0.08,0.07,0.04,0.03,0.03,0.04,0.04,0.04,0.05,0.04,0.05,0.07,
         0.14,0.40,0.78,1.10,1.54,2.59,2.97,3.33,3.75,4.05,4.27,4.41,
         4.42,4.55,4.59,4.66,4.69,4.74,4.93,5.07,5.27,5.40,5.45,5.44,
         5.43,5.43,5.45,5.55,5.52,5.47,5.30,5.27,5.26,5.25,5.24,5.24,
         5.07,4.84,4.61,4.38,4.27,4.15,4.10,4.08,4.05,4.02,3.98,3.94,
         3.89,3.85,3.81,3.77,3.73,3.69,3.65,3.61,3.57,3.53,3.49,3.45,],
        noise=0.04)

    hy_spread_m = ts(dates_m,
        [3.8,3.6,3.5,3.3,3.4,3.6,3.5,3.4,3.5,3.6,3.7,3.9,
         3.8,3.7,3.6,3.5,3.6,3.7,4.0,4.2,4.5,10.8,8.6,6.5,
         5.8,5.6,5.2,4.8,4.5,4.2,4.0,3.8,3.7,3.5,3.4,3.3,
         3.2,3.1,3.0,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,4.7,
         4.5,4.3,4.1,3.9,3.8,3.7,3.6,3.5,3.5,3.6,3.7,3.8,
         3.9,4.0,4.1,4.2,4.3,4.2,4.1,4.0,3.9,3.8,3.7,3.6,
         3.5,3.4,3.3,3.4,3.3,3.3,3.2,3.2,3.1,3.1,3.2,3.1,
         3.1,3.0,3.1,3.0,3.1,3.0,3.1,3.0,3.1,3.0,3.1,3.0,],
        noise=0.12)

    sentiment_m = ts(dates_m,
        [95,95,97,98,99,99,98,96,95,97,97,99,
         98,96,97,96,97,95,95,94,95,71,73,74,
         74,76,81,84,82,84,82,82,81,82,81,83,
         88,84,76,65,59,58,50,51,58,55,57,61,
         64,67,63,70,71,67,62,57,61,59,61,60,
         63,64,63,68,69,67,66,65,68,70,71,73,
         72,73,74,74,75,74,73,74,73,74,73,74,
         73,74,73,74,73,74,73,74,73,74,73,74,],
        noise=1.0)

    housing_m = ts(dates_m,
        [1250,1300,1290,1270,1310,1340,1220,1170,1210,1230,1190,1080,
         1070,1120,1200,1280,1250,1240,1220,1240,1380,900,974,1070,
         1120,1160,1200,1570,1580,1600,1510,1470,1530,1540,1480,1550,
         1600,1550,1430,1420,1440,1580,1540,1450,1520,1430,1350,1450,
         1380,1480,1380,1430,1460,1420,1360,1310,1350,1320,1350,1280,
         1310,1320,1340,1350,1360,1370,1360,1350,1360,1370,1360,1370,
         1370,1380,1370,1380,1370,1380,1380,1390,1380,1390,1380,1390,
         1380,1390,1380,1390,1380,1390,1380,1390,1380,1390,1380,1390,],
        noise=40)

    return {
        "gdp":         gdp_q,
        "indpro":      indpro_m,
        "unrate":      unrate_m,
        "cpi":         cpi_m,
        "fedfunds":    fedfunds_m,
        "t10y":        t10y_m,
        "t2y":         t2y_m,
        "t3m":         t3m_m,
        "hy_spread":   hy_spread_m,
        "sentiment":   sentiment_m,
        "housing":     housing_m,
        "pce":         cpi_m * 0.97 + 0.1,
        "breakeven5y": cpi_m * 0.85 + 0.3,
        "ig_spread":   hy_spread_m * 0.28,
        "m2":          pd.Series(
            np.linspace(14000, 22000, len(dates_m)),
            index=dates_m),
        "permits":     housing_m * 0.95,
        "retail":      pd.Series(
            np.linspace(480000, 720000, len(dates_m)) +
            np.random.normal(0, 5000, len(dates_m)),
            index=dates_m),
        "claims":      pd.Series(
            200 + 50 * np.random.randn(len(dates_d)),
            index=dates_d),
    }


# ── REGIME SCORING ────────────────────────────────────────────────────────────

def compute_yoy(series: pd.Series, periods: int = 12) -> pd.Series:
    return series.pct_change(periods) * 100


def safe_last(series: pd.Series, n: int = 1):
    s = series.dropna()
    if len(s) < n:
        return None
    return s.iloc[-n]


def score_regime(data: dict) -> dict:
    """
    Score each indicator against expansion/peak/contraction/recovery signals.
    Returns a dict with per-phase scores and a winner.

    Scoring logic (each indicator contributes 0-2 pts to one or more phases):
    - GDP growth momentum
    - Industrial production trend
    - Unemployment direction
    - CPI level and trend
    - Fed Funds vs neutral rate
    - Yield curve slope (10y-2y, 10y-3m)
    - Credit spread level
    - Consumer sentiment trend
    - Housing starts trend
    """
    scores = {"Expansion": 0, "Peak": 0, "Contraction": 0, "Recovery": 0}
    details = {}

    # ── 1. GDP GROWTH ────────────────────────────────────────────────────────
    gdp = data.get("gdp", pd.Series(dtype=float)).dropna()
    if len(gdp) >= 5:
        gdp_growth = gdp.pct_change(4).dropna() * 100  # YoY quarterly
        g_now = safe_last(gdp_growth)
        g_prev = safe_last(gdp_growth, 2)
        if g_now is not None and g_prev is not None:
            if g_now > 2.5 and g_now > g_prev:
                scores["Expansion"] += 2
                details["GDP"] = f"Strong & accelerating ({g_now:.1f}% YoY)"
            elif g_now > 2.5 and g_now <= g_prev:
                scores["Peak"] += 2
                details["GDP"] = f"Strong but decelerating ({g_now:.1f}% YoY)"
            elif 0 < g_now <= 2.5:
                scores["Recovery"] += 1
                scores["Peak"]     += 1
                details["GDP"] = f"Moderate growth ({g_now:.1f}% YoY)"
            elif g_now <= 0:
                scores["Contraction"] += 2
                details["GDP"] = f"Contraction ({g_now:.1f}% YoY)"
            else:
                details["GDP"] = "Insufficient data"

    # ── 2. INDUSTRIAL PRODUCTION ─────────────────────────────────────────────
    ip = data.get("indpro", pd.Series(dtype=float)).dropna()
    if len(ip) >= 13:
        ip_yoy = compute_yoy(ip).dropna()
        ip_now  = safe_last(ip_yoy)
        ip_prev = safe_last(ip_yoy, 4)
        if ip_now is not None:
            if ip_now > 3 and (ip_prev is None or ip_now > ip_prev):
                scores["Expansion"] += 2
                details["IndustrialProduction"] = f"Expanding rapidly ({ip_now:.1f}% YoY)"
            elif ip_now > 0:
                scores["Expansion"] += 1
                scores["Recovery"]  += 1
                details["IndustrialProduction"] = f"Modest growth ({ip_now:.1f}% YoY)"
            elif -3 < ip_now <= 0:
                scores["Peak"]       += 1
                scores["Contraction"] += 1
                details["IndustrialProduction"] = f"Stagnating ({ip_now:.1f}% YoY)"
            else:
                scores["Contraction"] += 2
                details["IndustrialProduction"] = f"Contracting ({ip_now:.1f}% YoY)"

    # ── 3. UNEMPLOYMENT ──────────────────────────────────────────────────────
    ur = data.get("unrate", pd.Series(dtype=float)).dropna()
    if len(ur) >= 7:
        ur_now  = safe_last(ur)
        ur_6m   = safe_last(ur, 6)
        ur_trend = ur_now - ur_6m if (ur_now and ur_6m) else None
        if ur_now is not None:
            if ur_now < 4.5 and (ur_trend is None or ur_trend < 0):
                scores["Expansion"] += 2
                details["Unemployment"] = f"Low & falling ({ur_now:.1f}%)"
            elif ur_now < 4.5 and ur_trend and ur_trend >= 0:
                scores["Peak"] += 2
                details["Unemployment"] = f"Low but rising ({ur_now:.1f}%)"
            elif 4.5 <= ur_now < 6.0:
                scores["Contraction"] += 1
                scores["Recovery"]    += 1
                details["Unemployment"] = f"Elevated ({ur_now:.1f}%)"
            else:
                scores["Contraction"] += 2
                details["Unemployment"] = f"High ({ur_now:.1f}%)"

    # ── 4. CPI INFLATION ─────────────────────────────────────────────────────
    cpi = data.get("cpi", pd.Series(dtype=float)).dropna()
    if len(cpi) >= 13:
        cpi_yoy  = compute_yoy(cpi).dropna()
        cpi_now  = safe_last(cpi_yoy)
        cpi_prev = safe_last(cpi_yoy, 6)
        if cpi_now is not None:
            if cpi_now < 3.0 and (cpi_prev is None or cpi_now >= cpi_prev):
                scores["Expansion"] += 1
                scores["Recovery"]  += 1
                details["CPI"] = f"Moderate, rising ({cpi_now:.1f}%)"
            elif cpi_now >= 3.0 and cpi_prev and cpi_now > cpi_prev:
                scores["Peak"] += 2
                details["CPI"] = f"High & accelerating ({cpi_now:.1f}%)"
            elif cpi_now >= 3.0 and cpi_prev and cpi_now <= cpi_prev:
                scores["Peak"]       += 1
                scores["Contraction"] += 1
                details["CPI"] = f"High but decelerating ({cpi_now:.1f}%)"
            elif cpi_now < 2.0:
                scores["Recovery"]    += 1
                scores["Contraction"] += 1
                details["CPI"] = f"Below target ({cpi_now:.1f}%)"

    # ── 5. FED FUNDS vs NEUTRAL ──────────────────────────────────────────────
    ff = data.get("fedfunds", pd.Series(dtype=float)).dropna()
    if len(ff) >= 3:
        ff_now  = safe_last(ff)
        ff_prev = safe_last(ff, 6)
        if ff_now is not None:
            NEUTRAL = 3.0  # approximate neutral rate estimate
            if ff_now < NEUTRAL and (ff_prev is None or ff_now <= ff_prev):
                scores["Recovery"]  += 1
                scores["Expansion"] += 1
                details["FedFunds"] = f"Accommodative ({ff_now:.2f}% < neutral {NEUTRAL}%)"
            elif ff_now < NEUTRAL and ff_prev and ff_now > ff_prev:
                scores["Expansion"] += 1
                details["FedFunds"] = f"Normalising ({ff_now:.2f}%)"
            elif ff_now >= NEUTRAL and ff_prev and ff_now > ff_prev:
                scores["Peak"] += 2
                details["FedFunds"] = f"Restrictive & tightening ({ff_now:.2f}%)"
            elif ff_now >= NEUTRAL and ff_prev and ff_now <= ff_prev:
                scores["Contraction"] += 1
                scores["Recovery"]    += 1
                details["FedFunds"] = f"Restrictive but easing ({ff_now:.2f}%)"

    # ── 6. YIELD CURVE ───────────────────────────────────────────────────────
    t10 = data.get("t10y", pd.Series(dtype=float)).dropna()
    t2  = data.get("t2y",  pd.Series(dtype=float)).dropna()
    t3m = data.get("t3m",  pd.Series(dtype=float)).dropna()
    if len(t10) > 0 and len(t2) > 0:
        slope_10_2 = safe_last(t10) - safe_last(t2)
        slope_10_3m = (safe_last(t10) - safe_last(t3m)
                       if len(t3m) > 0 else None)
        if slope_10_2 is not None:
            if slope_10_2 > 1.0:
                scores["Recovery"]  += 2
                scores["Expansion"] += 1
                details["YieldCurve10s2s"] = f"Steeply normal ({slope_10_2:+.2f}%)"
            elif 0 < slope_10_2 <= 1.0:
                scores["Expansion"] += 2
                details["YieldCurve10s2s"] = f"Normal ({slope_10_2:+.2f}%)"
            elif -0.5 < slope_10_2 <= 0:
                scores["Peak"] += 2
                details["YieldCurve10s2s"] = f"Flat ({slope_10_2:+.2f}%)"
            else:
                scores["Contraction"] += 2
                details["YieldCurve10s2s"] = f"Inverted ({slope_10_2:+.2f}%)"
        if slope_10_3m is not None:
            if slope_10_3m > 0.5:
                scores["Recovery"] += 1
            elif slope_10_3m < -0.3:
                scores["Contraction"] += 1
                details["YieldCurve10s3m"] = f"Inverted ({slope_10_3m:+.2f}%)"
            else:
                details["YieldCurve10s3m"] = f"Flat/normal ({slope_10_3m:+.2f}%)"

    # ── 7. CREDIT SPREADS ────────────────────────────────────────────────────
    hy = data.get("hy_spread", pd.Series(dtype=float)).dropna()
    if len(hy) >= 3:
        hy_now  = safe_last(hy)
        hy_prev = safe_last(hy, 6)
        if hy_now is not None:
            if hy_now < 3.5:
                scores["Expansion"] += 2
                details["HYSpread"] = f"Tight — risk-on ({hy_now:.1f}%)"
            elif 3.5 <= hy_now < 5.0:
                scores["Peak"] += 1
                details["HYSpread"] = f"Widening ({hy_now:.1f}%)"
            elif 5.0 <= hy_now < 8.0:
                scores["Contraction"] += 1
                details["HYSpread"] = f"Elevated ({hy_now:.1f}%)"
            else:
                scores["Contraction"] += 2
                details["HYSpread"] = f"Distressed ({hy_now:.1f}%)"

    # ── 8. CONSUMER SENTIMENT ────────────────────────────────────────────────
    sent = data.get("sentiment", pd.Series(dtype=float)).dropna()
    if len(sent) >= 4:
        s_now  = safe_last(sent)
        s_prev = safe_last(sent, 4)
        if s_now is not None:
            if s_now > 85 and (s_prev is None or s_now > s_prev):
                scores["Expansion"] += 1
                details["Sentiment"] = f"High & improving ({s_now:.0f})"
            elif s_now > 70:
                scores["Expansion"] += 1
                details["Sentiment"] = f"Moderate ({s_now:.0f})"
            elif 55 < s_now <= 70:
                scores["Recovery"] += 1
                details["Sentiment"] = f"Recovering ({s_now:.0f})"
            else:
                scores["Contraction"] += 1
                details["Sentiment"] = f"Pessimistic ({s_now:.0f})"

    # ── 9. HOUSING STARTS ────────────────────────────────────────────────────
    hs = data.get("housing", pd.Series(dtype=float)).dropna()
    if len(hs) >= 7:
        hs_now  = safe_last(hs)
        hs_6m   = safe_last(hs, 6)
        if hs_now is not None and hs_6m is not None:
            hs_chg = (hs_now - hs_6m) / hs_6m * 100
            if hs_chg > 5:
                scores["Recovery"]  += 1
                scores["Expansion"] += 1
                details["Housing"] = f"Accelerating ({hs_chg:+.1f}% 6M)"
            elif -5 < hs_chg <= 5:
                scores["Expansion"] += 1
                details["Housing"] = f"Stable ({hs_chg:+.1f}% 6M)"
            else:
                scores["Contraction"] += 1
                details["Housing"] = f"Weakening ({hs_chg:+.1f}% 6M)"

    # ── DETERMINE WINNER ─────────────────────────────────────────────────────
    total = sum(scores.values()) or 1
    pct   = {k: v / total * 100 for k, v in scores.items()}
    phase = max(scores, key=scores.get)

    return {
        "scores":   scores,
        "pct":      pct,
        "phase":    phase,
        "details":  details,
        "total":    total,
    }


# ── ALLOCATION RECOMMENDATIONS ───────────────────────────────────────────────

PHASE_ALLOCATIONS = {
    "Expansion": {
        "Global Equities":        30,
        "Emerging Markets":       10,
        "Private Equity":         22,
        "High Yield Bonds":        8,
        "Real Estate":            10,
        "Commodities":             5,
        "Investment Grade Bonds":  5,
        "Government Bonds":        2,
        "Gold":                    5,
        "Cash":                    3,
    },
    "Peak": {
        "Global Equities":        20,
        "Emerging Markets":        5,
        "Private Equity":         18,
        "High Yield Bonds":        5,
        "Real Estate":             8,
        "Commodities":            12,
        "Investment Grade Bonds":  8,
        "Government Bonds":        5,
        "Gold":                   10,
        "Cash":                    9,
    },
    "Contraction": {
        "Global Equities":         8,
        "Emerging Markets":        2,
        "Private Equity":          4,
        "High Yield Bonds":        2,
        "Real Estate":             3,
        "Commodities":             3,
        "Investment Grade Bonds":  20,
        "Government Bonds":        28,
        "Gold":                   15,
        "Cash":                   15,
    },
    "Recovery": {
        "Global Equities":        25,
        "Emerging Markets":        8,
        "Private Equity":         18,
        "High Yield Bonds":        8,
        "Real Estate":             8,
        "Commodities":             7,
        "Investment Grade Bonds":  8,
        "Government Bonds":        5,
        "Gold":                    8,
        "Cash":                    5,
    },
}

PHASE_MACRO = {
    "Expansion":   {"GDP": "Accelerating (2.5%+)", "Inflation": "Rising moderately",
                    "Central Bank": "Neutral to tightening", "Spreads": "Tightening",
                    "Yield Curve": "Steepening", "Earnings": "Strong beats"},
    "Peak":        {"GDP": "Slowing from peak", "Inflation": "High, persistent",
                    "Central Bank": "Actively tightening", "Spreads": "Widening",
                    "Yield Curve": "Flat / inverting", "Earnings": "Margin pressure"},
    "Contraction": {"GDP": "Negative (recession)", "Inflation": "Falling",
                    "Central Bank": "Cutting rates", "Spreads": "Sharply wide",
                    "Yield Curve": "Deeply inverted", "Earnings": "Declining"},
    "Recovery":    {"GDP": "Turning positive", "Inflation": "Low, stable",
                    "Central Bank": "Accommodative", "Spreads": "Tightening",
                    "Yield Curve": "Re-steepening", "Earnings": "Recovering"},
}


# ── VISUALISATION ─────────────────────────────────────────────────────────────

def make_dashboard(data: dict, result: dict, demo: bool = False):
    phase  = result["phase"]
    scores = result["scores"]
    pct    = result["pct"]
    p_col  = PHASE_COLORS[phase]
    alloc  = PHASE_ALLOCATIONS[phase]
    macro  = PHASE_MACRO[phase]

    fig = plt.figure(figsize=(24, 28), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(5, 3, figure=fig,
                            hspace=0.45, wspace=0.32)

    def sax(ax, title="", xl="", yl=""):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=WHITE, labelsize=8)
        ax.spines[:].set_color(GREY)
        for l in ax.get_xticklabels() + ax.get_yticklabels():
            l.set_color(WHITE)
        if title: ax.set_title(title, color=GOLD, fontsize=10,
                               fontweight="bold", pad=8)
        if xl:    ax.set_xlabel(xl, color=WHITE, fontsize=8)
        if yl:    ax.set_ylabel(yl, color=WHITE, fontsize=8)

    pct_f = FuncFormatter(lambda x, _: f"{x:.1f}%")
    pct_i = FuncFormatter(lambda x, _: f"{x:.0f}%")

    # ── TITLE ROW ────────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor(DARK_BG); ax0.axis("off")
    mode_tag = "  [DEMO MODE]" if demo else "  [LIVE DATA — FRED]"
    ax0.text(0.5, 0.88,
             "MACRO DASHBOARD — ECONOMIC REGIME DETECTOR" + mode_tag,
             ha="center", color=GOLD, fontsize=20, fontweight="bold",
             transform=ax0.transAxes)
    ax0.text(0.5, 0.62,
             f"Current Detected Regime:  {phase.upper()}",
             ha="center", color=p_col, fontsize=16, fontweight="bold",
             transform=ax0.transAxes)
    ax0.text(0.5, 0.36,
             f"Score: {scores[phase]}/{result['total']} points  |  "
             f"Confidence: {pct[phase]:.0f}%  |  "
             f"Last Updated: {datetime.today().strftime('%B %Y')}",
             ha="center", color=WHITE, fontsize=11,
             transform=ax0.transAxes)
    ax0.text(0.5, 0.12,
             "The Meridian Playbook  |  themeridianplaybook.com",
             ha="center", color=MID, fontsize=9, transform=ax0.transAxes)
    ax0.axhline(0.04, color=GOLD, linewidth=0.8, xmin=0.05, xmax=0.95)

    # ── 1. REGIME CONFIDENCE GAUGE ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor(DARK_BG); ax1.axis("off")
    ax1.set_title("Regime Score Distribution", color=GOLD,
                  fontsize=10, fontweight="bold", pad=8)
    phases_ord = ["Expansion", "Peak", "Contraction", "Recovery"]
    bar_vals   = [pct[p] for p in phases_ord]
    bar_cols   = [PHASE_COLORS[p] for p in phases_ord]
    ax1b = ax1.inset_axes([0.0, 0.05, 1.0, 0.85])
    ax1b.set_facecolor(DARK_BG)
    y_pos = np.arange(len(phases_ord))
    bars = ax1b.barh(y_pos, bar_vals, color=bar_cols, alpha=0.85)
    ax1b.set_yticks(y_pos)
    ax1b.set_yticklabels(phases_ord)
    for bar, val in zip(bars, bar_vals):
        ax1b.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                  f"{val:.0f}%", va="center", color=WHITE, fontsize=9)
    ax1b.xaxis.set_major_formatter(pct_i)
    ax1b.tick_params(colors=WHITE, labelsize=9)
    ax1b.spines[:].set_color(GREY)
    for l in ax1b.get_xticklabels() + ax1b.get_yticklabels():
        l.set_color(WHITE)
    ax1b.set_xlim(0, 55)

    # ── 2. CYCLE WHEEL ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_facecolor(DARK_BG); ax2.axis("off")
    ax2.set_title("Cycle Clock", color=GOLD, fontsize=10,
                  fontweight="bold", pad=8)
    ax2.set_xlim(-1.4, 1.4); ax2.set_ylim(-1.4, 1.4)
    ax2_inner = ax2.inset_axes([0.0, 0.0, 1.0, 1.0])
    ax2_inner.set_facecolor(DARK_BG); ax2_inner.axis("off")
    ax2_inner.set_xlim(-1.4, 1.4); ax2_inner.set_ylim(-1.4, 1.4)
    ax2_inner.set_aspect("equal")
    from matplotlib.patches import Wedge
    quads = [
        ("Recovery",    0,   90,  0.60,  0.60),
        ("Expansion",  90,  180, -0.60,  0.60),
        ("Peak",       180, 270, -0.60, -0.60),
        ("Contraction",270, 360,  0.60, -0.60),
    ]
    for qname, a1, a2, tx, ty in quads:
        is_cur = (qname == phase)
        w = Wedge((0, 0), 1.1, a1, a2,
                  facecolor=PHASE_COLORS[qname],
                  alpha=0.9 if is_cur else 0.30,
                  edgecolor=DARK_BG, linewidth=2)
        ax2_inner.add_patch(w)
        ax2_inner.text(tx, ty, qname, ha="center", va="center",
                       color=WHITE if is_cur else MID,
                       fontsize=8.5,
                       fontweight="bold" if is_cur else "normal")
        if is_cur:
            ax2_inner.text(tx, ty - 0.22, "◄ NOW", ha="center",
                           color=PHASE_COLORS[qname], fontsize=7,
                           fontweight="bold")
    centre = plt.Circle((0, 0), 0.28, color=DARK_BG, zorder=5)
    ax2_inner.add_patch(centre)
    ax2_inner.text(0, 0.05, "CYCLE", ha="center", color=GOLD,
                   fontsize=8, fontweight="bold", zorder=6)
    ax2_inner.text(0, -0.10, "CLOCK", ha="center", color=GOLD,
                   fontsize=8, fontweight="bold", zorder=6)
    for angle in [45, 135, 225, 315]:
        rad = np.radians(angle)
        ax2_inner.annotate(
            "", xy=(1.28*np.cos(rad+0.15), 1.28*np.sin(rad+0.15)),
            xytext=(1.28*np.cos(rad-0.15), 1.28*np.sin(rad-0.15)),
            arrowprops=dict(arrowstyle="->", color=MID, lw=1.0))

    # ── 3. OPTIMAL ALLOCATION PIE ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    pie_colors = [GOLD, BLUE, PURPLE, RED, GREEN, ORANGE,
                  WHITE, MID, "#c9a84c", "#58a6ff"]
    short_lbls = [k.replace("Investment Grade ", "IG ")
                   .replace("Government ", "Govt ")
                   .replace("Emerging ", "EM ")
                   .replace("Global ", "Glb ")
                   for k in alloc.keys()]
    wedges, texts, ats = ax3.pie(
        list(alloc.values()),
        labels=short_lbls,
        autopct="%1.0f%%",
        colors=pie_colors,
        textprops={"color": WHITE, "fontsize": 6.5},
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 1.2},
        startangle=90,
    )
    for at in ats:
        at.set_color(DARK_BG); at.set_fontsize(6.5)
    ax3.set_facecolor(DARK_BG)
    sax(ax3, f"Recommended Allocation\n{phase} Phase")

    # ── 4. YIELD CURVE ───────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    t10 = data.get("t10y", pd.Series(dtype=float)).dropna()
    t2  = data.get("t2y",  pd.Series(dtype=float)).dropna()
    t3m = data.get("t3m",  pd.Series(dtype=float)).dropna()
    if len(t10) > 0 and len(t2) > 0:
        # Resample to monthly
        t10m = t10.resample("ME").last()
        t2m  = t2.resample("ME").last()
        spread = (t10m - t2m).dropna()
        spread = spread[spread.index >= "2018-01-01"]
        color_fill = [RED if v < 0 else GREEN for v in spread.values]
        ax4.bar(spread.index, spread.values,
                color=[RED if v < 0 else GREEN for v in spread.values],
                alpha=0.75, width=20)
        ax4.axhline(0, color=WHITE, linewidth=0.8, linestyle=":")
        ax4.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter("%Y"))
    sax(ax4, "Yield Curve Slope: 10Y − 2Y (%)",
        xl="Date", yl="Spread (%)")

    # ── 5. CPI INFLATION ─────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    cpi = data.get("cpi", pd.Series(dtype=float)).dropna()
    if len(cpi) >= 13:
        cpi_yoy = compute_yoy(cpi).dropna()
        cpi_yoy = cpi_yoy[cpi_yoy.index >= "2018-01-01"]
        ax5.fill_between(cpi_yoy.index, cpi_yoy.values,
                         alpha=0.2, color=ORANGE)
        ax5.plot(cpi_yoy.index, cpi_yoy.values,
                 color=ORANGE, linewidth=2)
        ax5.axhline(2.0, color=GREEN, linewidth=1,
                    linestyle="--", alpha=0.7, label="2% target")
        ax5.legend(fontsize=8, labelcolor=WHITE,
                   facecolor=GREY, edgecolor=GREY)
    sax(ax5, "CPI Inflation YoY (%)", yl="YoY %")

    # ── 6. FED FUNDS + 10Y ───────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ff  = data.get("fedfunds", pd.Series(dtype=float)).dropna()
    if len(ff) > 0 and len(t10) > 0:
        ff_m  = ff.resample("ME").last()
        t10_m = t10.resample("ME").last()
        common = ff_m.index.intersection(t10_m.index)
        common = common[common >= pd.Timestamp("2018-01-01")]
        ax6.plot(common, ff_m.loc[common].values,
                 color=BLUE, linewidth=2, label="Fed Funds")
        ax6.plot(common, t10_m.loc[common].values,
                 color=GOLD, linewidth=2, linestyle="--",
                 label="10Y Treasury")
        ax6.axhline(3.0, color=MID, linewidth=0.8,
                    linestyle=":", alpha=0.6, label="Neutral rate (~3%)")
        ax6.legend(fontsize=7.5, labelcolor=WHITE,
                   facecolor=GREY, edgecolor=GREY)
    sax(ax6, "Fed Funds Rate vs 10Y Treasury (%)", yl="Rate (%)")

    # ── 7. UNEMPLOYMENT ──────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    ur = data.get("unrate", pd.Series(dtype=float)).dropna()
    if len(ur) > 0:
        ur2 = ur[ur.index >= "2018-01-01"]
        ax7.fill_between(ur2.index, ur2.values, alpha=0.2, color=BLUE)
        ax7.plot(ur2.index, ur2.values, color=BLUE, linewidth=2)
        ax7.axhline(4.0, color=GREEN, linewidth=1, linestyle="--",
                    alpha=0.7, label="Full employment (~4%)")
        ax7.legend(fontsize=8, labelcolor=WHITE,
                   facecolor=GREY, edgecolor=GREY)
    sax(ax7, "Unemployment Rate (%)", yl="%")

    # ── 8. HY CREDIT SPREAD ──────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    hy = data.get("hy_spread", pd.Series(dtype=float)).dropna()
    if len(hy) > 0:
        hy2 = hy.resample("ME").last() if hy.index.freq != "MS" else hy
        hy2 = hy2[hy2.index >= "2018-01-01"]
        ax8.fill_between(hy2.index, hy2.values, alpha=0.2, color=RED)
        ax8.plot(hy2.index, hy2.values, color=RED, linewidth=2)
        ax8.axhline(4.0, color=ORANGE, linewidth=1,
                    linestyle="--", alpha=0.7, label="Stress threshold (4%)")
        ax8.legend(fontsize=8, labelcolor=WHITE,
                   facecolor=GREY, edgecolor=GREY)
    sax(ax8, "HY Credit Spread OAS (%)", yl="Spread (%)")

    # ── 9. INDUSTRIAL PRODUCTION ─────────────────────────────────────────────
    ax9 = fig.add_subplot(gs[3, 2])
    ip = data.get("indpro", pd.Series(dtype=float)).dropna()
    if len(ip) >= 13:
        ip_yoy = compute_yoy(ip).dropna()
        ip_yoy = ip_yoy[ip_yoy.index >= "2018-01-01"]
        cols_ip = [GREEN if v > 0 else RED for v in ip_yoy.values]
        ax9.bar(ip_yoy.index, ip_yoy.values,
                color=cols_ip, alpha=0.75, width=20)
        ax9.axhline(0, color=WHITE, linewidth=0.7, linestyle=":")
    sax(ax9, "Industrial Production YoY (%)", yl="YoY %")

    # ── 10. INDICATOR SCORECARD TABLE ────────────────────────────────────────
    ax10 = fig.add_subplot(gs[4, :2])
    ax10.set_facecolor(DARK_BG); ax10.axis("off")
    ax10.set_title("Indicator Scorecard — Signal Details",
                   color=GOLD, fontsize=10, fontweight="bold", pad=8)

    headers = ["Indicator", "Current Signal", "Points to Phase"]
    col_x   = [0.01, 0.38, 0.78]
    ax10.text(col_x[0], 0.94, headers[0], transform=ax10.transAxes,
              color=GOLD, fontsize=8.5, fontweight="bold")
    ax10.text(col_x[1], 0.94, headers[1], transform=ax10.transAxes,
              color=GOLD, fontsize=8.5, fontweight="bold")
    ax10.text(col_x[2], 0.94, headers[2], transform=ax10.transAxes,
              color=GOLD, fontsize=8.5, fontweight="bold")
    ax10.plot([0.01, 0.99], [0.90, 0.90], color=GOLD, linewidth=0.5,
              transform=ax10.transAxes)

    details = result["details"]
    items   = list(details.items())
    for i, (ind, sig) in enumerate(items[:9]):
        y  = 0.83 - i * 0.092
        c  = WHITE if i % 2 == 0 else MID
        ax10.text(col_x[0], y, ind.replace("YieldCurve", "Yield Curve "),
                  transform=ax10.transAxes, color=c, fontsize=8)
        ax10.text(col_x[1], y, sig,
                  transform=ax10.transAxes, color=c, fontsize=8)
        # Determine which phase it signals
        sig_lower = sig.lower()
        if any(w in sig_lower for w in ["strong","acceler","tight","low","boom"]):
            pt = "Expansion"
        elif any(w in sig_lower for w in ["decelerat","high & accel","flat","widen"]):
            pt = "Peak"
        elif any(w in sig_lower for w in ["contract","negative","inverted","distress","high"]):
            pt = "Contraction"
        else:
            pt = "Recovery"
        ax10.text(col_x[2], y, pt,
                  transform=ax10.transAxes,
                  color=PHASE_COLORS.get(pt, WHITE), fontsize=8,
                  fontweight="bold")

    # ── 11. MACRO REGIME TABLE ───────────────────────────────────────────────
    ax11 = fig.add_subplot(gs[4, 2])
    ax11.set_facecolor(DARK_BG); ax11.axis("off")
    ax11.set_title(f"Macro Profile: {phase}",
                   color=p_col, fontsize=10, fontweight="bold", pad=8)
    for i, (k, v) in enumerate(macro.items()):
        y = 0.88 - i * 0.145
        ax11.text(0.02, y, f"{k}:", transform=ax11.transAxes,
                  color=GOLD, fontsize=8.5, fontweight="bold")
        ax11.text(0.02, y - 0.065, v, transform=ax11.transAxes,
                  color=WHITE, fontsize=8)

    # FOOTER
    fig.text(0.5, 0.005,
             "The Meridian Playbook  |  Research on Capital Allocation & Financial Systems"
             "  |  themeridianplaybook.com  |  "
             "Data: Federal Reserve FRED API",
             ha="center", color=GREY, fontsize=8)

    os.makedirs("outputs", exist_ok=True)
    out = "outputs/macro_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"    ✓ Dashboard saved → {out}\n")
    return out


# ── CONSOLE SUMMARY ───────────────────────────────────────────────────────────

def print_summary(result: dict):
    phase  = result["phase"]
    scores = result["scores"]
    pct    = result["pct"]
    p_col  = "\033[93m" if phase == "Expansion" else \
             "\033[91m" if phase == "Contraction" else \
             "\033[94m" if phase == "Recovery" else "\033[33m"
    RST    = "\033[0m"

    print("=" * 65)
    print(f"  ECONOMIC REGIME DETECTION — {datetime.today().strftime('%B %Y').upper()}")
    print("=" * 65)
    print(f"\n  {p_col}Detected Phase: {phase.upper()}{RST}")
    print(f"  Confidence:    {pct[phase]:.0f}%\n")
    print("  Score breakdown:")
    for p in ["Expansion", "Peak", "Contraction", "Recovery"]:
        bar = "█" * scores[p]
        marker = "  ◄" if p == phase else ""
        print(f"    {p:<14} {bar:<12} {pct[p]:.0f}%{marker}")
    print("\n  Indicator signals:")
    for k, v in result["details"].items():
        print(f"    {k:<26} {v}")
    print("\n  Recommended allocation:")
    for asset, w in PHASE_ALLOCATIONS[phase].items():
        print(f"    {asset:<30} {w}%")
    print("\n" + "=" * 65)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Macro Dashboard — Economic Regime Detector"
    )
    parser.add_argument("--api-key", type=str, default="",
                        help="FRED API key (free at fred.stlouisfed.org)")
    parser.add_argument("--demo",    action="store_true",
                        help="Run in demo mode with simulated data")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════╗")
    print("║   MACRO DASHBOARD — ECONOMIC REGIME DETECTOR     ║")
    print("║   The Meridian Playbook                          ║")
    print("╚══════════════════════════════════════════════════╝\n")

    demo_mode = args.demo or not args.api_key

    if demo_mode:
        print("  Running in DEMO MODE (simulated data).")
        print("  For live data: python src/macro_dashboard.py --api-key YOUR_KEY\n")
        data = generate_demo_data()
    else:
        print(f"  FRED API key detected. Fetching live data...\n")
        data = fetch_all_fred(args.api_key)

    print("  Scoring economic regime...")
    result = score_regime(data)
    print_summary(result)

    print("  Generating dashboard...")
    make_dashboard(data, result, demo=demo_mode)

    print("✅  Done.  Open outputs/macro_dashboard.png\n")


if __name__ == "__main__":
    main()
