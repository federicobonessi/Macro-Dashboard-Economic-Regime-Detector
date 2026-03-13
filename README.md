# Macro Dashboard — Economic Regime Detector

**Aggregates real macroeconomic data from the FRED API and automatically classifies the current economic regime across four phases: Expansion, Peak, Contraction, and Recovery — with cycle-consistent allocation recommendations.**

Built as part of [The Meridian Playbook](https://themeridianplaybook.com) — a research project on capital allocation, portfolio strategy and global financial systems.

---

## What It Does

This tool reads live macroeconomic data from the Federal Reserve's FRED API and applies a multi-indicator scoring model to determine the current economic phase. It answers the question every family office CIO asks every quarter:

> *Where are we in the cycle — and how should capital be positioned?*

The tool scores nine indicators across four regime dimensions:

| Indicator | Signal Type |
|-----------|-------------|
| Real GDP Growth (YoY, quarterly) | Growth momentum |
| Industrial Production (YoY) | Real economy activity |
| Unemployment Rate | Labour market direction |
| CPI Inflation (YoY) | Price level and trend |
| Fed Funds Rate vs Neutral | Monetary policy stance |
| Yield Curve Slope (10Y−2Y, 10Y−3M) | Credit conditions |
| HY Credit Spread (OAS) | Risk appetite |
| Consumer Sentiment | Forward-looking demand |
| Housing Starts | Leading real economy signal |

---

## Output

A single high-resolution dashboard (`outputs/macro_dashboard.png`) with eleven panels:

1. **Regime title bar** — detected phase, confidence score, timestamp
2. **Regime score distribution** — percentage score per phase
3. **Cycle clock** — four-quadrant wheel with current phase highlighted
4. **Optimal allocation pie** — cycle-consistent recommended weights
5. **Yield curve slope (10Y−2Y)** — recession signal visualisation
6. **CPI inflation YoY** — with 2% target reference
7. **Fed Funds vs 10Y Treasury** — monetary policy positioning
8. **Unemployment rate** — with full employment reference
9. **HY credit spread** — risk appetite indicator
10. **Industrial production YoY** — real economy signal
11. **Indicator scorecard table** — signal-by-signal detail
12. **Macro regime profile** — current phase characteristics

---

## Installation

```bash
git clone https://github.com/your-username/macro-dashboard.git
cd macro-dashboard
pip install -r requirements.txt
```

---

## Usage

### Demo Mode (no API key required)
```bash
python src/macro_dashboard.py --demo
```

### Live Mode (FRED API key)
Get a free API key at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

```bash
python src/macro_dashboard.py --api-key YOUR_FRED_API_KEY
```

---

## Regime Scoring Logic

Each indicator contributes 0–2 points toward each phase:

```
Expansion:   High GDP growth + accelerating | Low unemployment | Tight spreads
Peak:        Slowing GDP | High inflation | Inverted/flat curve | Fed tightening
Contraction: Negative GDP | Rising unemployment | Wide spreads | Fed cutting
Recovery:    Positive but modest GDP | Low inflation | Steepening curve | Accommodative Fed
```

The phase with the highest cumulative score is declared the current regime. Confidence is expressed as the share of total points assigned to the winning phase.

---

## Allocation Framework

Cycle-consistent recommended allocations per phase:

| Asset Class | Expansion | Peak | Contraction | Recovery |
|-------------|-----------|------|-------------|---------|
| Global Equities | 30% | 20% | 8% | 25% |
| Emerging Markets | 10% | 5% | 2% | 8% |
| Private Equity | 22% | 18% | 4% | 18% |
| High Yield Bonds | 8% | 5% | 2% | 8% |
| Real Estate | 10% | 8% | 3% | 8% |
| Commodities | 5% | 12% | 3% | 7% |
| Investment Grade Bonds | 5% | 8% | 20% | 8% |
| Government Bonds | 2% | 5% | 28% | 5% |
| Gold | 5% | 10% | 15% | 8% |
| Cash | 3% | 9% | 15% | 5% |

---

## FRED Series Used

| Series ID | Description |
|-----------|-------------|
| `GDPC1` | Real GDP (Quarterly, SA) |
| `INDPRO` | Industrial Production Index |
| `UNRATE` | Unemployment Rate |
| `CPIAUCSL` | CPI All Items |
| `FEDFUNDS` | Federal Funds Rate |
| `DGS10` | 10Y Treasury Yield |
| `DGS2` | 2Y Treasury Yield |
| `DGS3MO` | 3M Treasury Yield |
| `BAMLH0A0HYM2` | HY Credit Spread (OAS) |
| `UMCSENT` | U. Michigan Consumer Sentiment |
| `HOUST` | Housing Starts |
| `T5YIE` | 5Y Breakeven Inflation |
| `M2SL` | M2 Money Supply |
| `RSAFS` | Retail Sales |

---

## The Trilogy + Dashboard

| Project | Focus |
|---------|-------|
| [Portfolio Optimizer](https://github.com/your-username/portfolio-analyzer) | Efficient frontier, Sharpe optimization |
| [Risk Scoring Model](https://github.com/your-username/risk-scoring-model) | Multi-dimensional UHNW risk assessment |
| [Wealth Projection Tool](https://github.com/your-username/wealth-projection) | Long-term wealth preservation scenarios |
| [Cycle Allocation](https://github.com/your-username/cycle-allocation) | Static macro-driven allocation framework |
| **Macro Dashboard** | Live regime detection with real FRED data |

---

*Federico Bonessi — MSc Finance, IÉSEG School of Management*
*[LinkedIn](https://www.linkedin.com/in/federico-bonessi/) | [The Meridian Playbook](https://themeridianplaybook.com)*
