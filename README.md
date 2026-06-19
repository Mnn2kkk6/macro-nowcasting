# 📈 Multi-horizon Macroeconomic Nowcasting & Forecasting

> **Nowcasting problem**: GDP is published quarterly with a ~1–2 month lag. This project forecasts GDP growth, CPI inflation, and unemployment at multiple horizons (1, 2, 4 months ahead) using high-frequency leading indicators — exactly the approach used by the Fed's GDPNow and NY Fed Nowcast models.

---

## 🎯 Motivation

Official macroeconomic statistics are published with significant delays. The **nowcasting problem** asks: *can we estimate the current state of the economy before the official data is released?*

This project answers that using:
- **Weekly / daily indicators** (jobless claims, yield curve spread) published well before official GDP
- **Machine learning** to combine many indicators into a single forecast
- **Rigorous backtesting** to simulate real-world forecasting conditions

---

## 🏗️ Architecture

```
Raw FRED data (10 series, mixed frequency)
         │
         ▼
  Resample → Monthly (month-start)
         │
         ▼
  Stationarity transform (% change)
         │
         ▼
  Lag features (t-1 to t-12) + Rolling stats (3/6/12m)
         │
         ▼
  ┌──────┴──────┐
  │             │
ARIMA        LightGBM
(baseline)    (ML model)
  │             │
  └──────┬──────┘
         ▼
  Walk-forward backtest (no data leakage)
         │
         ▼
  RMSE / MAE / MAPE by horizon + SHAP explanations
         │
         ▼
  Streamlit dashboard
```

---

## 📊 Results

| Model | Target | Horizon | RMSE | MAE |
|-------|--------|---------|------|-----|
| Naive | CPI | h=1 | — | — |
| ARIMA(2,0,1) | CPI | h=1 | — | — |
| **LightGBM** | **CPI** | **h=1** | **—** | **—** |

*Table auto-populated after running notebook 05.*

---

## 🔑 Key Technical Points

- **Walk-forward (expanding window) validation** — never shuffles time series data; each prediction is made only using information available at that point in time
- **Mixed-frequency handling** — daily/weekly series (yield curve, jobless claims) are resampled to monthly via last-value aggregation before feature engineering
- **Publication lag awareness** — leading indicators are selected precisely because they are published before the target variable (GDP)
- **SHAP explainability** — tells us which indicators are driving each forecast, connecting model output to economic intuition

---

## 📦 Data Sources

All data from [FRED (Federal Reserve Bank of St. Louis)](https://fred.stlouisfed.org/):

| Series | ID | Frequency |
|--------|----|-----------|
| Real GDP | GDPC1 | Quarterly |
| CPI | CPIAUCSL | Monthly |
| Unemployment | UNRATE | Monthly |
| Nonfarm Payrolls | PAYEMS | Monthly |
| Initial Jobless Claims | ICSA | **Weekly** |
| Industrial Production | INDPRO | Monthly |
| Retail Sales | RSAFS | Monthly |
| Housing Starts | HOUST | Monthly |
| Consumer Sentiment | UMCSENT | Monthly |
| Yield Curve (10Y-2Y) | T10Y2Y | **Daily** |
| Money Supply M2 | M2SL | Monthly |
| Producer Price Index | PPIACO | Monthly |

---

## 🚀 Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/macro-nowcasting
cd macro-nowcasting
pip install -r requirements.txt

# 2. Add your FRED API key (free at fred.stlouisfed.org)
cp .env.example .env
# Edit .env: FRED_API_KEY=your_key_here

# 3. Fetch data
python -m src.data_loader

# 4. Run notebooks in order (01 → 05) or launch dashboard
streamlit run app/dashboard.py
```

---

## 🗂️ Project Structure

```
macro-nowcasting/
├── src/               # Production code (importable, testable)
│   ├── data_loader.py # FRED API fetching + caching
│   ├── features.py    # Resampling, lag, rolling features
│   ├── backtest.py    # Walk-forward validation engine
│   ├── evaluate.py    # Metrics + SHAP plots
│   └── models/
│       ├── baseline.py   # ARIMA, SARIMA, Naive
│       ├── ml_models.py  # LightGBM, XGBoost
│       └── dl_models.py  # LSTM (PyTorch)
├── notebooks/         # Exploratory analysis (01–05)
├── app/               # Streamlit dashboard
├── data/              # Raw + processed data (gitignored)
├── outputs/           # Figures + metrics CSV
├── tests/             # Unit tests (pytest)
└── config.yaml        # All parameters in one place
```

**Note:** Business logic lives in `src/`, not notebooks. Notebooks are for exploration only — all reusable code is extracted into importable modules.

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📚 References

- [Fed GDPNow Model](https://www.atlantafed.org/cqer/research/gdpnow)
- [NY Fed Staff Nowcast](https://www.newyorkfed.org/research/policy/nowcast)
- Giannone, D., Reichlin, L., Small, D. (2008). *Nowcasting: The real-time informational content of macroeconomic data*. Journal of Monetary Economics.
- [MIDAS regression — mixed-frequency data](https://en.wikipedia.org/wiki/Mixed-data_sampling)
