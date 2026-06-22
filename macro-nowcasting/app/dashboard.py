"""
dashboard.py
------------
Streamlit dashboard for the macroeconomic nowcasting project.
Run: streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Macro Nowcasting",
    page_icon="📈",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
target = st.sidebar.selectbox("Target variable", ["gdp", "cpi", "unemployment"])
horizon = st.sidebar.selectbox("Forecast horizon (months)", [1, 2, 4, 6, 12])
model_choice = st.sidebar.selectbox("Model", ["LightGBM", "ARIMA", "Naive"])
run_btn = st.sidebar.button("▶ Run forecast", type="primary")

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("📈 Multi-horizon Macroeconomic Nowcasting")
st.caption("Forecasting GDP, CPI, and Unemployment using FRED leading indicators")

col1, col2, col3 = st.columns(3)
col1.metric("Target", target.upper())
col2.metric("Horizon", f"{horizon} month(s) ahead")
col3.metric("Model", model_choice)

st.divider()

if run_btn:
    with st.spinner("Loading data and running forecast..."):
        try:
            from src.data_loader import get_fred_client, fetch_series, load_config
            from src.features import build_feature_matrix
            from src.backtest import walk_forward_backtest
            from src.evaluate import compile_metrics

            cfg = load_config()
            fred = get_fred_client()
            all_series = {**cfg["fred"]["target_series"], **cfg["fred"]["leading_indicators"]}
            df_raw = fetch_series(fred, all_series, cfg["fred"]["start_date"], None)

            X, y = build_feature_matrix(df_raw, target_col=target, forecast_horizon=horizon)

            if model_choice == "LightGBM":
                from src.models.ml_models import LightGBMForecaster
                model = LightGBMForecaster()
            elif model_choice == "ARIMA":
                from src.models.baseline import ARIMAWrapper
                model = ARIMAWrapper()
            else:
                from src.models.baseline import NaiveLastValue
                model = NaiveLastValue()

            result = walk_forward_backtest(X, y, model, model_name=model_choice, horizon=horizon)

            # Metrics
            st.subheader("📊 Performance Metrics")
            m = result.summary()
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("RMSE", f"{m['rmse']:.4f}")
            mc2.metric("MAE", f"{m['mae']:.4f}")
            mc3.metric("MAPE", f"{m['mape_%']:.2f}%")

            # Prediction chart
            st.subheader("🔮 Predicted vs Actual")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(result.actuals, label="Actual", color="steelblue", linewidth=1.5)
            ax.plot(result.predictions, label=f"Predicted ({model_choice})",
                    color="tomato", linewidth=1.5, linestyle="--")
            ax.set_title(f"{target.upper()} | h={horizon} | {model_choice}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # SHAP (LightGBM only)
            if model_choice == "LightGBM":
                st.subheader("🔍 Feature Importance (SHAP)")
                try:
                    import shap
                    # Fit on full data for SHAP display
                    model.fit(X, y)
                    explainer = shap.TreeExplainer(model._model)
                    shap_values = explainer.shap_values(X)
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=15)
                    st.pyplot(fig2)
                except Exception as e:
                    st.warning(f"SHAP plot unavailable: {e}")

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure FRED_API_KEY is set in your .env file and data has been loaded.")

else:
    st.info("👈 Configure settings in the sidebar and click **Run forecast** to start.")

    # Show sample metrics if available
    metrics_path = ROOT / "outputs" / "metrics.csv"
    if metrics_path.exists():
        st.subheader("📋 Previous Backtest Results")
        df_metrics = pd.read_csv(metrics_path)
        st.dataframe(df_metrics, use_container_width=True)
