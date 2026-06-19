"""
evaluate.py
-----------
Metrics comparison across models and horizons.
SHAP explainability wrapper for tree-based models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.use("Agg")  # non-interactive backend for saving figures

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Metrics table ──────────────────────────────────────────────────────────────

def compile_metrics(results: list) -> pd.DataFrame:
    """
    Takes a list of BacktestResult objects, returns a tidy DataFrame
    suitable for saving to outputs/metrics.csv or displaying in Streamlit.
    """
    rows = [r.summary() for r in results]
    df = pd.DataFrame(rows).sort_values(["horizon", "rmse"])
    return df


def save_metrics(df: pd.DataFrame, path: Path | None = None):
    path = path or ROOT / "outputs" / "metrics.csv"
    df.to_csv(path, index=False)
    print(f"Metrics saved → {path}")


# ── Prediction plot ────────────────────────────────────────────────────────────

def plot_predictions(result, title: str = "", save: bool = True):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(result.actuals, label="Actual", color="steelblue", linewidth=1.5)
    ax.plot(result.predictions, label=f"Predicted ({result.model_name})",
            color="tomato", linewidth=1.5, linestyle="--")
    ax.set_title(title or f"{result.model_name} | h={result.horizon} | RMSE={result.rmse:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        fname = FIG_DIR / f"pred_{result.model_name}_h{result.horizon}.png"
        fig.savefig(fname, dpi=150)
        print(f"Figure saved → {fname}")
    return fig


# ── SHAP wrapper ───────────────────────────────────────────────────────────────

def plot_shap_summary(model, X: pd.DataFrame, model_name: str = "", save: bool = True):
    """
    Generate SHAP beeswarm + bar summary plots for a tree-based model.
    Requires shap package.
    """
    try:
        import shap
    except ImportError:
        print("shap not installed. Run: pip install shap")
        return

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Bar plot (mean |SHAP|)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    fig1.tight_layout()
    if save:
        fname = FIG_DIR / f"shap_bar_{model_name}.png"
        fig1.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"SHAP bar plot saved → {fname}")

    # Beeswarm plot
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    shap.summary_plot(shap_values, X, show=False)
    fig2.tight_layout()
    if save:
        fname = FIG_DIR / f"shap_beeswarm_{model_name}.png"
        fig2.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"SHAP beeswarm saved → {fname}")

    plt.close("all")
