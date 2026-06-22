"""
features.py
-----------
Mixed-frequency alignment, lag features, rolling statistics.
All functions take/return DataFrames with DatetimeIndex.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ── 1. Frequency alignment ─────────────────────────────────────────────────────

def resample_to_monthly(df: pd.DataFrame, freq: str = "MS") -> pd.DataFrame:
    """
    Resample a mixed-frequency raw DataFrame to monthly (month-start).
    - Daily/weekly series  → take last value in period (most recent observation)
    - Monthly series       → forward-fill to align to month-start index
    - Quarterly series     → forward-fill (GDP will appear once per quarter)
    """
    # Resample each column individually using last() to preserve latest reading
    resampled = df.resample(freq).last()
    # Forward-fill remaining NaNs (quarterly GDP fills across 3 months)
    resampled = resampled.ffill()
    return resampled


def align_to_quarterly(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Downsample monthly DataFrame to quarterly (quarter-end) for GDP-target models.
    Uses last value in quarter (most recently available monthly data).
    """
    return df_monthly.resample("QE").last()


# ── 2. Stationarity transforms ────────────────────────────────────────────────

def make_stationary(df: pd.DataFrame, method: str = "pct_change") -> pd.DataFrame:
    """
    Transform series to be (approximately) stationary.
    method: 'pct_change' | 'diff' | 'log_diff'
    """
    if method == "pct_change":
        return df.pct_change().replace([np.inf, -np.inf], np.nan)
    elif method == "diff":
        return df.diff()
    elif method == "log_diff":
        return np.log(df.clip(lower=1e-9)).diff()
    else:
        raise ValueError(f"Unknown method: {method}")


# ── 3. Lag features ───────────────────────────────────────────────────────────

def make_lag_features(df: pd.DataFrame, lags: list[int], cols: list[str] | None = None) -> pd.DataFrame:
    """
    Create lagged versions of selected columns.
    cols=None → lag all columns.
    Returns original df + new lag columns.
    """
    cols = cols or df.columns.tolist()
    lag_frames = [df]
    for lag in lags:
        lagged = df[cols].shift(lag)
        lagged.columns = [f"{c}_lag{lag}" for c in cols]
        lag_frames.append(lagged)
    return pd.concat(lag_frames, axis=1)


# ── 4. Rolling statistics ─────────────────────────────────────────────────────

def make_rolling_features(
    df: pd.DataFrame,
    windows: list[int],
    cols: list[str] | None = None,
    stats: list[str] = ("mean", "std"),
) -> pd.DataFrame:
    """
    Create rolling mean and std for selected columns.
    """
    cols = cols or df.columns.tolist()
    roll_frames = [df]
    for window in windows:
        for stat in stats:
            rolled = df[cols].rolling(window=window, min_periods=window // 2)
            if stat == "mean":
                result = rolled.mean()
            elif stat == "std":
                result = rolled.std()
            else:
                raise ValueError(f"Unsupported stat: {stat}")
            result.columns = [f"{c}_roll{window}_{stat}" for c in cols]
            roll_frames.append(result)
    return pd.concat(roll_frames, axis=1)


# ── 5. Full pipeline ──────────────────────────────────────────────────────────

def build_feature_matrix(
    df_raw: pd.DataFrame,
    target_col: str,
    forecast_horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full feature engineering pipeline:
      1. Resample to monthly
      2. Stationarity transform
      3. Lag features
      4. Rolling features
      5. Shift target by forecast_horizon (so model predicts h steps ahead)
      6. Drop NaN rows
    Returns (X, y).
    """
    cfg = load_config()
    feat_cfg = cfg["features"]

    # Step 1: resample
    df = resample_to_monthly(df_raw, freq=feat_cfg["resample_freq"])

    # Step 2: stationarity (apply pct_change to level series)
    df = make_stationary(df, method="pct_change")

    # Step 3 & 4: lag + rolling on all indicator columns (not target)
    indicator_cols = [c for c in df.columns if c != target_col]
    df = make_lag_features(df, lags=feat_cfg["lag_periods"], cols=indicator_cols)
    df = make_rolling_features(df, windows=feat_cfg["rolling_windows"], cols=indicator_cols)

    # Step 5: target is h-period-ahead value of target_col
    y = df[target_col].shift(-forecast_horizon)
    X = df.drop(columns=[target_col])

    # Step 6: align and drop NaN
    combined = pd.concat([X, y.rename("target")], axis=1).dropna()
    X_clean = combined.drop(columns=["target"])
    y_clean = combined["target"]

    return X_clean, y_clean
