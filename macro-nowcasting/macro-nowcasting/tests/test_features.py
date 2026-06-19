"""
test_features.py
----------------
Unit tests for feature engineering functions.
Run: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import pandas as pd
import numpy as np
from src.features import (
    resample_to_monthly,
    make_stationary,
    make_lag_features,
    make_rolling_features,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def daily_df():
    """Synthetic daily data mimicking FRED series."""
    idx = pd.date_range("2010-01-01", "2023-12-31", freq="D")
    np.random.seed(42)
    return pd.DataFrame({
        "yield_curve": np.random.randn(len(idx)).cumsum() * 0.01,
        "gdp": np.nan,  # quarterly — mostly NaN at daily freq
    }, index=idx)


@pytest.fixture
def monthly_df():
    """Synthetic monthly DataFrame."""
    idx = pd.date_range("2000-01-01", "2023-12-01", freq="MS")
    np.random.seed(0)
    return pd.DataFrame({
        "cpi": 100 + np.cumsum(np.random.randn(len(idx)) * 0.3),
        "unemployment": 5 + np.random.randn(len(idx)) * 0.5,
        "payrolls": 130_000 + np.cumsum(np.random.randn(len(idx)) * 50),
    }, index=idx)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestResample:
    def test_output_freq_is_monthly(self, daily_df):
        result = resample_to_monthly(daily_df)
        assert result.index.freq in (pd.tseries.offsets.MonthBegin(1), None)
        # All dates should be month-start
        assert all(d.day == 1 for d in result.index)

    def test_no_all_nan_columns(self, daily_df):
        result = resample_to_monthly(daily_df)
        # yield_curve should have values after resample
        assert result["yield_curve"].notna().sum() > 0

    def test_shape_reduced(self, daily_df):
        result = resample_to_monthly(daily_df)
        assert len(result) < len(daily_df)


class TestStationarity:
    def test_pct_change_removes_trend(self, monthly_df):
        result = make_stationary(monthly_df, method="pct_change")
        # First row is NaN (no previous value for pct_change)
        assert result.iloc[0].isna().all()
        # After first row, should have values
        assert result.iloc[1:].notna().all().all()

    def test_diff_shape_preserved(self, monthly_df):
        result = make_stationary(monthly_df, method="diff")
        assert result.shape == monthly_df.shape

    def test_invalid_method_raises(self, monthly_df):
        with pytest.raises(ValueError):
            make_stationary(monthly_df, method="nonexistent")


class TestLagFeatures:
    def test_lag_columns_created(self, monthly_df):
        result = make_lag_features(monthly_df, lags=[1, 3, 6])
        for lag in [1, 3, 6]:
            for col in monthly_df.columns:
                assert f"{col}_lag{lag}" in result.columns

    def test_total_column_count(self, monthly_df):
        lags = [1, 2, 3]
        result = make_lag_features(monthly_df, lags=lags)
        expected = len(monthly_df.columns) * (1 + len(lags))
        assert result.shape[1] == expected

    def test_lag1_shift(self, monthly_df):
        result = make_lag_features(monthly_df, lags=[1], cols=["cpi"])
        # lag1 of row i should equal original row i-1
        assert result["cpi_lag1"].iloc[5] == pytest.approx(monthly_df["cpi"].iloc[4])

    def test_index_preserved(self, monthly_df):
        result = make_lag_features(monthly_df, lags=[1])
        assert result.index.equals(monthly_df.index)


class TestRollingFeatures:
    def test_rolling_columns_created(self, monthly_df):
        result = make_rolling_features(monthly_df, windows=[3, 6])
        for w in [3, 6]:
            for col in monthly_df.columns:
                assert f"{col}_roll{w}_mean" in result.columns
                assert f"{col}_roll{w}_std" in result.columns

    def test_rolling_mean_correct(self, monthly_df):
        result = make_rolling_features(monthly_df, windows=[3], cols=["cpi"])
        # Row 10: rolling mean of rows 8,9,10
        expected = monthly_df["cpi"].iloc[8:11].mean()
        assert result["cpi_roll3_mean"].iloc[10] == pytest.approx(expected, rel=1e-5)

    def test_invalid_stat_raises(self, monthly_df):
        with pytest.raises(ValueError):
            make_rolling_features(monthly_df, windows=[3], stats=["median"])
