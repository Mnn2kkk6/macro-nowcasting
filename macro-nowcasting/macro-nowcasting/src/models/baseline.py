"""
baseline.py
-----------
Sklearn-compatible wrappers for ARIMA and Seasonal ARIMA baselines.
Uses statsmodels under the hood; exposes fit/predict interface
so they plug into walk_forward_backtest() without modification.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMAWrapper:
    """
    Sklearn-compatible ARIMA wrapper.
    order: (p, d, q) tuple — default (2, 0, 1) works well on stationary series.
    """

    def __init__(self, order: tuple = (2, 0, 1)):
        self.order = order
        self._fitted = None
        self._y_train = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # ARIMA ignores X (univariate); kept for API compatibility
        self._y_train = y.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(y, order=self.order)
            self._fitted = model.fit()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Predict 1 step ahead from end of training data
        forecast = self._fitted.forecast(steps=1)
        return np.array([forecast.iloc[0]])

    def __repr__(self):
        return f"ARIMAWrapper(order={self.order})"


class SARIMAWrapper:
    """
    Sklearn-compatible SARIMA wrapper.
    order: (p, d, q), seasonal_order: (P, D, Q, s)
    Default seasonal_order assumes monthly data (s=12).
    """

    def __init__(self, order: tuple = (1, 0, 1), seasonal_order: tuple = (1, 0, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self._fitted = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            self._fitted = model.fit(disp=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        forecast = self._fitted.forecast(steps=1)
        return np.array([forecast.iloc[0]])

    def __repr__(self):
        return f"SARIMAWrapper(order={self.order}, seasonal={self.seasonal_order})"


class NaiveLastValue:
    """
    Simplest possible baseline: predict last observed value.
    Useful as a sanity check — any real model should beat this.
    """

    def __init__(self):
        self._last_value = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._last_value = y.iloc[-1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._last_value] * len(X))

    def __repr__(self):
        return "NaiveLastValue()"
