"""
backtest.py
-----------
Walk-forward (expanding window) validation for time series models.
This is NOT sklearn's cross_val_score — shuffling time series data leaks future info.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class BacktestResult:
    horizon: int
    model_name: str
    predictions: pd.Series
    actuals: pd.Series
    train_sizes: list[int] = field(default_factory=list)

    @property
    def rmse(self) -> float:
        err = self.actuals - self.predictions
        return float(np.sqrt((err ** 2).mean()))

    @property
    def mae(self) -> float:
        return float((self.actuals - self.predictions).abs().mean())

    @property
    def mape(self) -> float:
        mask = self.actuals != 0
        return float(((self.actuals[mask] - self.predictions[mask]).abs() / self.actuals[mask].abs()).mean() * 100)

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "horizon": self.horizon,
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "mape_%": round(self.mape, 2),
            "n_predictions": len(self.predictions),
        }


def walk_forward_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    model_name: str,
    horizon: int,
    min_train_size: int = 60,   # at least 5 years of monthly data
    step: int = 1,
) -> BacktestResult:
    """
    Expanding-window walk-forward validation.

    At each step t:
      - Train on X[:t], y[:t]
      - Predict X[t]
      - Compare to y[t]
    t starts at min_train_size and advances by `step` each iteration.

    Parameters
    ----------
    model : any sklearn-compatible model (fit/predict interface)
    min_train_size : minimum number of training samples before first prediction
    step : number of periods to advance between predictions
    """
    assert X.index.equals(y.index), "X and y must share the same DatetimeIndex"
    assert len(X) == len(y)

    n = len(X)
    preds = {}
    actuals = {}
    train_sizes = []

    for t in range(min_train_size, n, step):
        X_train, y_train = X.iloc[:t], y.iloc[:t]
        X_test = X.iloc[[t]]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        idx = X.index[t]
        preds[idx] = pred
        actuals[idx] = y.iloc[t]
        train_sizes.append(t)

    return BacktestResult(
        horizon=horizon,
        model_name=model_name,
        predictions=pd.Series(preds, name="prediction"),
        actuals=pd.Series(actuals, name="actual"),
        train_sizes=train_sizes,
    )
