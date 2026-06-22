"""
ml_models.py
------------
LightGBM and XGBoost wrappers for multi-horizon forecasting.
These accept the full feature matrix from features.py (lags, rolling stats, etc.)
and are the main ML workhorses before moving to DL models.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class LightGBMForecaster(BaseEstimator, RegressorMixin):
    """
    LightGBM regression wrapper with sensible macro-forecasting defaults.
    Inherits from sklearn base → works directly with walk_forward_backtest().
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 10,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        import lightgbm as lgb

        self._model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,  # Tận dụng toàn bộ lõi CPU
            verbose=-1,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Expose for SHAP compatibility."""
        return self._model.feature_importances_

    @property
    def booster_(self):
        """Expose raw booster for shap.TreeExplainer."""
        return self._model.booster_

    def __repr__(self):
        return f"LightGBMForecaster(n_estimators={self.n_estimators}, lr={self.learning_rate})"


class XGBoostForecaster(BaseEstimator, RegressorMixin):
    """XGBoost alternative — useful for comparison."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        tree_method: str = "hist",  # Bật Histogram-based algorithm để chạy nhanh hơn trên CPU
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.tree_method = tree_method
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        from xgboost import XGBRegressor

        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            tree_method=self.tree_method,
            n_jobs=-1,  # Tận dụng toàn bộ lõi CPU
            verbosity=0,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)