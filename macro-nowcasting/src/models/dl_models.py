"""
dl_models.py
------------
LSTM forecaster (PyTorch) for multi-horizon macroeconomic forecasting.
Temporal Fusion Transformer (TFT) via pytorch-forecasting is also scaffolded
but requires more data prep — implement after LightGBM is validated.

Requires: pip install torch
"""

import numpy as np
import pandas as pd


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class LSTMForecaster:
    """
    Vanilla LSTM for sequence-to-one forecasting.
    Wraps PyTorch; exposes fit/predict like sklearn for backtest compatibility.

    Usage:
        model = LSTMForecaster(input_size=X.shape[1], seq_len=12)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        seq_len: int = 12,          # lookback window in months
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self._net = None
        self._scaler_X = None
        self._scaler_y = None

    def _build_sequences(self, X: np.ndarray, y: np.ndarray):
        Xs, ys = [], []
        for i in range(self.seq_len, len(X)):
            Xs.append(X[i - self.seq_len:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch not installed. Run: pip install torch")

        from sklearn.preprocessing import StandardScaler

        # Scale features and target
        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_sc = self._scaler_X.fit_transform(X.values)
        y_sc = self._scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

        Xs, ys = self._build_sequences(X_sc, y_sc)
        if len(Xs) == 0:
            raise ValueError(f"Not enough data for seq_len={self.seq_len}")

        tensor_X = torch.tensor(Xs, dtype=torch.float32).to(self.device)
        tensor_y = torch.tensor(ys, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(tensor_X, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self._net = _build_lstm_net(
            input_size=X.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self._net.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self._net(xb).squeeze()
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # For walk-forward: X here is a single row; we need the last seq_len rows
        # Note: caller should pass X_context (last seq_len rows before test point)
        # This is handled by LSTMWalkForward wrapper below for proper backtesting
        raise NotImplementedError(
            "Use LSTMWalkForwardWrapper for backtesting. "
            "Direct predict() requires full sequence context."
        )


class _LSTMNet:
    """Defined lazily — instantiated only inside LSTMForecaster.fit()"""
    pass


def _build_lstm_net(input_size, hidden_size, num_layers, dropout):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    return Net()


# ── TFT placeholder ───────────────────────────────────────────────────────────

class TemporalFusionTransformerForecaster:
    """
    Placeholder for TFT via pytorch-forecasting.
    Implement after LightGBM baseline is validated.

    TFT is more complex to set up (requires TimeSeriesDataSet format)
    but handles multi-horizon natively and gives interpretable attention weights.

    Reference: https://pytorch-forecasting.readthedocs.io/
    """

    def __init__(self):
        raise NotImplementedError(
            "TFT not yet implemented. "
            "Start with LightGBMForecaster, then come back here once "
            "the pipeline is validated end-to-end."
        )
