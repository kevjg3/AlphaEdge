"""1D Temporal Convolutional Network forecaster with dilated causal convolutions."""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from alphaedge.forecasting.base import BaseForecaster, ForecastPoint, ForecastResult
from alphaedge.forecasting.dl_common import (
    check_torch,
    build_sequences,
    prepare_features,
    TORCH_AVAILABLE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyTorch model components
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalConv1d(nn.Module):
        """1D convolution with left-only (causal) padding."""

        def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.pad(x, (self.padding, 0))
            return self.conv(x)

    class ResidualBlock(nn.Module):
        """Dilated causal conv residual block."""

        def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
            super().__init__()
            self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
            self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            out = self.relu(self.conv1(x))
            out = self.dropout(out)
            out = self.conv2(out)
            return self.relu(out + residual)

    class TemporalCNN(nn.Module):
        """WaveNet-style dilated causal 1D CNN.

        Architecture:
            CausalConv1d(n_features -> 32)
            ResidualBlock x4 (dilations 1, 2, 4, 8)
            AdaptiveAvgPool1d -> Dropout -> Linear(32 -> 3)
            Outputs: [median, q05, q95] predicted returns
        """

        def __init__(
            self,
            n_features: int,
            channels: int = 32,
            kernel_size: int = 3,
            n_blocks: int = 4,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.input_conv = CausalConv1d(n_features, channels, kernel_size)
            self.relu = nn.ReLU()
            self.blocks = nn.ModuleList(
                [
                    ResidualBlock(channels, kernel_size, dilation=2**i, dropout=dropout)
                    for i in range(n_blocks)
                ]
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(0.3)
            self.head = nn.Linear(channels, 3)  # median, q05, q95

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, n_features, seq_len)
            x = self.relu(self.input_conv(x))
            for block in self.blocks:
                x = block(x)
            x = self.pool(x).squeeze(-1)  # (batch, channels)
            x = self.dropout(x)
            return self.head(x)  # (batch, 3)

    def quantile_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Combined MSE (median) + pinball loss for quantile predictions."""
        median_pred = pred[:, 0]
        q05_pred = pred[:, 1]
        q95_pred = pred[:, 2]

        mse = F.mse_loss(median_pred, target)

        err_05 = target - q05_pred
        loss_05 = torch.where(err_05 >= 0, 0.05 * err_05, (0.05 - 1) * err_05).mean()

        err_95 = target - q95_pred
        loss_95 = torch.where(err_95 >= 0, 0.95 * err_95, (0.95 - 1) * err_95).mean()

        return mse + loss_05 + loss_95


class CNNForecaster(BaseForecaster):
    """Dilated causal CNN forecaster — extends BaseForecaster."""

    name = "cnn"

    def __init__(self, seed: int = 42, window: int = 60):
        self.seed = seed
        self.window = window
        self._model = None
        self._last_window: np.ndarray | None = None
        self._n_features: int | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._prices: pd.Series | None = None

    # ------------------------------------------------------------------
    def fit(self, train_data: pd.DataFrame) -> None:
        try:
            check_torch()
        except ImportError:
            logger.debug("CNN: torch unavailable, skipping fit")
            return

        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if isinstance(train_data, pd.Series):
            train_data = pd.DataFrame({"Close": train_data})

        features, target = prepare_features(train_data)
        X_seq, y_seq = build_sequences(features, target, window=self.window)

        if len(X_seq) < 50:
            logger.warning("CNN: insufficient sequences (%d < 50)", len(X_seq))
            return

        # Normalise per-feature
        self._feature_mean = X_seq.mean(axis=(0, 1))
        self._feature_std = X_seq.std(axis=(0, 1)) + 1e-8
        X_norm = (X_seq - self._feature_mean) / self._feature_std

        # Transpose for Conv1d: (N, window, features) -> (N, features, window)
        X_t = torch.FloatTensor(X_norm).transpose(1, 2)
        y_t = torch.FloatTensor(y_seq)

        self._n_features = X_seq.shape[2]
        self._model = TemporalCNN(n_features=self._n_features)

        # Custom training loop with quantile loss
        self._train_quantile(self._model, X_t, y_t)

        # Store last window (un-normalised, seq-first format) for prediction
        self._last_window = X_seq[-1:]  # (1, window, features)
        self._prices = train_data["Close"] if "Close" in train_data.columns else train_data.iloc[:, 0]
        logger.debug("CNN fitted: %d features, %d sequences", self._n_features, len(X_seq))

    def _train_quantile(
        self,
        model,
        X_train,  # (N, features, window)
        y_train,  # (N,)
        epochs: int = 100,
        lr: float = 1e-3,
        patience: int = 10,
    ) -> None:
        import torch

        n = len(X_train)
        val_size = max(int(n * 0.15), 1)
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        model.train()
        for _epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_tr)  # (N, 3)
            loss = quantile_loss(pred, y_tr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = quantile_loss(val_pred, y_val).item()
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

    # ------------------------------------------------------------------
    def predict(self, horizon_days: int, current_price: float) -> ForecastResult:
        warns: list[str] = []

        if self._model is None or self._last_window is None:
            return ForecastResult(
                model_name=self.name,
                horizon=f"{horizon_days}d",
                current_price=current_price,
                forecasts=[],
                predicted_return=0.0,
                direction="flat",
                confidence=0.0,
                warnings=["CNN model not fitted"],
            )

        import torch

        try:
            price = current_price
            points: list[ForecastPoint] = []
            last_date = self._prices.index[-1] if self._prices is not None else None
            window_data = self._last_window.copy()  # (1, window, features)

            self._model.eval()
            for i in range(horizon_days):
                X_norm = (window_data - self._feature_mean) / self._feature_std
                # Transpose to channels-first for Conv1d
                X_t = torch.FloatTensor(X_norm).transpose(1, 2)

                with torch.no_grad():
                    out = self._model(X_t)  # (1, 3)

                median_ret = float(out[0, 0])
                q05_ret = float(out[0, 1])
                q95_ret = float(out[0, 2])

                pred_price = price * np.exp(median_ret)
                lower_price = price * np.exp(q05_ret)
                upper_price = price * np.exp(q95_ret)

                # Ensure lower <= predicted <= upper
                lower_price = min(lower_price, pred_price)
                upper_price = max(upper_price, pred_price)

                if last_date is not None and hasattr(last_date, "date"):
                    date_str = (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
                else:
                    date_str = f"T+{i + 1}"

                ci_width = upper_price - lower_price
                conf = max(0.0, 1.0 - ci_width / pred_price) if pred_price > 0 else 0.5

                points.append(
                    ForecastPoint(
                        date=date_str,
                        predicted=round(float(pred_price), 4),
                        lower_bound=round(float(max(lower_price, 0.01)), 4),
                        upper_bound=round(float(upper_price), 4),
                        confidence=min(round(float(conf), 4), 1.0),
                    )
                )

                # Shift window
                new_row = window_data[0, -1:, :].copy()
                if window_data.shape[2] > 0:
                    new_row[0, 0] = median_ret
                window_data = np.concatenate(
                    [window_data[:, 1:, :], new_row.reshape(1, 1, -1)], axis=1
                )
                price = pred_price

            final_price = points[-1].predicted if points else current_price
            predicted_return = (final_price / current_price - 1) * 100
            avg_conf = float(np.mean([p.confidence for p in points]))

            return ForecastResult(
                model_name=self.name,
                horizon=f"{horizon_days}d",
                current_price=current_price,
                forecasts=points,
                predicted_return=predicted_return,
                direction=self._direction(predicted_return / 100),
                confidence=avg_conf,
                metrics={},
                warnings=warns,
            )

        except Exception as e:
            logger.warning("CNN predict failed: %s", e)
            return ForecastResult(
                model_name=self.name,
                horizon=f"{horizon_days}d",
                current_price=current_price,
                forecasts=[],
                predicted_return=0.0,
                direction="flat",
                confidence=0.0,
                warnings=[f"CNN prediction failed: {e}"],
            )
