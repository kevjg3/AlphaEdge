"""Transformer-based forecaster with temporal self-attention."""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from alphaedge.forecasting.base import BaseForecaster, ForecastPoint, ForecastResult
from alphaedge.forecasting.dl_common import (
    check_torch,
    build_sequences,
    mc_dropout_predict,
    prepare_features,
    train_model,
    TORCH_AVAILABLE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyTorch model components (only defined when torch is available)
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding."""

        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model % 2 == 0:
                pe[:, 1::2] = torch.cos(position * div_term)
            else:
                pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)

    class TemporalTransformer(nn.Module):
        """Small transformer encoder for time-series forecasting.

        Architecture:
            Linear(n_features -> d_model=32)
            PositionalEncoding
            TransformerEncoder x 2 layers (nhead=4, ff=64, causal mask)
            AdaptiveAvgPool1d -> Dropout -> Linear(32 -> 1)
        """

        def __init__(
            self,
            n_features: int,
            d_model: int = 32,
            nhead: int = 4,
            num_layers: int = 2,
            dim_ff: int = 64,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=256)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, n_features)
            x = self.input_proj(x)
            x = self.pos_encoder(x)

            # Causal mask — position i can only attend to positions <= i
            seq_len = x.size(1)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            x = self.encoder(x, mask=mask)

            # Pool over time: (batch, d_model, seq) -> (batch, d_model)
            x = x.transpose(1, 2)
            x = self.pool(x).squeeze(-1)
            x = self.dropout(x)
            return self.head(x)


class TransformerForecaster(BaseForecaster):
    """Temporal Transformer forecaster — extends BaseForecaster."""

    name = "transformer"

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
            logger.debug("Transformer: torch unavailable, skipping fit")
            return

        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if isinstance(train_data, pd.Series):
            train_data = pd.DataFrame({"Close": train_data})

        features, target = prepare_features(train_data)
        X_seq, y_seq = build_sequences(features, target, window=self.window)

        if len(X_seq) < 50:
            logger.warning("Transformer: insufficient sequences (%d < 50)", len(X_seq))
            return

        # Per-feature normalisation
        self._feature_mean = X_seq.mean(axis=(0, 1))
        self._feature_std = X_seq.std(axis=(0, 1)) + 1e-8
        X_norm = (X_seq - self._feature_mean) / self._feature_std

        X_t = torch.FloatTensor(X_norm)
        y_t = torch.FloatTensor(y_seq)

        self._n_features = X_seq.shape[2]
        self._model = TemporalTransformer(n_features=self._n_features)

        train_model(self._model, X_t, y_t, epochs=100, lr=1e-3, patience=10)

        # Store last window (un-normalised) for iterative prediction
        self._last_window = X_seq[-1:]
        self._prices = train_data["Close"] if "Close" in train_data.columns else train_data.iloc[:, 0]
        logger.debug("Transformer fitted: %d features, %d sequences", self._n_features, len(X_seq))

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
                warnings=["Transformer model not fitted"],
            )

        import torch

        try:
            price = current_price
            points: list[ForecastPoint] = []
            last_date = self._prices.index[-1] if self._prices is not None else None
            window_data = self._last_window.copy()  # (1, window, features)

            for i in range(horizon_days):
                # Normalise
                X_norm = (window_data - self._feature_mean) / self._feature_std
                X_t = torch.FloatTensor(X_norm)

                mean_ret, std_ret = mc_dropout_predict(self._model, X_t, n_samples=30)

                pred_price = price * np.exp(mean_ret)
                lower_price = price * np.exp(mean_ret - 1.645 * std_ret)
                upper_price = price * np.exp(mean_ret + 1.645 * std_ret)

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

                # Shift window: drop oldest row, append approximation for new row
                new_row = window_data[0, -1:, :].copy()
                if window_data.shape[2] > 0:
                    new_row[0, 0] = mean_ret  # first feature ≈ ret_1d
                window_data = np.concatenate(
                    [window_data[:, 1:, :], new_row[:, np.newaxis, :].reshape(1, 1, -1)],
                    axis=1,
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
            logger.warning("Transformer predict failed: %s", e)
            return ForecastResult(
                model_name=self.name,
                horizon=f"{horizon_days}d",
                current_price=current_price,
                forecasts=[],
                predicted_return=0.0,
                direction="flat",
                confidence=0.0,
                warnings=[f"Transformer prediction failed: {e}"],
            )
