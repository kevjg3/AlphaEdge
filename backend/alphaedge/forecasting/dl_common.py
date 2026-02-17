"""Shared utilities for deep learning forecasters."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    logger.debug("PyTorch not available; DL forecasters will be disabled")


def check_torch() -> None:
    """Raise ImportError if torch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required for deep learning forecasters")


def build_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    window: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert tabular features + target into sliding-window sequences.

    Returns
    -------
    X : np.ndarray of shape (n_samples, window, n_features)
    y : np.ndarray of shape (n_samples,)
    """
    common_idx = features.index.intersection(target.dropna().index)
    feat_arr = features.loc[common_idx].values.astype(np.float32)
    tgt_arr = target.loc[common_idx].values.astype(np.float32)

    X_list, y_list = [], []
    for i in range(window, len(feat_arr)):
        X_list.append(feat_arr[i - window : i])
        y_list.append(tgt_arr[i])

    if not X_list:
        return np.empty((0, window, feat_arr.shape[1]), dtype=np.float32), np.empty(0, dtype=np.float32)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def prepare_features(
    train_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build features and 1-day forward return target from train_data.

    Uses FeatureStore when full OHLCV columns are present, otherwise builds a
    minimal feature set from Close prices only.
    """
    has_ohlcv = all(
        c in train_data.columns for c in ("Open", "High", "Low", "Close", "Volume")
    )

    if has_ohlcv:
        from alphaedge.technicals.feature_store import FeatureStore

        features = FeatureStore.build_features(train_data)
        target = FeatureStore.build_target(train_data, horizon=1)
    else:
        c = train_data["Close"]
        features = pd.DataFrame(index=train_data.index)

        # Lagged returns
        for lag in [1, 2, 3, 5, 10, 21]:
            features[f"ret_{lag}d"] = np.log(c / c.shift(lag))

        # Rolling volatility
        log_ret = features["ret_1d"]
        for w in [5, 21]:
            features[f"vol_{w}d"] = log_ret.rolling(w).std()

        # RSI
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        features["rsi_14"] = 100 - (100 / (1 + rs))

        features = features.dropna()
        target = np.log(c.shift(-1) / c)
        target.name = "target_1d"

    return features, target


def mc_dropout_predict(
    model,  # nn.Module
    X_tensor,  # torch.Tensor
    n_samples: int = 30,
) -> tuple[float, float]:
    """Run MC Dropout inference: n forward passes with dropout enabled.

    Returns (mean_pred, std_pred).
    """
    import torch

    model.train()  # enables dropout
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(X_tensor)
            preds.append(out.item())
    model.eval()
    return float(np.mean(preds)), float(max(np.std(preds), 1e-8))


def train_model(
    model,  # nn.Module
    X_train,  # torch.Tensor  (N, window, features)
    y_train,  # torch.Tensor  (N,)
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    val_fraction: float = 0.15,
) -> float:
    """Generic single-batch training loop with early stopping on validation.

    Validation set is the temporal tail (no shuffling).
    Returns best validation loss.
    """
    import torch
    import torch.nn as nn

    n = len(X_train)
    val_size = max(int(n * val_fraction), 1)

    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    model.train()
    for _epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tr).squeeze(-1)
        loss = criterion(pred, y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).squeeze(-1)
            val_loss = criterion(val_pred, y_val).item()
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
    return best_val_loss
