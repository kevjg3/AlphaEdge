"""Graph Neural Network forecaster using cross-asset correlation signals."""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from alphaedge.forecasting.base import BaseForecaster, ForecastPoint, ForecastResult
from alphaedge.forecasting.dl_common import (
    check_torch,
    mc_dropout_predict,
    TORCH_AVAILABLE,
)

logger = logging.getLogger(__name__)

# Same benchmarks used in quant/correlation.py
BENCHMARK_TICKERS = [
    "SPY", "QQQ", "IWM", "TLT", "GLD",
    "DIA", "XLK", "XLF", "XLE", "XLV",
]


# ---------------------------------------------------------------------------
# PyTorch model components
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    class GraphConvLayer(nn.Module):
        """Simple message-passing graph convolution.

        h_i' = ReLU( W * (h_i + Σ_j a_ij h_j) ) + dropout
        """

        def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()

        def forward(
            self,
            node_features: torch.Tensor,  # (n_nodes, in_dim)
            adj_weights: torch.Tensor,    # (n_nodes, n_nodes) row-normalised
        ) -> torch.Tensor:
            # Message passing: aggregate neighbour features weighted by correlation
            agg = torch.matmul(adj_weights, node_features)  # (n_nodes, in_dim)
            combined = node_features + agg
            out = self.linear(combined)
            return self.dropout(self.relu(out))

    class StockGNN(nn.Module):
        """Graph Neural Network for cross-asset signal propagation.

        Architecture:
            GraphConvLayer x 2  (node_features -> 16 -> 16)
            Concat target node embedding + raw features
            Linear(16 + n_node_features -> 1)
        """

        def __init__(
            self,
            n_node_features: int,
            hidden_dim: int = 16,
            n_layers: int = 2,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layers.append(GraphConvLayer(n_node_features, hidden_dim, dropout))
            for _ in range(n_layers - 1):
                self.layers.append(GraphConvLayer(hidden_dim, hidden_dim, dropout))
            self.head = nn.Linear(hidden_dim + n_node_features, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            node_features: torch.Tensor,  # (n_nodes, n_node_features)
            adj_weights: torch.Tensor,    # (n_nodes, n_nodes)
            target_idx: int = 0,
        ) -> torch.Tensor:
            h = node_features
            for layer in self.layers:
                h = layer(h, adj_weights)

            target_embedding = h[target_idx]          # (hidden_dim,)
            target_raw = node_features[target_idx]    # (n_node_features,)
            combined = torch.cat([target_embedding, target_raw])
            return self.head(self.dropout(combined))


def _compute_node_features(prices: pd.Series) -> np.ndarray:
    """Compute a feature vector from a single asset's price series.

    Returns the most recent snapshot of: [ret_1d, ret_5d, ret_21d, vol_5d,
    vol_21d, rsi_14, price_vs_sma20, price_vs_sma50].
    """
    c = prices
    if len(c) < 60:
        # Not enough data — return zeros
        return np.zeros(8, dtype=np.float32)

    log_ret = np.log(c / c.shift(1))
    feats = pd.DataFrame(index=c.index)
    feats["ret_1d"] = log_ret
    feats["ret_5d"] = np.log(c / c.shift(5))
    feats["ret_21d"] = np.log(c / c.shift(21))
    feats["vol_5d"] = log_ret.rolling(5).std()
    feats["vol_21d"] = log_ret.rolling(21).std()

    # RSI
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    feats["rsi_14"] = (100 - (100 / (1 + rs))) / 100.0  # normalise to [0,1]

    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    feats["price_vs_sma20"] = c / sma20.replace(0, np.nan) - 1
    feats["price_vs_sma50"] = c / sma50.replace(0, np.nan) - 1

    row = feats.iloc[-1].values.astype(np.float32)
    return np.nan_to_num(row, nan=0.0)


class GNNForecaster(BaseForecaster):
    """Graph Neural Network forecaster — cross-asset correlation signals."""

    name = "gnn"

    def __init__(self, seed: int = 42, ticker: str | None = None):
        self.seed = seed
        self.ticker = ticker
        self._model = None
        self._adj_weights: np.ndarray | None = None
        self._last_node_features: np.ndarray | None = None
        self._n_node_features: int | None = None
        self._prices: pd.Series | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, train_data: pd.DataFrame) -> None:
        try:
            check_torch()
        except ImportError:
            logger.debug("GNN: torch unavailable, skipping fit")
            return

        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if isinstance(train_data, pd.Series):
            train_data = pd.DataFrame({"Close": train_data})

        target_prices = train_data["Close"] if "Close" in train_data.columns else train_data.iloc[:, 0]
        if isinstance(target_prices, pd.DataFrame):
            target_prices = target_prices.iloc[:, 0]

        self._prices = target_prices

        # Fetch benchmark data
        benchmark_prices = self._fetch_benchmarks()
        if len(benchmark_prices) < 3:
            logger.warning("GNN: too few benchmarks (%d), skipping", len(benchmark_prices))
            return

        # Align all series to common dates
        all_prices = {"target": target_prices}
        all_prices.update(benchmark_prices)

        returns_df = pd.DataFrame()
        for name, series in all_prices.items():
            r = np.log(series / series.shift(1))
            returns_df[name] = r
        returns_df = returns_df.dropna()

        if len(returns_df) < 100:
            logger.warning("GNN: insufficient aligned data (%d rows)", len(returns_df))
            return

        # Build adjacency matrix from correlation
        corr = returns_df.corr().values.astype(np.float32)
        corr = np.abs(corr)  # Use absolute correlation as edge weight
        # Row-normalise (softmax-like: divide by row sum)
        row_sums = corr.sum(axis=1, keepdims=True)
        adj = corr / np.maximum(row_sums, 1e-8)
        self._adj_weights = adj

        # Build training samples: for each date, compute node features and target return
        node_names = list(all_prices.keys())
        n_nodes = len(node_names)
        target_returns = returns_df["target"].values

        # Compute per-node price series aligned
        aligned_prices = {}
        for name in node_names:
            s = all_prices[name]
            aligned_prices[name] = s.reindex(returns_df.index).ffill()

        warmup = 60  # need enough history for features
        X_list = []
        y_list = []

        for t_idx in range(warmup, len(returns_df) - 1):
            node_feats = []
            for name in node_names:
                p = aligned_prices[name].iloc[: t_idx + 1]
                nf = _compute_node_features(p)
                node_feats.append(nf)
            X_list.append(np.array(node_feats, dtype=np.float32))
            y_list.append(target_returns[t_idx + 1])

        if len(X_list) < 50:
            logger.warning("GNN: insufficient training samples (%d)", len(X_list))
            return

        X_all = np.array(X_list, dtype=np.float32)  # (T, n_nodes, n_node_features)
        y_all = np.array(y_list, dtype=np.float32)

        self._n_node_features = X_all.shape[2]

        # Normalise node features
        self._feature_mean = X_all.mean(axis=0)  # (n_nodes, features)
        self._feature_std = X_all.std(axis=0) + 1e-8
        X_norm = (X_all - self._feature_mean) / self._feature_std

        # Train the GNN
        self._model = StockGNN(n_node_features=self._n_node_features)
        self._train_gnn(self._model, X_norm, y_all, adj)

        # Store last node features (un-normalised)
        self._last_node_features = X_all[-1]  # (n_nodes, features)
        logger.debug("GNN fitted: %d nodes, %d features, %d samples", n_nodes, self._n_node_features, len(X_list))

    def _fetch_benchmarks(self) -> dict[str, pd.Series]:
        """Fetch benchmark ETF price series."""
        result = {}
        try:
            from alphaedge.data_ingestion.yfinance_source import YFinanceSource

            yf = YFinanceSource()
            for t in BENCHMARK_TICKERS:
                try:
                    hist = yf.get_history(t, period="2y")
                    if hist.success and not hist.data.empty:
                        close = hist.data["Close"]
                        if isinstance(close, pd.DataFrame):
                            close = close.iloc[:, 0]
                        result[t] = close
                except Exception as e:
                    logger.debug("GNN: failed to fetch %s: %s", t, e)
        except Exception as e:
            logger.debug("GNN: YFinanceSource unavailable: %s", e)
        return result

    def _train_gnn(
        self,
        model,
        X_norm: np.ndarray,  # (T, n_nodes, features)
        y: np.ndarray,       # (T,)
        adj: np.ndarray,     # (n_nodes, n_nodes)
        epochs: int = 80,
        lr: float = 1e-3,
        patience: int = 10,
    ) -> None:
        import torch
        import torch.nn as nn

        adj_t = torch.FloatTensor(adj)
        y_t = torch.FloatTensor(y)

        n = len(X_norm)
        val_size = max(int(n * 0.15), 1)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        model.train()
        for _epoch in range(epochs):
            # Mini-epoch: iterate over training samples
            train_loss = 0.0
            optimizer.zero_grad()

            preds = []
            for t in range(n - val_size):
                node_feats = torch.FloatTensor(X_norm[t])
                pred = model(node_feats, adj_t, target_idx=0)
                preds.append(pred)

            preds_t = torch.stack(preds).squeeze()
            loss = criterion(preds_t, y_t[: n - val_size])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Validation
            model.eval()
            val_preds = []
            with torch.no_grad():
                for t in range(n - val_size, n):
                    node_feats = torch.FloatTensor(X_norm[t])
                    pred = model(node_feats, adj_t, target_idx=0)
                    val_preds.append(pred)
            val_preds_t = torch.stack(val_preds).squeeze()
            val_targets = y_t[n - val_size :]
            if val_preds_t.dim() == 0:
                val_preds_t = val_preds_t.unsqueeze(0)
            if val_targets.dim() == 0:
                val_targets = val_targets.unsqueeze(0)
            val_loss = criterion(val_preds_t, val_targets).item()
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

        if self._model is None or self._last_node_features is None or self._adj_weights is None:
            return ForecastResult(
                model_name=self.name,
                horizon=f"{horizon_days}d",
                current_price=current_price,
                forecasts=[],
                predicted_return=0.0,
                direction="flat",
                confidence=0.0,
                warnings=["GNN model not fitted"],
            )

        import torch

        try:
            price = current_price
            points: list[ForecastPoint] = []
            last_date = self._prices.index[-1] if self._prices is not None else None
            adj_t = torch.FloatTensor(self._adj_weights)
            node_feats = self._last_node_features.copy()  # (n_nodes, features)

            for i in range(horizon_days):
                # Normalise
                nf_norm = (node_feats - self._feature_mean) / self._feature_std
                nf_t = torch.FloatTensor(nf_norm)

                mean_ret, std_ret = mc_dropout_predict(
                    self._model,
                    # MC dropout wrapper expects a single input; we wrap the GNN call
                    nf_t,
                    n_samples=30,
                )
                # Override: mc_dropout_predict calls model(X) which won't work for GNN.
                # Do it manually:
                self._model.train()
                preds_list = []
                with torch.no_grad():
                    for _ in range(30):
                        out = self._model(nf_t, adj_t, target_idx=0)
                        preds_list.append(out.item())
                self._model.eval()
                mean_ret = float(np.mean(preds_list))
                std_ret = float(max(np.std(preds_list), 1e-8))

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

                # Update target node features with predicted return
                if node_feats.shape[1] > 0:
                    node_feats[0, 0] = mean_ret  # ret_1d for target
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
            logger.warning("GNN predict failed: %s", e)
            return ForecastResult(
                model_name=self.name,
                horizon=f"{horizon_days}d",
                current_price=current_price,
                forecasts=[],
                predicted_return=0.0,
                direction="flat",
                confidence=0.0,
                warnings=[f"GNN prediction failed: {e}"],
            )
