"""XGBoost-based forecasting with quantile regression for confidence bands."""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from alphaedge.forecasting.base import BaseForecaster, ForecastPoint, ForecastResult

logger = logging.getLogger(__name__)


class XGBoostForecaster(BaseForecaster):
    """XGBoost gradient boosting with quantile regression."""

    name = "xgboost"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._model_median = None
        self._model_lower = None
        self._model_upper = None
        self._feature_cols: list[str] = []
        self._last_features = None
        self._prices = None

    def fit(self, train_data: pd.DataFrame) -> None:
        from xgboost import XGBRegressor

        if isinstance(train_data, pd.Series):
            df = pd.DataFrame({"Close": train_data})
        else:
            df = train_data.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        self._prices = df["Close"].copy()

        features = self._build_features(df)
        target = np.log(df["Close"].shift(-1) / df["Close"])
        target.name = "target"

        # Align
        common = features.index.intersection(target.dropna().index)
        if len(common) < 50:
            logger.warning("XGBoost: insufficient data after feature engineering (%d)", len(common))
            return

        X = features.loc[common]
        y = target.loc[common]
        self._feature_cols = list(X.columns)

        # Store last row for iterative prediction
        self._last_features = features.iloc[-1:].copy()

        params = dict(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=10,
            random_state=self.seed,
            verbosity=0,
        )

        try:
            # Median model
            self._model_median = XGBRegressor(**params, objective="reg:squarederror")
            self._model_median.fit(X, y)

            # Lower bound (5th percentile)
            self._model_lower = XGBRegressor(
                **params, objective="reg:quantileerror", quantile_alpha=0.05,
            )
            self._model_lower.fit(X, y)

            # Upper bound (95th percentile)
            self._model_upper = XGBRegressor(
                **params, objective="reg:quantileerror", quantile_alpha=0.95,
            )
            self._model_upper.fit(X, y)

            logger.debug("XGBoost fitted with %d features, %d samples", len(self._feature_cols), len(X))
        except Exception as e:
            logger.warning("XGBoost fit failed: %s", e)
            self._model_median = None

    def predict(self, horizon_days: int, current_price: float) -> ForecastResult:
        warns: list[str] = []

        if self._model_median is None or self._last_features is None:
            return ForecastResult(
                model_name=self.name,
                horizon=f"{horizon_days}d",
                current_price=current_price,
                forecasts=[],
                predicted_return=0.0,
                direction="flat",
                confidence=0.0,
                warnings=["XGBoost model not fitted"],
            )

        try:
            # Iterative multi-step prediction
            price = current_price
            points: list[ForecastPoint] = []
            last_date = self._prices.index[-1] if self._prices is not None else None

            # Use the last feature row as starting point
            current_features = self._last_features.copy()

            for i in range(horizon_days):
                X = current_features[self._feature_cols]

                pred_ret = float(self._model_median.predict(X)[0])
                lower_ret = float(self._model_lower.predict(X)[0])
                upper_ret = float(self._model_upper.predict(X)[0])

                pred_price = price * np.exp(pred_ret)
                lower_price = price * np.exp(lower_ret)
                upper_price = price * np.exp(upper_ret)

                if last_date is not None and hasattr(last_date, "date"):
                    date = last_date + timedelta(days=i + 1)
                    date_str = date.strftime("%Y-%m-%d")
                else:
                    date_str = f"T+{i+1}"

                ci_width = upper_price - lower_price
                conf_score = max(0, 1 - ci_width / pred_price) if pred_price > 0 else 0.5

                points.append(ForecastPoint(
                    date=date_str,
                    predicted=round(float(pred_price), 4),
                    lower_bound=round(float(max(lower_price, 0.01)), 4),
                    upper_bound=round(float(upper_price), 4),
                    confidence=min(round(float(conf_score), 4), 1.0),
                ))

                # Update features for next step (shift returns)
                price = pred_price
                current_features = self._update_features(current_features, pred_ret)

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
            logger.warning("XGBoost predict failed: %s", e)
            return ForecastResult(
                model_name=self.name,
                horizon=f"{horizon_days}d",
                current_price=current_price,
                forecasts=[],
                predicted_return=0.0,
                direction="flat",
                confidence=0.0,
                warnings=[f"XGBoost prediction failed: {e}"],
            )

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature DataFrame from OHLCV data."""
        # Handle multi-level columns from newer yfinance
        work = df.copy()
        if isinstance(work.columns, pd.MultiIndex):
            work.columns = [col[0] if isinstance(col, tuple) else col for col in work.columns]

        c = work["Close"].squeeze() if "Close" in work.columns else work.iloc[:, 0]

        has_volume = "Volume" in work.columns and float(work["Volume"].sum()) > 0
        v = work["Volume"] if has_volume else None

        features = pd.DataFrame(index=work.index)

        # Lagged returns
        for lag in [1, 2, 3, 5, 10, 21]:
            features[f"ret_{lag}d"] = np.log(c / c.shift(lag))

        # Rolling volatility
        log_ret = np.log(c / c.shift(1))
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

        # MACD signal
        fast_ema = c.ewm(span=12, adjust=False).mean()
        slow_ema = c.ewm(span=26, adjust=False).mean()
        macd = fast_ema - slow_ema
        features["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

        # Volume ratio (skip if no volume data to avoid all-NaN column)
        if v is not None:
            vol_sma20 = v.rolling(20).mean()
            features["volume_ratio"] = v / vol_sma20.replace(0, np.nan)

        # Calendar
        try:
            idx = pd.to_datetime(work.index)
            features["day_of_week"] = idx.dayofweek
            features["month"] = idx.month
        except Exception:
            features["day_of_week"] = 0
            features["month"] = 1

        features = features.dropna()
        return features

    def _update_features(self, features: pd.DataFrame, new_return: float) -> pd.DataFrame:
        """Update feature row for next iterative step (approximate)."""
        updated = features.copy()
        # Shift returns: ret_1d = new_return, older lags stay (approximation)
        if "ret_1d" in updated.columns:
            updated["ret_1d"] = new_return
        return updated
