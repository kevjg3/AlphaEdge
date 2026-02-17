"""Feature store: build ML-ready feature matrix from OHLCV data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaedge.technicals.indicators import TechnicalIndicators


class FeatureStore:
    """Build a comprehensive feature DataFrame from OHLCV data."""

    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix from OHLCV DataFrame.

        Returns DataFrame with same DatetimeIndex, leading NaN rows dropped.
        """
        c = df["Close"]
        h = df["High"]
        l = df["Low"]
        v = df["Volume"]
        ti = TechnicalIndicators

        features = pd.DataFrame(index=df.index)

        # Returns
        features["ret_1d"] = np.log(c / c.shift(1))
        features["ret_5d"] = np.log(c / c.shift(5))
        features["ret_10d"] = np.log(c / c.shift(10))
        features["ret_21d"] = np.log(c / c.shift(21))

        # Volatility
        log_ret = features["ret_1d"]
        features["vol_5d"] = log_ret.rolling(5).std()
        features["vol_10d"] = log_ret.rolling(10).std()
        features["vol_21d"] = log_ret.rolling(21).std()
        features["vol_63d"] = log_ret.rolling(63).std()

        # Momentum
        features["rsi_14"] = ti.rsi(c, 14)
        macd_data = ti.macd(c)
        features["macd_signal"] = macd_data["signal"]
        features["macd_hist"] = macd_data["histogram"]
        features["roc_5d"] = c.pct_change(5)
        features["roc_21d"] = c.pct_change(21)

        # Volume
        vol_sma20 = ti.sma(v, 20)
        features["volume_ratio"] = v / vol_sma20.replace(0, np.nan)
        features["obv_slope"] = ti.obv(c, v).diff(5)

        # Trend: price vs SMA ratios
        sma20 = ti.sma(c, 20)
        sma50 = ti.sma(c, 50)
        sma200 = ti.sma(c, 200)
        features["price_sma20_ratio"] = c / sma20.replace(0, np.nan)
        features["price_sma50_ratio"] = c / sma50.replace(0, np.nan)
        features["price_sma200_ratio"] = c / sma200.replace(0, np.nan)

        # Bollinger
        bb = ti.bollinger_bands(c)
        features["bb_pct_b"] = bb["pct_b"]
        features["bb_bandwidth"] = bb["bandwidth"]

        # Stochastic
        stoch = ti.stochastic(h, l, c)
        features["stoch_k"] = stoch["k"]
        features["stoch_d"] = stoch["d"]

        # ATR normalized
        atr = ti.atr(h, l, c, 14)
        features["atr_norm"] = atr / c.replace(0, np.nan)

        # Calendar
        if hasattr(df.index, "dayofweek"):
            features["day_of_week"] = df.index.dayofweek
            features["month"] = df.index.month
            features["quarter"] = df.index.quarter
        else:
            idx = pd.to_datetime(df.index)
            features["day_of_week"] = idx.dayofweek
            features["month"] = idx.month
            features["quarter"] = idx.quarter

        # Drop rows with NaN (from rolling windows)
        features = features.dropna()
        return features

    @staticmethod
    def build_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """Build forward return target for ML training."""
        c = df["Close"]
        target = np.log(c.shift(-horizon) / c)
        target.name = f"target_{horizon}d"
        return target
