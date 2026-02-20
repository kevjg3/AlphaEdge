"""RSI and MACD divergence detection with price."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DivergenceDetector:
    """Detect bullish and bearish divergences between price and oscillators."""

    def __init__(self, lookback: int = 60, extrema_order: int = 5):
        self.lookback = lookback
        self.extrema_order = extrema_order

    def detect(self, df: pd.DataFrame) -> dict:
        """Detect RSI and MACD divergences from OHLCV DataFrame."""
        if len(df) < self.lookback + 20:
            return {"rsi_divergence": _no_div(), "macd_divergence": _no_div(), "has_any_divergence": False}

        close = df["Close"]

        # Compute RSI(14)
        rsi = self._rsi(close, 14)

        # Compute MACD histogram
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_hist = ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()

        # Use last `lookback` bars
        price_window = close.iloc[-self.lookback:].values
        rsi_window = rsi.iloc[-self.lookback:].values
        macd_window = macd_hist.iloc[-self.lookback:].values

        rsi_div = self._check_divergence(price_window, rsi_window, "RSI")
        macd_div = self._check_divergence(price_window, macd_window, "MACD")

        has_any = rsi_div["type"] != "none" or macd_div["type"] != "none"

        return {
            "rsi_divergence": rsi_div,
            "macd_divergence": macd_div,
            "has_any_divergence": has_any,
        }

    def _check_divergence(self, price: np.ndarray, indicator: np.ndarray, name: str) -> dict:
        """Check for bullish/bearish divergence between price and an indicator."""
        try:
            # Find local peaks and troughs
            peaks = self._find_extrema(price, mode="peak")
            troughs = self._find_extrema(price, mode="trough")

            # Check bearish divergence (price higher highs, indicator lower highs)
            bearish = self._check_bearish(price, indicator, peaks, name)
            if bearish["type"] != "none":
                return bearish

            # Check bullish divergence (price lower lows, indicator higher lows)
            bullish = self._check_bullish(price, indicator, troughs, name)
            if bullish["type"] != "none":
                return bullish

        except Exception as e:
            logger.debug("Divergence detection failed for %s: %s", name, e)

        return _no_div()

    def _check_bearish(
        self, price: np.ndarray, indicator: np.ndarray, peaks: list[int], name: str
    ) -> dict:
        """Bearish divergence: price higher highs, indicator lower highs."""
        if len(peaks) < 2:
            return _no_div()

        # Check last 2-3 peaks
        recent_peaks = peaks[-3:]
        n_divergent = 0

        for i in range(len(recent_peaks) - 1):
            p1, p2 = recent_peaks[i], recent_peaks[i + 1]
            if price[p2] > price[p1] and indicator[p2] < indicator[p1]:
                n_divergent += 1

        if n_divergent > 0:
            strength = "strong" if n_divergent >= 2 else "moderate"
            return {
                "type": "bearish",
                "strength": strength,
                "description": f"Price making higher highs while {name} makes lower highs — potential reversal down",
            }

        return _no_div()

    def _check_bullish(
        self, price: np.ndarray, indicator: np.ndarray, troughs: list[int], name: str
    ) -> dict:
        """Bullish divergence: price lower lows, indicator higher lows."""
        if len(troughs) < 2:
            return _no_div()

        recent_troughs = troughs[-3:]
        n_divergent = 0

        for i in range(len(recent_troughs) - 1):
            t1, t2 = recent_troughs[i], recent_troughs[i + 1]
            if price[t2] < price[t1] and indicator[t2] > indicator[t1]:
                n_divergent += 1

        if n_divergent > 0:
            strength = "strong" if n_divergent >= 2 else "moderate"
            return {
                "type": "bullish",
                "strength": strength,
                "description": f"Price making lower lows while {name} makes higher lows — potential reversal up",
            }

        return _no_div()

    def _find_extrema(self, data: np.ndarray, mode: str = "peak") -> list[int]:
        """Find local peaks or troughs using rolling comparison."""
        order = self.extrema_order
        n = len(data)
        indices = []

        for i in range(order, n - order):
            if mode == "peak":
                if all(data[i] >= data[i - j] for j in range(1, order + 1)) and \
                   all(data[i] >= data[i + j] for j in range(1, order + 1)):
                    indices.append(i)
            else:
                if all(data[i] <= data[i - j] for j in range(1, order + 1)) and \
                   all(data[i] <= data[i + j] for j in range(1, order + 1)):
                    indices.append(i)

        return indices

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI (same logic as TechnicalIndicators.rsi)."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))


def _no_div() -> dict:
    return {"type": "none", "strength": "none", "description": ""}
