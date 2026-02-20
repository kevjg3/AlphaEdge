"""Ichimoku Cloud indicator — trend, momentum, support/resistance in one system."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IchimokuAnalyzer:
    """Compute Ichimoku Cloud components and derive trading signals."""

    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52):
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period

    def compute(self, df: pd.DataFrame) -> dict:
        """Compute all Ichimoku components and signals from OHLCV DataFrame."""
        if len(df) < self.senkou_b_period + self.kijun_period:
            return {}

        high, low, close = df["High"], df["Low"], df["Close"]
        price = float(close.iloc[-1])

        def _midpoint(h: pd.Series, l: pd.Series, period: int) -> pd.Series:
            return (h.rolling(period).max() + l.rolling(period).min()) / 2

        # Core lines
        tenkan = _midpoint(high, low, self.tenkan_period)
        kijun = _midpoint(high, low, self.kijun_period)
        senkou_a = (tenkan + kijun) / 2  # normally shifted 26 forward
        senkou_b = _midpoint(high, low, self.senkou_b_period)  # normally shifted 26 forward

        # Current values (most recent bar)
        tenkan_val = self._last(tenkan)
        kijun_val = self._last(kijun)
        senkou_a_val = self._last(senkou_a)
        senkou_b_val = self._last(senkou_b)
        chikou_val = float(close.iloc[-1]) if len(close) > 0 else None

        if any(v is None for v in [tenkan_val, kijun_val, senkou_a_val, senkou_b_val]):
            return {}

        # --- Signals ---
        # Tenkan-Kijun cross
        tk_cross = "bullish" if tenkan_val > kijun_val else "bearish" if tenkan_val < kijun_val else "neutral"

        # Price vs cloud
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        if price > cloud_top:
            price_vs_cloud = "above"
        elif price < cloud_bottom:
            price_vs_cloud = "below"
        else:
            price_vs_cloud = "inside"

        # Cloud color (bullish = green when Span A > Span B)
        cloud_color = "green" if senkou_a_val > senkou_b_val else "red"

        # Cloud thickness relative to price
        cloud_thickness = abs(senkou_a_val - senkou_b_val) / price if price > 0 else 0

        # Overall signal
        overall_signal = self._overall_signal(tk_cross, price_vs_cloud, cloud_color)

        return {
            "tenkan_sen": round(tenkan_val, 4),
            "kijun_sen": round(kijun_val, 4),
            "senkou_span_a": round(senkou_a_val, 4),
            "senkou_span_b": round(senkou_b_val, 4),
            "chikou_span": round(chikou_val, 4) if chikou_val else None,
            "tk_cross": tk_cross,
            "price_vs_cloud": price_vs_cloud,
            "cloud_color": cloud_color,
            "cloud_thickness": round(cloud_thickness, 4),
            "overall_signal": overall_signal,
        }

    @staticmethod
    def _last(s: pd.Series):
        if len(s) == 0:
            return None
        val = s.iloc[-1]
        return float(val) if pd.notna(val) else None

    @staticmethod
    def _overall_signal(tk_cross: str, price_vs_cloud: str, cloud_color: str) -> str:
        """Derive overall Ichimoku signal from component signals."""
        bullish_points = 0
        bearish_points = 0

        # TK cross
        if tk_cross == "bullish":
            bullish_points += 1
        elif tk_cross == "bearish":
            bearish_points += 1

        # Price vs cloud
        if price_vs_cloud == "above":
            bullish_points += 1
        elif price_vs_cloud == "below":
            bearish_points += 1

        # Cloud color
        if cloud_color == "green":
            bullish_points += 1
        else:
            bearish_points += 1

        if bullish_points == 3:
            return "strong_bullish"
        elif bullish_points == 2:
            return "bullish"
        elif bearish_points == 3:
            return "strong_bearish"
        elif bearish_points == 2:
            return "bearish"
        return "neutral"
