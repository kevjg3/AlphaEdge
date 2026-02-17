"""Support and resistance level detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PriceLevel:
    price: float
    level_type: str  # "support" or "resistance"
    strength: float  # 0-1
    touches: int
    last_tested: str

    def to_dict(self) -> dict:
        return {
            "price": round(self.price, 4),
            "level_type": self.level_type,
            "strength": round(self.strength, 3),
            "touches": self.touches,
            "last_tested": self.last_tested,
        }


class SupportResistance:
    """Detect support and resistance levels using multiple methods."""

    def __init__(self, lookback: int = 252, tolerance_pct: float = 0.02):
        self.lookback = lookback
        self.tolerance_pct = tolerance_pct

    def detect(self, df: pd.DataFrame) -> dict:
        """Detect S/R levels from OHLCV DataFrame."""
        df = df.tail(self.lookback).copy()
        if len(df) < 20:
            return {"levels": [], "pivot_points": {}, "fibonacci_levels": {}, "warnings": ["Insufficient data"]}

        h, l, c = df["High"], df["Low"], df["Close"]
        current_price = float(c.iloc[-1])

        # Method 1: Local extrema clustering
        levels = self._find_extrema_levels(h, l, c, current_price)

        # Method 2: Classic pivot points
        pivot_points = self._pivot_points(h, l, c)

        # Method 3: Fibonacci retracements
        fib_levels = self._fibonacci_levels(h, l, current_price)

        # Find nearest
        supports = [lv for lv in levels if lv.level_type == "support"]
        resistances = [lv for lv in levels if lv.level_type == "resistance"]
        nearest_support = max((lv.price for lv in supports), default=None) if supports else None
        nearest_resistance = min((lv.price for lv in resistances), default=None) if resistances else None

        return {
            "levels": [lv.to_dict() for lv in levels],
            "pivot_points": pivot_points,
            "fibonacci_levels": fib_levels,
            "current_price": current_price,
            "nearest_support": round(nearest_support, 4) if nearest_support else None,
            "nearest_resistance": round(nearest_resistance, 4) if nearest_resistance else None,
        }

    def _find_extrema_levels(self, high: pd.Series, low: pd.Series, close: pd.Series, current_price: float) -> list[PriceLevel]:
        """Find S/R from local highs/lows using scipy if available, else rolling."""
        levels: list[PriceLevel] = []

        try:
            from scipy.signal import argrelextrema

            order = max(5, len(high) // 20)
            high_idx = argrelextrema(high.values, np.greater, order=order)[0]
            low_idx = argrelextrema(low.values, np.less, order=order)[0]
        except ImportError:
            # Fallback: rolling max/min
            window = max(5, len(high) // 20)
            roll_max = high.rolling(window, center=True).max()
            roll_min = low.rolling(window, center=True).min()
            high_idx = np.where(high.values == roll_max.values)[0]
            low_idx = np.where(low.values == roll_min.values)[0]

        # Cluster nearby prices
        resistance_prices = high.iloc[high_idx].values if len(high_idx) > 0 else np.array([])
        support_prices = low.iloc[low_idx].values if len(low_idx) > 0 else np.array([])

        # Cluster resistance levels
        for cluster_price, touches, last_idx in self._cluster_prices(resistance_prices, high_idx, high):
            if cluster_price > current_price:
                level_type = "resistance"
            else:
                level_type = "support"

            date_str = str(high.index[last_idx].date()) if hasattr(high.index[last_idx], "date") else str(high.index[last_idx])
            strength = min(touches / 5, 1.0)
            levels.append(PriceLevel(
                price=cluster_price,
                level_type=level_type,
                strength=strength,
                touches=touches,
                last_tested=date_str,
            ))

        # Cluster support levels
        for cluster_price, touches, last_idx in self._cluster_prices(support_prices, low_idx, low):
            if cluster_price < current_price:
                level_type = "support"
            else:
                level_type = "resistance"

            date_str = str(low.index[last_idx].date()) if hasattr(low.index[last_idx], "date") else str(low.index[last_idx])
            strength = min(touches / 5, 1.0)
            levels.append(PriceLevel(
                price=cluster_price,
                level_type=level_type,
                strength=strength,
                touches=touches,
                last_tested=date_str,
            ))

        # Deduplicate close levels
        levels = self._deduplicate(levels)
        levels.sort(key=lambda x: x.price)
        return levels

    def _cluster_prices(self, prices: np.ndarray, indices: np.ndarray, series: pd.Series):
        """Cluster nearby prices and count touches."""
        if len(prices) == 0:
            return

        sorted_idx = np.argsort(prices)
        prices = prices[sorted_idx]
        indices = indices[sorted_idx]

        clusters: list[tuple[float, int, int]] = []
        current_cluster = [prices[0]]
        current_indices = [indices[0]]

        for i in range(1, len(prices)):
            if abs(prices[i] - np.mean(current_cluster)) / np.mean(current_cluster) < self.tolerance_pct:
                current_cluster.append(prices[i])
                current_indices.append(indices[i])
            else:
                cluster_price = float(np.mean(current_cluster))
                touches = len(current_cluster)
                last_idx = int(max(current_indices))
                clusters.append((cluster_price, touches, last_idx))
                current_cluster = [prices[i]]
                current_indices = [indices[i]]

        # Last cluster
        cluster_price = float(np.mean(current_cluster))
        touches = len(current_cluster)
        last_idx = int(max(current_indices))
        clusters.append((cluster_price, touches, last_idx))

        yield from clusters

    def _deduplicate(self, levels: list[PriceLevel]) -> list[PriceLevel]:
        """Remove levels that are too close to each other."""
        if not levels:
            return levels

        levels.sort(key=lambda x: x.price)
        deduped = [levels[0]]
        for lv in levels[1:]:
            if abs(lv.price - deduped[-1].price) / deduped[-1].price > self.tolerance_pct:
                deduped.append(lv)
            else:
                # Keep the stronger one
                if lv.strength > deduped[-1].strength:
                    deduped[-1] = lv
        return deduped

    def _pivot_points(self, high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
        """Classic pivot points from last trading session."""
        h = float(high.iloc[-1])
        l = float(low.iloc[-1])
        c = float(close.iloc[-1])
        p = (h + l + c) / 3

        return {
            "pivot": round(p, 4),
            "r1": round(2 * p - l, 4),
            "r2": round(p + (h - l), 4),
            "r3": round(h + 2 * (p - l), 4),
            "s1": round(2 * p - h, 4),
            "s2": round(p - (h - l), 4),
            "s3": round(l - 2 * (h - p), 4),
        }

    def _fibonacci_levels(self, high: pd.Series, low: pd.Series, current_price: float) -> dict:
        """Fibonacci retracement levels from the swing high/low in the lookback period."""
        swing_high = float(high.max())
        swing_low = float(low.min())
        diff = swing_high - swing_low

        if diff <= 0:
            return {}

        fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        levels = {}

        # If price is in a downtrend (closer to low), use retracement from high
        for ratio in fib_ratios:
            level = swing_high - diff * ratio
            levels[f"fib_{ratio}"] = round(level, 4)

        levels["swing_high"] = round(swing_high, 4)
        levels["swing_low"] = round(swing_low, 4)
        return levels
