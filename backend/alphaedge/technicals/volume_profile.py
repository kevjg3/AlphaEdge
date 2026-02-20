"""Volume Profile analysis — price-level volume distribution, POC, Value Area, VWAP bands."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VolumeProfileAnalyzer:
    """Compute volume profile and VWAP bands from OHLCV data."""

    def __init__(self, n_bins: int = 30, value_area_pct: float = 0.70):
        self.n_bins = n_bins
        self.value_area_pct = value_area_pct

    def analyze(self, df: pd.DataFrame) -> dict:
        """Compute volume profile, POC, Value Area, and VWAP bands."""
        if len(df) < 20 or "Volume" not in df.columns:
            return {}

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]
        price = float(close.iloc[-1])

        result: dict = {}

        # --- Volume Profile ---
        try:
            self._volume_profile(result, close, volume, price)
        except Exception as e:
            logger.warning("Volume profile failed: %s", e)

        # --- VWAP Bands ---
        try:
            self._vwap_bands(result, high, low, close, volume, price)
        except Exception as e:
            logger.warning("VWAP bands failed: %s", e)

        return result

    def _volume_profile(
        self, result: dict, close: pd.Series, volume: pd.Series, price: float
    ) -> None:
        """Build price-volume histogram, find POC and Value Area."""
        # Use typical price as midpoint for each bar
        prices = close.values
        volumes = volume.values

        price_min = float(np.min(prices))
        price_max = float(np.max(prices))
        if price_max <= price_min:
            return

        # Create bins
        bin_edges = np.linspace(price_min, price_max, self.n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Assign volume to bins
        bin_volumes = np.zeros(self.n_bins)
        bin_indices = np.digitize(prices, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        for i, vol in zip(bin_indices, volumes):
            bin_volumes[i] += vol

        total_vol = bin_volumes.sum()
        if total_vol <= 0:
            return

        # Point of Control — price level with highest volume
        poc_idx = int(np.argmax(bin_volumes))
        poc_price = float(bin_centers[poc_idx])

        # Value Area — price range containing value_area_pct of total volume
        # Start from POC and expand outward
        sorted_indices = np.argsort(bin_volumes)[::-1]
        cumulative = 0.0
        va_indices = []
        for idx in sorted_indices:
            va_indices.append(idx)
            cumulative += bin_volumes[idx]
            if cumulative / total_vol >= self.value_area_pct:
                break

        va_low = float(bin_edges[min(va_indices)])
        va_high = float(bin_edges[max(va_indices) + 1])

        # Price position
        if abs(price - poc_price) / price < 0.005:
            price_vs_poc = "at"
        elif price > poc_price:
            price_vs_poc = "above"
        else:
            price_vs_poc = "below"

        if price > va_high:
            price_vs_va = "above"
        elif price < va_low:
            price_vs_va = "below"
        else:
            price_vs_va = "inside"

        # Build histogram data (top 20 bins for JSON)
        profile_data = []
        for i in range(self.n_bins):
            if bin_volumes[i] > 0:
                profile_data.append({
                    "price": round(float(bin_centers[i]), 2),
                    "volume": round(float(bin_volumes[i]), 0),
                    "pct": round(float(bin_volumes[i] / total_vol * 100), 1),
                })

        result["poc_price"] = round(poc_price, 2)
        result["value_area_high"] = round(va_high, 2)
        result["value_area_low"] = round(va_low, 2)
        result["price_vs_poc"] = price_vs_poc
        result["price_vs_value_area"] = price_vs_va
        result["volume_profile"] = profile_data

    @staticmethod
    def _vwap_bands(
        result: dict,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        price: float,
    ) -> None:
        """Compute VWAP and standard deviation bands."""
        typical = (high + low + close) / 3
        cum_vol = volume.cumsum()
        cum_tp_vol = (typical * volume).cumsum()

        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

        # Standard deviation of (price - VWAP) for bands
        deviation = typical - vwap
        # Rolling squared deviation weighted by volume
        cum_sq_dev = ((deviation ** 2) * volume).cumsum()
        variance = cum_sq_dev / cum_vol.replace(0, np.nan)
        std = np.sqrt(variance)

        vwap_val = float(vwap.iloc[-1]) if pd.notna(vwap.iloc[-1]) else None
        std_val = float(std.iloc[-1]) if pd.notna(std.iloc[-1]) else None

        if vwap_val and std_val:
            result["vwap"] = round(vwap_val, 2)
            result["vwap_upper_1"] = round(vwap_val + std_val, 2)
            result["vwap_lower_1"] = round(vwap_val - std_val, 2)
            result["vwap_upper_2"] = round(vwap_val + 2 * std_val, 2)
            result["vwap_lower_2"] = round(vwap_val - 2 * std_val, 2)
