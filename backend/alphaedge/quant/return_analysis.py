"""Return distribution analysis and rolling metrics time series."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


class ReturnAnalyzer:
    """Analyze return distribution and produce rolling metric time series."""

    def __init__(self, risk_free_rate: float = 0.045):
        self.rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1

    def analyze(self, prices: pd.Series) -> dict:
        """Full return distribution analysis."""
        if len(prices) < 30:
            return {"error": "Insufficient data"}

        log_returns = np.log(prices / prices.shift(1)).dropna()

        return {
            "distribution": self._distribution_stats(log_returns),
            "histogram": self._histogram(log_returns),
            "rolling_metrics": self._rolling_metrics(prices, log_returns),
            "seasonality": self._seasonality(prices),
        }

    def _distribution_stats(self, returns: pd.Series) -> dict:
        """Return distribution statistics."""
        arr = returns.values

        # Jarque-Bera normality test
        jb_stat, jb_pval = stats.jarque_bera(arr)

        # Percentiles
        percentiles = {
            "p1": float(np.percentile(arr, 1)),
            "p5": float(np.percentile(arr, 5)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

        # Tail ratios
        left_tail = float(np.mean(arr[arr < np.percentile(arr, 5)]))
        right_tail = float(np.mean(arr[arr > np.percentile(arr, 95)]))

        return {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
            "skewness": round(float(stats.skew(arr)), 4),
            "kurtosis": round(float(stats.kurtosis(arr)), 4),
            "jarque_bera_stat": round(float(jb_stat), 2),
            "jarque_bera_pval": round(float(jb_pval), 4),
            "is_normal": bool(jb_pval > 0.05),
            "percentiles": {k: round(v, 6) for k, v in percentiles.items()},
            "left_tail_mean": round(left_tail, 6),
            "right_tail_mean": round(right_tail, 6),
            "tail_ratio": round(abs(right_tail / left_tail), 4) if left_tail != 0 else 0.0,
            "n_observations": len(arr),
        }

    def _histogram(self, returns: pd.Series, n_bins: int = 50) -> dict:
        """Return histogram bin data for charting."""
        arr = returns.values
        counts, edges = np.histogram(arr, bins=n_bins)

        # Normal distribution overlay
        x = np.linspace(arr.min(), arr.max(), 200)
        normal_pdf = stats.norm.pdf(x, loc=arr.mean(), scale=arr.std())
        # Scale to match histogram
        bin_width = edges[1] - edges[0]
        normal_scaled = normal_pdf * len(arr) * bin_width

        bins = []
        for i in range(len(counts)):
            bins.append({
                "x": round(float((edges[i] + edges[i + 1]) / 2), 6),
                "count": int(counts[i]),
                "pct": round(float(counts[i] / len(arr) * 100), 2),
            })

        normal_overlay = [
            {"x": round(float(xi), 6), "y": round(float(yi), 2)}
            for xi, yi in zip(x[::4], normal_scaled[::4])  # downsample for JSON
        ]

        return {
            "bins": bins,
            "normal_overlay": normal_overlay,
            "bin_width": round(float(bin_width), 6),
        }

    def _rolling_metrics(self, prices: pd.Series, log_returns: pd.Series) -> dict:
        """Rolling time series for charts."""
        # Rolling volatility
        roll_vol_21 = log_returns.rolling(21).std() * np.sqrt(TRADING_DAYS)
        roll_vol_63 = log_returns.rolling(63).std() * np.sqrt(TRADING_DAYS)

        # Rolling Sharpe (21d window, annualized)
        excess = log_returns - self.rf_daily
        roll_sharpe = (
            excess.rolling(63).mean() / excess.rolling(63).std() * np.sqrt(TRADING_DAYS)
        )

        # Rolling beta vs SPY (we approximate from the data itself — no external fetch)
        # Instead, compute rolling skewness
        roll_skew = log_returns.rolling(63).skew()

        # Cumulative return
        cum_return = (1 + prices.pct_change().fillna(0)).cumprod() - 1

        def _series_to_list(s: pd.Series, decimals: int = 4) -> list[dict]:
            s = s.dropna()
            # Downsample to ~250 points max for JSON efficiency
            if len(s) > 250:
                step = max(1, len(s) // 250)
                s = s.iloc[::step]
            return [
                {"date": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
                 "value": round(float(v), decimals)}
                for d, v in s.items()
                if np.isfinite(v)
            ]

        return {
            "rolling_vol_21d": _series_to_list(roll_vol_21),
            "rolling_vol_63d": _series_to_list(roll_vol_63),
            "rolling_sharpe_63d": _series_to_list(roll_sharpe),
            "rolling_skewness_63d": _series_to_list(roll_skew),
            "cumulative_return": _series_to_list(cum_return),
        }

    def _seasonality(self, prices: pd.Series) -> dict:
        """Monthly seasonality statistics."""
        try:
            monthly_ret = prices.resample("ME").last().pct_change().dropna()
            monthly_ret_by_month = monthly_ret.groupby(monthly_ret.index.month)

            months = {}
            for m, rets in monthly_ret_by_month:
                months[int(m)] = {
                    "mean_return": round(float(rets.mean()), 4),
                    "median_return": round(float(rets.median()), 4),
                    "win_rate": round(float((rets > 0).mean()), 4),
                    "n_samples": len(rets),
                }
            return {"monthly": months}
        except Exception:
            return {}
