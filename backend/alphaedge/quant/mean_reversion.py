"""Mean reversion detection: Hurst exponent (R/S analysis) and Ornstein-Uhlenbeck half-life."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


@dataclass
class MeanReversionResult:
    """Result of mean reversion analysis."""

    hurst_exponent: float = 0.5
    hurst_regime: str = "random_walk"  # "trending" | "random_walk" | "mean_reverting"
    half_life_days: float | None = None
    ou_theta: float | None = None  # speed of mean reversion
    ou_mu: float | None = None  # long-run equilibrium (log price level)
    confidence: str = "low"  # "low" | "medium" | "high"

    def to_dict(self) -> dict:
        return {
            "hurst_exponent": round(self.hurst_exponent, 4),
            "hurst_regime": self.hurst_regime,
            "half_life_days": round(self.half_life_days, 1) if self.half_life_days is not None else None,
            "ou_theta": round(self.ou_theta, 6) if self.ou_theta is not None else None,
            "ou_mu": round(self.ou_mu, 4) if self.ou_mu is not None else None,
            "confidence": self.confidence,
        }


class MeanReversionAnalyzer:
    """Classify price regime via Hurst exponent and estimate mean reversion half-life."""

    def analyze(self, prices: pd.Series) -> MeanReversionResult:
        """Run full mean reversion analysis on a price series."""
        if len(prices) < 100:
            return MeanReversionResult()

        result = MeanReversionResult()

        # Confidence based on data length
        n = len(prices)
        if n >= 500:
            result.confidence = "high"
        elif n >= 200:
            result.confidence = "medium"
        else:
            result.confidence = "low"

        # --- Hurst Exponent via Rescaled Range (R/S) Analysis ---
        try:
            result.hurst_exponent = self._hurst_rs(prices)
            if result.hurst_exponent > 0.55:
                result.hurst_regime = "trending"
            elif result.hurst_exponent < 0.45:
                result.hurst_regime = "mean_reverting"
            else:
                result.hurst_regime = "random_walk"
        except Exception as e:
            logger.warning("Hurst exponent calculation failed: %s", e)

        # --- Ornstein-Uhlenbeck Half-Life via OLS ---
        try:
            theta, mu, half_life = self._ou_half_life(prices)
            if theta is not None and theta > 0:
                result.ou_theta = theta
                result.ou_mu = mu
                result.half_life_days = half_life
        except Exception as e:
            logger.warning("OU half-life calculation failed: %s", e)

        return result

    @staticmethod
    def _hurst_rs(prices: pd.Series) -> float:
        """Compute Hurst exponent via Rescaled Range (R/S) analysis.

        For a range of sub-period sizes n, split the return series into
        non-overlapping windows. For each window compute R/S (range of
        cumulative deviation / std). The slope of log(R/S) vs log(n) gives H.
        """
        log_returns = np.diff(np.log(prices.values))
        n_total = len(log_returns)

        # Generate sub-period sizes: powers of 2 and intermediate points
        min_n = 10
        max_n = n_total // 2
        if max_n < min_n:
            return 0.5

        # Use logarithmically spaced sizes for better coverage
        sizes = np.unique(
            np.logspace(np.log10(min_n), np.log10(max_n), num=20).astype(int)
        )
        sizes = sizes[(sizes >= min_n) & (sizes <= max_n)]

        if len(sizes) < 4:
            return 0.5

        log_sizes = []
        log_rs = []

        for size in sizes:
            n_windows = n_total // size
            if n_windows < 1:
                continue

            rs_values = []
            for w in range(n_windows):
                window = log_returns[w * size : (w + 1) * size]
                if len(window) < size:
                    continue
                mean_w = window.mean()
                dev = np.cumsum(window - mean_w)
                r = dev.max() - dev.min()
                s = window.std(ddof=1)
                if s > 1e-12:
                    rs_values.append(r / s)

            if rs_values:
                avg_rs = np.mean(rs_values)
                if avg_rs > 0:
                    log_sizes.append(np.log(size))
                    log_rs.append(np.log(avg_rs))

        if len(log_sizes) < 4:
            return 0.5

        # Linear regression: slope = Hurst exponent
        coeffs = np.polyfit(log_sizes, log_rs, 1)
        hurst = float(coeffs[0])

        # Clamp to reasonable range
        return max(0.01, min(0.99, hurst))

    @staticmethod
    def _ou_half_life(prices: pd.Series) -> tuple[float | None, float | None, float | None]:
        """Estimate Ornstein-Uhlenbeck half-life via OLS.

        Regression: Δlog(P) = α + β·log(P)_{t-1} + ε
        θ = -β  (speed of reversion)
        half-life = ln(2) / θ
        μ = -α / β  (equilibrium level)

        Returns (theta, mu, half_life) or (None, None, None) on failure.
        """
        log_prices = np.log(prices.values)
        y = np.diff(log_prices)  # Δlog(P)
        x = log_prices[:-1]  # log(P)_{t-1}

        x_with_const = sm.add_constant(x)
        model = sm.OLS(y, x_with_const).fit()

        alpha = model.params[0]
        beta = model.params[1]

        if beta >= 0:
            # No mean reversion — beta should be negative for mean-reverting
            return None, None, None

        theta = -beta
        half_life = np.log(2) / theta
        mu = -alpha / beta  # equilibrium log price

        # Sanity check: half-life should be positive and reasonable (< 5 years)
        if half_life <= 0 or half_life > 1260:
            return theta, mu, None

        return float(theta), float(mu), float(half_life)
