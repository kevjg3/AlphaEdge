"""Multi-timeframe momentum scoring and Omega ratio analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class MomentumResult:
    """Result of momentum scoring and Omega ratio analysis."""

    # Multi-timeframe returns
    return_1m: float | None = None
    return_3m: float | None = None
    return_6m: float | None = None
    return_12m: float | None = None

    # Momentum z-scores (current return vs rolling history of N-day returns)
    momentum_zscore_1m: float = 0.0
    momentum_zscore_3m: float = 0.0
    momentum_zscore_6m: float = 0.0
    momentum_zscore_12m: float = 0.0
    composite_momentum_zscore: float = 0.0

    # Acceleration (current vs prior period)
    acceleration_1m: float = 0.0
    acceleration_3m: float = 0.0
    momentum_regime: str = "neutral"  # "accelerating" | "decelerating" | "neutral"

    # Omega ratio at various thresholds
    omega_ratios: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "return_1m": round(self.return_1m, 4) if self.return_1m is not None else None,
            "return_3m": round(self.return_3m, 4) if self.return_3m is not None else None,
            "return_6m": round(self.return_6m, 4) if self.return_6m is not None else None,
            "return_12m": round(self.return_12m, 4) if self.return_12m is not None else None,
            "momentum_zscore_1m": round(self.momentum_zscore_1m, 4),
            "momentum_zscore_3m": round(self.momentum_zscore_3m, 4),
            "momentum_zscore_6m": round(self.momentum_zscore_6m, 4),
            "momentum_zscore_12m": round(self.momentum_zscore_12m, 4),
            "composite_momentum_zscore": round(self.composite_momentum_zscore, 4),
            "acceleration_1m": round(self.acceleration_1m, 4),
            "acceleration_3m": round(self.acceleration_3m, 4),
            "momentum_regime": self.momentum_regime,
            "omega_ratios": {k: round(v, 4) for k, v in self.omega_ratios.items()},
        }


class MomentumAnalyzer:
    """Compute multi-timeframe momentum scores and Omega ratios."""

    def __init__(self, risk_free_rate: float = 0.045):
        self.rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1

    def analyze(self, prices: pd.Series) -> MomentumResult:
        """Full momentum and Omega ratio analysis."""
        if len(prices) < 63:
            return MomentumResult()

        result = MomentumResult()
        daily_returns = prices.pct_change().dropna()
        log_returns = np.log(prices / prices.shift(1)).dropna()

        # --- Multi-timeframe Returns ---
        self._compute_returns(result, prices)

        # --- Momentum Z-Scores ---
        self._compute_zscores(result, prices)

        # --- Acceleration ---
        self._compute_acceleration(result, prices)

        # --- Omega Ratio ---
        self._compute_omega(result, daily_returns)

        return result

    @staticmethod
    def _compute_returns(result: MomentumResult, prices: pd.Series) -> None:
        """Compute lookback returns for standard momentum windows."""
        n = len(prices)
        p_now = prices.iloc[-1]

        if n >= 21:
            result.return_1m = float(p_now / prices.iloc[-21] - 1)
        if n >= 63:
            result.return_3m = float(p_now / prices.iloc[-63] - 1)
        if n >= 126:
            result.return_6m = float(p_now / prices.iloc[-126] - 1)
        if n >= 252:
            result.return_12m = float(p_now / prices.iloc[-252] - 1)

    @staticmethod
    def _compute_zscores(result: MomentumResult, prices: pd.Series) -> None:
        """Compute z-scores: how unusual is the current N-day return vs. history."""
        n = len(prices)
        lookbacks = {
            "1m": 21,
            "3m": 63,
            "6m": 126,
            "12m": 252,
        }

        z_scores = {}
        for label, window in lookbacks.items():
            if n < window + 63:  # need at least 63 extra points for rolling history
                continue

            # Rolling N-day returns
            rolling_ret = prices.pct_change(window).dropna()
            if len(rolling_ret) < 10:
                continue

            current = rolling_ret.iloc[-1]
            mean = rolling_ret.mean()
            std = rolling_ret.std()

            if std > 1e-10:
                z = float((current - mean) / std)
                z_scores[label] = z

        result.momentum_zscore_1m = z_scores.get("1m", 0.0)
        result.momentum_zscore_3m = z_scores.get("3m", 0.0)
        result.momentum_zscore_6m = z_scores.get("6m", 0.0)
        result.momentum_zscore_12m = z_scores.get("12m", 0.0)

        # Composite: weighted average (classic momentum factor weighting)
        weights = {"1m": 0.1, "3m": 0.2, "6m": 0.3, "12m": 0.4}
        total_weight = sum(weights[k] for k in z_scores)
        if total_weight > 0:
            result.composite_momentum_zscore = sum(
                z_scores[k] * weights[k] for k in z_scores
            ) / total_weight

    @staticmethod
    def _compute_acceleration(result: MomentumResult, prices: pd.Series) -> None:
        """Compute momentum acceleration: is momentum speeding up or slowing down?"""
        n = len(prices)

        if n >= 42:  # need 2x 21 days
            ret_1m_current = prices.iloc[-1] / prices.iloc[-21] - 1
            ret_1m_prior = prices.iloc[-21] / prices.iloc[-42] - 1
            result.acceleration_1m = float(ret_1m_current - ret_1m_prior)

        if n >= 126:  # need 2x 63 days
            ret_3m_current = prices.iloc[-1] / prices.iloc[-63] - 1
            ret_3m_prior = prices.iloc[-63] / prices.iloc[-126] - 1
            result.acceleration_3m = float(ret_3m_current - ret_3m_prior)

        # Classify regime
        if result.acceleration_1m > 0.005 and result.acceleration_3m > 0.005:
            result.momentum_regime = "accelerating"
        elif result.acceleration_1m < -0.005 and result.acceleration_3m < -0.005:
            result.momentum_regime = "decelerating"
        else:
            result.momentum_regime = "neutral"

    def _compute_omega(self, result: MomentumResult, daily_returns: pd.Series) -> None:
        """Compute Omega ratio at various threshold returns.

        Omega(L) = sum(max(r - L, 0)) / sum(max(L - r, 0))
        A higher Omega means more probability-weighted gain relative to loss.
        """
        r = daily_returns.values
        thresholds = {
            "0.0%": 0.0,
            "-0.5%": -0.005,
            "+0.5%": 0.005,
            "Risk-Free": self.rf_daily,
        }

        for label, threshold in thresholds.items():
            gains = np.sum(np.maximum(r - threshold, 0))
            losses = np.sum(np.maximum(threshold - r, 0))
            if losses > 1e-12:
                result.omega_ratios[label] = float(gains / losses)
            else:
                result.omega_ratios[label] = float("inf") if gains > 0 else 1.0
