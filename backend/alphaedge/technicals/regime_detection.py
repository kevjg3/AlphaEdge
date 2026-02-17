"""Market regime detection using Hidden Markov Models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_LABELS = {0: "low_vol", 1: "medium_vol", 2: "high_vol"}


@dataclass
class RegimeInfo:
    current_regime: str
    regime_probability: float
    n_regimes: int
    regime_history: list[dict] = field(default_factory=list)
    regime_stats: dict = field(default_factory=dict)
    transition_matrix: list[list[float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "current_regime": self.current_regime,
            "regime_probability": round(self.regime_probability, 4),
            "n_regimes": self.n_regimes,
            "regime_history": self.regime_history[-20:],
            "regime_stats": self.regime_stats,
            "transition_matrix": [
                [round(v, 4) for v in row] for row in self.transition_matrix
            ] if self.transition_matrix else [],
            "warnings": self.warnings,
        }


class RegimeDetector:
    """Detect market regimes using Gaussian HMM on returns + volatility."""

    def __init__(self, n_regimes: int = 3, seed: int = 42):
        self.n_regimes = n_regimes
        self.seed = seed

    def detect(self, prices: pd.Series) -> RegimeInfo:
        """Fit HMM and classify regimes. Falls back to vol-percentile method."""
        if len(prices) < 60:
            return RegimeInfo(
                current_regime="unknown",
                regime_probability=0.0,
                n_regimes=0,
                warnings=["Insufficient data for regime detection (need >= 60 days)"],
            )

        # Compute features
        log_ret = np.log(prices / prices.shift(1)).dropna()
        vol_21 = log_ret.rolling(21).std().dropna()

        # Align
        common_idx = log_ret.index.intersection(vol_21.index)
        if len(common_idx) < 30:
            return self._fallback(log_ret, prices)

        features = np.column_stack([
            log_ret.loc[common_idx].values,
            vol_21.loc[common_idx].values,
        ])

        # Try HMM
        try:
            from hmmlearn.hmm import GaussianHMM

            model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=200,
                random_state=self.seed,
            )
            model.fit(features)
            states = model.predict(features)
            probs = model.predict_proba(features)

            # Label regimes by volatility (sort means by vol column)
            vol_means = model.means_[:, 1]
            order = np.argsort(vol_means)
            label_map = {}
            labels = ["low_vol", "medium_vol", "high_vol"]
            for rank, orig_idx in enumerate(order):
                label_map[orig_idx] = labels[min(rank, len(labels) - 1)]

            current_state = int(states[-1])
            current_label = label_map[current_state]
            current_prob = float(probs[-1, current_state])

            # Build history
            history = []
            dates = common_idx
            prev_state = None
            for i in range(len(states)):
                if states[i] != prev_state:
                    history.append({
                        "date": str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i]),
                        "regime": label_map[int(states[i])],
                        "probability": round(float(probs[i, states[i]]), 4),
                    })
                    prev_state = states[i]

            # Stats per regime
            regime_stats = {}
            for state_idx, label in label_map.items():
                mask = states == state_idx
                if mask.sum() > 0:
                    rets = features[mask, 0]
                    # Duration: average consecutive run length
                    runs = []
                    current_run = 0
                    for s in states:
                        if s == state_idx:
                            current_run += 1
                        elif current_run > 0:
                            runs.append(current_run)
                            current_run = 0
                    if current_run > 0:
                        runs.append(current_run)

                    regime_stats[label] = {
                        "mean_daily_return": round(float(np.mean(rets)), 6),
                        "annualized_return": round(float(np.mean(rets) * 252), 4),
                        "daily_volatility": round(float(np.std(rets)), 6),
                        "annualized_volatility": round(float(np.std(rets) * np.sqrt(252)), 4),
                        "avg_duration_days": round(float(np.mean(runs)), 1) if runs else 0,
                        "pct_time": round(float(mask.sum() / len(mask) * 100), 1),
                    }

            # Transition matrix
            trans_matrix = model.transmat_.tolist()

            return RegimeInfo(
                current_regime=current_label,
                regime_probability=current_prob,
                n_regimes=self.n_regimes,
                regime_history=history,
                regime_stats=regime_stats,
                transition_matrix=trans_matrix,
            )

        except Exception as e:
            logger.debug("HMM failed, falling back: %s", e)
            return self._fallback(log_ret, prices)

    def _fallback(self, log_ret: pd.Series, prices: pd.Series) -> RegimeInfo:
        """Volatility-percentile fallback regime detection."""
        vol_21 = log_ret.rolling(21).std().dropna()
        if vol_21.empty:
            return RegimeInfo(
                current_regime="unknown",
                regime_probability=0.0,
                n_regimes=0,
                warnings=["Insufficient data for regime detection"],
            )

        p33 = vol_21.quantile(0.33)
        p66 = vol_21.quantile(0.66)
        current_vol = float(vol_21.iloc[-1])

        if current_vol < p33:
            regime = "low_vol"
            prob = 0.7
        elif current_vol < p66:
            regime = "medium_vol"
            prob = 0.6
        else:
            regime = "high_vol"
            prob = 0.7

        return RegimeInfo(
            current_regime=regime,
            regime_probability=prob,
            n_regimes=3,
            regime_stats={
                "low_vol": {"threshold": round(float(p33), 6)},
                "medium_vol": {"threshold": round(float(p66), 6)},
                "high_vol": {"threshold": None},
            },
            warnings=["Using volatility-percentile fallback (HMM unavailable)"],
        )
