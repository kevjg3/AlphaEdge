"""Run forecasts across multiple horizons."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaedge.forecasting.arima_forecaster import ARIMAForecaster
from alphaedge.forecasting.ets_forecaster import ETSForecaster
from alphaedge.forecasting.xgboost_forecaster import XGBoostForecaster
from alphaedge.forecasting.ensemble import EnsembleForecaster, EnsembleResult

logger = logging.getLogger(__name__)

HORIZON_MAP = {
    "1D": 1,
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "12M": 252,
}


@dataclass
class MultiHorizonResult:
    forecasts: dict  # horizon_label -> EnsembleResult.to_dict()
    overall_direction: str
    short_term_outlook: str
    long_term_outlook: str
    model_agreement: float
    accuracy: dict = field(default_factory=dict)  # forecast accuracy metrics
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "forecasts": self.forecasts,
            "overall_direction": self.overall_direction,
            "short_term_outlook": self.short_term_outlook,
            "long_term_outlook": self.long_term_outlook,
            "model_agreement": round(self.model_agreement, 4),
            "accuracy": self.accuracy,
            "warnings": self.warnings,
        }


class HorizonForecaster:
    """Run ensemble forecasts across multiple time horizons."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def run_all_horizons(
        self,
        prices: pd.Series,
        current_price: float,
        ohlcv_df: pd.DataFrame | None = None,
        ticker: str | None = None,
    ) -> MultiHorizonResult:
        """Fit ensemble once, predict across all horizons."""
        warnings: list[str] = []

        # Build ensemble with default models
        base_models = [
            ARIMAForecaster(seed=self.seed),
            ETSForecaster(seed=self.seed),
            XGBoostForecaster(seed=self.seed),
        ]

        # Conditionally add deep-learning models (graceful if torch unavailable)
        try:
            from alphaedge.forecasting.transformer_forecaster import TransformerForecaster
            base_models.append(TransformerForecaster(seed=self.seed))
        except Exception as e:
            logger.debug("TransformerForecaster unavailable: %s", e)

        try:
            from alphaedge.forecasting.cnn_forecaster import CNNForecaster
            base_models.append(CNNForecaster(seed=self.seed))
        except Exception as e:
            logger.debug("CNNForecaster unavailable: %s", e)

        try:
            from alphaedge.forecasting.gnn_forecaster import GNNForecaster
            base_models.append(GNNForecaster(seed=self.seed, ticker=ticker))
        except Exception as e:
            logger.debug("GNNForecaster unavailable: %s", e)

        ensemble = EnsembleForecaster(base_models, seed=self.seed)

        # Fit once on full history — use OHLCV if available for richer features
        train_df = ohlcv_df if ohlcv_df is not None else pd.DataFrame({"Close": prices})
        try:
            ensemble.fit(train_df)
        except Exception as e:
            logger.warning("Ensemble fit failed: %s", e)
            warnings.append(f"Ensemble fit failed: {e}")

        # Predict each horizon
        horizon_results: dict[str, dict] = {}
        directions: list[str] = []

        for label, days in HORIZON_MAP.items():
            try:
                result = ensemble.predict(days, current_price)
                horizon_results[label] = result.to_dict()

                # Track direction for agreement metric
                pred = result.ensemble_prediction
                if isinstance(pred, dict):
                    directions.append(pred.get("direction", "flat"))
            except Exception as e:
                logger.warning("Horizon %s failed: %s", label, e)
                warnings.append(f"Horizon {label} failed: {e}")

        # Overall analysis
        up_count = sum(1 for d in directions if d == "up")
        down_count = sum(1 for d in directions if d == "down")
        n_dirs = len(directions) or 1

        if up_count > down_count:
            overall = "up"
        elif down_count > up_count:
            overall = "down"
        else:
            overall = "flat"

        # Agreement: fraction of models agreeing with overall direction
        agreement = max(up_count, down_count) / n_dirs if directions else 0.0

        # Short-term = 1D + 1W, Long-term = 3M + 12M
        short_dirs = [d for label, d in zip(HORIZON_MAP.keys(), directions) if label in ("1D", "1W")]
        long_dirs = [d for label, d in zip(HORIZON_MAP.keys(), directions) if label in ("3M", "12M")]

        def _outlook(dirs):
            if not dirs:
                return "neutral"
            ups = sum(1 for d in dirs if d == "up")
            downs = sum(1 for d in dirs if d == "down")
            if ups > downs:
                return "bullish"
            elif downs > ups:
                return "bearish"
            return "neutral"

        # --- Accuracy Tracking ---
        accuracy = {}
        try:
            from alphaedge.forecasting.accuracy_tracker import AccuracyTracker
            accuracy = AccuracyTracker(seed=self.seed).evaluate(prices, ensemble)
        except Exception as e:
            logger.debug("Accuracy tracking failed: %s", e)

        return MultiHorizonResult(
            forecasts=horizon_results,
            overall_direction=overall,
            short_term_outlook=_outlook(short_dirs),
            long_term_outlook=_outlook(long_dirs),
            model_agreement=agreement,
            accuracy=accuracy,
            warnings=warnings,
        )
