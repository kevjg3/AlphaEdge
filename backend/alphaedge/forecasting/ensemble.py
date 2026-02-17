"""Ensemble forecaster using stacking with Ridge meta-learner."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaedge.forecasting.base import BaseForecaster, ForecastPoint, ForecastResult

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    individual_forecasts: dict  # model_name -> ForecastResult.to_dict()
    ensemble_prediction: dict  # ForecastResult.to_dict()
    model_weights: dict  # model_name -> weight
    best_model: str
    confidence: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "individual_forecasts": self.individual_forecasts,
            "ensemble_prediction": self.ensemble_prediction,
            "model_weights": {k: round(v, 4) for k, v in self.model_weights.items()},
            "best_model": self.best_model,
            "confidence": round(self.confidence, 4),
            "warnings": self.warnings,
        }


class EnsembleForecaster:
    """Stacking ensemble combining multiple base forecasters."""

    name = "ensemble"

    def __init__(self, forecasters: list[BaseForecaster], seed: int = 42):
        self.forecasters = forecasters
        self.seed = seed
        self._weights: dict[str, float] = {}
        self._fitted = False

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit all base models. Attempt Ridge meta-learner for weights."""
        for fc in self.forecasters:
            try:
                fc.fit(train_data)
            except Exception as e:
                logger.warning("Failed to fit %s: %s", fc.name, e)

        # Try to learn weights via walk-forward CV
        try:
            self._learn_weights(train_data)
        except Exception as e:
            logger.debug("Meta-learner training failed, using equal weights: %s", e)
            n = len(self.forecasters)
            self._weights = {fc.name: 1.0 / n for fc in self.forecasters}

        self._fitted = True

    def _learn_weights(self, train_data: pd.DataFrame) -> None:
        """Learn model weights from walk-forward predictions."""
        from sklearn.linear_model import Ridge

        prices = train_data["Close"] if isinstance(train_data, pd.DataFrame) else train_data
        n = len(prices)

        min_train = min(252, n // 2)
        step = max(21, n // 10)
        horizon = 5  # short horizon for weight learning

        # Collect OOF predictions
        predictions: dict[str, list[float]] = {fc.name: [] for fc in self.forecasters}
        actuals: list[float] = []

        for start in range(min_train, n - horizon, step):
            actual = float(prices.iloc[min(start + horizon, n - 1)])
            current = float(prices.iloc[start - 1])
            actuals.append(actual)

            for fc in self.forecasters:
                try:
                    import copy
                    fc_copy = copy.deepcopy(fc)
                    if isinstance(train_data, pd.DataFrame):
                        fc_copy.fit(train_data.iloc[:start])
                    else:
                        fc_copy.fit(pd.DataFrame({"Close": prices.iloc[:start]}))
                    result = fc_copy.predict(horizon, current)
                    if result.forecasts:
                        predictions[fc.name].append(result.forecasts[-1].predicted)
                    else:
                        predictions[fc.name].append(current)  # naive: no change
                except Exception:
                    predictions[fc.name].append(current)

        if len(actuals) < 3:
            raise ValueError("Not enough CV points for meta-learner")

        # Stack predictions into matrix
        X = np.column_stack([predictions[fc.name] for fc in self.forecasters])
        y = np.array(actuals)

        # Ridge regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)

        # Extract weights (normalize to sum to 1)
        raw_weights = ridge.coef_
        raw_weights = np.maximum(raw_weights, 0)  # no negative weights
        total = raw_weights.sum()
        if total > 0:
            weights = raw_weights / total
        else:
            weights = np.ones(len(self.forecasters)) / len(self.forecasters)

        self._weights = {fc.name: float(w) for fc, w in zip(self.forecasters, weights)}
        logger.debug("Learned ensemble weights: %s", self._weights)

    def predict(self, horizon_days: int, current_price: float) -> EnsembleResult:
        """Generate ensemble prediction."""
        individual: dict[str, dict] = {}
        warnings: list[str] = []
        valid_results: list[tuple[str, ForecastResult]] = []

        for fc in self.forecasters:
            try:
                result = fc.predict(horizon_days, current_price)
                individual[fc.name] = result.to_dict()
                if result.forecasts:
                    valid_results.append((fc.name, result))
            except Exception as e:
                logger.warning("Ensemble: %s predict failed: %s", fc.name, e)
                warnings.append(f"{fc.name} failed: {e}")

        if not valid_results:
            empty_result = ForecastResult(
                model_name="ensemble",
                horizon=f"{horizon_days}d",
                current_price=current_price,
                forecasts=[],
                predicted_return=0.0,
                direction="flat",
                confidence=0.0,
                warnings=["All base models failed"],
            )
            return EnsembleResult(
                individual_forecasts=individual,
                ensemble_prediction=empty_result.to_dict(),
                model_weights=self._weights,
                best_model="none",
                confidence=0.0,
                warnings=warnings + ["All base models failed"],
            )

        # Weighted average of predictions
        if not self._weights:
            n = len(valid_results)
            self._weights = {name: 1.0 / n for name, _ in valid_results}

        # Find common forecast length
        min_len = min(len(r.forecasts) for _, r in valid_results)

        ensemble_points: list[ForecastPoint] = []
        for i in range(min_len):
            weighted_pred = 0.0
            weighted_lower = 0.0
            weighted_upper = 0.0
            weighted_conf = 0.0
            total_weight = 0.0

            for name, result in valid_results:
                w = self._weights.get(name, 1.0 / len(valid_results))
                pt = result.forecasts[i]
                weighted_pred += w * pt.predicted
                weighted_lower += w * pt.lower_bound
                weighted_upper += w * pt.upper_bound
                weighted_conf += w * pt.confidence
                total_weight += w

            if total_weight > 0:
                weighted_pred /= total_weight
                weighted_lower /= total_weight
                weighted_upper /= total_weight
                weighted_conf /= total_weight

            date_str = valid_results[0][1].forecasts[i].date
            ensemble_points.append(ForecastPoint(
                date=date_str,
                predicted=round(float(weighted_pred), 4),
                lower_bound=round(float(weighted_lower), 4),
                upper_bound=round(float(weighted_upper), 4),
                confidence=round(float(min(weighted_conf, 1.0)), 4),
            ))

        final_price = ensemble_points[-1].predicted if ensemble_points else current_price
        predicted_return = (final_price / current_price - 1) * 100
        avg_conf = float(np.mean([p.confidence for p in ensemble_points])) if ensemble_points else 0.0

        ensemble_result = ForecastResult(
            model_name="ensemble",
            horizon=f"{horizon_days}d",
            current_price=current_price,
            forecasts=ensemble_points,
            predicted_return=predicted_return,
            direction=BaseForecaster._direction(predicted_return / 100),
            confidence=avg_conf,
            warnings=warnings,
        )

        # Best model (highest confidence among valid)
        best = max(valid_results, key=lambda x: x[1].confidence)

        return EnsembleResult(
            individual_forecasts=individual,
            ensemble_prediction=ensemble_result.to_dict(),
            model_weights=self._weights,
            best_model=best[0],
            confidence=avg_conf,
            warnings=warnings,
        )
