"""Walk-forward (expanding window) cross-validation for time series."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaedge.forecasting.base import BaseForecaster

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    mse: float
    mae: float
    rmse: float
    directional_accuracy: float
    per_fold_metrics: list[dict] = field(default_factory=list)
    mean_metrics: dict = field(default_factory=dict)
    n_folds: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mse": round(self.mse, 8),
            "mae": round(self.mae, 8),
            "rmse": round(self.rmse, 8),
            "directional_accuracy": round(self.directional_accuracy, 4),
            "n_folds": self.n_folds,
            "per_fold_metrics": self.per_fold_metrics,
            "mean_metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in self.mean_metrics.items()},
            "warnings": self.warnings,
        }


class WalkForwardCV:
    """Walk-forward cross-validation with expanding window."""

    def __init__(self, n_splits: int = 5, min_train_size: int = 252, step_size: int = 63):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.step_size = step_size

    def validate(
        self,
        forecaster: BaseForecaster,
        data: pd.DataFrame,
        horizon_days: int,
    ) -> CVResult:
        """Run walk-forward CV and return aggregated metrics."""
        prices = data["Close"] if isinstance(data, pd.DataFrame) else data
        n = len(prices)

        # Check if we have enough data
        min_needed = self.min_train_size + horizon_days + self.step_size
        if n < min_needed:
            return CVResult(
                mse=0.0, mae=0.0, rmse=0.0,
                directional_accuracy=0.0, n_folds=0,
                warnings=[f"Insufficient data ({n} points, need {min_needed})"],
            )

        # Determine fold boundaries
        max_folds = (n - self.min_train_size - horizon_days) // self.step_size
        actual_folds = min(self.n_splits, max_folds)

        if actual_folds < 1:
            return CVResult(
                mse=0.0, mae=0.0, rmse=0.0,
                directional_accuracy=0.0, n_folds=0,
                warnings=["Cannot create any folds with current parameters"],
            )

        fold_metrics: list[dict] = []
        all_errors: list[float] = []
        direction_correct = 0
        direction_total = 0

        for fold in range(actual_folds):
            train_end = self.min_train_size + fold * self.step_size
            test_end = min(train_end + horizon_days, n)

            if test_end > n:
                break

            train = data.iloc[:train_end] if isinstance(data, pd.DataFrame) else pd.DataFrame({"Close": prices.iloc[:train_end]})
            actual_prices = prices.iloc[train_end:test_end]

            if len(actual_prices) == 0:
                continue

            try:
                # Clone and fit a new forecaster for each fold
                import copy
                fold_forecaster = copy.deepcopy(forecaster)
                fold_forecaster.fit(train)

                current_price = float(prices.iloc[train_end - 1])
                result = fold_forecaster.predict(len(actual_prices), current_price)

                if result.forecasts:
                    predicted_final = result.forecasts[-1].predicted
                    actual_final = float(actual_prices.iloc[-1])

                    error = predicted_final - actual_final
                    all_errors.append(error)

                    # Directional accuracy
                    actual_dir = actual_final - current_price
                    pred_dir = predicted_final - current_price
                    if (actual_dir > 0 and pred_dir > 0) or (actual_dir < 0 and pred_dir < 0):
                        direction_correct += 1
                    direction_total += 1

                    fold_metrics.append({
                        "fold": fold,
                        "train_size": train_end,
                        "predicted": round(predicted_final, 4),
                        "actual": round(actual_final, 4),
                        "error": round(error, 4),
                        "error_pct": round(error / actual_final * 100, 4) if actual_final else 0,
                    })

            except Exception as e:
                logger.debug("CV fold %d failed: %s", fold, e)
                fold_metrics.append({"fold": fold, "error": str(e)})

        if not all_errors:
            return CVResult(
                mse=0.0, mae=0.0, rmse=0.0,
                directional_accuracy=0.0, n_folds=0,
                warnings=["All CV folds failed"],
            )

        errors = np.array(all_errors)
        mse = float(np.mean(errors ** 2))
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(mse))
        dir_acc = direction_correct / direction_total if direction_total > 0 else 0.0

        return CVResult(
            mse=mse,
            mae=mae,
            rmse=rmse,
            directional_accuracy=dir_acc,
            per_fold_metrics=fold_metrics,
            mean_metrics={"mse": mse, "mae": mae, "rmse": rmse, "directional_accuracy": dir_acc},
            n_folds=len([f for f in fold_metrics if "actual" in f]),
        )
