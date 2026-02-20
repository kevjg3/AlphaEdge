"""Walk-forward forecast accuracy tracking — measures historical prediction quality."""

from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Horizons to evaluate (days -> label)
EVAL_HORIZONS = {5: "1W", 21: "1M", 63: "3M"}


class AccuracyTracker:
    """Evaluate forecast accuracy via walk-forward backtesting."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def evaluate(self, prices: pd.Series, ensemble) -> dict:
        """Run walk-forward backtest and compute accuracy metrics.

        Args:
            prices: Full historical price series.
            ensemble: A *fitted* EnsembleForecaster instance.

        Returns:
            Dict with per-horizon accuracy metrics and overall grade.
        """
        n = len(prices)
        if n < 300:
            return {}

        min_train = min(252, n // 2)
        step = max(21, n // 15)  # ~15 evaluation points

        horizon_results: dict[str, dict] = {}
        all_correct = 0
        all_total = 0

        for horizon_days, label in EVAL_HORIZONS.items():
            evals = self._walk_forward(prices, ensemble, min_train, step, horizon_days)
            if not evals:
                continue

            n_evals = len(evals)
            correct = sum(1 for e in evals if e["direction_correct"])
            mae_values = [e["abs_error_pct"] for e in evals]
            ci_hits = sum(1 for e in evals if e["within_ci"])

            directional_acc = correct / n_evals if n_evals > 0 else 0
            mean_abs_err = float(np.mean(mae_values)) if mae_values else 0
            ci_coverage = ci_hits / n_evals if n_evals > 0 else 0

            # Recent accuracy (last 5)
            recent = evals[-5:]
            recent_correct = sum(1 for e in recent if e["direction_correct"])
            recent_acc = recent_correct / len(recent) if recent else 0

            horizon_results[label] = {
                "directional_accuracy": round(directional_acc, 4),
                "mean_abs_error_pct": round(mean_abs_err, 4),
                "ci_coverage_rate": round(ci_coverage, 4),
                "n_evaluations": n_evals,
                "recent_accuracy_5": round(recent_acc, 4),
            }

            all_correct += correct
            all_total += n_evals

        if not horizon_results:
            return {}

        overall_acc = all_correct / all_total if all_total > 0 else 0

        # Grade
        if overall_acc >= 0.70:
            grade = "A"
        elif overall_acc >= 0.60:
            grade = "B"
        elif overall_acc >= 0.50:
            grade = "C"
        else:
            grade = "D"

        return {
            "horizons": horizon_results,
            "overall_directional_accuracy": round(overall_acc, 4),
            "reliability_grade": grade,
        }

    @staticmethod
    def _walk_forward(
        prices: pd.Series,
        ensemble,
        min_train: int,
        step: int,
        horizon_days: int,
    ) -> list[dict]:
        """Walk through history, generate forecasts, compare to actuals."""
        n = len(prices)
        evaluations: list[dict] = []

        for t in range(min_train, n - horizon_days, step):
            current_price = float(prices.iloc[t])
            actual_price = float(prices.iloc[t + horizon_days])
            actual_return = actual_price / current_price - 1

            try:
                # Re-fit a copy on data up to time t, predict forward
                ens_copy = copy.deepcopy(ensemble)
                train_slice = pd.DataFrame({"Close": prices.iloc[:t]})
                ens_copy.fit(train_slice)
                result = ens_copy.predict(horizon_days, current_price)

                pred = result.ensemble_prediction
                if isinstance(pred, dict):
                    pred_return = pred.get("predicted_return", 0) / 100  # convert from %
                    pred_direction = pred.get("direction", "flat")

                    # Check CI coverage from forecast points
                    forecasts = pred.get("forecasts", [])
                    within_ci = False
                    if forecasts:
                        last_fc = forecasts[-1]
                        lb = last_fc.get("lower_bound", 0)
                        ub = last_fc.get("upper_bound", float("inf"))
                        within_ci = lb <= actual_price <= ub

                    # Direction check
                    actual_direction = "up" if actual_return > 0.005 else ("down" if actual_return < -0.005 else "flat")
                    direction_correct = (
                        (pred_direction == "up" and actual_return > 0) or
                        (pred_direction == "down" and actual_return < 0) or
                        (pred_direction == "flat" and abs(actual_return) < 0.005)
                    )

                    evaluations.append({
                        "direction_correct": direction_correct,
                        "abs_error_pct": abs(pred_return - actual_return),
                        "within_ci": within_ci,
                    })

            except Exception as e:
                logger.debug("Walk-forward eval at t=%d failed: %s", t, e)
                continue

        return evaluations
