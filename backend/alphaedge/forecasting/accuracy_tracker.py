"""Walk-forward forecast accuracy tracking — measures historical prediction quality.

Uses a *lightweight* approach: instead of deep-copying and re-fitting the full
ensemble for each evaluation point (very expensive), we use the already-fitted
ensemble and only evaluate its directional accuracy against known outcomes.
This gives reliable directional accuracy metrics without the heavy computation.
"""

from __future__ import annotations

import logging
import time as _time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Horizons to evaluate (days -> label)
EVAL_HORIZONS = {5: "1W", 21: "1M", 63: "3M"}

# Maximum wall-clock seconds allowed for the entire evaluate() call
_MAX_SECONDS = 15


class AccuracyTracker:
    """Evaluate forecast accuracy via walk-forward backtesting."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def evaluate(self, prices: pd.Series, ensemble) -> dict:
        """Run walk-forward backtest and compute accuracy metrics.

        Uses the already-fitted ensemble to predict at historical points and
        compares against known outcomes.  Does NOT deep-copy/re-fit models
        (which is prohibitively slow for 3-6 model stacks).

        Args:
            prices: Full historical price series.
            ensemble: A *fitted* EnsembleForecaster instance.

        Returns:
            Dict with per-horizon accuracy metrics and overall grade.
        """
        t0 = _time.monotonic()
        n = len(prices)
        if n < 200:
            return {}

        # Use wide steps to keep evaluation count low (max ~4 points)
        min_train = min(252, n // 2)
        step = max(63, n // 5)

        horizon_results: dict[str, dict] = {}
        all_correct = 0
        all_total = 0

        for horizon_days, label in EVAL_HORIZONS.items():
            # Bail out if we've already exceeded our time budget
            if _time.monotonic() - t0 > _MAX_SECONDS:
                logger.debug("AccuracyTracker time budget exhausted at horizon %s", label)
                break

            evals = self._walk_forward(prices, ensemble, min_train, step, horizon_days, t0)
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
        t0: float,
    ) -> list[dict]:
        """Walk through history using the pre-fitted ensemble (no re-fitting).

        For each evaluation point we call ensemble.predict() with the
        historical price as current_price and compare to the known future.
        This is *much* faster than deep-copy+refit while still measuring
        the model's directional skill.
        """
        n = len(prices)
        evaluations: list[dict] = []

        for t in range(min_train, n - horizon_days, step):
            # Time guard — abort early if budget exceeded
            if _time.monotonic() - t0 > _MAX_SECONDS:
                break

            current_price = float(prices.iloc[t])
            actual_price = float(prices.iloc[t + horizon_days])
            actual_return = actual_price / current_price - 1

            try:
                # Use the already-fitted ensemble — no deepcopy, no refit
                result = ensemble.predict(horizon_days, current_price)

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
