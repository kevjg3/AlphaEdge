"""Exponential Smoothing (ETS) forecasting."""

from __future__ import annotations

import logging
import warnings as _warnings
from datetime import timedelta

import numpy as np
import pandas as pd

from alphaedge.forecasting.base import BaseForecaster, ForecastPoint, ForecastResult

logger = logging.getLogger(__name__)


class ETSForecaster(BaseForecaster):
    """Exponential Smoothing model using Holt-Winters."""

    name = "ets"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._fitted = None
        self._prices = None

    def fit(self, train_data: pd.DataFrame) -> None:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        prices = train_data["Close"] if isinstance(train_data, pd.DataFrame) else train_data
        self._prices = prices.dropna()

        if len(self._prices) < 20:
            logger.warning("ETS: insufficient data (%d points)", len(self._prices))
            return

        # Try Holt-Winters with additive trend
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    self._prices,
                    trend="add",
                    seasonal=None,
                    damped_trend=True,
                )
                self._fitted = model.fit(optimized=True)
                logger.debug("ETS Holt-Winters fitted successfully")
                return
        except Exception as e:
            logger.debug("Holt-Winters failed, trying simple: %s", e)

        # Fallback: simple exponential smoothing
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    self._prices,
                    trend=None,
                    seasonal=None,
                )
                self._fitted = model.fit(optimized=True)
                logger.debug("ETS simple smoothing fitted")
        except Exception as e:
            logger.warning("ETS fit failed entirely: %s", e)
            self._fitted = None

    def predict(self, horizon_days: int, current_price: float) -> ForecastResult:
        warns: list[str] = []

        if self._fitted is None or self._prices is None:
            return ForecastResult(
                model_name=self.name,
                horizon=f"{horizon_days}d",
                current_price=current_price,
                forecasts=[],
                predicted_return=0.0,
                direction="flat",
                confidence=0.0,
                warnings=["ETS model not fitted"],
            )

        try:
            forecast = self._fitted.forecast(horizon_days)
            mean_prices = forecast.values

            # Estimate confidence intervals from residuals
            residuals = self._fitted.resid
            resid_std = float(np.std(residuals.dropna()))

            last_date = self._prices.index[-1]
            points: list[ForecastPoint] = []

            for i in range(horizon_days):
                if hasattr(last_date, "date"):
                    date = last_date + timedelta(days=i + 1)
                    date_str = date.strftime("%Y-%m-%d")
                else:
                    date_str = f"T+{i+1}"

                # Wider CI for further horizons
                ci_mult = 1.645 * np.sqrt(i + 1)
                pred = float(mean_prices[i])
                lower = pred - ci_mult * resid_std
                upper = pred + ci_mult * resid_std

                ci_width = upper - lower
                conf_score = max(0, 1 - ci_width / pred) if pred > 0 else 0.5

                points.append(ForecastPoint(
                    date=date_str,
                    predicted=pred,
                    lower_bound=max(lower, 0.01),
                    upper_bound=upper,
                    confidence=min(conf_score, 1.0),
                ))

            final_price = float(mean_prices[-1])
            predicted_return = (final_price / current_price - 1) * 100
            avg_conf = float(np.mean([p.confidence for p in points]))

            return ForecastResult(
                model_name=self.name,
                horizon=f"{horizon_days}d",
                current_price=current_price,
                forecasts=points,
                predicted_return=predicted_return,
                direction=self._direction(predicted_return / 100),
                confidence=avg_conf,
                metrics={"residual_std": resid_std},
                warnings=warns,
            )

        except Exception as e:
            logger.warning("ETS predict failed: %s", e)
            return ForecastResult(
                model_name=self.name,
                horizon=f"{horizon_days}d",
                current_price=current_price,
                forecasts=[],
                predicted_return=0.0,
                direction="flat",
                confidence=0.0,
                warnings=[f"ETS prediction failed: {e}"],
            )
