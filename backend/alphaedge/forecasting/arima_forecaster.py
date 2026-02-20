"""ARIMA-based forecasting using statsmodels."""

from __future__ import annotations

import logging
import warnings as _warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from alphaedge.forecasting.base import BaseForecaster, ForecastPoint, ForecastResult

logger = logging.getLogger(__name__)


class ARIMAForecaster(BaseForecaster):
    """ARIMA model with automatic order selection via AIC."""

    name = "arima"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._model = None
        self._fitted = None
        self._log_prices = None

    def fit(self, train_data: pd.DataFrame) -> None:
        import time as _time
        from statsmodels.tsa.arima.model import ARIMA

        prices = train_data["Close"] if isinstance(train_data, pd.DataFrame) else train_data
        self._log_prices = np.log(prices.dropna())

        if len(self._log_prices) < 30:
            logger.warning("ARIMA: insufficient data (%d points)", len(self._log_prices))
            return

        # Grid search for best (p,d,q) by AIC — with 15s time budget
        _MAX_GRID_SECONDS = 15
        t0 = _time.monotonic()
        best_aic = np.inf
        best_order = (1, 1, 0)

        for p in [0, 1, 2]:
            for d in [0, 1]:
                for q in [0, 1, 2]:
                    if p == 0 and q == 0:
                        continue
                    if _time.monotonic() - t0 > _MAX_GRID_SECONDS:
                        logger.debug("ARIMA grid search timed out, using best so far %s", best_order)
                        break
                    try:
                        with _warnings.catch_warnings():
                            _warnings.simplefilter("ignore")
                            model = ARIMA(self._log_prices, order=(p, d, q))
                            result = model.fit()
                            if result.aic < best_aic:
                                best_aic = result.aic
                                best_order = (p, d, q)
                    except Exception:
                        continue

        # Fit with best order
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                self._model = ARIMA(self._log_prices, order=best_order)
                self._fitted = self._model.fit()
                logger.debug("ARIMA fitted with order %s, AIC=%.2f", best_order, self._fitted.aic)
        except Exception as e:
            logger.warning("ARIMA fit failed: %s", e)
            self._fitted = None

    def predict(self, horizon_days: int, current_price: float) -> ForecastResult:
        warns: list[str] = []

        if self._fitted is None or self._log_prices is None:
            return ForecastResult(
                model_name=self.name,
                horizon=f"{horizon_days}d",
                current_price=current_price,
                forecasts=[],
                predicted_return=0.0,
                direction="flat",
                confidence=0.0,
                warnings=["ARIMA model not fitted"],
            )

        try:
            forecast = self._fitted.get_forecast(steps=horizon_days)
            mean_log = forecast.predicted_mean.values
            conf = forecast.conf_int(alpha=0.10)  # 90% CI

            lower_log = conf.iloc[:, 0].values
            upper_log = conf.iloc[:, 1].values

            # Convert from log prices to price levels
            mean_prices = np.exp(mean_log)
            lower_prices = np.exp(lower_log)
            upper_prices = np.exp(upper_log)

            # Build forecast points
            last_date = self._log_prices.index[-1]
            points: list[ForecastPoint] = []
            for i in range(horizon_days):
                if hasattr(last_date, "date"):
                    date = last_date + timedelta(days=i + 1)
                    date_str = date.strftime("%Y-%m-%d")
                else:
                    date_str = f"T+{i+1}"

                ci_width = float(upper_prices[i] - lower_prices[i])
                price_level = float(mean_prices[i])
                conf_score = max(0, 1 - ci_width / price_level) if price_level > 0 else 0.5

                points.append(ForecastPoint(
                    date=date_str,
                    predicted=float(mean_prices[i]),
                    lower_bound=float(lower_prices[i]),
                    upper_bound=float(upper_prices[i]),
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
                metrics={"aic": float(self._fitted.aic)},
                warnings=warns,
            )

        except Exception as e:
            logger.warning("ARIMA predict failed: %s", e)
            return ForecastResult(
                model_name=self.name,
                horizon=f"{horizon_days}d",
                current_price=current_price,
                forecasts=[],
                predicted_return=0.0,
                direction="flat",
                confidence=0.0,
                warnings=[f"ARIMA prediction failed: {e}"],
            )
