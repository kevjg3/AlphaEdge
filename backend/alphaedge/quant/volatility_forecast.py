"""GARCH(1,1) volatility forecasting via manual MLE — no external arch library needed."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class GarchForecastResult:
    """Result of GARCH(1,1) volatility forecast."""

    # Model parameters
    omega: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    persistence: float = 0.0  # alpha + beta
    long_run_vol_annual: float = 0.0  # sqrt(omega / (1 - alpha - beta)) * sqrt(252)
    converged: bool = False

    # Forecasts (annualized vol)
    forecast_5d_vol: float = 0.0
    forecast_10d_vol: float = 0.0
    forecast_21d_vol: float = 0.0

    # Historical comparison
    realized_vol_21d: float = 0.0
    vol_ratio: float = 0.0  # forecast_21d / realized_21d — >1 means vol expansion expected

    # Time series for chart
    conditional_vol_series: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "omega": round(self.omega, 8),
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "persistence": round(self.persistence, 4),
            "long_run_vol_annual": round(self.long_run_vol_annual, 4),
            "converged": self.converged,
            "forecast_5d_vol": round(self.forecast_5d_vol, 4),
            "forecast_10d_vol": round(self.forecast_10d_vol, 4),
            "forecast_21d_vol": round(self.forecast_21d_vol, 4),
            "realized_vol_21d": round(self.realized_vol_21d, 4),
            "vol_ratio": round(self.vol_ratio, 4),
            "conditional_vol_series": self.conditional_vol_series,
        }


class GarchForecaster:
    """Forecast volatility using a manually-implemented GARCH(1,1) model."""

    def forecast(self, prices: pd.Series) -> GarchForecastResult:
        """Run GARCH(1,1) estimation and produce volatility forecasts."""
        if len(prices) < 100:
            return GarchForecastResult()

        result = GarchForecastResult()

        log_returns = np.log(prices / prices.shift(1)).dropna()
        eps = (log_returns - log_returns.mean()).values
        dates = log_returns.index

        # Realized vol for comparison
        result.realized_vol_21d = float(log_returns.iloc[-21:].std() * np.sqrt(TRADING_DAYS))

        # --- Fit GARCH(1,1) via MLE ---
        try:
            omega, alpha, beta, sigma2 = self._fit_garch(eps)
            result.omega = omega
            result.alpha = alpha
            result.beta = beta
            result.persistence = alpha + beta
            result.converged = True

            # Long-run variance
            if alpha + beta < 1:
                long_run_var = omega / (1 - alpha - beta)
                result.long_run_vol_annual = float(np.sqrt(long_run_var * TRADING_DAYS))

            # --- Forward forecasts ---
            last_sigma2 = sigma2[-1]
            last_eps2 = eps[-1] ** 2

            # 1-step ahead
            next_sigma2 = omega + alpha * last_eps2 + beta * last_sigma2

            # Multi-step ahead: E[σ²_{t+h}] = ω + (α+β) * E[σ²_{t+h-1}]
            # converges to unconditional variance
            forecast_vars = [next_sigma2]
            for _ in range(20):  # up to 21 steps
                next_var = omega + (alpha + beta) * forecast_vars[-1]
                forecast_vars.append(next_var)

            # Annualized vol = sqrt(mean_daily_var * 252)
            result.forecast_5d_vol = float(np.sqrt(np.mean(forecast_vars[:5]) * TRADING_DAYS))
            result.forecast_10d_vol = float(np.sqrt(np.mean(forecast_vars[:10]) * TRADING_DAYS))
            result.forecast_21d_vol = float(np.sqrt(np.mean(forecast_vars[:21]) * TRADING_DAYS))

            # Vol ratio
            if result.realized_vol_21d > 0:
                result.vol_ratio = result.forecast_21d_vol / result.realized_vol_21d

            # --- Conditional volatility time series ---
            result.conditional_vol_series = self._build_vol_series(sigma2, dates)

        except Exception as e:
            logger.warning("GARCH fitting failed, using EWMA fallback: %s", e)
            result.converged = False
            self._ewma_fallback(result, log_returns, dates)

        return result

    @staticmethod
    def _fit_garch(eps: np.ndarray) -> tuple[float, float, float, np.ndarray]:
        """Fit GARCH(1,1) via maximum likelihood estimation.

        Model: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

        Returns (omega, alpha, beta, sigma2_series).
        """
        n = len(eps)
        var_eps = np.var(eps)

        def neg_log_likelihood(params: np.ndarray) -> float:
            omega, alpha, beta = params
            # Stationarity constraint
            if alpha + beta >= 0.9999:
                return 1e10

            sigma2 = np.empty(n)
            sigma2[0] = var_eps  # initialize with unconditional variance

            for t in range(1, n):
                sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
                if sigma2[t] < 1e-14:
                    sigma2[t] = 1e-14

            # Log-likelihood (Gaussian)
            ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)
            return -ll  # minimize negative

        # Initial guess
        x0 = np.array([var_eps * 0.05, 0.08, 0.88])

        # Bounds
        bounds = [
            (1e-10, var_eps * 10),  # omega
            (1e-6, 0.5),  # alpha
            (0.3, 0.9999),  # beta
        ]

        result = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        if not result.success:
            raise RuntimeError(f"GARCH optimization did not converge: {result.message}")

        omega, alpha, beta = result.x

        # Recompute conditional variance with optimal params
        sigma2 = np.empty(n)
        sigma2[0] = var_eps
        for t in range(1, n):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
            sigma2[t] = max(sigma2[t], 1e-14)

        return float(omega), float(alpha), float(beta), sigma2

    @staticmethod
    def _build_vol_series(sigma2: np.ndarray, dates: pd.Index) -> list[dict]:
        """Convert conditional variance series to annualized vol chart data."""
        ann_vol = np.sqrt(sigma2 * TRADING_DAYS)
        # Downsample to ~250 points
        step = max(1, len(ann_vol) // 250)
        indices = list(range(0, len(ann_vol), step))
        if indices[-1] != len(ann_vol) - 1:
            indices.append(len(ann_vol) - 1)

        series = []
        for i in indices:
            d = dates[i]
            date_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
            series.append({"date": date_str, "value": round(float(ann_vol[i]), 4)})
        return series

    @staticmethod
    def _ewma_fallback(
        result: GarchForecastResult,
        log_returns: pd.Series,
        dates: pd.Index,
    ) -> None:
        """Fallback: use EWMA volatility when GARCH fails to converge."""
        ewma_var = log_returns.ewm(span=21).var()
        ann_vol = np.sqrt(ewma_var * TRADING_DAYS).dropna()

        if len(ann_vol) > 0:
            result.forecast_5d_vol = float(ann_vol.iloc[-1])
            result.forecast_10d_vol = float(ann_vol.iloc[-1])
            result.forecast_21d_vol = float(ann_vol.iloc[-1])

            if result.realized_vol_21d > 0:
                result.vol_ratio = result.forecast_21d_vol / result.realized_vol_21d

            # Build series
            step = max(1, len(ann_vol) // 250)
            indices = list(range(0, len(ann_vol), step))
            if indices[-1] != len(ann_vol) - 1:
                indices.append(len(ann_vol) - 1)

            for i in indices:
                d = ann_vol.index[i]
                date_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                result.conditional_vol_series.append(
                    {"date": date_str, "value": round(float(ann_vol.iloc[i]), 4)}
                )
