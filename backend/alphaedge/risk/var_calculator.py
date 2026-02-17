"""Value at Risk (VaR) and Expected Shortfall (CVaR) computation.

Supports parametric, historical, and Monte Carlo VaR methods.
MC VaR reuses mcopt's generate_paths() for efficient GBM simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

from alphaedge.data_ingestion.base import DataSourceName, SourceAttribution


@dataclass
class VaRResult:
    """Value at Risk computation result."""
    confidence_level: float
    horizon_days: int
    current_price: float
    parametric_var: float       # Dollar VaR (loss)
    parametric_var_pct: float   # Percentage VaR
    historical_var: float
    historical_var_pct: float
    monte_carlo_var: float
    monte_carlo_var_pct: float
    expected_shortfall: float   # CVaR (average loss beyond VaR)
    expected_shortfall_pct: float
    daily_vol: float
    annualized_vol: float
    attribution: SourceAttribution = field(
        default_factory=lambda: SourceAttribution(source=DataSourceName.DERIVED)
    )

    def to_dict(self) -> dict:
        return {
            "confidence_level": self.confidence_level,
            "horizon_days": self.horizon_days,
            "current_price": round(self.current_price, 2),
            "parametric_var": round(self.parametric_var, 2),
            "parametric_var_pct": round(self.parametric_var_pct, 4),
            "historical_var": round(self.historical_var, 2),
            "historical_var_pct": round(self.historical_var_pct, 4),
            "monte_carlo_var": round(self.monte_carlo_var, 2),
            "monte_carlo_var_pct": round(self.monte_carlo_var_pct, 4),
            "expected_shortfall": round(self.expected_shortfall, 2),
            "expected_shortfall_pct": round(self.expected_shortfall_pct, 4),
            "daily_vol": round(self.daily_vol, 6),
            "annualized_vol": round(self.annualized_vol, 4),
        }


class VaRCalculator:
    """Compute VaR using parametric, historical, and Monte Carlo methods."""

    def __init__(self, yf_source=None):
        self._yf_source = yf_source

    def compute(
        self,
        prices: pd.Series | None = None,
        ticker: str | None = None,
        confidence: float = 0.95,
        horizon_days: int = 1,
        lookback_days: int = 252,
        mc_paths: int = 50_000,
        mc_seed: int = 42,
    ) -> VaRResult:
        """Compute parametric, historical, and Monte Carlo VaR.

        Either provide `prices` directly or `ticker` to fetch via yf_source.
        """
        if prices is None and ticker and self._yf_source:
            hist_res = self._yf_source.get_history(ticker, period="2y")
            if hist_res.success and not hist_res.data.empty:
                prices = hist_res.data["Close"]

        if prices is None or len(prices) < 30:
            raise ValueError("Insufficient price data for VaR computation")

        # Compute log returns
        log_returns = np.log(prices / prices.shift(1)).dropna()
        recent_returns = log_returns.tail(lookback_days)

        current_price = float(prices.iloc[-1])
        mu = float(recent_returns.mean())
        sigma = float(recent_returns.std())
        sqrt_h = np.sqrt(horizon_days)

        # 1. Parametric VaR (normal assumption)
        z = norm.ppf(1 - confidence)
        param_var_pct = -(mu * horizon_days + z * sigma * sqrt_h)
        param_var = param_var_pct * current_price

        # 2. Historical VaR
        if horizon_days == 1:
            hist_losses = -recent_returns
        else:
            # Scale using overlapping windows
            rolling_returns = recent_returns.rolling(horizon_days).sum().dropna()
            hist_losses = -rolling_returns

        hist_var_pct = float(np.percentile(hist_losses, confidence * 100))
        hist_var = hist_var_pct * current_price

        # 3. Monte Carlo VaR
        mc_var, mc_var_pct = self._mc_var(
            current_price, mu, sigma, horizon_days, confidence,
            mc_paths, mc_seed,
        )

        # 4. Expected Shortfall (CVaR) — average loss beyond VaR
        tail = hist_losses[hist_losses >= hist_var_pct]
        if len(tail) > 0:
            es_pct = float(tail.mean())
        else:
            # Parametric ES for normal distribution
            pdf_at_z = norm.pdf(norm.ppf(1 - confidence))
            es_pct = sigma * sqrt_h * pdf_at_z / (1 - confidence) - mu * horizon_days
        es = es_pct * current_price

        return VaRResult(
            confidence_level=confidence,
            horizon_days=horizon_days,
            current_price=current_price,
            parametric_var=param_var,
            parametric_var_pct=param_var_pct,
            historical_var=hist_var,
            historical_var_pct=hist_var_pct,
            monte_carlo_var=mc_var,
            monte_carlo_var_pct=mc_var_pct,
            expected_shortfall=es,
            expected_shortfall_pct=es_pct,
            daily_vol=sigma,
            annualized_vol=sigma * np.sqrt(252),
        )

    def _mc_var(
        self, S0: float, mu: float, sigma: float,
        horizon: int, confidence: float,
        n_paths: int, seed: int,
    ) -> tuple[float, float]:
        """Monte Carlo VaR via GBM simulation."""
        try:
            # Try to use mcopt's optimized path generator
            import sys
            sys.path.insert(0, "/Users/kevinjiang/stuff/mcopt")
            from mcopt.models.dynamics import generate_paths, VolMode

            S = generate_paths(
                S0=S0, r=mu * 252, q=0.0,
                T=horizon / 252, sigma=sigma * np.sqrt(252),
                vol_mode=VolMode.CONSTANT,
                steps=horizon, paths=n_paths,
                seed=seed, antithetic=True,
            )
            final_prices = S[:, -1]
        except Exception:
            # Fallback: simple GBM simulation
            rng = np.random.default_rng(seed)
            dt = 1.0  # daily
            Z = rng.standard_normal((n_paths, horizon))
            log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            cum_returns = np.sum(log_returns, axis=1)
            final_prices = S0 * np.exp(cum_returns)

        losses = -(final_prices / S0 - 1)
        mc_var_pct = float(np.percentile(losses, confidence * 100))
        mc_var = mc_var_pct * S0

        return mc_var, mc_var_pct
