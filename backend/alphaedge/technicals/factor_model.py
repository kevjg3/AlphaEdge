"""Factor exposure analysis using ETF proxies."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from alphaedge.data_ingestion.yfinance_source import YFinanceSource

logger = logging.getLogger(__name__)


@dataclass
class FactorExposure:
    factor_name: str
    beta: float
    t_stat: float
    p_value: float

    def to_dict(self) -> dict:
        return {
            "factor_name": self.factor_name,
            "beta": round(self.beta, 4),
            "t_stat": round(self.t_stat, 4),
            "p_value": round(self.p_value, 4),
        }


class FactorModel:
    """Estimate factor exposures via OLS regression against ETF proxies."""

    FACTOR_ETFS = {
        "market": "SPY",
        "size_long": "IWM",
        "value_long": "IWD",
        "growth_long": "IWF",
    }

    def __init__(self, yf_source: YFinanceSource):
        self.yf = yf_source

    def compute_exposures(self, ticker: str, lookback_days: int = 252) -> dict:
        """Compute factor exposures via OLS."""
        warnings: list[str] = []

        # Fetch ticker returns
        hist_result = self.yf.get_history(ticker, period=f"{max(lookback_days // 252, 1)}y")
        if not hist_result.success or hist_result.data is None or hist_result.data.empty:
            return {"exposures": [], "warnings": ["No price data for ticker"]}

        ticker_prices = hist_result.data["Close"].tail(lookback_days)
        ticker_returns = np.log(ticker_prices / ticker_prices.shift(1)).dropna()

        # Fetch factor ETF returns
        factor_returns = {}
        for factor_name, etf in self.FACTOR_ETFS.items():
            try:
                etf_result = self.yf.get_history(etf, period=f"{max(lookback_days // 252, 1)}y")
                if etf_result.success and etf_result.data is not None and not etf_result.data.empty:
                    etf_prices = etf_result.data["Close"].tail(lookback_days)
                    factor_returns[factor_name] = np.log(etf_prices / etf_prices.shift(1)).dropna()
            except Exception as e:
                warnings.append(f"Failed to fetch {etf}: {e}")

        if "market" not in factor_returns:
            return {"exposures": [], "warnings": warnings + ["Market factor (SPY) unavailable"]}

        # Build factor matrix
        # Market = SPY returns
        # Size = IWM - SPY
        # Value = IWD - IWF
        factors: dict[str, pd.Series] = {"market": factor_returns["market"]}

        if "size_long" in factor_returns:
            common = factor_returns["size_long"].index.intersection(factor_returns["market"].index)
            factors["size"] = factor_returns["size_long"].loc[common] - factor_returns["market"].loc[common]

        if "value_long" in factor_returns and "growth_long" in factor_returns:
            common = factor_returns["value_long"].index.intersection(factor_returns["growth_long"].index)
            factors["value"] = factor_returns["value_long"].loc[common] - factor_returns["growth_long"].loc[common]

        # Align all series
        factor_df = pd.DataFrame(factors).dropna()
        common_idx = ticker_returns.index.intersection(factor_df.index)

        if len(common_idx) < 30:
            return {"exposures": [], "warnings": warnings + ["Insufficient overlapping data"]}

        y = ticker_returns.loc[common_idx].values
        X = factor_df.loc[common_idx].values
        X_with_const = np.column_stack([np.ones(len(y)), X])

        # OLS
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)
        except np.linalg.LinAlgError:
            return {"exposures": [], "warnings": warnings + ["OLS computation failed"]}

        # Compute statistics
        n, k = X_with_const.shape
        y_hat = X_with_const @ beta
        resid = y - y_hat
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        mse = ss_res / (n - k) if n > k else ss_res
        try:
            cov = mse * np.linalg.inv(X_with_const.T @ X_with_const)
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            se = np.ones(k) * np.nan

        t_stats = beta / np.where(se > 0, se, np.nan)

        # p-values (two-tailed, approximate using normal for large n)
        from scipy import stats as sp_stats
        p_values = 2 * (1 - sp_stats.t.cdf(np.abs(t_stats), df=max(n - k, 1)))

        # Build results
        factor_names = ["alpha"] + list(factor_df.columns)
        exposures = []
        for i, name in enumerate(factor_names):
            exposures.append(FactorExposure(
                factor_name=name,
                beta=float(beta[i]),
                t_stat=float(t_stats[i]) if not np.isnan(t_stats[i]) else 0.0,
                p_value=float(p_values[i]) if not np.isnan(p_values[i]) else 1.0,
            ))

        # Annualized alpha
        alpha_daily = float(beta[0])
        alpha_annual = alpha_daily * 252

        residual_vol = float(np.std(resid) * np.sqrt(252))

        return {
            "exposures": [e.to_dict() for e in exposures],
            "r_squared": round(r_squared, 4),
            "alpha_annualized": round(alpha_annual, 4),
            "residual_vol": round(residual_vol, 4),
            "n_observations": n,
            "warnings": warnings,
        }
