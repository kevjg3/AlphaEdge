"""Correlation analysis: cross-asset correlation matrix, rolling correlation, beta decomposition."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Benchmark ETFs for correlation analysis
BENCHMARK_TICKERS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "TLT": "20Y+ Treasury",
    "GLD": "Gold",
    "DIA": "Dow Jones",
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
}


class CorrelationAnalyzer:
    """Compute cross-asset correlations, rolling correlation, and beta decomposition."""

    def __init__(self, yf_source=None):
        self._yf = yf_source

    def analyze(
        self,
        ticker: str,
        prices: pd.Series,
        benchmarks: list[str] | None = None,
    ) -> dict:
        """Full correlation analysis."""
        if benchmarks is None:
            benchmarks = list(BENCHMARK_TICKERS.keys())

        result = {
            "correlation_matrix": {},
            "rolling_correlation": [],
            "beta_decomposition": {},
            "benchmark_labels": {},
        }

        if len(prices) < 60:
            return result

        # Fetch benchmark prices
        bench_data = self._fetch_benchmarks(benchmarks, len(prices))
        if not bench_data:
            return result

        # Align all returns
        stock_returns = np.log(prices / prices.shift(1)).dropna()
        stock_returns.name = ticker

        all_returns = pd.DataFrame({"ticker": stock_returns})
        for sym, series in bench_data.items():
            bench_ret = np.log(series / series.shift(1)).dropna()
            all_returns[sym] = bench_ret

        all_returns = all_returns.dropna()
        if len(all_returns) < 30:
            return result

        # ── Correlation matrix ──
        corr = all_returns.corr()
        matrix = {}
        for col in corr.columns:
            matrix[col] = {row: round(float(corr.loc[row, col]), 4) for row in corr.index}

        result["correlation_matrix"] = matrix
        result["benchmark_labels"] = {
            k: BENCHMARK_TICKERS.get(k, k) for k in bench_data.keys()
        }
        result["benchmark_labels"]["ticker"] = ticker

        # ── Rolling 63d correlation vs SPY ──
        if "SPY" in all_returns.columns:
            rolling_corr = all_returns["ticker"].rolling(63).corr(all_returns["SPY"])
            rolling_list = []
            rc = rolling_corr.dropna()
            if len(rc) > 250:
                step = max(1, len(rc) // 250)
                rc = rc.iloc[::step]
            for d, v in rc.items():
                if np.isfinite(v):
                    rolling_list.append({
                        "date": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
                        "value": round(float(v), 4),
                    })
            result["rolling_correlation"] = rolling_list

        # ── Beta decomposition (OLS vs SPY) ──
        if "SPY" in all_returns.columns:
            from numpy.linalg import lstsq

            X = all_returns["SPY"].values
            y = all_returns["ticker"].values
            X_mat = np.column_stack([np.ones(len(X)), X])
            coef, residuals, _, _ = lstsq(X_mat, y, rcond=None)

            alpha_daily = coef[0]
            beta = coef[1]
            y_hat = X_mat @ coef
            resid = y - y_hat

            # R-squared
            ss_res = np.sum(resid ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            result["beta_decomposition"] = {
                "beta": round(float(beta), 4),
                "alpha_daily": round(float(alpha_daily), 6),
                "alpha_annualized": round(float(alpha_daily * 252), 4),
                "r_squared": round(float(r_sq), 4),
                "residual_vol_annualized": round(float(np.std(resid) * np.sqrt(252)), 4),
                "tracking_error": round(float(np.std(y - X) * np.sqrt(252)), 4),
            }

        return result

    def _fetch_benchmarks(self, tickers: list[str], min_length: int) -> dict[str, pd.Series]:
        """Fetch benchmark price series via yfinance."""
        result = {}
        if not self._yf:
            return result

        for t in tickers[:10]:  # Cap at 10 benchmarks
            try:
                hist = self._yf.get_history(t, period="2y")
                if hist.success and not hist.data.empty:
                    close = hist.data["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    result[t] = close
            except Exception as e:
                logger.debug("Failed to fetch benchmark %s: %s", t, e)
                continue

        return result
