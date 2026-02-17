"""Scenario analysis: historical stress tests and custom Monte Carlo stress scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaedge.data_ingestion.base import DataSourceName, SourceAttribution


HISTORICAL_SCENARIOS = {
    "gfc_2008": {
        "start": "2008-09-15",
        "end": "2009-03-09",
        "description": "Global Financial Crisis",
        "spx_return": -0.46,
    },
    "covid_2020": {
        "start": "2020-02-19",
        "end": "2020-03-23",
        "description": "COVID-19 Crash",
        "spx_return": -0.34,
    },
    "dot_com_2000": {
        "start": "2000-03-10",
        "end": "2002-10-09",
        "description": "Dot-Com Bubble Burst",
        "spx_return": -0.49,
    },
    "rate_hike_2022": {
        "start": "2022-01-03",
        "end": "2022-10-12",
        "description": "2022 Rate Hiking Cycle",
        "spx_return": -0.25,
    },
    "flash_crash_2010": {
        "start": "2010-05-06",
        "end": "2010-05-07",
        "description": "2010 Flash Crash",
        "spx_return": -0.07,
    },
}


@dataclass
class ScenarioResult:
    """Result of a single scenario analysis."""
    name: str
    description: str
    period: str  # "2008-09-15 to 2009-03-09"
    stock_return: float | None
    benchmark_return: float | None
    beta_adjusted_return: float | None
    max_drawdown: float | None
    data_available: bool = True
    warning: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "period": self.period,
            "stock_return": round(self.stock_return, 4) if self.stock_return is not None else None,
            "benchmark_return": round(self.benchmark_return, 4) if self.benchmark_return is not None else None,
            "beta_adjusted_return": round(self.beta_adjusted_return, 4) if self.beta_adjusted_return is not None else None,
            "max_drawdown": round(self.max_drawdown, 4) if self.max_drawdown is not None else None,
            "data_available": self.data_available,
            "warning": self.warning,
        }


@dataclass
class CustomStressResult:
    """Result of a custom Monte Carlo stress test."""
    shock_pct: float
    vol_multiplier: float
    horizon_days: int
    mean_return: float
    median_return: float
    worst_5pct: float  # 5th percentile
    best_5pct: float  # 95th percentile
    prob_loss: float

    def to_dict(self) -> dict:
        return {
            "shock_pct": self.shock_pct,
            "vol_multiplier": self.vol_multiplier,
            "horizon_days": self.horizon_days,
            "mean_return": round(self.mean_return, 4),
            "median_return": round(self.median_return, 4),
            "worst_5pct": round(self.worst_5pct, 4),
            "best_5pct": round(self.best_5pct, 4),
            "prob_loss": round(self.prob_loss, 4),
        }


class ScenarioAnalyzer:
    """Historical and custom stress test analysis."""

    def __init__(self, yf_source=None):
        self._yf_source = yf_source

    def stress_test_historical(
        self, ticker: str, beta: float = 1.0,
    ) -> list[ScenarioResult]:
        """Compute how a ticker would have performed in historical scenarios.

        If historical data unavailable for the period, use beta-adjusted
        benchmark return as a proxy.
        """
        results: list[ScenarioResult] = []

        for name, scenario in HISTORICAL_SCENARIOS.items():
            start = scenario["start"]
            end = scenario["end"]
            desc = scenario["description"]
            spx_ret = scenario["spx_return"]

            stock_ret = None
            max_dd = None
            data_avail = False
            warning = ""

            # Try to fetch actual historical data
            if self._yf_source:
                try:
                    import yfinance as yf
                    hist = yf.Ticker(ticker).history(start=start, end=end)
                    if not hist.empty and len(hist) > 1:
                        stock_ret = float(hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1)
                        running_max = hist["Close"].expanding().max()
                        dd = (hist["Close"] - running_max) / running_max
                        max_dd = float(dd.min())
                        data_avail = True
                except Exception:
                    warning = f"Historical data unavailable for {ticker} during {desc}"

            # Beta-adjusted proxy
            beta_adj = spx_ret * beta if beta else spx_ret

            results.append(ScenarioResult(
                name=name,
                description=desc,
                period=f"{start} to {end}",
                stock_return=stock_ret,
                benchmark_return=spx_ret,
                beta_adjusted_return=beta_adj,
                max_drawdown=max_dd,
                data_available=data_avail,
                warning=warning,
            ))

        return results

    def custom_stress_test(
        self,
        current_price: float,
        daily_vol: float,
        shock_pct: float = -0.10,
        vol_multiplier: float = 2.0,
        horizon_days: int = 21,
        n_paths: int = 50_000,
        seed: int = 42,
    ) -> CustomStressResult:
        """Monte Carlo stress test with custom shock + elevated volatility."""
        rng = np.random.default_rng(seed)

        # Apply initial shock
        shocked_price = current_price * (1 + shock_pct)

        # Simulate from shocked price with elevated volatility
        stressed_vol = daily_vol * vol_multiplier
        dt = 1.0
        Z = rng.standard_normal((n_paths, horizon_days))
        log_returns = -0.5 * stressed_vol**2 * dt + stressed_vol * np.sqrt(dt) * Z
        cum_returns = np.sum(log_returns, axis=1)
        final_prices = shocked_price * np.exp(cum_returns)

        # Total return from original price
        total_returns = final_prices / current_price - 1

        return CustomStressResult(
            shock_pct=shock_pct,
            vol_multiplier=vol_multiplier,
            horizon_days=horizon_days,
            mean_return=float(np.mean(total_returns)),
            median_return=float(np.median(total_returns)),
            worst_5pct=float(np.percentile(total_returns, 5)),
            best_5pct=float(np.percentile(total_returns, 95)),
            prob_loss=float(np.mean(total_returns < 0)),
        )
