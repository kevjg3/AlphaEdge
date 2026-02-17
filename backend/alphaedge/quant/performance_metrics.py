"""Performance metrics: Sharpe, Sortino, Calmar, Information Ratio, win rates, period returns."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class PerformanceMetrics:
    """Full suite of performance metrics."""
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: Optional[float] = None

    # Volatility
    daily_vol: float = 0.0
    annualized_vol: float = 0.0
    downside_vol: float = 0.0

    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    period_returns: dict = field(default_factory=dict)  # 1M, 3M, 6M, YTD, 1Y, 2Y

    # Win rate analysis
    daily_win_rate: float = 0.0
    weekly_win_rate: float = 0.0
    monthly_win_rate: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    best_month: float = 0.0
    worst_month: float = 0.0
    avg_positive_day: float = 0.0
    avg_negative_day: float = 0.0
    profit_factor: float = 0.0

    # Tail risk
    skewness: float = 0.0
    kurtosis: float = 0.0

    def to_dict(self) -> dict:
        return {
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "information_ratio": round(self.information_ratio, 4) if self.information_ratio is not None else None,
            "daily_vol": round(self.daily_vol, 6),
            "annualized_vol": round(self.annualized_vol, 4),
            "downside_vol": round(self.downside_vol, 6),
            "total_return": round(self.total_return, 4),
            "annualized_return": round(self.annualized_return, 4),
            "period_returns": {k: round(v, 4) for k, v in self.period_returns.items()},
            "daily_win_rate": round(self.daily_win_rate, 4),
            "weekly_win_rate": round(self.weekly_win_rate, 4),
            "monthly_win_rate": round(self.monthly_win_rate, 4),
            "best_day": round(self.best_day, 4),
            "worst_day": round(self.worst_day, 4),
            "best_month": round(self.best_month, 4),
            "worst_month": round(self.worst_month, 4),
            "avg_positive_day": round(self.avg_positive_day, 6),
            "avg_negative_day": round(self.avg_negative_day, 6),
            "profit_factor": round(self.profit_factor, 4),
            "skewness": round(self.skewness, 4),
            "kurtosis": round(self.kurtosis, 4),
        }


class PerformanceAnalyzer:
    """Compute comprehensive performance metrics from a price series."""

    def __init__(self, risk_free_rate: float = 0.045):
        self.rf = risk_free_rate
        self.rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1

    def compute(
        self,
        prices: pd.Series,
        benchmark_prices: pd.Series | None = None,
    ) -> PerformanceMetrics:
        if len(prices) < 10:
            return PerformanceMetrics()

        log_returns = np.log(prices / prices.shift(1)).dropna()
        simple_returns = prices.pct_change().dropna()

        n_days = len(log_returns)
        n_years = n_days / TRADING_DAYS

        # ── Basic stats ──
        daily_vol = float(log_returns.std())
        ann_vol = daily_vol * np.sqrt(TRADING_DAYS)
        total_ret = float(prices.iloc[-1] / prices.iloc[0] - 1)
        ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

        # ── Sharpe ──
        excess = log_returns - self.rf_daily
        sharpe = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)) if excess.std() > 0 else 0.0

        # ── Sortino ──
        downside = log_returns[log_returns < 0]
        downside_vol = float(downside.std()) if len(downside) > 1 else daily_vol
        sortino = float(excess.mean() / downside_vol * np.sqrt(TRADING_DAYS)) if downside_vol > 0 else 0.0

        # ── Max drawdown for Calmar ──
        running_max = prices.expanding().max()
        underwater = (prices - running_max) / running_max
        max_dd = float(underwater.min())
        calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0

        # ── Information ratio (vs benchmark) ──
        info_ratio = None
        if benchmark_prices is not None and len(benchmark_prices) >= len(prices):
            try:
                bench_ret = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()
                aligned = pd.DataFrame({"stock": log_returns, "bench": bench_ret}).dropna()
                if len(aligned) > 20:
                    active_ret = aligned["stock"] - aligned["bench"]
                    te = float(active_ret.std()) * np.sqrt(TRADING_DAYS)
                    info_ratio = float(active_ret.mean() * TRADING_DAYS / te) if te > 0 else 0.0
            except Exception:
                pass

        # ── Period returns ──
        period_returns = self._period_returns(prices)

        # ── Win rates ──
        daily_wins = float((simple_returns > 0).mean())
        weekly_returns = simple_returns.resample("W").sum() if hasattr(simple_returns.index, "freq") or True else simple_returns
        try:
            weekly_ret = prices.resample("W-FRI").last().pct_change().dropna()
            weekly_wins = float((weekly_ret > 0).mean()) if len(weekly_ret) > 0 else 0.0
        except Exception:
            weekly_wins = 0.0

        try:
            monthly_ret = prices.resample("ME").last().pct_change().dropna()
            monthly_wins = float((monthly_ret > 0).mean()) if len(monthly_ret) > 0 else 0.0
            best_month = float(monthly_ret.max()) if len(monthly_ret) > 0 else 0.0
            worst_month = float(monthly_ret.min()) if len(monthly_ret) > 0 else 0.0
        except Exception:
            monthly_wins = best_month = worst_month = 0.0

        # ── Avg positive/negative day ──
        pos_days = simple_returns[simple_returns > 0]
        neg_days = simple_returns[simple_returns < 0]
        avg_pos = float(pos_days.mean()) if len(pos_days) > 0 else 0.0
        avg_neg = float(neg_days.mean()) if len(neg_days) > 0 else 0.0

        # Profit factor
        gross_profit = float(pos_days.sum()) if len(pos_days) > 0 else 0.0
        gross_loss = float(abs(neg_days.sum())) if len(neg_days) > 0 else 1.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # ── Tail stats ──
        skew = float(log_returns.skew())
        kurt = float(log_returns.kurtosis())

        return PerformanceMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            daily_vol=daily_vol,
            annualized_vol=ann_vol,
            downside_vol=downside_vol,
            total_return=total_ret,
            annualized_return=ann_ret,
            period_returns=period_returns,
            daily_win_rate=daily_wins,
            weekly_win_rate=weekly_wins,
            monthly_win_rate=monthly_wins,
            best_day=float(simple_returns.max()),
            worst_day=float(simple_returns.min()),
            best_month=best_month,
            worst_month=worst_month,
            avg_positive_day=avg_pos,
            avg_negative_day=avg_neg,
            profit_factor=profit_factor,
            skewness=skew,
            kurtosis=kurt,
        )

    def _period_returns(self, prices: pd.Series) -> dict:
        """Calculate returns for standard periods."""
        now_idx = len(prices) - 1
        results = {}
        periods = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504}

        for label, days in periods.items():
            if now_idx >= days:
                ret = float(prices.iloc[-1] / prices.iloc[-days] - 1)
                results[label] = ret

        # YTD
        try:
            year_start = prices.index[0].year
            current_year = prices.index[-1].year
            if current_year > year_start or True:
                ytd_mask = prices.index >= pd.Timestamp(f"{prices.index[-1].year}-01-01", tz=prices.index.tz)
                ytd_prices = prices[ytd_mask]
                if len(ytd_prices) > 1:
                    results["YTD"] = float(ytd_prices.iloc[-1] / ytd_prices.iloc[0] - 1)
        except Exception:
            pass

        return results
