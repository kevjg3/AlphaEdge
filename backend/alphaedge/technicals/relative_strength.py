"""Relative strength analysis — compare stock performance to sector ETF."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
}


class RelativeStrengthAnalyzer:
    """Compare stock's rolling performance to its sector ETF."""

    def analyze(
        self,
        ticker: str,
        prices: pd.Series,
        yf_source,
        sector: str | None = None,
    ) -> dict:
        """Compute relative strength metrics vs sector ETF.

        Args:
            ticker: Stock ticker symbol
            prices: Stock price series (Close)
            yf_source: YFinanceSource instance for fetching sector ETF data
            sector: Sector name from yfinance info
        """
        if not sector or sector not in SECTOR_ETFS:
            return {}

        etf_ticker = SECTOR_ETFS[sector]

        # Fetch sector ETF prices
        try:
            lookback_years = max(1, len(prices) // 252)
            etf_result = yf_source.get_history(etf_ticker, period=f"{lookback_years}y")
            if not etf_result.success or etf_result.data.empty:
                return {}
            etf_prices = etf_result.data["Close"]
        except Exception as e:
            logger.warning("Failed to fetch sector ETF %s: %s", etf_ticker, e)
            return {}

        # Align on common dates
        common_idx = prices.index.intersection(etf_prices.index)
        if len(common_idx) < 21:
            return {}

        stock = prices.reindex(common_idx)
        sector_px = etf_prices.reindex(common_idx)

        # Relative returns at different horizons
        n = len(stock)
        result: dict = {
            "sector": sector,
            "sector_etf": etf_ticker,
        }

        # 1M relative return
        if n >= 21:
            stock_1m = float(stock.iloc[-1] / stock.iloc[-21] - 1)
            sector_1m = float(sector_px.iloc[-1] / sector_px.iloc[-21] - 1)
            result["relative_return_1m"] = round(stock_1m - sector_1m, 4)

        # 3M relative return
        if n >= 63:
            stock_3m = float(stock.iloc[-1] / stock.iloc[-63] - 1)
            sector_3m = float(sector_px.iloc[-1] / sector_px.iloc[-63] - 1)
            result["relative_return_3m"] = round(stock_3m - sector_3m, 4)

        # 6M relative return
        if n >= 126:
            stock_6m = float(stock.iloc[-1] / stock.iloc[-126] - 1)
            sector_6m = float(sector_px.iloc[-1] / sector_px.iloc[-126] - 1)
            result["relative_return_6m"] = round(stock_6m - sector_6m, 4)

        # RS ratio (current price relative to sector)
        stock_ret = stock.pct_change().dropna()
        sector_ret = sector_px.pct_change().dropna()
        rs_line = (1 + stock_ret).cumprod() / (1 + sector_ret).cumprod()
        result["rs_ratio"] = round(float(rs_line.iloc[-1]), 4)

        # Outperformance streak: consecutive days stock > sector daily return
        relative_daily = stock_ret - sector_ret
        streak = 0
        for val in reversed(relative_daily.values):
            if val > 0:
                streak += 1
            else:
                break
        result["outperformance_streak_days"] = streak

        # RS trend
        if n >= 21:
            rs_21d_ago = float(rs_line.iloc[-21]) if len(rs_line) >= 21 else float(rs_line.iloc[0])
            rs_now = float(rs_line.iloc[-1])
            if rs_now > rs_21d_ago * 1.005:
                result["rs_trend"] = "outperforming"
            elif rs_now < rs_21d_ago * 0.995:
                result["rs_trend"] = "underperforming"
            else:
                result["rs_trend"] = "inline"
        else:
            result["rs_trend"] = "inline"

        return result
