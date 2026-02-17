"""YFinance data source — wraps yfinance for pricing, fundamentals, and company info."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from diskcache import Cache

from alphaedge.config import settings
from alphaedge.data_ingestion.base import (
    DataSource, DataSourceName, FetchResult, SourceAttribution,
)


# Curated sector → ETF/representative peers mapping for peer selection
_SECTOR_PEERS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "ADBE", "CRM", "ORCL", "INTC", "AMD", "AVGO", "CSCO", "IBM", "NOW", "SHOP", "SQ", "PYPL", "NFLX", "UBER", "ABNB"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY", "AMGN", "GILD", "ISRG", "CVS", "CI", "HCA", "ZTS", "VRTX", "REGN", "MRNA"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB", "PNC", "TFC", "COF", "CME", "ICE", "SPGI", "MCO", "AON", "MMC", "MET"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TGT", "LOW", "TJX", "ROST", "DHI", "LEN", "GM", "F", "BKNG", "MAR", "HLT", "CMG", "DPZ", "YUM"],
    "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST", "CL", "MDLZ", "GIS", "K", "KHC", "HSY", "MO", "PM", "EL", "CLX", "KMB", "SJM", "CAG", "CPB", "HRL"],
    "Industrials": ["HON", "UNP", "UPS", "RTX", "BA", "CAT", "DE", "LMT", "GD", "NOC", "GE", "MMM", "EMR", "ITW", "ETN", "PH", "ROK", "SWK", "IR", "WM"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "PXD", "HES", "DVN", "FANG", "HAL", "BKR", "KMI", "WMB", "OKE", "LNG", "ET"],
    "Communication Services": ["GOOGL", "META", "DIS", "CMCSA", "NFLX", "T", "VZ", "TMUS", "CHTR", "EA", "ATVI", "TTWO", "RBLX", "SNAP", "PINS", "MTCH", "ZM", "SPOT", "LYV", "PARA"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "WEC", "ED", "PEG", "ES", "FE", "DTE", "ETR", "AEE", "CMS", "CNP", "ATO", "NI"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB", "EQR", "ARE", "MAA", "UDR", "VTR", "BXP", "SLG", "KIM", "REG", "HST"],
    "Basic Materials": ["LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "DOW", "NUE", "VMC", "MLM", "ALB", "FMC", "CE", "EMN", "PPG", "RPM", "IFF", "CTVA", "MOS"],
}


class YFinanceSource(DataSource):
    """Unified yfinance data source with caching."""

    def __init__(self, cache_ttl: int | None = None, cache_dir: str | None = None):
        self._cache_ttl = cache_ttl or settings.cache_ttl
        self._cache = Cache(cache_dir or settings.cache_dir)

    def is_available(self) -> bool:
        try:
            yf.Ticker("AAPL").fast_info
            return True
        except Exception:
            return False

    def _attr(self, url: str | None = None, cache_hit: bool = False) -> SourceAttribution:
        return SourceAttribution(
            source=DataSourceName.YFINANCE, url=url, cache_hit=cache_hit,
        )

    # --- Spot & History ---

    def get_spot(self, ticker: str) -> FetchResult:
        key = f"yf:spot:{ticker}"
        cached = self._cache.get(key)
        if cached is not None:
            return FetchResult(data=cached, attribution=self._attr(cache_hit=True))

        t = yf.Ticker(ticker)
        try:
            price = t.fast_info["last_price"]
        except (KeyError, AttributeError, TypeError):
            hist = t.history(period="1d")
            if hist.empty:
                return FetchResult(data=None, attribution=self._attr(),
                                   success=False, warnings=[f"No spot for {ticker}"])
            price = float(hist["Close"].iloc[-1])

        self._cache.set(key, float(price), expire=self._cache_ttl)
        return FetchResult(data=float(price), attribution=self._attr())

    def get_history(
        self, ticker: str, period: str = "2y", interval: str = "1d",
    ) -> FetchResult:
        key = f"yf:hist:{ticker}:{period}:{interval}"
        cached = self._cache.get(key)
        if cached is not None:
            return FetchResult(data=cached, attribution=self._attr(cache_hit=True))

        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return FetchResult(data=pd.DataFrame(), attribution=self._attr(),
                               success=False, warnings=[f"No history for {ticker}"])
        self._cache.set(key, df, expire=self._cache_ttl)
        return FetchResult(data=df, attribution=self._attr())

    # --- Company Info ---

    def get_company_info(self, ticker: str) -> FetchResult:
        key = f"yf:info:{ticker}"
        cached = self._cache.get(key)
        if cached is not None:
            return FetchResult(data=cached, attribution=self._attr(cache_hit=True))

        t = yf.Ticker(ticker)
        ws: list[str] = []
        try:
            info = dict(t.info)
        except Exception as e:
            return FetchResult(data={}, attribution=self._attr(),
                               success=False, warnings=[f"Info fetch failed: {e}"])

        result = {
            # ── Identity ──
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "description": info.get("longBusinessSummary", ""),
            "employees": info.get("fullTimeEmployees"),
            "country": info.get("country", ""),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", ""),
            "website": info.get("website", ""),

            # ── Market data ──
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales_trailing_12_months": info.get("priceToSalesTrailing12Months"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
            "shares_outstanding": info.get("sharesOutstanding"),

            # ── Financials (needed by DCF + comps) ──
            "total_revenue": info.get("totalRevenue"),
            "revenue_growth": info.get("revenueGrowth"),
            "gross_margins": info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins"),
            "profit_margins": info.get("profitMargins"),
            "ebitda": info.get("ebitda"),
            "total_debt": info.get("totalDebt"),
            "total_cash": info.get("totalCash"),
            "return_on_equity": info.get("returnOnEquity"),
            "return_on_assets": info.get("returnOnAssets"),
            "current_ratio": info.get("currentRatio"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cashflow": info.get("freeCashflow"),
            "operating_cashflow": info.get("operatingCashflow"),
            "earnings_growth": info.get("earningsGrowth"),
            "revenue_per_share": info.get("revenuePerShare"),
        }

        # Quality checks
        if result["market_cap"] is None:
            ws.append("market_cap unavailable")
        if result["beta"] is None:
            ws.append("beta unavailable — will estimate from returns")

        self._cache.set(key, result, expire=self._cache_ttl)
        return FetchResult(data=result, attribution=self._attr(), warnings=ws)

    # --- Financial Statements ---

    def get_financials(self, ticker: str) -> FetchResult:
        key = f"yf:fin:{ticker}"
        cached = self._cache.get(key)
        if cached is not None:
            return FetchResult(data=cached, attribution=self._attr(cache_hit=True))

        t = yf.Ticker(ticker)
        ws: list[str] = []

        try:
            income = t.income_stmt
        except Exception:
            income = pd.DataFrame()
            ws.append("income statement unavailable")

        try:
            balance = t.balance_sheet
        except Exception:
            balance = pd.DataFrame()
            ws.append("balance sheet unavailable")

        try:
            cashflow = t.cashflow
        except Exception:
            cashflow = pd.DataFrame()
            ws.append("cash flow statement unavailable")

        try:
            q_income = t.quarterly_income_stmt
        except Exception:
            q_income = pd.DataFrame()

        result = {
            "income_stmt": income,
            "balance_sheet": balance,
            "cashflow": cashflow,
            "quarterly_income": q_income,
        }

        success = not (income.empty and balance.empty and cashflow.empty)
        self._cache.set(key, result, expire=self._cache_ttl)
        return FetchResult(data=result, attribution=self._attr(),
                           success=success, warnings=ws)

    # --- Institutional Holders ---

    def get_institutional_holders(self, ticker: str) -> FetchResult:
        t = yf.Ticker(ticker)
        try:
            holders = t.institutional_holders
            if holders is None or holders.empty:
                return FetchResult(data=pd.DataFrame(), attribution=self._attr(),
                                   warnings=["No institutional holder data"])
            return FetchResult(data=holders, attribution=self._attr())
        except Exception as e:
            return FetchResult(data=pd.DataFrame(), attribution=self._attr(),
                               success=False, warnings=[str(e)])

    # --- Risk-Free Rate ---

    def get_risk_free_rate(self, proxy: str = "^IRX", fallback: float = 0.045) -> FetchResult:
        try:
            t = yf.Ticker(proxy)
            hist = t.history(period="5d")
            if not hist.empty:
                rate = float(hist["Close"].iloc[-1]) / 100.0
                if 0 < rate < 0.20:
                    return FetchResult(data=rate, attribution=self._attr())
        except Exception:
            pass
        return FetchResult(
            data=fallback, attribution=self._attr(),
            warnings=[f"Using fallback rate {fallback}"],
        )

    # --- Dividend Yield ---

    def get_dividend_yield(self, ticker: str) -> FetchResult:
        t = yf.Ticker(ticker)
        try:
            info = t.info
            dy = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
            if dy is not None and dy > 0:
                dy = float(dy)
                if dy > 0.20:
                    return FetchResult(
                        data=0.0, attribution=self._attr(),
                        warnings=[f"Dividend yield {dy:.1%} unreasonable; defaulting to 0"],
                    )
                return FetchResult(data=dy, attribution=self._attr())
        except Exception:
            pass
        return FetchResult(data=0.0, attribution=self._attr(),
                           warnings=["No dividend yield data"])

    # --- Peer Selection ---

    def get_peers(self, ticker: str, n: int = 8) -> FetchResult:
        """Select comparable peers by sector + industry + market cap proximity."""
        info_res = self.get_company_info(ticker)
        if not info_res.success:
            return FetchResult(data=[], attribution=self._attr(),
                               success=False, warnings=["Cannot get info for peer selection"])

        info = info_res.data
        sector = info.get("sector", "")
        market_cap = info.get("market_cap") or 0
        ws: list[str] = []

        candidates = _SECTOR_PEERS.get(sector, [])
        candidates = [c for c in candidates if c != ticker.upper()]

        if not candidates:
            ws.append(f"No curated peers for sector '{sector}'")
            return FetchResult(data=[], attribution=self._attr(), warnings=ws)

        # Filter by market cap proximity (0.2x – 5x)
        peers_with_cap: list[tuple[str, float]] = []
        for c in candidates:
            try:
                ct = yf.Ticker(c)
                c_cap = ct.info.get("marketCap", 0) or 0
                if c_cap > 0 and market_cap > 0:
                    ratio = c_cap / market_cap
                    if 0.2 <= ratio <= 5.0:
                        peers_with_cap.append((c, abs(1 - ratio)))
            except Exception:
                continue

        if not peers_with_cap:
            # Fallback: just return top N from sector without cap filter
            ws.append("Market cap filtering yielded no peers; using sector list")
            return FetchResult(data=candidates[:n], attribution=self._attr(), warnings=ws)

        # Sort by closest market cap ratio to 1.0
        peers_with_cap.sort(key=lambda x: x[1])
        selected = [p[0] for p in peers_with_cap[:n]]
        return FetchResult(data=selected, attribution=self._attr(), warnings=ws)

    # --- Earnings & Calendar ---

    def get_earnings_dates(self, ticker: str) -> FetchResult:
        t = yf.Ticker(ticker)
        try:
            dates = t.earnings_dates
            if dates is not None and not dates.empty:
                return FetchResult(data=dates, attribution=self._attr())
        except Exception:
            pass
        return FetchResult(data=pd.DataFrame(), attribution=self._attr(),
                           warnings=["No earnings dates available"])

    def get_calendar(self, ticker: str) -> FetchResult:
        t = yf.Ticker(ticker)
        try:
            cal = t.calendar
            if cal is not None:
                return FetchResult(data=cal, attribution=self._attr())
        except Exception:
            pass
        return FetchResult(data={}, attribution=self._attr(),
                           warnings=["No calendar data available"])
