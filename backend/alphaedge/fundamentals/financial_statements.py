"""Financial statement normalization and analysis."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from alphaedge.data_ingestion.yfinance_source import YFinanceSource
from alphaedge.data_ingestion.edgar_source import EdgarSource

logger = logging.getLogger(__name__)


def _safe_float(val: Any) -> Optional[float]:
    """Convert to Python float or None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


class FinancialStatements:
    """Normalize and analyze financial statements from yfinance + EDGAR."""

    def __init__(self, yf_source: YFinanceSource, edgar_source: EdgarSource):
        self.yf = yf_source
        self.edgar = edgar_source

    def analyze(self, ticker: str) -> dict:
        """Fetch and analyze financial statements."""
        warnings: list[str] = []
        attribution: list[dict] = []

        # Fetch from yfinance
        fin_result = self.yf.get_financials(ticker)
        attribution.append(fin_result.attribution.to_dict())
        warnings.extend(fin_result.warnings)

        info_result = self.yf.get_company_info(ticker)
        info = info_result.data or {}
        warnings.extend(info_result.warnings)

        fin = fin_result.data or {}
        income = fin.get("income_stmt")
        balance = fin.get("balance_sheet")
        cashflow = fin.get("cashflow")

        # Supplement from EDGAR
        try:
            edgar_result = self.edgar.get_company_facts(ticker)
            edgar_data = edgar_result.data or {}
            attribution.append(edgar_result.attribution.to_dict())
        except Exception as e:
            logger.debug("EDGAR data unavailable for %s: %s", ticker, e)
            edgar_data = {}

        income_summary = self._analyze_income(income, info)
        balance_summary = self._analyze_balance(balance, info)
        cashflow_summary = self._analyze_cashflow(cashflow, income, info)
        quality = self._quality_scores(income, cashflow, balance)

        return {
            "income_summary": income_summary,
            "balance_summary": balance_summary,
            "cashflow_summary": cashflow_summary,
            "quality_scores": quality,
            "warnings": warnings,
            "attribution": attribution,
        }

    def _analyze_income(self, income: Optional[pd.DataFrame], info: dict) -> dict:
        """Analyze income statement."""
        result: dict[str, Any] = {}
        if income is None or income.empty:
            return result

        latest = income.iloc[:, 0] if len(income.columns) > 0 else pd.Series()

        def _get(row_name: str) -> Optional[float]:
            for name in [row_name, row_name.replace(" ", "")]:
                if name in latest.index:
                    return _safe_float(latest[name])
            return None

        revenue = _get("Total Revenue") or _get("TotalRevenue")
        gross_profit = _get("Gross Profit") or _get("GrossProfit")
        operating_income = _get("Operating Income") or _get("OperatingIncome") or _get("EBIT")
        net_income = _get("Net Income") or _get("NetIncome")
        ebitda = _get("EBITDA") or _get("Normalized EBITDA")

        result["revenue"] = revenue
        result["gross_profit"] = gross_profit
        result["operating_income"] = operating_income
        result["net_income"] = net_income
        result["ebitda"] = ebitda
        result["gross_margin"] = _safe_div(gross_profit, revenue)
        result["operating_margin"] = _safe_div(operating_income, revenue)
        result["net_margin"] = _safe_div(net_income, revenue)

        # Revenue growth YoY
        if income.shape[1] >= 2:
            rev_cols = []
            for col_idx in range(min(income.shape[1], 4)):
                col = income.iloc[:, col_idx]
                r = None
                for name in ["Total Revenue", "TotalRevenue"]:
                    if name in col.index:
                        r = _safe_float(col[name])
                        break
                rev_cols.append(r)
            if rev_cols[0] is not None and rev_cols[1] is not None and rev_cols[1] != 0:
                result["revenue_growth_yoy"] = (rev_cols[0] - rev_cols[1]) / abs(rev_cols[1])
            else:
                result["revenue_growth_yoy"] = None
        else:
            result["revenue_growth_yoy"] = None

        # EPS from info
        result["eps"] = _safe_float(info.get("trailingEps"))

        return result

    def _analyze_balance(self, balance: Optional[pd.DataFrame], info: dict) -> dict:
        result: dict[str, Any] = {}
        if balance is None or balance.empty:
            return result

        latest = balance.iloc[:, 0] if len(balance.columns) > 0 else pd.Series()

        def _get(row_name: str) -> Optional[float]:
            for name in [row_name, row_name.replace(" ", "")]:
                if name in latest.index:
                    return _safe_float(latest[name])
            return None

        total_assets = _get("Total Assets") or _get("TotalAssets")
        total_debt = _get("Total Debt") or _get("TotalDebt") or _get("Long Term Debt")
        cash = _get("Cash And Cash Equivalents") or _get("Cash Cash Equivalents And Short Term Investments")
        current_assets = _get("Current Assets") or _get("CurrentAssets")
        current_liabilities = _get("Current Liabilities") or _get("CurrentLiabilities")
        total_equity = _get("Stockholders Equity") or _get("StockholdersEquity") or _get("Total Equity Gross Minority Interest")
        shares = _safe_float(info.get("sharesOutstanding"))

        result["total_assets"] = total_assets
        result["total_debt"] = total_debt
        result["cash"] = cash
        result["net_debt"] = (total_debt - cash) if total_debt and cash else None
        result["current_ratio"] = _safe_div(current_assets, current_liabilities)
        result["debt_to_equity"] = _safe_div(total_debt, total_equity)
        result["total_equity"] = total_equity
        result["book_value_per_share"] = _safe_div(total_equity, shares)

        return result

    def _analyze_cashflow(self, cashflow: Optional[pd.DataFrame], income: Optional[pd.DataFrame], info: dict) -> dict:
        result: dict[str, Any] = {}
        if cashflow is None or cashflow.empty:
            return result

        latest = cashflow.iloc[:, 0] if len(cashflow.columns) > 0 else pd.Series()

        def _get(row_name: str) -> Optional[float]:
            for name in [row_name, row_name.replace(" ", "")]:
                if name in latest.index:
                    return _safe_float(latest[name])
            return None

        operating_cf = _get("Operating Cash Flow") or _get("Cash Flow From Continuing Operating Activities")
        capex = _get("Capital Expenditure") or _get("CapitalExpenditure")
        free_cf = _get("Free Cash Flow") or _get("FreeCashFlow")
        if free_cf is None and operating_cf is not None and capex is not None:
            free_cf = operating_cf + capex  # capex is typically negative

        revenue = None
        if income is not None and not income.empty:
            col = income.iloc[:, 0]
            for name in ["Total Revenue", "TotalRevenue"]:
                if name in col.index:
                    revenue = _safe_float(col[name])
                    break

        market_cap = _safe_float(info.get("marketCap"))

        result["operating_cf"] = operating_cf
        result["capex"] = capex
        result["free_cf"] = free_cf
        result["fcf_margin"] = _safe_div(free_cf, revenue)
        result["fcf_yield"] = _safe_div(free_cf, market_cap)

        return result

    def _quality_scores(self, income: Optional[pd.DataFrame], cashflow: Optional[pd.DataFrame], balance: Optional[pd.DataFrame]) -> dict:
        result: dict[str, Any] = {}

        # Accruals ratio = (Net Income - Operating CF) / Total Assets
        net_income = None
        operating_cf = None
        total_assets = None

        if income is not None and not income.empty:
            col = income.iloc[:, 0]
            for name in ["Net Income", "NetIncome"]:
                if name in col.index:
                    net_income = _safe_float(col[name])
                    break

        if cashflow is not None and not cashflow.empty:
            col = cashflow.iloc[:, 0]
            for name in ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]:
                if name in col.index:
                    operating_cf = _safe_float(col[name])
                    break

        if balance is not None and not balance.empty:
            col = balance.iloc[:, 0]
            for name in ["Total Assets", "TotalAssets"]:
                if name in col.index:
                    total_assets = _safe_float(col[name])
                    break

        if net_income is not None and operating_cf is not None and total_assets and total_assets != 0:
            result["accruals_ratio"] = (net_income - operating_cf) / abs(total_assets)
        else:
            result["accruals_ratio"] = None

        # Revenue consistency (std of YoY growth rates)
        if income is not None and income.shape[1] >= 3:
            rev_vals = []
            for col_idx in range(min(income.shape[1], 5)):
                col = income.iloc[:, col_idx]
                for name in ["Total Revenue", "TotalRevenue"]:
                    if name in col.index:
                        v = _safe_float(col[name])
                        if v is not None:
                            rev_vals.append(v)
                        break
            if len(rev_vals) >= 3:
                growths = []
                for i in range(len(rev_vals) - 1):
                    if rev_vals[i + 1] and rev_vals[i + 1] != 0:
                        growths.append((rev_vals[i] - rev_vals[i + 1]) / abs(rev_vals[i + 1]))
                if growths:
                    result["revenue_growth_std"] = float(np.std(growths))
                else:
                    result["revenue_growth_std"] = None
            else:
                result["revenue_growth_std"] = None
        else:
            result["revenue_growth_std"] = None

        return result
