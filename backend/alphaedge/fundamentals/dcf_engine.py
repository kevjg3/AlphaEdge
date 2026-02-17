"""Discounted Cash Flow (DCF) valuation engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from alphaedge.data_ingestion.yfinance_source import YFinanceSource

logger = logging.getLogger(__name__)


def _sf(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        v = float(val)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


@dataclass
class DCFAssumptions:
    # Conservative defaults: GDP-like growth, average S&P 500 margins
    revenue_growth_rates: list[float] = field(default_factory=lambda: [0.04, 0.035, 0.03, 0.025, 0.02])
    terminal_growth_rate: float = 0.02
    operating_margin: float = 0.10
    tax_rate: float = 0.21
    capex_pct_revenue: float = 0.07
    nwc_pct_revenue: float = 0.08
    depreciation_pct_revenue: float = 0.035
    wacc: Optional[float] = None
    projection_years: int = 5
    shares_outstanding: float = 1.0

    def to_dict(self) -> dict:
        return {
            "revenue_growth_rates": [round(r, 4) for r in self.revenue_growth_rates],
            "terminal_growth_rate": self.terminal_growth_rate,
            "operating_margin": self.operating_margin,
            "tax_rate": self.tax_rate,
            "capex_pct_revenue": self.capex_pct_revenue,
            "nwc_pct_revenue": self.nwc_pct_revenue,
            "depreciation_pct_revenue": self.depreciation_pct_revenue,
            "wacc": self.wacc,
            "projection_years": self.projection_years,
            "shares_outstanding": self.shares_outstanding,
        }


class DCFEngine:
    """Discounted Cash Flow model with sensitivity analysis."""

    def __init__(self, yf_source: YFinanceSource, risk_free_rate: float = 0.04):
        self.yf = yf_source
        self.rf = risk_free_rate

    def estimate_wacc(
        self,
        info: dict,
        beta: float,
        equity_risk_premium: float = 0.05,
    ) -> float:
        """Estimate WACC via CAPM."""
        # Cost of equity = Rf + Beta * ERP
        cost_of_equity = self.rf + beta * equity_risk_premium

        # Cost of debt (approximate from interest expense / total debt)
        interest_expense = _sf(info.get("interest_expense"))
        total_debt = _sf(info.get("total_debt")) or 0
        cost_of_debt = 0.05  # default
        if interest_expense and total_debt and total_debt > 0:
            cost_of_debt = abs(interest_expense) / total_debt
            cost_of_debt = min(max(cost_of_debt, 0.01), 0.15)  # bound

        # Weights
        market_cap = _sf(info.get("marketCap")) or _sf(info.get("market_cap")) or 0
        total_value = market_cap + total_debt
        if total_value == 0:
            return cost_of_equity

        we = market_cap / total_value
        wd = total_debt / total_value
        tax_rate = 0.21

        wacc = we * cost_of_equity + wd * cost_of_debt * (1 - tax_rate)
        return max(wacc, 0.04)  # floor at 4%

    def run_dcf(
        self,
        ticker: str,
        assumptions: DCFAssumptions | None = None,
    ) -> dict:
        """Run full DCF valuation."""
        warnings: list[str] = []
        attribution: list[dict] = []

        # Fetch data
        info_result = self.yf.get_company_info(ticker)
        info = info_result.data or {}
        attribution.append(info_result.attribution.to_dict())
        warnings.extend(info_result.warnings)

        fin_result = self.yf.get_financials(ticker)
        fin = fin_result.data or {}
        warnings.extend(fin_result.warnings)

        spot_result = self.yf.get_spot(ticker)
        current_price = spot_result.data

        # Derive assumptions if not provided
        if assumptions is None:
            assumptions = self._derive_assumptions(info, fin)

        # Estimate WACC if not set
        beta = _sf(info.get("beta")) or 1.0
        if assumptions.wacc is None:
            assumptions.wacc = self.estimate_wacc(info, beta)

        shares = _sf(info.get("sharesOutstanding")) or _sf(info.get("shares_outstanding")) or assumptions.shares_outstanding
        assumptions.shares_outstanding = shares

        # Get base revenue
        base_revenue = _sf(info.get("total_revenue")) or _sf(info.get("totalRevenue"))
        if base_revenue is None:
            income = fin.get("income_stmt")
            if income is not None and not income.empty:
                col = income.iloc[:, 0]
                for name in ["Total Revenue", "TotalRevenue"]:
                    if name in col.index:
                        base_revenue = _sf(col[name])
                        break

        if base_revenue is None or base_revenue <= 0:
            return {
                "warnings": warnings + ["No revenue data — cannot run DCF"],
                "attribution": attribution,
            }

        # Project FCFs
        projected = self._project_fcf(base_revenue, assumptions)

        # Discount
        wacc = assumptions.wacc
        discount_factors = [(1 + wacc) ** (i + 1) for i in range(assumptions.projection_years)]
        pv_fcfs = [fcf["fcf"] / df for fcf, df in zip(projected, discount_factors)]
        sum_pv_fcf = sum(pv_fcfs)

        # Terminal value (Gordon Growth)
        terminal_fcf = projected[-1]["fcf"] * (1 + assumptions.terminal_growth_rate)
        terminal_value = terminal_fcf / (wacc - assumptions.terminal_growth_rate)
        pv_terminal = terminal_value / discount_factors[-1]

        enterprise_value = sum_pv_fcf + pv_terminal

        # Equity value
        total_debt = _sf(info.get("total_debt")) or _sf(info.get("totalDebt")) or 0
        cash = _sf(info.get("total_cash")) or _sf(info.get("totalCash")) or 0
        equity_value = enterprise_value - total_debt + cash
        implied_price = equity_value / shares if shares > 0 else None

        # Sensitivity grid
        sensitivity = self._sensitivity_grid(
            projected, assumptions, total_debt, cash, shares,
        )

        upside = None
        if implied_price and current_price and current_price > 0:
            upside = round((implied_price / current_price - 1) * 100, 2)

        return {
            "assumptions_used": assumptions.to_dict(),
            "base_revenue": round(base_revenue, 0),
            "projected_fcf": [
                {k: round(v, 0) if isinstance(v, (int, float)) else v for k, v in p.items()}
                for p in projected
            ],
            "sum_pv_fcf": round(sum_pv_fcf, 0),
            "terminal_value": round(terminal_value, 0),
            "pv_terminal": round(pv_terminal, 0),
            "enterprise_value": round(enterprise_value, 0),
            "equity_value": round(equity_value, 0),
            "implied_price": round(implied_price, 2) if implied_price else None,
            "current_price": current_price,
            "upside_downside_pct": upside,
            "sensitivity_grid": sensitivity,
            "warnings": warnings,
            "attribution": attribution,
        }

    def _derive_assumptions(self, info: dict, fin: dict) -> DCFAssumptions:
        """Auto-derive assumptions from historical data with conservative adjustments."""
        a = DCFAssumptions()

        sector = (info.get("sector", "") or "").lower()

        # Sector-aware capex and NWC defaults
        _SECTOR_CAPEX = {
            "technology": 0.05, "communication services": 0.06,
            "healthcare": 0.06, "financial services": 0.03,
            "consumer cyclical": 0.07, "consumer defensive": 0.06,
            "industrials": 0.08, "basic materials": 0.09,
            "energy": 0.12, "utilities": 0.14, "real estate": 0.05,
        }
        _SECTOR_NWC = {
            "technology": 0.06, "communication services": 0.05,
            "healthcare": 0.10, "financial services": 0.03,
            "consumer cyclical": 0.12, "consumer defensive": 0.08,
            "industrials": 0.14, "basic materials": 0.12,
            "energy": 0.10, "utilities": 0.05, "real estate": 0.04,
        }
        a.capex_pct_revenue = _SECTOR_CAPEX.get(sector, 0.07)
        a.nwc_pct_revenue = _SECTOR_NWC.get(sector, 0.12)

        # Revenue growth — use actual growth but apply conservative decay
        rev_growth = _sf(info.get("revenue_growth")) or _sf(info.get("revenueGrowth"))
        if rev_growth is not None:
            # Cap at 20% and floor at -5%; decay 25% per year toward terminal
            base = min(max(rev_growth, -0.05), 0.20)
            terminal = a.terminal_growth_rate
            a.revenue_growth_rates = [
                round(base + (terminal - base) * (i / a.projection_years), 4)
                for i in range(a.projection_years)
            ]

        # Operating margin — use actual with light mean-reversion toward sector
        op_margin = _sf(info.get("operating_margins")) or _sf(info.get("operatingMargins"))
        if op_margin is not None:
            # Sector median operating margins (sourced from S&P 500 sector averages)
            sector_avg = {
                "technology": 0.25, "communication services": 0.20,
                "healthcare": 0.15, "financial services": 0.30,
                "consumer cyclical": 0.10, "consumer defensive": 0.15,
                "industrials": 0.12, "basic materials": 0.12,
                "energy": 0.10, "utilities": 0.18, "real estate": 0.30,
            }.get(sector, 0.12)
            # Blend 85% actual + 15% sector average (light mean reversion)
            blended = 0.85 * op_margin + 0.15 * sector_avg
            a.operating_margin = min(max(blended, 0.02), 0.50)
        else:
            a.operating_margin = 0.12  # conservative fallback

        # Shares
        shares = _sf(info.get("sharesOutstanding")) or _sf(info.get("shares_outstanding"))
        if shares:
            a.shares_outstanding = shares

        return a

    def _project_fcf(self, base_revenue: float, a: DCFAssumptions) -> list[dict]:
        """Project Free Cash Flow for each year."""
        projected = []
        revenue = base_revenue

        for i in range(a.projection_years):
            growth = a.revenue_growth_rates[i] if i < len(a.revenue_growth_rates) else a.revenue_growth_rates[-1]
            revenue = revenue * (1 + growth)
            ebit = revenue * a.operating_margin
            nopat = ebit * (1 - a.tax_rate)
            depreciation = revenue * a.depreciation_pct_revenue
            capex = revenue * a.capex_pct_revenue
            delta_nwc = revenue * a.nwc_pct_revenue * growth  # change in NWC

            fcf = nopat + depreciation - capex - delta_nwc

            projected.append({
                "year": i + 1,
                "revenue": revenue,
                "ebit": ebit,
                "nopat": nopat,
                "depreciation": depreciation,
                "capex": capex,
                "delta_nwc": delta_nwc,
                "fcf": fcf,
            })

        return projected

    def _sensitivity_grid(
        self,
        projected: list[dict],
        assumptions: DCFAssumptions,
        total_debt: float,
        cash: float,
        shares: float,
    ) -> dict:
        """Build sensitivity table varying WACC and terminal growth."""
        wacc_base = assumptions.wacc
        tg_base = assumptions.terminal_growth_rate

        wacc_range = [round(wacc_base + delta, 4) for delta in [-0.02, -0.01, 0, 0.01, 0.02]]
        tg_range = [round(tg_base + delta, 4) for delta in [-0.01, -0.005, 0, 0.005, 0.01]]

        grid: dict[str, dict[str, float | None]] = {}

        for wacc in wacc_range:
            wacc_key = f"{wacc:.2%}"
            grid[wacc_key] = {}
            for tg in tg_range:
                tg_key = f"{tg:.2%}"
                if wacc <= tg:
                    grid[wacc_key][tg_key] = None  # invalid
                    continue

                discount_factors = [(1 + wacc) ** (i + 1) for i in range(len(projected))]
                pv_fcfs = sum(p["fcf"] / df for p, df in zip(projected, discount_factors))

                terminal_fcf = projected[-1]["fcf"] * (1 + tg)
                tv = terminal_fcf / (wacc - tg)
                pv_tv = tv / discount_factors[-1]

                ev = pv_fcfs + pv_tv
                eq = ev - total_debt + cash
                price = eq / shares if shares > 0 else None
                grid[wacc_key][tg_key] = round(price, 2) if price else None

        return {
            "wacc_values": [f"{w:.2%}" for w in wacc_range],
            "terminal_growth_values": [f"{t:.2%}" for t in tg_range],
            "implied_prices": grid,
        }
