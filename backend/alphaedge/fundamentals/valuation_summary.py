"""Combined valuation summary — ties together financials, comps, and DCF."""

from __future__ import annotations

import logging
from typing import Any

from alphaedge.data_ingestion.yfinance_source import YFinanceSource
from alphaedge.data_ingestion.edgar_source import EdgarSource
from alphaedge.fundamentals.financial_statements import FinancialStatements
from alphaedge.fundamentals.comps_engine import CompsEngine
from alphaedge.fundamentals.dcf_engine import DCFEngine

logger = logging.getLogger(__name__)


class ValuationSummary:
    """Orchestrate all fundamental valuation analyses."""

    def __init__(self, yf_source: YFinanceSource, edgar_source: EdgarSource):
        self.yf = yf_source
        self.fs = FinancialStatements(yf_source, edgar_source)
        self.comps = CompsEngine(yf_source)
        self.dcf = DCFEngine(yf_source)

    def full_valuation(self, ticker: str) -> dict:
        """Run all valuation analyses and combine into a summary."""
        warnings: list[str] = []
        attribution: list[dict] = []

        # Get shares outstanding for converting market cap → per-share price
        info_result = self.yf.get_company_info(ticker)
        info = info_result.data or {}
        shares = info.get("shares_outstanding") or info.get("sharesOutstanding")

        # Financial health
        financial_health: dict[str, Any] = {}
        try:
            financial_health = self.fs.analyze(ticker)
            warnings.extend(financial_health.pop("warnings", []))
            attribution.extend(financial_health.pop("attribution", []))
        except Exception as e:
            logger.warning("Financial statements analysis failed: %s", e)
            warnings.append(f"Financial statements analysis failed: {e}")

        # Comps
        comps_valuation: dict[str, Any] = {}
        try:
            comps_valuation = self.comps.run_comps(ticker)
            warnings.extend(comps_valuation.pop("warnings", []))
            attribution.extend(comps_valuation.pop("attribution", []))
        except Exception as e:
            logger.warning("Comps analysis failed: %s", e)
            warnings.append(f"Comps analysis failed: {e}")

        # DCF
        dcf_valuation: dict[str, Any] = {}
        try:
            dcf_valuation = self.dcf.run_dcf(ticker)
            warnings.extend(dcf_valuation.pop("warnings", []))
            attribution.extend(dcf_valuation.pop("attribution", []))
        except Exception as e:
            logger.warning("DCF analysis failed: %s", e)
            warnings.append(f"DCF analysis failed: {e}")

        # Combine valuation ranges (comps gives market cap, DCF gives per-share)
        combined_range = self._combine_ranges(comps_valuation, dcf_valuation, shares)

        # Verdict
        current_price = dcf_valuation.get("current_price")
        if current_price is None:
            spot = self.yf.get_spot(ticker)
            current_price = spot.data
        verdict = self._verdict(combined_range, current_price)

        return {
            "financial_health": financial_health,
            "comps_valuation": comps_valuation,
            "dcf_valuation": dcf_valuation,
            "combined_range": combined_range,
            "verdict": verdict,
            "current_price": current_price,
            "warnings": warnings,
            "attribution": attribution,
        }

    def _combine_ranges(
        self, comps: dict, dcf: dict, shares: float | None,
    ) -> dict:
        """Combine valuation ranges from comps and DCF into per-share prices.

        Uses the DCF point estimate and comps low/mid/high (converted from
        market cap to per-share). The sensitivity grid is NOT included — it's
        variations of a single model and would over-weight DCF in the median.
        """
        per_share_prices: list[float] = []

        # Comps range — values are implied MARKET CAPs, convert to per-share
        comps_combined = comps.get("valuation_range", {}).get("combined", {})
        if comps_combined and shares and shares > 0:
            for key in ("low", "mid", "high"):
                mc = comps_combined.get(key)
                if mc is not None:
                    per_share = float(mc) / shares
                    if per_share > 0:
                        per_share_prices.append(per_share)

        # DCF implied price (single point estimate, already per-share)
        dcf_price = dcf.get("implied_price")
        if dcf_price is not None and dcf_price > 0:
            per_share_prices.append(float(dcf_price))

        if not per_share_prices:
            return {}

        return {
            "low": round(min(per_share_prices), 2),
            "mid": round(float(sorted(per_share_prices)[len(per_share_prices) // 2]), 2),
            "high": round(max(per_share_prices), 2),
            "methodology_weights": {
                "dcf": 0.5,
                "comps": 0.5,
            },
        }

    def _verdict(self, combined: dict, current_price: float | None) -> dict:
        if not combined or current_price is None:
            return {"label": "insufficient_data", "reasoning": "Not enough data for a valuation verdict"}

        mid = combined.get("mid")
        if mid is None:
            return {"label": "insufficient_data", "reasoning": "No mid-point valuation available"}

        upside = (mid / current_price - 1) * 100

        # Use ±10% band — standard valuation practice
        if upside > 10:
            label = "undervalued"
            reasoning = f"Fair value estimate ({mid:.2f}) is {upside:.1f}% above current price ({current_price:.2f})"
        elif upside < -10:
            label = "overvalued"
            reasoning = f"Fair value estimate ({mid:.2f}) is {abs(upside):.1f}% below current price ({current_price:.2f})"
        else:
            label = "fairly_valued"
            reasoning = f"Fair value estimate ({mid:.2f}) is within 10% of current price ({current_price:.2f})"

        return {
            "label": label,
            "upside_pct": round(upside, 2),
            "fair_value_mid": round(mid, 2),
            "reasoning": reasoning,
        }
