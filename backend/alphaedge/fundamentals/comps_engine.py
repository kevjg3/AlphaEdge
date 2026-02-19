"""Comparable company (comps) analysis engine."""

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
class CompMetrics:
    ticker: str
    name: str = ""
    market_cap: Optional[float] = None
    ev: Optional[float] = None
    pe: Optional[float] = None
    forward_pe: Optional[float] = None
    ev_ebitda: Optional[float] = None
    ev_revenue: Optional[float] = None
    ps: Optional[float] = None
    pb: Optional[float] = None
    roe: Optional[float] = None
    revenue_growth: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


class CompsEngine:
    """Run comparable company analysis."""

    def __init__(self, yf_source: YFinanceSource):
        self.yf = yf_source

    def _fetch_metrics(self, ticker: str) -> Optional[CompMetrics]:
        """Fetch and compute valuation metrics for a single ticker."""
        try:
            info_result = self.yf.get_company_info(ticker)
            info = info_result.data or {}
            if not info:
                return None

            market_cap = _sf(info.get("market_cap"))
            total_debt = _sf(info.get("total_debt", 0)) or 0
            cash = _sf(info.get("total_cash", 0)) or 0
            ev = (market_cap + total_debt - cash) if market_cap else None

            ebitda = _sf(info.get("ebitda"))
            revenue = _sf(info.get("total_revenue"))

            return CompMetrics(
                ticker=ticker.upper(),
                name=info.get("name", ""),
                market_cap=market_cap,
                ev=ev,
                pe=_sf(info.get("trailing_pe")),
                forward_pe=_sf(info.get("forward_pe")),
                ev_ebitda=ev / ebitda if ev and ebitda and ebitda > 0 else None,
                ev_revenue=ev / revenue if ev and revenue and revenue > 0 else None,
                ps=_sf(info.get("price_to_sales_trailing_12_months")),
                pb=_sf(info.get("price_to_book")),
                roe=_sf(info.get("return_on_equity")),
                revenue_growth=_sf(info.get("revenue_growth")),
                gross_margin=_sf(info.get("gross_margins")),
                operating_margin=_sf(info.get("operating_margins")),
            )
        except Exception as e:
            logger.debug("Failed to fetch metrics for %s: %s", ticker, e)
            return None

    def run_comps(self, ticker: str, peers: list[str] | None = None) -> dict:
        """Run comparable company analysis.

        Returns dict with target, peers, valuation_range, percentile_rank.
        """
        warnings: list[str] = []
        attribution: list[dict] = []

        # Target metrics
        target = self._fetch_metrics(ticker)
        if target is None:
            return {"warnings": ["Could not fetch target metrics"], "attribution": []}

        # Get peers
        if not peers:
            peers_result = self.yf.get_peers(ticker)
            peers = peers_result.data or []
            warnings.extend(peers_result.warnings)
            attribution.append(peers_result.attribution.to_dict())

        # Fetch peer metrics in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed

        filtered_peers = [p for p in peers[:6] if p.upper() != ticker.upper()]
        peer_metrics: list[CompMetrics] = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(self._fetch_metrics, p): p for p in filtered_peers}
            for fut in as_completed(futures):
                m = fut.result()
                if m is not None and m.market_cap is not None:
                    peer_metrics.append(m)

        if not peer_metrics:
            warnings.append("No peer data available for comps analysis")
            return {
                "target": target.to_dict(),
                "peers": [],
                "valuation_range": {},
                "percentile_rank": {},
                "warnings": warnings,
                "attribution": attribution,
            }

        # Compute valuation range from peer medians
        valuation_range = self._implied_valuation(target, peer_metrics)

        # Percentile rank
        percentile_rank = self._percentile_rank(target, peer_metrics)

        return {
            "target": target.to_dict(),
            "peers": [p.to_dict() for p in peer_metrics],
            "peer_count": len(peer_metrics),
            "valuation_range": valuation_range,
            "percentile_rank": percentile_rank,
            "warnings": warnings,
            "attribution": attribution,
        }

    def _implied_valuation(self, target: CompMetrics, peers: list[CompMetrics]) -> dict:
        """Compute implied market cap from peer median multiples.

        Each approach computes an implied market cap, then the combined range
        gives low/mid/high across all approaches.
        """
        result: dict[str, Any] = {}
        implied_mcs: list[float] = []

        # P/E approach — ratio of median peer PE to target PE applied to market cap
        peer_pes = [p.pe for p in peers if p.pe and 0 < p.pe < 200]
        if peer_pes and target.pe and target.pe > 0 and target.market_cap:
            median_pe = float(np.median(peer_pes))
            implied_mc = median_pe / target.pe * target.market_cap
            upside = (implied_mc / target.market_cap - 1) * 100
            result["pe_approach"] = {
                "peer_median_pe": round(median_pe, 2),
                "target_pe": round(target.pe, 2),
                "implied_market_cap": round(implied_mc, 0),
                "implied_upside": round(upside, 2),
            }
            implied_mcs.append(implied_mc)

        # EV/EBITDA approach
        peer_ev_ebitda = [p.ev_ebitda for p in peers if p.ev_ebitda and 0 < p.ev_ebitda < 80]
        if peer_ev_ebitda and target.ev_ebitda and target.ev_ebitda > 0 and target.ev:
            median_ev_ebitda = float(np.median(peer_ev_ebitda))
            ebitda_est = target.ev / target.ev_ebitda
            implied_ev = median_ev_ebitda * ebitda_est
            # Net debt = EV - market_cap
            net_debt = (target.ev - target.market_cap) if target.ev and target.market_cap else 0
            implied_mc = implied_ev - net_debt
            if implied_mc > 0 and target.market_cap:
                upside = (implied_mc / target.market_cap - 1) * 100
                result["ev_ebitda_approach"] = {
                    "peer_median_ev_ebitda": round(median_ev_ebitda, 2),
                    "target_ev_ebitda": round(target.ev_ebitda, 2),
                    "implied_market_cap": round(implied_mc, 0),
                    "implied_upside": round(upside, 2),
                }
                implied_mcs.append(implied_mc)

        # EV/Revenue approach
        peer_ev_rev = [p.ev_revenue for p in peers if p.ev_revenue and 0 < p.ev_revenue < 50]
        if peer_ev_rev and target.ev_revenue and target.ev_revenue > 0 and target.ev:
            median_ev_rev = float(np.median(peer_ev_rev))
            rev_est = target.ev / target.ev_revenue
            implied_ev = median_ev_rev * rev_est
            net_debt = (target.ev - target.market_cap) if target.ev and target.market_cap else 0
            implied_mc = implied_ev - net_debt
            if implied_mc > 0 and target.market_cap:
                upside = (implied_mc / target.market_cap - 1) * 100
                result["ev_revenue_approach"] = {
                    "peer_median_ev_revenue": round(median_ev_rev, 2),
                    "target_ev_revenue": round(target.ev_revenue, 2),
                    "implied_market_cap": round(implied_mc, 0),
                    "implied_upside": round(upside, 2),
                }
                implied_mcs.append(implied_mc)

        # Combined range of implied market caps
        if implied_mcs:
            result["combined"] = {
                "low": round(min(implied_mcs), 0),
                "mid": round(float(np.median(implied_mcs)), 0),
                "high": round(max(implied_mcs), 0),
            }

        return result

    def _percentile_rank(self, target: CompMetrics, peers: list[CompMetrics]) -> dict:
        """Rank target vs peers on key metrics."""
        result = {}
        metrics = ["pe", "ev_ebitda", "ps", "pb", "roe", "revenue_growth", "gross_margin", "operating_margin"]

        for metric in metrics:
            target_val = getattr(target, metric, None)
            peer_vals = [getattr(p, metric) for p in peers if getattr(p, metric, None) is not None]
            if target_val is not None and peer_vals:
                below = sum(1 for v in peer_vals if v < target_val)
                pctile = below / len(peer_vals) * 100
                result[metric] = round(pctile, 1)

        return result
