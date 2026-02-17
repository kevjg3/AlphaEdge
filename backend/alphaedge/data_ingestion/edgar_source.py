"""SEC EDGAR data source for filings (10-K, 10-Q, 8-K) and XBRL data."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from diskcache import Cache

from alphaedge.config import settings
from alphaedge.data_ingestion.base import (
    DataSource, DataSourceName, FetchResult, SourceAttribution,
)


_CIK_LOOKUP_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&forms=10-K"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


class EdgarSource(DataSource):
    """Fetch SEC EDGAR filings and structured XBRL financial data."""

    def __init__(self, cache_dir: str | None = None):
        self._cache = Cache(cache_dir or settings.cache_dir)
        self._headers = {"User-Agent": settings.edgar_user_agent}
        self._last_request_time = 0.0

    def is_available(self) -> bool:
        try:
            r = httpx.get("https://data.sec.gov", headers=self._headers, timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _attr(self, url: str | None = None, cache_hit: bool = False) -> SourceAttribution:
        return SourceAttribution(source=DataSourceName.EDGAR, url=url, cache_hit=cache_hit)

    def _rate_limit(self) -> None:
        """Enforce 10 requests/second limit for SEC EDGAR."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self._last_request_time = time.time()

    def _get_json(self, url: str) -> dict | None:
        self._rate_limit()
        try:
            r = httpx.get(url, headers=self._headers, timeout=15, follow_redirects=True)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    # --- CIK Lookup ---

    def resolve_cik(self, ticker: str) -> Optional[str]:
        """Resolve ticker to CIK number."""
        key = f"edgar:cik:{ticker}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        data = self._get_json(_COMPANY_TICKERS_URL)
        if data:
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    self._cache.set(key, cik, expire=86400)
                    return cik
        return None

    # --- Filings Index ---

    def get_filings_index(
        self, ticker: str, form_types: list[str] | None = None,
    ) -> FetchResult:
        """Fetch recent filing metadata from EDGAR submissions API."""
        cik = self.resolve_cik(ticker)
        if not cik:
            return FetchResult(
                data=[], attribution=self._attr(),
                success=False, warnings=[f"Cannot resolve CIK for {ticker}"],
            )

        key = f"edgar:filings:{ticker}"
        cached = self._cache.get(key)
        if cached is not None:
            return FetchResult(data=cached, attribution=self._attr(cache_hit=True))

        url = _SUBMISSIONS_URL.format(cik=cik)
        data = self._get_json(url)
        if not data:
            return FetchResult(
                data=[], attribution=self._attr(url=url),
                success=False, warnings=["EDGAR submissions API returned no data"],
            )

        form_types = form_types or ["10-K", "10-Q", "8-K"]
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        descriptions = recent.get("primaryDocDescription", [])

        filings = []
        for i, form in enumerate(forms):
            if form in form_types:
                filings.append({
                    "form_type": form,
                    "filing_date": dates[i] if i < len(dates) else None,
                    "accession_number": accessions[i] if i < len(accessions) else None,
                    "description": descriptions[i] if i < len(descriptions) else "",
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accessions[i].replace('-', '')}"
                           if i < len(accessions) else None,
                })

        self._cache.set(key, filings, expire=3600)
        return FetchResult(data=filings, attribution=self._attr(url=url))

    # --- XBRL Company Facts ---

    def get_company_facts(self, ticker: str) -> FetchResult:
        """Fetch structured financial data via XBRL companyfacts API."""
        cik = self.resolve_cik(ticker)
        if not cik:
            return FetchResult(
                data={}, attribution=self._attr(),
                success=False, warnings=[f"Cannot resolve CIK for {ticker}"],
            )

        key = f"edgar:facts:{ticker}"
        cached = self._cache.get(key)
        if cached is not None:
            return FetchResult(data=cached, attribution=self._attr(cache_hit=True))

        url = _COMPANY_FACTS_URL.format(cik=cik)
        data = self._get_json(url)
        if not data:
            return FetchResult(
                data={}, attribution=self._attr(url=url),
                success=False, warnings=["XBRL companyfacts API returned no data"],
            )

        # Extract key financial facts
        facts = data.get("facts", {})
        us_gaap = facts.get("us-gaap", {})

        extracted = {}
        for concept_name, concept_data in us_gaap.items():
            units = concept_data.get("units", {})
            # Get USD values
            usd_vals = units.get("USD", [])
            if usd_vals:
                # Get most recent annual (10-K) filing value
                annual = [v for v in usd_vals if v.get("form") == "10-K"]
                if annual:
                    most_recent = max(annual, key=lambda v: v.get("end", ""))
                    extracted[concept_name] = {
                        "value": most_recent.get("val"),
                        "end_date": most_recent.get("end"),
                        "form": most_recent.get("form"),
                    }

        self._cache.set(key, extracted, expire=3600)
        return FetchResult(data=extracted, attribution=self._attr(url=url))
