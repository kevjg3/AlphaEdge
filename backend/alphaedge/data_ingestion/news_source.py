"""News fetching from free sources: RSS feeds and GDELT."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
from diskcache import Cache

from alphaedge.config import settings
from alphaedge.data_ingestion.base import (
    DataSource, DataSourceName, FetchResult, SourceAttribution,
)


@dataclass
class NewsArticle:
    """Normalized news article."""
    title: str
    url: str
    source: str
    published_at: datetime
    snippet: str = ""
    full_text: Optional[str] = None
    attribution: Optional[SourceAttribution] = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at if isinstance(self.published_at, str) else self.published_at.isoformat(),
            "snippet": self.snippet[:500],
        }


class NewsSource(DataSource):
    """Multi-source news fetcher using free APIs."""

    # RSS feed templates — {query} gets replaced with company name or ticker
    RSS_FEEDS = [
        "https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en",
    ]
    GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, cache_dir: str | None = None):
        self._cache = Cache(cache_dir or settings.cache_dir)

    def is_available(self) -> bool:
        try:
            r = httpx.get("https://news.google.com", timeout=5, follow_redirects=True)
            return r.status_code < 500
        except Exception:
            return False

    def _attr(self, source_name: DataSourceName = DataSourceName.NEWS_RSS) -> SourceAttribution:
        return SourceAttribution(source=source_name)

    def fetch_articles(
        self,
        ticker: str,
        company_name: str = "",
        days_back: int = 90,
        max_articles: int = 100,
    ) -> FetchResult:
        """Fetch, deduplicate, and sort articles from all sources."""
        key = f"news:{ticker}:{days_back}"
        cached = self._cache.get(key)
        if cached is not None:
            articles = [self._from_cache(a) if isinstance(a, dict) else a for a in cached]
            return FetchResult(data=articles, attribution=self._attr(DataSourceName.NEWS_RSS),
                               warnings=[] if articles else ["No cached articles"])

        all_articles: list[NewsArticle] = []
        ws: list[str] = []

        # RSS feeds
        rss_articles = self._fetch_rss(ticker, company_name)
        all_articles.extend(rss_articles)

        # GDELT
        gdelt_articles = self._fetch_gdelt(company_name or ticker, days_back)
        all_articles.extend(gdelt_articles)

        if not all_articles:
            ws.append("No articles found from any source")

        # Deduplicate
        deduped = self._deduplicate(all_articles)

        # Filter by date range
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        deduped = [a for a in deduped if a.published_at >= cutoff]

        # Sort by date (newest first)
        deduped.sort(key=lambda a: a.published_at, reverse=True)

        # Limit
        deduped = deduped[:max_articles]

        # Cache as dicts
        self._cache.set(key, [a.to_dict() for a in deduped], expire=1800)

        return FetchResult(
            data=deduped,
            attribution=self._attr(DataSourceName.NEWS_RSS),
            warnings=ws,
        )

    def _fetch_rss(self, ticker: str, company_name: str) -> list[NewsArticle]:
        """Fetch from RSS feeds."""
        articles: list[NewsArticle] = []
        query = company_name or ticker

        for feed_template in self.RSS_FEEDS:
            url = feed_template.format(query=query.replace(" ", "+"))
            try:
                import feedparser
                r = httpx.get(url, timeout=10, follow_redirects=True)
                feed = feedparser.parse(r.text)
                for entry in feed.entries:
                    pub_date = self._parse_date(entry.get("published", ""))
                    title = entry.get("title", "").strip()
                    link = entry.get("link", "")
                    snippet = entry.get("summary", "")
                    # Clean HTML from snippet
                    snippet = re.sub(r"<[^>]+>", "", snippet).strip()

                    if title and link:
                        articles.append(NewsArticle(
                            title=title,
                            url=link,
                            source="Google News RSS",
                            published_at=pub_date,
                            snippet=snippet[:500],
                            attribution=self._attr(DataSourceName.NEWS_RSS),
                        ))
            except Exception:
                continue

        return articles

    def _fetch_gdelt(self, query: str, days_back: int) -> list[NewsArticle]:
        """Fetch from GDELT DOC API (free, no key needed)."""
        articles: list[NewsArticle] = []
        try:
            params = {
                "query": query,
                "mode": "artlist",
                "maxrecords": "50",
                "format": "json",
                "timespan": f"{days_back}d" if days_back <= 90 else "90d",
            }
            r = httpx.get(self.GDELT_URL, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                for item in data.get("articles", []):
                    title = item.get("title", "").strip()
                    url = item.get("url", "")
                    source = item.get("domain", "GDELT")
                    seendate = item.get("seendate", "")
                    pub_date = self._parse_gdelt_date(seendate)

                    if title and url:
                        articles.append(NewsArticle(
                            title=title,
                            url=url,
                            source=source,
                            published_at=pub_date,
                            snippet="",
                            attribution=self._attr(DataSourceName.NEWS_GDELT),
                        ))
        except Exception:
            pass

        return articles

    @staticmethod
    def _from_cache(d: dict) -> NewsArticle:
        """Reconstruct a NewsArticle from a cached dict, parsing date strings back."""
        pub = d.get("published_at")
        if isinstance(pub, str):
            try:
                pub = datetime.fromisoformat(pub)
            except (ValueError, TypeError):
                pub = datetime.now(timezone.utc)
        elif pub is None:
            pub = datetime.now(timezone.utc)
        return NewsArticle(
            title=d.get("title", ""),
            url=d.get("url", ""),
            source=d.get("source", ""),
            published_at=pub,
            snippet=d.get("snippet", ""),
        )

    def _deduplicate(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        """Remove near-duplicate articles using Jaccard similarity on titles."""
        if len(articles) <= 1:
            return articles

        seen_titles: list[set[str]] = []
        deduped: list[NewsArticle] = []

        for article in articles:
            tokens = set(article.title.lower().split())
            is_dup = False
            for seen in seen_titles:
                if not tokens or not seen:
                    continue
                jaccard = len(tokens & seen) / len(tokens | seen)
                if jaccard > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                seen_titles.append(tokens)
                deduped.append(article)

        return deduped

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        """Parse RSS date string to UTC datetime."""
        from email.utils import parsedate_to_datetime
        try:
            dt = parsedate_to_datetime(date_str)
            return dt.astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    @staticmethod
    def _parse_gdelt_date(date_str: str) -> datetime:
        """Parse GDELT date format (YYYYMMDDTHHmmssZ)."""
        try:
            return datetime.strptime(date_str[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)
