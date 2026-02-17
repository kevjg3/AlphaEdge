"""Base types for the data ingestion layer.

Every piece of data carries a SourceAttribution for explainability and audit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class DataSourceName(str, Enum):
    YFINANCE = "yfinance"
    EDGAR = "sec_edgar"
    NEWS_RSS = "news_rss"
    NEWS_GDELT = "news_gdelt"
    DERIVED = "derived"


@dataclass
class SourceAttribution:
    """Every data point traces back to its source."""
    source: DataSourceName
    url: Optional[str] = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cache_hit: bool = False
    stale: bool = False

    def to_dict(self) -> dict:
        return {
            "source": self.source.value,
            "url": self.url,
            "fetched_at": self.fetched_at.isoformat(),
            "cache_hit": self.cache_hit,
            "stale": self.stale,
        }


@dataclass
class FetchResult:
    """Standard wrapper for all fetched data."""
    data: Any
    attribution: SourceAttribution
    warnings: list[str] = field(default_factory=list)
    success: bool = True

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class DataSource(ABC):
    """Protocol for pluggable data sources."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data source is reachable."""
        ...
