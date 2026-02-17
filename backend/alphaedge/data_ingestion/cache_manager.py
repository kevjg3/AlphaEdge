"""Two-tier caching: Redis (hot) + disk (cold)."""

from __future__ import annotations

import json
import pickle
from typing import Any, Optional

from diskcache import Cache

from alphaedge.config import settings


class CacheManager:
    """Two-tier cache: Redis (fast, short TTL) + disk (persistent, long TTL)."""

    def __init__(
        self,
        redis_url: str | None = None,
        disk_dir: str | None = None,
        hot_ttl: int = 300,
        cold_ttl: int = 86400,
    ):
        self._hot_ttl = hot_ttl
        self._cold_ttl = cold_ttl
        self._disk = Cache(disk_dir or settings.cache_dir)

        # Redis is optional — degrade to disk-only if unavailable
        self._redis = None
        try:
            import redis as redis_lib
            self._redis = redis_lib.Redis.from_url(
                redis_url or settings.redis_url, decode_responses=False,
            )
            self._redis.ping()
        except Exception:
            self._redis = None

    def get(self, key: str) -> Optional[Any]:
        """Get from hot tier first, then cold tier."""
        # Hot tier (Redis)
        if self._redis:
            try:
                val = self._redis.get(key)
                if val is not None:
                    return pickle.loads(val)
            except Exception:
                pass

        # Cold tier (disk)
        val = self._disk.get(key)
        return val

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set in both tiers."""
        # Hot tier
        if self._redis:
            try:
                self._redis.setex(key, ttl or self._hot_ttl, pickle.dumps(value))
            except Exception:
                pass

        # Cold tier
        self._disk.set(key, value, expire=ttl or self._cold_ttl)

    def invalidate(self, key: str) -> None:
        """Remove from both tiers."""
        if self._redis:
            try:
                self._redis.delete(key)
            except Exception:
                pass
        self._disk.delete(key)
