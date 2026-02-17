"""Health-check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from alphaedge.api.schemas import HealthResponse
from alphaedge.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health and dependency status."""
    services: dict[str, bool] = {}

    # Check Redis
    try:
        import redis as redis_lib

        r = redis_lib.Redis.from_url(settings.redis_url, socket_timeout=2)
        r.ping()
        services["redis"] = True
    except Exception:
        services["redis"] = False

    # Check PostgreSQL
    try:
        from sqlalchemy import text
        from alphaedge.db import async_session_factory

        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        services["postgres"] = True
    except Exception:
        services["postgres"] = False

    return HealthResponse(
        status="ok",
        version="0.1.0",
        services=services,
    )
