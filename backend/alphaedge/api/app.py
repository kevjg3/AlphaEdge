"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from alphaedge.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    # Startup: warm caches, check dependencies
    import logging

    logger = logging.getLogger("alphaedge")
    logger.info("AlphaEdge starting up")
    yield
    # Shutdown
    logger.info("AlphaEdge shutting down")


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(
        title="AlphaEdge",
        description="Investment-grade analysis: valuation, technicals, forecasting, news",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    from alphaedge.api.routes import api_router

    app.include_router(api_router, prefix="/api/v1")

    return app


app = create_app()
