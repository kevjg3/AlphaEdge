"""API route registration."""

from __future__ import annotations

from fastapi import APIRouter

from alphaedge.api.routes.health import router as health_router
from alphaedge.api.routes.snapshot import router as snapshot_router
from alphaedge.api.routes.analysis import router as analysis_router
from alphaedge.api.routes.report import router as report_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(snapshot_router, prefix="/snapshot", tags=["snapshot"])
api_router.include_router(analysis_router, prefix="/analysis", tags=["analysis"])
api_router.include_router(report_router, prefix="/analysis", tags=["reports"])
