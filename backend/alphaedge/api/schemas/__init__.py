"""Pydantic schemas for the AlphaEdge API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ──────────────────────── Enums ────────────────────────

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Horizon(str, Enum):
    ONE_DAY = "1D"
    ONE_WEEK = "1W"
    ONE_MONTH = "1M"
    THREE_MONTHS = "3M"
    TWELVE_MONTHS = "12M"


# ──────────────────────── Request Schemas ────────────────────────

class AnalysisRequest(BaseModel):
    """Request to run a full analysis on a ticker."""
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    lookback_years: int = Field(default=2, ge=1, le=10)
    include_fundamentals: bool = True
    include_technicals: bool = True
    include_news: bool = True
    include_forecast: bool = True
    include_risk: bool = True


class SnapshotRequest(BaseModel):
    """Quick snapshot request."""
    ticker: str = Field(..., min_length=1, max_length=10)


# ──────────────────────── Response Schemas ────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    services: dict[str, bool] = Field(default_factory=dict)


class SnapshotResponse(BaseModel):
    """Quick market snapshot for a ticker."""
    ticker: str
    name: str = ""
    price: Optional[float] = None
    change_1d: Optional[float] = None
    change_1d_pct: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    beta: Optional[float] = None
    sector: str = ""
    industry: str = ""
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    avg_volume: Optional[int] = None
    description: str = ""
    country: str = ""
    employees: Optional[int] = None
    website: str = ""
    dividend_yield: Optional[float] = None
    forward_pe: Optional[float] = None
    revenue_growth: Optional[float] = None
    operating_margins: Optional[float] = None
    profit_margins: Optional[float] = None
    return_on_equity: Optional[float] = None
    debt_to_equity: Optional[float] = None
    free_cashflow: Optional[float] = None
    total_revenue: Optional[float] = None
    warnings: list[str] = Field(default_factory=list)
    attribution: list[dict] = Field(default_factory=list)


class AnalysisStatusResponse(BaseModel):
    """Status of a running or completed analysis."""
    run_id: str
    ticker: str
    status: AnalysisStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    current_step: str = ""
    warnings: list[str] = Field(default_factory=list)


class FundamentalsResponse(BaseModel):
    """Fundamentals analysis results."""
    financial_health: dict = Field(default_factory=dict)
    comps_valuation: dict = Field(default_factory=dict)
    dcf_valuation: dict = Field(default_factory=dict)
    combined_range: dict = Field(default_factory=dict)
    investment_thesis: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class TechnicalsResponse(BaseModel):
    """Technical analysis results."""
    indicators: dict = Field(default_factory=dict)
    regime: dict = Field(default_factory=dict)
    support_resistance: dict = Field(default_factory=dict)
    factor_exposures: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class NewsResponse(BaseModel):
    """News and NLP analysis results."""
    articles: list[dict] = Field(default_factory=list)
    sentiment: dict = Field(default_factory=dict)
    events: dict = Field(default_factory=dict)
    synthesis: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ForecastResponse(BaseModel):
    """Forecast results across horizons."""
    horizons: dict = Field(default_factory=dict)
    overall_direction: str = ""
    model_weights: dict = Field(default_factory=dict)
    model_agreement: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class RiskResponse(BaseModel):
    """Risk analysis results."""
    var: dict = Field(default_factory=dict)
    drawdown: dict = Field(default_factory=dict)
    scenarios: list[dict] = Field(default_factory=list)
    stress_tests: list[dict] = Field(default_factory=list)
    upcoming_events: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FullAnalysisResponse(BaseModel):
    """Complete analysis response combining all modules."""
    model_config = {"arbitrary_types_allowed": True}

    run_id: str
    ticker: str
    status: AnalysisStatus
    snapshot: SnapshotResponse
    fundamentals: Optional[Any] = None
    technicals: Optional[Any] = None
    news: Optional[Any] = None
    forecast: Optional[Any] = None
    risk: Optional[Any] = None
    quant: Optional[Any] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    warnings: list[str] = Field(default_factory=list)
    attribution: list[dict] = Field(default_factory=list)
