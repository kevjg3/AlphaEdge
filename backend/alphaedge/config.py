"""Application settings via Pydantic BaseSettings."""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://alphaedge:alphaedge_dev@localhost:5432/alphaedge"
    database_url_sync: str = "postgresql://alphaedge:alphaedge_dev@localhost:5432/alphaedge"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Caching
    cache_dir: str = "/tmp/alphaedge_cache"
    cache_ttl: int = 300  # seconds

    # Data storage
    data_dir: str = "/tmp/alphaedge_data"

    # SEC EDGAR
    edgar_user_agent: str = "AlphaEdge/0.1 (dev@example.com)"

    # Analysis defaults
    default_seed: int = 42
    default_lookback_years: int = 2
    forecast_horizons: list[str] = ["1D", "1W", "1M", "3M", "12M"]

    # NLP
    finbert_model: str = "ProsusAI/finbert"
    spacy_model: str = "en_core_web_sm"

    # CORS
    cors_origins: list[str] = ["*"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
