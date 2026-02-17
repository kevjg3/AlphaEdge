"""Reproducible run ID generation."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone


def generate_run_id(ticker: str, seed: int = 42) -> str:
    """Generate a deterministic run ID from ticker + seed + timestamp.

    Format: {TICKER}_{seed}_{timestamp_hash8}
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    raw = f"{ticker}:{seed}:{ts}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"{ticker.upper()}_{seed}_{h}"
