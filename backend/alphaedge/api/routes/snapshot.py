"""Quick market snapshot endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from alphaedge.api.schemas import SnapshotResponse
from alphaedge.data_ingestion.yfinance_source import YFinanceSource

router = APIRouter()


@router.get("/{ticker}", response_model=SnapshotResponse)
async def get_snapshot(ticker: str) -> SnapshotResponse:
    """Fetch a quick market snapshot for a ticker."""
    ticker = ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")

    yf = YFinanceSource()
    warnings: list[str] = []
    attribution: list[dict] = []

    # Company info
    info_result = yf.get_company_info(ticker)
    info = info_result.data or {}
    warnings.extend(info_result.warnings)
    attribution.append(info_result.attribution.to_dict())

    if not info:
        raise HTTPException(status_code=404, detail=f"No data found for ticker '{ticker}'")

    # Spot price
    spot_result = yf.get_spot(ticker)
    price = spot_result.data
    warnings.extend(spot_result.warnings)

    # 1-day change
    hist_result = yf.get_history(ticker, period="5d")
    change_1d = None
    change_1d_pct = None
    if hist_result.success and hist_result.data is not None and len(hist_result.data) >= 2:
        closes = hist_result.data["Close"]
        prev = float(closes.iloc[-2])
        curr = float(closes.iloc[-1])
        change_1d = round(curr - prev, 4)
        change_1d_pct = round((curr - prev) / prev * 100, 4) if prev else None

    return SnapshotResponse(
        ticker=ticker,
        name=info.get("name", ""),
        price=price,
        change_1d=change_1d,
        change_1d_pct=change_1d_pct,
        market_cap=info.get("market_cap"),
        pe_ratio=info.get("trailing_pe"),
        beta=info.get("beta"),
        sector=info.get("sector", ""),
        industry=info.get("industry", ""),
        high_52w=info.get("52w_high") or info.get("fifty_two_week_high"),
        low_52w=info.get("52w_low") or info.get("fifty_two_week_low"),
        avg_volume=info.get("avg_volume") or info.get("average_volume"),
        warnings=warnings,
        attribution=attribution,
    )
