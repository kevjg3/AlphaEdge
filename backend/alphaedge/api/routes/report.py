"""Report generation endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from alphaedge.api.routes.analysis import _runs
from alphaedge.api.schemas import AnalysisStatus
from alphaedge.reporting.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_analysis_dict(run_id: str, run: dict) -> dict:
    """Build a plain-dict analysis payload from an in-memory run."""
    analysis = {
        "ticker": run["ticker"],
        "run_id": run_id,
        "snapshot": run.get("snapshot"),
        "fundamentals": run.get("fundamentals"),
        "technicals": run.get("technicals"),
        "news": run.get("news"),
        "forecast": run.get("forecast"),
        "risk": run.get("risk"),
        "quant": run.get("quant"),
        "warnings": run.get("warnings", []),
    }
    # Convert snapshot Pydantic model to dict if needed
    snap = analysis["snapshot"]
    if snap and hasattr(snap, "model_dump"):
        analysis["snapshot"] = snap.model_dump()
    return analysis


@router.get("/report/{run_id}", response_class=HTMLResponse)
async def get_report(run_id: str) -> HTMLResponse:
    """Generate and return an HTML report for a completed analysis run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")

    run = _runs[run_id]
    if run["status"] != AnalysisStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis not yet completed")

    analysis = _build_analysis_dict(run_id, run)
    generator = ReportGenerator()
    html = generator.generate_html(analysis)
    return HTMLResponse(content=html)


@router.get("/pitch-deck/{run_id}")
async def get_pitch_deck(run_id: str) -> StreamingResponse:
    """Generate and return a PPTX pitch deck for a completed analysis run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")

    run = _runs[run_id]
    if run["status"] != AnalysisStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis not yet completed")

    analysis = _build_analysis_dict(run_id, run)

    from alphaedge.reporting.pitch_deck import PitchDeckGenerator

    generator = PitchDeckGenerator()
    pptx_bytes = generator.generate(analysis)

    ticker = run["ticker"]
    filename = f"{ticker}_pitch_deck.pptx"
    logger.info("Generated pitch deck for %s (run %s)", ticker, run_id)

    return StreamingResponse(
        pptx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
