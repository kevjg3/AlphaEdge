"""Report generation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from alphaedge.api.routes.analysis import _runs
from alphaedge.api.schemas import AnalysisStatus
from alphaedge.reporting.report_generator import ReportGenerator

router = APIRouter()


@router.get("/report/{run_id}", response_class=HTMLResponse)
async def get_report(run_id: str) -> HTMLResponse:
    """Generate and return an HTML report for a completed analysis run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")

    run = _runs[run_id]
    if run["status"] != AnalysisStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis not yet completed")

    generator = ReportGenerator()
    analysis = {
        "ticker": run["ticker"],
        "run_id": run_id,
        "snapshot": run.get("snapshot"),
        "fundamentals": run.get("fundamentals"),
        "technicals": run.get("technicals"),
        "news": run.get("news"),
        "forecast": run.get("forecast"),
        "risk": run.get("risk"),
        "warnings": run.get("warnings", []),
    }

    # Convert snapshot Pydantic model to dict if needed
    snap = analysis["snapshot"]
    if snap and hasattr(snap, "model_dump"):
        analysis["snapshot"] = snap.model_dump()

    html = generator.generate_html(analysis)
    return HTMLResponse(content=html)
