"""Full analysis endpoints — submit, poll status, get results."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from alphaedge.api.schemas import (
    AnalysisRequest,
    AnalysisStatus,
    AnalysisStatusResponse,
    FullAnalysisResponse,
    SnapshotResponse,
)
from alphaedge.run_id import generate_run_id

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory store for analysis runs (production would use DB)
_runs: dict[str, dict[str, Any]] = {}


_PIPELINE_TIMEOUT = 180  # seconds — hard limit for entire pipeline


def _run_analysis(run_id: str, req: AnalysisRequest) -> None:
    """Execute the full analysis pipeline in the background."""
    import time as _time

    run = _runs[run_id]
    run["status"] = AnalysisStatus.RUNNING
    ticker = req.ticker.upper()
    warnings: list[str] = []
    attribution: list[dict] = []
    _t0 = _time.time()

    def _lap(label: str):
        elapsed = _time.time() - _t0
        logger.info("[%s] %s — %.1fs elapsed", ticker, label, elapsed)

    def _budget_left() -> float:
        """Seconds remaining before hard pipeline timeout."""
        return max(0, _PIPELINE_TIMEOUT - (_time.time() - _t0))

    try:
        # ── Data ingestion ──
        from alphaedge.data_ingestion.yfinance_source import YFinanceSource
        from alphaedge.data_ingestion.edgar_source import EdgarSource
        from alphaedge.data_ingestion.news_source import NewsSource

        yf = YFinanceSource()
        edgar = EdgarSource()
        news_src = NewsSource()

        run["current_step"] = "fetching_data"
        run["progress"] = 0.05

        # Snapshot — fetch info, spot, history in parallel
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=3) as pool:
            info_fut = pool.submit(yf.get_company_info, ticker)
            spot_fut = pool.submit(yf.get_spot, ticker)
            hist_fut = pool.submit(yf.get_history, ticker, f"{req.lookback_years}y")

            info_result = info_fut.result(timeout=30)
            spot_result = spot_fut.result(timeout=30)
            hist_result = hist_fut.result(timeout=30)

        info = info_result.data or {}
        warnings.extend(info_result.warnings)
        attribution.append(info_result.attribution.to_dict())

        current_price = spot_result.data

        hist_df = hist_result.data
        warnings.extend(hist_result.warnings)

        if hist_df is None or hist_df.empty:
            raise ValueError(f"No historical data for {ticker}")
        _lap("data fetched")

        # 1-day change
        change_1d = change_1d_pct = None
        if len(hist_df) >= 2:
            closes = hist_df["Close"]
            prev, curr = float(closes.iloc[-2]), float(closes.iloc[-1])
            change_1d = round(curr - prev, 4)
            change_1d_pct = round((curr - prev) / prev * 100, 4) if prev else None

        snapshot = SnapshotResponse(
            ticker=ticker,
            name=info.get("name", ""),
            price=current_price,
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
            description=info.get("description", ""),
            country=info.get("country", ""),
            employees=info.get("employees"),
            website=info.get("website", ""),
            dividend_yield=info.get("dividend_yield"),
            forward_pe=info.get("forward_pe"),
            revenue_growth=info.get("revenue_growth"),
            operating_margins=info.get("operating_margins"),
            profit_margins=info.get("profit_margins"),
            return_on_equity=info.get("return_on_equity"),
            debt_to_equity=info.get("debt_to_equity"),
            free_cashflow=info.get("free_cashflow"),
            total_revenue=info.get("total_revenue"),
            warnings=[],
            attribution=attribution[:],
        )
        run["snapshot"] = snapshot
        run["progress"] = 0.15

        # ── Fundamentals ──
        fundamentals_data = None
        if req.include_fundamentals:
            run["current_step"] = "fundamentals"
            try:
                from alphaedge.fundamentals.valuation_summary import ValuationSummary

                vs = ValuationSummary(yf, edgar)
                fundamentals_data = vs.full_valuation(ticker)
                _lap("fundamentals done")
                run["progress"] = 0.30

                # Generate investment thesis
                try:
                    from alphaedge.fundamentals.investment_thesis import generate_investment_thesis

                    thesis = generate_investment_thesis(
                        info=info,
                        financial_health=fundamentals_data.get("financial_health", {}),
                        comps_valuation=fundamentals_data.get("comps_valuation", {}),
                        dcf_valuation=fundamentals_data.get("dcf_valuation", {}),
                        verdict=fundamentals_data.get("verdict", {}),
                        combined_range=fundamentals_data.get("combined_range", {}),
                        current_price=current_price,
                    )
                    fundamentals_data["investment_thesis"] = thesis
                except Exception as te:
                    logger.warning("Investment thesis generation failed: %s", te)
                    warnings.append(f"Investment thesis generation failed: {te}")

            except Exception as e:
                logger.warning("Fundamentals failed: %s", e)
                warnings.append(f"Fundamentals analysis failed: {e}")

        # ── Technicals ──
        technicals_data = None
        if req.include_technicals:
            run["current_step"] = "technicals"
            try:
                from alphaedge.technicals.indicators import TechnicalIndicators
                from alphaedge.technicals.regime_detection import RegimeDetector
                from alphaedge.technicals.support_resistance import SupportResistance
                from alphaedge.technicals.ichimoku import IchimokuAnalyzer
                from alphaedge.technicals.volume_profile import VolumeProfileAnalyzer
                from alphaedge.technicals.confluence import ConfluenceScorer
                from alphaedge.technicals.divergence import DivergenceDetector
                from alphaedge.technicals.relative_strength import RelativeStrengthAnalyzer

                indicators = TechnicalIndicators.compute_all(hist_df)
                regime = RegimeDetector(seed=req.seed).detect(hist_df["Close"])
                sr = SupportResistance().detect(hist_df)

                factor_data = None
                try:
                    from alphaedge.technicals.factor_model import FactorModel

                    fm = FactorModel(yf)
                    factor_data = fm.compute_exposures(ticker)
                except Exception as e:
                    warnings.append(f"Factor model failed: {e}")

                # New technicals (each wrapped independently)
                ichimoku = {}
                try:
                    ichimoku = IchimokuAnalyzer().compute(hist_df)
                except Exception as e:
                    logger.warning("Ichimoku failed: %s", e)

                volume_profile = {}
                try:
                    volume_profile = VolumeProfileAnalyzer().analyze(hist_df)
                except Exception as e:
                    logger.warning("Volume profile failed: %s", e)

                divergence = {}
                try:
                    divergence = DivergenceDetector().detect(hist_df)
                except Exception as e:
                    logger.warning("Divergence detection failed: %s", e)

                confluence = {}
                try:
                    confluence = ConfluenceScorer().score(indicators, sr)
                except Exception as e:
                    logger.warning("Confluence scoring failed: %s", e)

                rel_strength = {}
                try:
                    sector = info.get("sector")
                    if sector:
                        rel_strength = RelativeStrengthAnalyzer().analyze(
                            ticker, hist_df["Close"], yf, sector
                        )
                except Exception as e:
                    logger.warning("Relative strength failed: %s", e)

                _lap("technicals computed")
                technicals_data = {
                    "indicators": indicators,
                    "regime": regime.to_dict() if hasattr(regime, "to_dict") else regime,
                    "support_resistance": sr,
                    "factor_exposures": factor_data or {},
                    "ichimoku": ichimoku,
                    "volume_profile": volume_profile,
                    "confluence": confluence,
                    "divergence": divergence,
                    "relative_strength": rel_strength,
                }
                run["progress"] = 0.45
            except Exception as e:
                logger.warning("Technicals failed: %s", e)
                warnings.append(f"Technical analysis failed: {e}")

        # ── News NLP ──
        news_data = None
        if req.include_news:
            run["current_step"] = "news_nlp"
            try:
                from alphaedge.news_nlp.sentiment import SentimentAnalyzer
                from alphaedge.news_nlp.event_detection import EventDetector
                from alphaedge.news_nlp.synthesizer import NewsSynthesizer

                articles_result = news_src.fetch_articles(ticker)
                articles = articles_result.data or []
                warnings.extend(articles_result.warnings)

                sentiment_analyzer = SentimentAnalyzer()
                event_detector = EventDetector()
                synthesizer = NewsSynthesizer(sentiment_analyzer, event_detector)
                synthesis = synthesizer.synthesize(articles, ticker)

                _lap("news NLP done")
                news_data = {
                    "articles": [
                        {"title": a.title, "source": a.source, "url": a.url}
                        for a in articles[:20]
                    ],
                    "sentiment": synthesis.to_dict() if hasattr(synthesis, "to_dict") else {},
                    "events": {},
                    "synthesis": synthesis.to_dict() if hasattr(synthesis, "to_dict") else {},
                }
                run["progress"] = 0.60
            except Exception as e:
                logger.warning("News NLP failed: %s", e)
                warnings.append(f"News analysis failed: {e}")

        # ── Update thesis with news sentiment if available ──
        if fundamentals_data and news_data:
            try:
                from alphaedge.fundamentals.investment_thesis import generate_investment_thesis

                thesis = generate_investment_thesis(
                    info=info,
                    financial_health=fundamentals_data.get("financial_health", {}),
                    comps_valuation=fundamentals_data.get("comps_valuation", {}),
                    dcf_valuation=fundamentals_data.get("dcf_valuation", {}),
                    verdict=fundamentals_data.get("verdict", {}),
                    combined_range=fundamentals_data.get("combined_range", {}),
                    current_price=current_price,
                    news_sentiment=news_data.get("sentiment", {}),
                )
                fundamentals_data["investment_thesis"] = thesis
            except Exception:
                pass  # Keep the thesis without sentiment

        # ── Forecasting ──
        forecast_data = None
        if req.include_forecast and _budget_left() > 15:
            run["current_step"] = "forecasting"
            try:
                from concurrent.futures import ThreadPoolExecutor as _TPE, TimeoutError as _Timeout
                from alphaedge.forecasting.horizons import HorizonForecaster

                hf = HorizonForecaster(seed=req.seed)

                # Run forecasting in a thread with a time cap
                def _do_forecast():
                    return hf.run_all_horizons(
                        hist_df["Close"], current_price,
                        ohlcv_df=hist_df, ticker=ticker,
                    )

                forecast_timeout = min(60, _budget_left() - 10)
                with _TPE(max_workers=1) as pool:
                    fut = pool.submit(_do_forecast)
                    multi = fut.result(timeout=forecast_timeout)

                forecast_data = multi.to_dict() if hasattr(multi, "to_dict") else {}
                _lap("forecasting done")
                run["progress"] = 0.80
            except _Timeout:
                logger.warning("Forecasting timed out after %.0fs", forecast_timeout)
                warnings.append("Forecasting timed out")
                run["progress"] = 0.80
            except Exception as e:
                logger.warning("Forecasting failed: %s", e)
                warnings.append(f"Forecasting failed: {e}")

        # ── Risk ──
        risk_data = None
        if req.include_risk:
            run["current_step"] = "risk"
            try:
                from alphaedge.risk.var_calculator import VaRCalculator
                from alphaedge.risk.drawdown import DrawdownAnalyzer
                from alphaedge.risk.scenario_analysis import ScenarioAnalyzer
                from alphaedge.risk.event_calendar import EventCalendar

                var_calc = VaRCalculator()
                var_result = var_calc.compute(hist_df["Close"])

                dd_analyzer = DrawdownAnalyzer()
                dd_result = dd_analyzer.analyze(hist_df["Close"])

                scenario = ScenarioAnalyzer(yf)
                beta = info.get("beta") or 1.0
                stress = scenario.stress_test_historical(
                    ticker, beta=float(beta)
                )

                events = EventCalendar(yf)
                upcoming = events.get_upcoming_events(ticker)

                _lap("risk analysis done")
                risk_data = {
                    "var": var_result.to_dict() if hasattr(var_result, "to_dict") else var_result,
                    "drawdown": dd_result.to_dict() if hasattr(dd_result, "to_dict") else dd_result,
                    "scenarios": [
                        s.to_dict() if hasattr(s, "to_dict") else s
                        for s in (stress if isinstance(stress, list) else [])
                    ],
                    "stress_tests": [],
                    "upcoming_events": [
                        e.to_dict() if hasattr(e, "to_dict") else e
                        for e in (upcoming if isinstance(upcoming, list) else [])
                    ],
                }
                run["progress"] = 0.95
            except Exception as e:
                logger.warning("Risk analysis failed: %s", e)
                warnings.append(f"Risk analysis failed: {e}")

        # ── Quantitative Analysis (run sub-tasks in parallel) ──
        quant_data = None
        run["current_step"] = "quantitative"
        run["progress"] = 0.96
        try:
            from concurrent.futures import ThreadPoolExecutor
            from alphaedge.quant.performance_metrics import PerformanceAnalyzer
            from alphaedge.quant.return_analysis import ReturnAnalyzer
            from alphaedge.quant.correlation import CorrelationAnalyzer
            from alphaedge.quant.monte_carlo import MonteCarloSimulator
            from alphaedge.quant.signal_backtest import SignalBacktester
            from alphaedge.quant.mean_reversion import MeanReversionAnalyzer
            from alphaedge.quant.volatility_forecast import GarchForecaster
            from alphaedge.quant.momentum_signals import MomentumAnalyzer
            from alphaedge.quant.alpha_intelligence import AlphaIntelligenceAnalyzer

            prices = hist_df["Close"]

            # Pre-fetch SPY for performance metrics (will also be cached for correlation)
            def _perf():
                analyzer = PerformanceAnalyzer()
                spy_prices = None
                try:
                    spy_res = yf.get_history("SPY", period=f"{req.lookback_years}y")
                    if spy_res.success and not spy_res.data.empty:
                        spy_prices = spy_res.data["Close"]
                except Exception:
                    pass
                return analyzer.compute(prices, spy_prices)

            def _returns():
                return ReturnAnalyzer().analyze(prices)

            def _corr():
                return CorrelationAnalyzer(yf).analyze(ticker, prices)

            def _mc():
                return MonteCarloSimulator(seed=req.seed).simulate(prices, horizon_days=252, n_paths=500)

            def _backtest():
                return SignalBacktester().backtest_all(hist_df)

            def _mean_reversion():
                return MeanReversionAnalyzer().analyze(prices)

            def _garch():
                return GarchForecaster().forecast(prices)

            def _momentum():
                return MomentumAnalyzer().analyze(prices)

            def _alpha_intel():
                return AlphaIntelligenceAnalyzer(seed=req.seed).analyze(prices, hist_df)

            quant_timeout = min(60, _budget_left() - 5)
            with ThreadPoolExecutor(max_workers=9) as pool:
                perf_fut = pool.submit(_perf)
                ret_fut = pool.submit(_returns)
                corr_fut = pool.submit(_corr)
                mc_fut = pool.submit(_mc)
                bt_fut = pool.submit(_backtest)
                mr_fut = pool.submit(_mean_reversion)
                garch_fut = pool.submit(_garch)
                mom_fut = pool.submit(_momentum)
                ai_fut = pool.submit(_alpha_intel)

                performance = perf_fut.result(timeout=quant_timeout)
                return_analysis = ret_fut.result(timeout=quant_timeout)
                correlation = corr_fut.result(timeout=quant_timeout)
                monte_carlo = mc_fut.result(timeout=quant_timeout)
                signal_backtest = bt_fut.result(timeout=quant_timeout)

                # New alpha signal analyzers — each wrapped independently
                try:
                    mean_reversion = mr_fut.result(timeout=quant_timeout)
                except Exception as e:
                    logger.warning("Mean reversion analysis failed: %s", e)
                    mean_reversion = {}
                try:
                    garch_forecast = garch_fut.result(timeout=quant_timeout)
                except Exception as e:
                    logger.warning("GARCH forecast failed: %s", e)
                    garch_forecast = {}
                try:
                    momentum = mom_fut.result(timeout=quant_timeout)
                except Exception as e:
                    logger.warning("Momentum analysis failed: %s", e)
                    momentum = {}
                try:
                    alpha_intel = ai_fut.result(timeout=quant_timeout)
                except Exception as e:
                    logger.warning("Alpha intelligence failed: %s", e)
                    alpha_intel = {}

            _lap("quant (all parallel) done")

            # Price series for chart (downsampled)
            price_series = []
            p = prices
            if len(p) > 500:
                step = max(1, len(p) // 500)
                p = p.iloc[::step]
            for d, v in p.items():
                price_series.append({
                    "date": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
                    "price": round(float(v), 2),
                })

            quant_data = {
                "performance": performance.to_dict(),
                "return_analysis": return_analysis,
                "correlation": correlation,
                "monte_carlo": monte_carlo,
                "signal_backtest": signal_backtest,
                "price_series": price_series,
                "mean_reversion": mean_reversion.to_dict() if hasattr(mean_reversion, "to_dict") else mean_reversion,
                "garch_forecast": garch_forecast.to_dict() if hasattr(garch_forecast, "to_dict") else garch_forecast,
                "momentum": momentum.to_dict() if hasattr(momentum, "to_dict") else momentum,
                "alpha_intelligence": alpha_intel.to_dict() if hasattr(alpha_intel, "to_dict") else alpha_intel,
            }
        except Exception as e:
            logger.warning("Quantitative analysis failed: %s", e)
            warnings.append(f"Quantitative analysis failed: {e}")

        _lap("ALL DONE")
        # ── Assemble result ──
        run["current_step"] = "completed"
        run["progress"] = 1.0
        run["status"] = AnalysisStatus.COMPLETED
        run["completed_at"] = datetime.now(timezone.utc)
        run["fundamentals"] = fundamentals_data
        run["technicals"] = technicals_data
        run["news"] = news_data
        run["forecast"] = forecast_data
        run["risk"] = risk_data
        run["quant"] = quant_data
        run["warnings"] = warnings
        run["attribution"] = attribution

    except Exception as e:
        logger.exception("Analysis pipeline failed for %s", ticker)
        run["status"] = AnalysisStatus.FAILED
        run["current_step"] = "failed"
        run["warnings"] = warnings + [f"Pipeline error: {e}"]
        run["completed_at"] = datetime.now(timezone.utc)


@router.post("/run", response_model=AnalysisStatusResponse)
async def start_analysis(
    req: AnalysisRequest, background_tasks: BackgroundTasks
) -> AnalysisStatusResponse:
    """Submit a new analysis job. Returns a run_id to poll for results."""
    ticker = req.ticker.upper().strip()
    run_id = generate_run_id(ticker, req.seed)

    _runs[run_id] = {
        "run_id": run_id,
        "ticker": ticker,
        "status": AnalysisStatus.PENDING,
        "started_at": datetime.now(timezone.utc),
        "completed_at": None,
        "progress": 0.0,
        "current_step": "queued",
        "warnings": [],
        "snapshot": None,
        "fundamentals": None,
        "technicals": None,
        "news": None,
        "forecast": None,
        "risk": None,
        "quant": None,
        "attribution": [],
    }

    background_tasks.add_task(_run_analysis, run_id, req)

    return AnalysisStatusResponse(
        run_id=run_id,
        ticker=ticker,
        status=AnalysisStatus.PENDING,
        started_at=_runs[run_id]["started_at"],
        progress=0.0,
        current_step="queued",
    )


@router.get("/status/{run_id}", response_model=AnalysisStatusResponse)
async def get_status(run_id: str) -> AnalysisStatusResponse:
    """Poll the status of a running analysis."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")

    run = _runs[run_id]
    return AnalysisStatusResponse(
        run_id=run_id,
        ticker=run["ticker"],
        status=run["status"],
        started_at=run["started_at"],
        completed_at=run.get("completed_at"),
        progress=run["progress"],
        current_step=run["current_step"],
        warnings=run.get("warnings", []),
    )


@router.get("/result/{run_id}", response_model=FullAnalysisResponse)
async def get_result(run_id: str) -> FullAnalysisResponse:
    """Retrieve the full analysis result."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")

    run = _runs[run_id]
    if run["status"] not in (AnalysisStatus.COMPLETED, AnalysisStatus.FAILED):
        raise HTTPException(status_code=202, detail="Analysis still running")

    snapshot = run.get("snapshot")
    if snapshot is None:
        snapshot = SnapshotResponse(ticker=run["ticker"])

    return FullAnalysisResponse(
        run_id=run_id,
        ticker=run["ticker"],
        status=run["status"],
        snapshot=snapshot,
        fundamentals=run.get("fundamentals"),
        technicals=run.get("technicals"),
        news=run.get("news"),
        forecast=run.get("forecast"),
        risk=run.get("risk"),
        quant=run.get("quant"),
        started_at=run["started_at"],
        completed_at=run.get("completed_at"),
        warnings=run.get("warnings", []),
        attribution=run.get("attribution", []),
    )


@router.get("/runs", response_model=list[AnalysisStatusResponse])
async def list_runs() -> list[AnalysisStatusResponse]:
    """List all analysis runs."""
    return [
        AnalysisStatusResponse(
            run_id=r["run_id"],
            ticker=r["ticker"],
            status=r["status"],
            started_at=r["started_at"],
            completed_at=r.get("completed_at"),
            progress=r["progress"],
            current_step=r["current_step"],
        )
        for r in _runs.values()
    ]


@router.post("/sync", response_model=FullAnalysisResponse)
async def run_analysis_sync(req: AnalysisRequest) -> FullAnalysisResponse:
    """Run the full analysis synchronously and return the result.

    This avoids in-memory state issues on single-worker deployments (Railway).
    The client waits for the full result — no polling needed.
    """
    import asyncio

    ticker = req.ticker.upper().strip()
    run_id = generate_run_id(ticker, req.seed)

    _runs[run_id] = {
        "run_id": run_id,
        "ticker": ticker,
        "status": AnalysisStatus.PENDING,
        "started_at": datetime.now(timezone.utc),
        "completed_at": None,
        "progress": 0.0,
        "current_step": "queued",
        "warnings": [],
        "snapshot": None,
        "fundamentals": None,
        "technicals": None,
        "news": None,
        "forecast": None,
        "risk": None,
        "quant": None,
        "attribution": [],
    }

    # Run the heavy analysis in a thread so we don't block the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_analysis, run_id, req)

    run = _runs[run_id]
    snapshot = run.get("snapshot")
    if snapshot is None:
        snapshot = SnapshotResponse(ticker=ticker)

    return FullAnalysisResponse(
        run_id=run_id,
        ticker=ticker,
        status=run["status"],
        snapshot=snapshot,
        fundamentals=run.get("fundamentals"),
        technicals=run.get("technicals"),
        news=run.get("news"),
        forecast=run.get("forecast"),
        risk=run.get("risk"),
        quant=run.get("quant"),
        started_at=run["started_at"],
        completed_at=run.get("completed_at"),
        warnings=run.get("warnings", []),
        attribution=run.get("attribution", []),
    )


@router.post("/stream")
async def run_analysis_stream(req: AnalysisRequest):
    """Run analysis with SSE progress streaming.

    Streams `event: progress` messages with step/progress data,
    then a final `event: result` with the full analysis JSON.
    """
    import asyncio
    import json

    from starlette.responses import StreamingResponse

    ticker = req.ticker.upper().strip()
    run_id = generate_run_id(ticker, req.seed)

    _runs[run_id] = {
        "run_id": run_id,
        "ticker": ticker,
        "status": AnalysisStatus.PENDING,
        "started_at": datetime.now(timezone.utc),
        "completed_at": None,
        "progress": 0.0,
        "current_step": "queued",
        "warnings": [],
        "snapshot": None,
        "fundamentals": None,
        "technicals": None,
        "news": None,
        "forecast": None,
        "risk": None,
        "quant": None,
        "attribution": [],
    }

    async def _event_stream():
        loop = asyncio.get_event_loop()
        # Start analysis in background thread
        analysis_future = loop.run_in_executor(None, _run_analysis, run_id, req)

        last_step = ""
        last_progress = 0.0
        while not analysis_future.done():
            run = _runs.get(run_id, {})
            step = run.get("current_step", "")
            progress = run.get("progress", 0.0)
            if step != last_step or abs(progress - last_progress) > 0.01:
                msg = json.dumps({"step": step, "progress": round(progress, 2)})
                yield f"event: progress\ndata: {msg}\n\n"
                last_step = step
                last_progress = progress
            await asyncio.sleep(0.3)

        # Ensure we catch any exception from the analysis
        try:
            analysis_future.result()
        except Exception:
            pass

        # Send final progress
        yield f"event: progress\ndata: {json.dumps({'step': 'completed', 'progress': 1.0})}\n\n"

        # Build and send the full result
        run = _runs[run_id]
        snapshot = run.get("snapshot")
        if snapshot is None:
            snapshot = SnapshotResponse(ticker=ticker)

        result = FullAnalysisResponse(
            run_id=run_id,
            ticker=ticker,
            status=run["status"],
            snapshot=snapshot,
            fundamentals=run.get("fundamentals"),
            technicals=run.get("technicals"),
            news=run.get("news"),
            forecast=run.get("forecast"),
            risk=run.get("risk"),
            quant=run.get("quant"),
            started_at=run["started_at"],
            completed_at=run.get("completed_at"),
            warnings=run.get("warnings", []),
            attribution=run.get("attribution", []),
        )

        result_json = result.model_dump_json() if hasattr(result, "model_dump_json") else json.dumps(result.dict(), default=str)
        yield f"event: result\ndata: {result_json}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
