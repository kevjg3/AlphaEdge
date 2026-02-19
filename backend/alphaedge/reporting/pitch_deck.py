"""Generate professional PPTX pitch decks from analysis results."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from io import BytesIO
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
NAVY = RGBColor(0x0B, 0x14, 0x2B)
DARK_PANEL = RGBColor(0x10, 0x1D, 0x3A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xA0, 0xAE, 0xC4)
MID_GRAY = RGBColor(0x64, 0x74, 0x8B)
ACCENT_BLUE = RGBColor(0x63, 0x66, 0xF1)   # indigo-500
ACCENT_GREEN = RGBColor(0x10, 0xB9, 0x81)
ACCENT_RED = RGBColor(0xEF, 0x44, 0x44)
ACCENT_GOLD = RGBColor(0xF5, 0x9E, 0x0B)
BORDER_COLOR = RGBColor(0x1E, 0x29, 0x3B)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# matplotlib dark theme
_MPL_RC = {
    "figure.facecolor": "#0B142B",
    "axes.facecolor": "#101D3A",
    "text.color": "#FFFFFF",
    "axes.labelcolor": "#A0AEC4",
    "xtick.color": "#A0AEC4",
    "ytick.color": "#A0AEC4",
    "axes.edgecolor": "#1E293B",
    "grid.color": "#1E293B",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sg(data: dict | None, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts."""
    cur = data
    for k in keys:
        if cur is None or not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _fmt_large(n: Any) -> str:
    if n is None:
        return "\u2014"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "\u2014"
    if abs(n) >= 1e12:
        return f"${n / 1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"${n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n / 1e6:.1f}M"
    if abs(n) >= 1e3:
        return f"${n / 1e3:.1f}K"
    return f"${n:,.0f}"


def _fmt_pct(n: Any, decimals: int = 1) -> str:
    if n is None:
        return "\u2014"
    try:
        return f"{float(n) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "\u2014"


def _fmt_ratio(n: Any, decimals: int = 2) -> str:
    if n is None:
        return "\u2014"
    try:
        return f"{float(n):.{decimals}f}"
    except (TypeError, ValueError):
        return "\u2014"


def _fmt_price(n: Any) -> str:
    if n is None:
        return "\u2014"
    try:
        return f"${float(n):,.2f}"
    except (TypeError, ValueError):
        return "\u2014"


# ---------------------------------------------------------------------------
# PitchDeckGenerator
# ---------------------------------------------------------------------------

class PitchDeckGenerator:
    """Generate a PPTX pitch deck from AlphaEdge analysis results."""

    def __init__(self) -> None:
        self._prs: Presentation | None = None
        self._charts_dir: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, analysis: dict) -> BytesIO:
        """Build pitch deck and return as in-memory bytes."""
        self._charts_dir = tempfile.mkdtemp(prefix="alphaedge_deck_")
        try:
            self._prs = Presentation()
            self._prs.slide_width = SLIDE_W
            self._prs.slide_height = SLIDE_H

            self._add_recommendation_slide(analysis)
            self._add_business_overview_slide(analysis)
            self._add_financial_summary_slide(analysis)
            self._add_industry_slide(analysis)
            self._add_why_mispriced_slide(analysis)
            self._add_thesis_slides(analysis)
            self._add_downside_slide(analysis)
            self._add_valuation_slide(analysis)
            self._add_risks_slide(analysis)
            self._add_conclusion_slide(analysis)

            buf = BytesIO()
            self._prs.save(buf)
            buf.seek(0)
            return buf
        finally:
            shutil.rmtree(self._charts_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Slide helpers
    # ------------------------------------------------------------------

    def _blank_slide(self):
        layout = self._prs.slide_layouts[6]  # blank
        slide = self._prs.slides.add_slide(layout)
        bg = slide.background.fill
        bg.solid()
        bg.fore_color.rgb = NAVY
        return slide

    def _title_bar(self, slide, text: str):
        """Add a thin accent title bar at the top of a slide."""
        # accent line
        ln = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(0.45), Inches(0.08), Inches(0.35),
        )
        ln.fill.solid()
        ln.fill.fore_color.rgb = ACCENT_BLUE
        ln.line.fill.background()
        # title text
        self._text_box(slide, Inches(0.9), Inches(0.35), Inches(10), Inches(0.55),
                       text, size=24, bold=True, color=WHITE)

    def _text_box(self, slide, left, top, width, height, text: str, *,
                  size: int = 14, bold: bool = False, color=LIGHT_GRAY,
                  align=PP_ALIGN.LEFT, font_name: str = "Calibri"):
        txbox = slide.shapes.add_textbox(left, top, width, height)
        tf = txbox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = str(text)
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = font_name
        p.alignment = align
        return txbox

    def _metric_card(self, slide, left, top, label: str, value: str,
                     width=Inches(2.4), height=Inches(0.9), *,
                     value_color=WHITE, label_color=MID_GRAY):
        """Render a small metric card (label on top, big value below)."""
        # background panel
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height,
        )
        box.fill.solid()
        box.fill.fore_color.rgb = DARK_PANEL
        box.line.color.rgb = BORDER_COLOR
        box.line.width = Pt(0.75)
        # label
        self._text_box(slide, left + Inches(0.15), top + Inches(0.08),
                       width - Inches(0.3), Inches(0.25),
                       label.upper(), size=8, bold=True, color=label_color)
        # value
        self._text_box(slide, left + Inches(0.15), top + Inches(0.35),
                       width - Inches(0.3), Inches(0.45),
                       value, size=18, bold=True, color=value_color)

    def _footer(self, slide, ticker: str = ""):
        """Add footer with branding and disclaimer."""
        self._text_box(
            slide, Inches(0.6), Inches(6.95), Inches(6), Inches(0.35),
            "AlphaEdge Analysis Platform  \u2022  For informational purposes only",
            size=7, color=MID_GRAY,
        )
        if ticker:
            self._text_box(
                slide, Inches(10), Inches(6.95), Inches(2.7), Inches(0.35),
                ticker, size=7, color=MID_GRAY, align=PP_ALIGN.RIGHT,
            )

    def _add_picture(self, slide, path: str, left, top, width=None, height=None):
        """Embed a chart image."""
        kwargs = {"left": left, "top": top}
        if width:
            kwargs["width"] = width
        if height:
            kwargs["height"] = height
        slide.shapes.add_picture(path, **kwargs)

    def _divider(self, slide, top):
        ln = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.6), top, Inches(12.13), Pt(1),
        )
        ln.fill.solid()
        ln.fill.fore_color.rgb = BORDER_COLOR
        ln.line.fill.background()

    # ------------------------------------------------------------------
    # SLIDE 1: Recommendation
    # ------------------------------------------------------------------

    def _add_recommendation_slide(self, analysis: dict):
        slide = self._blank_slide()
        snap = analysis.get("snapshot") or {}
        fund = analysis.get("fundamentals") or {}
        verdict = _sg(fund, "verdict") or {}
        thesis = _sg(fund, "investment_thesis") or {}

        name = snap.get("name", analysis.get("ticker", ""))
        ticker = snap.get("ticker", analysis.get("ticker", ""))
        price = snap.get("price")
        label = verdict.get("label", "")
        upside = verdict.get("upside_pct")
        fair_mid = verdict.get("fair_value_mid")

        is_long = label in ("undervalued", "fairly_valued")
        rec_text = "LONG" if is_long else "SHORT"
        rec_color = ACCENT_GREEN if is_long else ACCENT_RED

        # Recommendation badge
        badge = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(0.8), Inches(1.2), Inches(2.2), Inches(0.7),
        )
        badge.fill.solid()
        badge.fill.fore_color.rgb = rec_color
        badge.line.fill.background()
        tf = badge.text_frame
        tf.word_wrap = False
        p = tf.paragraphs[0]
        p.text = rec_text
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.font.name = "Calibri"
        p.alignment = PP_ALIGN.CENTER
        tf.paragraphs[0].space_before = Pt(4)

        # Company name + ticker
        self._text_box(slide, Inches(0.8), Inches(2.2), Inches(11), Inches(0.7),
                       f"{name} ({ticker})", size=36, bold=True, color=WHITE)

        # Key metrics row
        x = Inches(0.8)
        y = Inches(3.3)
        gap = Inches(3.1)
        if price is not None:
            self._metric_card(slide, x, y, "Current Price", _fmt_price(price), width=Inches(2.8))
        if fair_mid is not None:
            self._metric_card(slide, x + gap, y, "Price Target", _fmt_price(fair_mid), width=Inches(2.8),
                              value_color=rec_color)
        if upside is not None:
            sign = "+" if upside > 0 else ""
            self._metric_card(slide, x + gap * 2, y, "Implied Upside",
                              f"{sign}{upside:.1f}%", width=Inches(2.8),
                              value_color=ACCENT_GREEN if upside > 0 else ACCENT_RED)

        # One-line thesis
        thesis_text = thesis.get("investment_thesis", "")
        if thesis_text:
            first_sentence = thesis_text.split(". ")[0] + "."
            if len(first_sentence) > 250:
                first_sentence = first_sentence[:247] + "..."
            self._text_box(slide, Inches(0.8), Inches(4.8), Inches(11.5), Inches(1.2),
                           first_sentence, size=16, color=LIGHT_GRAY)

        # Sector / Industry
        sector = snap.get("sector", "")
        industry = snap.get("industry", "")
        if sector or industry:
            self._text_box(slide, Inches(0.8), Inches(6.2), Inches(8), Inches(0.35),
                           f"{sector}  \u2022  {industry}", size=11, color=MID_GRAY)

        self._footer(slide, ticker)

    # ------------------------------------------------------------------
    # SLIDE 2: Business Overview
    # ------------------------------------------------------------------

    def _add_business_overview_slide(self, analysis: dict):
        slide = self._blank_slide()
        self._title_bar(slide, "Business Overview")

        snap = analysis.get("snapshot") or {}
        thesis = _sg(analysis, "fundamentals", "investment_thesis") or {}

        # Left column — company description
        overview = thesis.get("company_overview", snap.get("description", ""))
        if overview:
            if len(overview) > 700:
                overview = overview[:697] + "..."
            self._text_box(slide, Inches(0.6), Inches(1.2), Inches(6.5), Inches(3.5),
                           overview, size=13, color=LIGHT_GRAY)

        # Right column — key stats
        x = Inches(7.8)
        y = Inches(1.2)
        w = Inches(2.3)
        h = Inches(0.85)
        gap_y = Inches(0.95)

        stats = [
            ("Market Cap", _fmt_large(snap.get("market_cap"))),
            ("Revenue", _fmt_large(snap.get("total_revenue"))),
            ("Sector", snap.get("sector", "\u2014")),
            ("Employees", f"{snap.get('employees', 0):,}" if snap.get("employees") else "\u2014"),
            ("Operating Margin", _fmt_pct(snap.get("operating_margins"))),
            ("P/E Ratio", _fmt_ratio(snap.get("pe_ratio"))),
        ]

        for i, (label, value) in enumerate(stats):
            row = i // 2
            col = i % 2
            self._metric_card(slide, x + col * Inches(2.5), y + row * gap_y,
                              label, str(value), width=w, height=h)

        self._footer(slide, snap.get("ticker", ""))

    # ------------------------------------------------------------------
    # SLIDE 3: Financial Summary
    # ------------------------------------------------------------------

    def _add_financial_summary_slide(self, analysis: dict):
        slide = self._blank_slide()
        self._title_bar(slide, "Financial Summary")

        fh = _sg(analysis, "fundamentals", "financial_health") or {}
        income = fh.get("income_summary") or {}
        balance = fh.get("balance_summary") or {}
        cashflow = fh.get("cashflow_summary") or {}
        snap = analysis.get("snapshot") or {}

        col_w = Inches(3.8)
        col_gap = Inches(0.35)
        y_start = Inches(1.3)

        columns = [
            ("Income Statement", [
                ("Revenue", _fmt_large(income.get("revenue"))),
                ("Revenue Growth", _fmt_pct(income.get("revenue_growth_yoy"))),
                ("Gross Margin", _fmt_pct(income.get("gross_margin"))),
                ("Operating Margin", _fmt_pct(income.get("operating_margin"))),
                ("Net Margin", _fmt_pct(income.get("net_margin"))),
                ("EPS", _fmt_price(income.get("eps"))),
            ]),
            ("Balance Sheet", [
                ("Total Assets", _fmt_large(balance.get("total_assets"))),
                ("Total Debt", _fmt_large(balance.get("total_debt"))),
                ("Cash & Equiv", _fmt_large(balance.get("cash"))),
                ("Net Debt", _fmt_large(balance.get("net_debt"))),
                ("Current Ratio", _fmt_ratio(balance.get("current_ratio"))),
                ("Debt / Equity", _fmt_ratio(balance.get("debt_to_equity"))),
            ]),
            ("Cash Flow", [
                ("Operating CF", _fmt_large(cashflow.get("operating_cf"))),
                ("CapEx", _fmt_large(cashflow.get("capex"))),
                ("Free Cash Flow", _fmt_large(cashflow.get("free_cf"))),
                ("FCF Margin", _fmt_pct(cashflow.get("fcf_margin"))),
                ("FCF Yield", _fmt_pct(cashflow.get("fcf_yield"))),
                ("Dividend Yield", _fmt_pct(snap.get("dividend_yield"))),
            ]),
        ]

        for ci, (col_title, items) in enumerate(columns):
            x = Inches(0.6) + ci * (col_w + col_gap)
            # Column header
            hdr = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE, x, y_start, col_w, Inches(0.4),
            )
            hdr.fill.solid()
            hdr.fill.fore_color.rgb = ACCENT_BLUE
            hdr.line.fill.background()
            self._text_box(slide, x + Inches(0.15), y_start + Inches(0.02),
                           col_w - Inches(0.3), Inches(0.35),
                           col_title, size=11, bold=True, color=WHITE)

            # Rows
            for ri, (label, value) in enumerate(items):
                ry = y_start + Inches(0.55) + ri * Inches(0.72)
                # background stripe
                stripe = slide.shapes.add_shape(
                    MSO_SHAPE.RECTANGLE, x, ry, col_w, Inches(0.62),
                )
                stripe.fill.solid()
                stripe.fill.fore_color.rgb = DARK_PANEL if ri % 2 == 0 else NAVY
                stripe.line.fill.background()

                self._text_box(slide, x + Inches(0.15), ry + Inches(0.05),
                               Inches(1.8), Inches(0.5),
                               label, size=10, color=MID_GRAY)
                self._text_box(slide, x + Inches(2.0), ry + Inches(0.05),
                               Inches(1.6), Inches(0.5),
                               value, size=12, bold=True, color=WHITE, align=PP_ALIGN.RIGHT)

        self._footer(slide, snap.get("ticker", ""))

    # ------------------------------------------------------------------
    # SLIDE 4: Industry & Competitive Position
    # ------------------------------------------------------------------

    def _add_industry_slide(self, analysis: dict):
        slide = self._blank_slide()
        self._title_bar(slide, "Industry & Competitive Position")

        fund = analysis.get("fundamentals") or {}
        thesis = fund.get("investment_thesis") or {}
        comps = fund.get("comps_valuation") or {}
        snap = analysis.get("snapshot") or {}

        # Left — industry analysis text
        ind_text = thesis.get("industry_analysis", "")
        if ind_text:
            if len(ind_text) > 600:
                ind_text = ind_text[:597] + "..."
            self._text_box(slide, Inches(0.6), Inches(1.2), Inches(6), Inches(3),
                           ind_text, size=12, color=LIGHT_GRAY)

        # Right — peer comparison chart
        chart_path = self._create_peer_chart(comps, snap)
        if chart_path:
            self._add_picture(slide, chart_path, Inches(7.2), Inches(1.1),
                              width=Inches(5.5), height=Inches(3.3))

        # Bottom — percentile rank badges
        pct_rank = comps.get("percentile_rank") or {}
        if pct_rank:
            self._text_box(slide, Inches(0.6), Inches(4.6), Inches(12), Inches(0.35),
                           "PERCENTILE RANK VS PEERS", size=9, bold=True, color=MID_GRAY)
            bx = Inches(0.6)
            by = Inches(5.0)
            for metric, pctile in list(pct_rank.items())[:8]:
                try:
                    pval = float(pctile)
                except (TypeError, ValueError):
                    continue
                color = ACCENT_GREEN if pval >= 60 else (ACCENT_RED if pval <= 30 else LIGHT_GRAY)
                label_clean = metric.replace("_", " ").title()
                self._metric_card(slide, bx, by, label_clean, f"{pval:.0f}th",
                                  width=Inches(1.4), height=Inches(0.7), value_color=color)
                bx += Inches(1.55)

        self._footer(slide, snap.get("ticker", ""))

    # ------------------------------------------------------------------
    # SLIDE 5: Why Mispriced?
    # ------------------------------------------------------------------

    def _add_why_mispriced_slide(self, analysis: dict):
        slide = self._blank_slide()
        self._title_bar(slide, "Why Is This Mispriced?")

        fund = analysis.get("fundamentals") or {}
        verdict = fund.get("verdict") or {}
        news = _sg(analysis, "news", "synthesis") or {}
        thesis = fund.get("investment_thesis") or {}
        combined = fund.get("combined_range") or {}
        snap = analysis.get("snapshot") or {}

        price = snap.get("price")
        low = combined.get("low")
        mid = combined.get("mid")
        high = combined.get("high")

        # Fair value range chart
        chart_path = self._create_fair_value_chart(price, low, mid, high)
        if chart_path:
            self._add_picture(slide, chart_path, Inches(0.6), Inches(1.1),
                              width=Inches(12), height=Inches(2.1))

        # Left column — Sentiment
        y = Inches(3.5)
        sentiment = news.get("overall_sentiment", "")
        momentum = news.get("news_momentum", "")
        narrative = news.get("narrative", "")

        self._text_box(slide, Inches(0.6), y, Inches(3), Inches(0.35),
                       "MARKET SENTIMENT", size=9, bold=True, color=MID_GRAY)
        if sentiment:
            s_color = ACCENT_GREEN if sentiment == "bullish" else (
                ACCENT_RED if sentiment == "bearish" else LIGHT_GRAY)
            self._text_box(slide, Inches(0.6), y + Inches(0.35), Inches(3), Inches(0.4),
                           sentiment.upper(), size=18, bold=True, color=s_color)
        if momentum:
            self._text_box(slide, Inches(0.6), y + Inches(0.8), Inches(5), Inches(0.3),
                           f"News Momentum: {momentum}", size=11, color=LIGHT_GRAY)
        if narrative:
            trunc = narrative[:300] + "..." if len(narrative) > 300 else narrative
            self._text_box(slide, Inches(0.6), y + Inches(1.2), Inches(5.5), Inches(2),
                           trunc, size=11, color=LIGHT_GRAY)

        # Right column — Catalysts
        catalysts = thesis.get("catalysts", [])
        if catalysts:
            self._text_box(slide, Inches(7), y, Inches(3), Inches(0.35),
                           "KEY CATALYSTS", size=9, bold=True, color=MID_GRAY)
            for ci, cat in enumerate(catalysts[:5]):
                self._text_box(slide, Inches(7), y + Inches(0.45 + ci * 0.55),
                               Inches(5.8), Inches(0.5),
                               f"\u25B8  {cat}", size=11, color=LIGHT_GRAY)

        # Reasoning
        reasoning = verdict.get("reasoning", "")
        if reasoning:
            self._text_box(slide, Inches(0.6), Inches(6.3), Inches(12), Inches(0.55),
                           reasoning[:250], size=10, color=MID_GRAY)

        self._footer(slide, snap.get("ticker", ""))

    # ------------------------------------------------------------------
    # SLIDES 6-8: Core Theses
    # ------------------------------------------------------------------

    def _add_thesis_slides(self, analysis: dict):
        fund = analysis.get("fundamentals") or {}
        thesis = fund.get("investment_thesis") or {}
        snap = analysis.get("snapshot") or {}
        forecast_data = analysis.get("forecast") or {}

        strengths = thesis.get("strengths", [])
        catalysts = thesis.get("catalysts", [])
        key_metrics = thesis.get("key_metrics", [])

        # Build thesis items: combine strengths + catalysts
        items: list[tuple[str, str | None]] = []
        for s in strengths[:3]:
            # Find a related catalyst
            related_cat = None
            if catalysts:
                related_cat = catalysts.pop(0) if catalysts else None
            items.append((s, related_cat))

        # Ensure at least 2 slides
        if not items:
            items = [("Investment opportunity based on fundamental analysis.", None)]

        for i, (strength, catalyst) in enumerate(items[:3]):
            slide = self._blank_slide()
            self._title_bar(slide, f"Core Thesis {i + 1}")

            # Main thesis text
            self._text_box(slide, Inches(0.6), Inches(1.3), Inches(11.5), Inches(1.5),
                           strength, size=18, bold=True, color=WHITE)

            # Catalyst
            if catalyst:
                self._text_box(slide, Inches(0.6), Inches(3.0), Inches(3), Inches(0.3),
                               "CATALYST", size=9, bold=True, color=ACCENT_GOLD)
                self._text_box(slide, Inches(0.6), Inches(3.35), Inches(11), Inches(1),
                               catalyst, size=13, color=LIGHT_GRAY)

            # Supporting metrics
            y_met = Inches(4.5)
            shown = 0
            for km in key_metrics:
                if shown >= 4:
                    break
                label = km.get("label", "")
                value = km.get("value", "")
                if label and value:
                    self._metric_card(slide, Inches(0.6) + shown * Inches(3.1), y_met,
                                      label, str(value), width=Inches(2.8), height=Inches(0.85))
                    shown += 1

            # Forecast horizon if available
            horizons = forecast_data.get("forecasts") or {}
            horizon_12m = horizons.get("12M") or {}
            ep = horizon_12m.get("ensemble_prediction") or {}
            pred_ret = ep.get("predicted_return")
            confidence = ep.get("confidence")
            if pred_ret is not None and i == 0:
                self._text_box(slide, Inches(0.6), Inches(5.7), Inches(8), Inches(0.35),
                               f"12-Month Forecast: {pred_ret:+.1f}% return "
                               f"({confidence * 100:.0f}% confidence)" if confidence else
                               f"12-Month Forecast: {pred_ret:+.1f}% return",
                               size=11, color=ACCENT_BLUE)

            self._footer(slide, snap.get("ticker", ""))

    # ------------------------------------------------------------------
    # SLIDE 9: Downside Protection
    # ------------------------------------------------------------------

    def _add_downside_slide(self, analysis: dict):
        slide = self._blank_slide()
        self._title_bar(slide, "Downside Protection")

        risk = analysis.get("risk") or {}
        var_data = risk.get("var") or {}
        scenarios = risk.get("scenarios") or []
        fund = analysis.get("fundamentals") or {}
        fh = fund.get("financial_health") or {}
        cashflow = fh.get("cashflow_summary") or {}
        balance = fh.get("balance_summary") or {}
        quant = analysis.get("quant") or {}
        perf = quant.get("performance") or {}
        snap = analysis.get("snapshot") or {}

        # Top-left: VaR metrics
        self._text_box(slide, Inches(0.6), Inches(1.2), Inches(4), Inches(0.3),
                       "VALUE AT RISK (95%, 1-DAY)", size=9, bold=True, color=MID_GRAY)

        var_items = [
            ("Parametric VaR", _fmt_pct(var_data.get("parametric_var_pct"))),
            ("Historical VaR", _fmt_pct(var_data.get("historical_var_pct"))),
            ("Monte Carlo VaR", _fmt_pct(var_data.get("monte_carlo_var_pct"))),
            ("CVaR (Exp. Shortfall)", _fmt_pct(var_data.get("expected_shortfall_pct"))),
        ]
        for vi, (label, value) in enumerate(var_items):
            y = Inches(1.6) + vi * Inches(0.55)
            self._text_box(slide, Inches(0.6), y, Inches(2.8), Inches(0.45),
                           label, size=10, color=LIGHT_GRAY)
            self._text_box(slide, Inches(3.5), y, Inches(1.2), Inches(0.45),
                           value, size=11, bold=True, color=ACCENT_RED, align=PP_ALIGN.RIGHT)

        # Top-right: Cash & liquidity
        self._text_box(slide, Inches(5.5), Inches(1.2), Inches(4), Inches(0.3),
                       "CASH FLOW RESILIENCE", size=9, bold=True, color=MID_GRAY)

        cash_items = [
            ("Free Cash Flow", _fmt_large(cashflow.get("free_cf"))),
            ("FCF Margin", _fmt_pct(cashflow.get("fcf_margin"))),
            ("Cash Position", _fmt_large(balance.get("cash"))),
            ("Current Ratio", _fmt_ratio(balance.get("current_ratio"))),
        ]
        for vi, (label, value) in enumerate(cash_items):
            y = Inches(1.6) + vi * Inches(0.55)
            self._text_box(slide, Inches(5.5), y, Inches(2.8), Inches(0.45),
                           label, size=10, color=LIGHT_GRAY)
            self._text_box(slide, Inches(8.5), y, Inches(1.5), Inches(0.45),
                           value, size=11, bold=True, color=WHITE, align=PP_ALIGN.RIGHT)

        # Far right: risk-adjusted returns
        self._text_box(slide, Inches(10.5), Inches(1.2), Inches(2.5), Inches(0.3),
                       "RISK METRICS", size=9, bold=True, color=MID_GRAY)
        risk_items = [
            ("Sharpe Ratio", _fmt_ratio(perf.get("sharpe_ratio"))),
            ("Sortino Ratio", _fmt_ratio(perf.get("sortino_ratio"))),
            ("Max Drawdown", _fmt_pct(perf.get("max_drawdown"))),
            ("Ann. Volatility", _fmt_pct(perf.get("annualized_vol"))),
        ]
        for vi, (label, value) in enumerate(risk_items):
            y = Inches(1.6) + vi * Inches(0.55)
            self._text_box(slide, Inches(10.5), y, Inches(1.6), Inches(0.45),
                           label, size=9, color=LIGHT_GRAY)
            self._text_box(slide, Inches(12.1), y, Inches(1), Inches(0.45),
                           value, size=10, bold=True, color=WHITE, align=PP_ALIGN.RIGHT)

        # Bottom: Historical stress tests
        self._divider(slide, Inches(4.0))
        self._text_box(slide, Inches(0.6), Inches(4.2), Inches(4), Inches(0.3),
                       "HISTORICAL STRESS TESTS", size=9, bold=True, color=MID_GRAY)

        if isinstance(scenarios, list) and scenarios:
            # Header row
            headers = ["Scenario", "Stock Return", "Benchmark", "Max Drawdown"]
            hx_positions = [Inches(0.6), Inches(5), Inches(7.5), Inches(10)]
            hx_widths = [Inches(4.2), Inches(2.3), Inches(2.3), Inches(2.3)]
            for hi, header in enumerate(headers):
                self._text_box(slide, hx_positions[hi], Inches(4.55),
                               hx_widths[hi], Inches(0.3),
                               header, size=9, bold=True, color=MID_GRAY)

            for si, sc in enumerate(scenarios[:5]):
                if not isinstance(sc, dict):
                    continue
                sy = Inches(4.9) + si * Inches(0.4)
                name = sc.get("name", sc.get("scenario", f"Scenario {si + 1}"))
                stock_ret = sc.get("stock_return") or sc.get("return")
                bench_ret = sc.get("benchmark_return")
                mdd = sc.get("max_drawdown")

                self._text_box(slide, hx_positions[0], sy, hx_widths[0], Inches(0.35),
                               str(name)[:40], size=10, color=LIGHT_GRAY)
                self._text_box(slide, hx_positions[1], sy, hx_widths[1], Inches(0.35),
                               _fmt_pct(stock_ret), size=10, bold=True,
                               color=ACCENT_RED if stock_ret and float(stock_ret) < 0 else ACCENT_GREEN)
                self._text_box(slide, hx_positions[2], sy, hx_widths[2], Inches(0.35),
                               _fmt_pct(bench_ret), size=10, color=LIGHT_GRAY)
                self._text_box(slide, hx_positions[3], sy, hx_widths[3], Inches(0.35),
                               _fmt_pct(mdd), size=10, color=LIGHT_GRAY)

        self._footer(slide, snap.get("ticker", ""))

    # ------------------------------------------------------------------
    # SLIDE 10: Valuation
    # ------------------------------------------------------------------

    def _add_valuation_slide(self, analysis: dict):
        slide = self._blank_slide()
        self._title_bar(slide, "Valuation")

        fund = analysis.get("fundamentals") or {}
        dcf = fund.get("dcf_valuation") or {}
        comps = fund.get("comps_valuation") or {}
        combined = fund.get("combined_range") or {}
        snap = analysis.get("snapshot") or {}
        price = snap.get("price")

        # Left: DCF section
        self._text_box(slide, Inches(0.6), Inches(1.2), Inches(4), Inches(0.3),
                       "DCF VALUATION", size=9, bold=True, color=MID_GRAY)

        dcf_items = [
            ("Implied Price", _fmt_price(dcf.get("implied_price"))),
            ("Enterprise Value", _fmt_large(dcf.get("enterprise_value"))),
            ("Equity Value", _fmt_large(dcf.get("equity_value"))),
        ]
        for di, (label, value) in enumerate(dcf_items):
            self._metric_card(slide, Inches(0.6) + di * Inches(2.1), Inches(1.55),
                              label, value, width=Inches(1.95), height=Inches(0.8))

        # DCF assumptions
        assumptions = dcf.get("assumptions_used") or {}
        if assumptions:
            self._text_box(slide, Inches(0.6), Inches(2.55), Inches(3), Inches(0.25),
                           "ASSUMPTIONS", size=8, bold=True, color=MID_GRAY)
            a_items = [
                f"WACC: {_fmt_pct(assumptions.get('wacc'))}",
                f"Terminal Growth: {_fmt_pct(assumptions.get('terminal_growth_rate'))}",
                f"Op Margin: {_fmt_pct(assumptions.get('operating_margin'))}",
            ]
            self._text_box(slide, Inches(0.6), Inches(2.8), Inches(6), Inches(0.5),
                           "   \u2022   ".join(a_items), size=10, color=LIGHT_GRAY)

        # Right: Comps section
        val_range = comps.get("valuation_range") or {}
        self._text_box(slide, Inches(7.5), Inches(1.2), Inches(5), Inches(0.3),
                       "COMPARABLE COMPANY ANALYSIS", size=9, bold=True, color=MID_GRAY)

        approaches = [
            ("P/E Approach", val_range.get("pe_approach") or {}),
            ("EV/EBITDA Approach", val_range.get("ev_ebitda_approach") or {}),
            ("EV/Revenue Approach", val_range.get("ev_revenue_approach") or {}),
        ]
        for ai, (approach_name, approach_data) in enumerate(approaches):
            y = Inches(1.6) + ai * Inches(0.55)
            implied = approach_data.get("implied_value") or approach_data.get("implied_market_cap")
            self._text_box(slide, Inches(7.5), y, Inches(3), Inches(0.45),
                           approach_name, size=10, color=LIGHT_GRAY)
            self._text_box(slide, Inches(10.5), y, Inches(2.2), Inches(0.45),
                           _fmt_large(implied), size=11, bold=True, color=WHITE, align=PP_ALIGN.RIGHT)

        # Bottom: Sensitivity heatmap + Bull/Base/Bear
        self._divider(slide, Inches(3.4))

        # Sensitivity heatmap
        heatmap_path = self._create_sensitivity_heatmap(dcf, price)
        if heatmap_path:
            self._add_picture(slide, heatmap_path, Inches(0.4), Inches(3.6),
                              width=Inches(6.5), height=Inches(3.5))

        # Bull / Base / Bear range
        range_path = self._create_range_chart(combined, price)
        if range_path:
            self._add_picture(slide, range_path, Inches(7.2), Inches(3.6),
                              width=Inches(5.5), height=Inches(1.8))

        # Combined range metrics
        if combined:
            y_range = Inches(5.7)
            self._metric_card(slide, Inches(7.2), y_range, "Bear Case",
                              _fmt_price(combined.get("low")),
                              width=Inches(1.7), height=Inches(0.75), value_color=ACCENT_RED)
            self._metric_card(slide, Inches(9.1), y_range, "Base Case",
                              _fmt_price(combined.get("mid")),
                              width=Inches(1.7), height=Inches(0.75), value_color=ACCENT_BLUE)
            self._metric_card(slide, Inches(11), y_range, "Bull Case",
                              _fmt_price(combined.get("high")),
                              width=Inches(1.7), height=Inches(0.75), value_color=ACCENT_GREEN)

        self._footer(slide, snap.get("ticker", ""))

    # ------------------------------------------------------------------
    # SLIDE 11: Risks & Mitigants
    # ------------------------------------------------------------------

    def _add_risks_slide(self, analysis: dict):
        slide = self._blank_slide()
        self._title_bar(slide, "Risks & Mitigants")

        fund = analysis.get("fundamentals") or {}
        thesis = fund.get("investment_thesis") or {}
        snap = analysis.get("snapshot") or {}

        risks = thesis.get("risks", [])
        strengths = thesis.get("strengths", [])

        # Build risk/mitigant pairs
        pairs: list[tuple[str, str]] = []
        for i, risk_text in enumerate(risks[:5]):
            mitigant = strengths[i] if i < len(strengths) else "Fundamental value provides margin of safety."
            pairs.append((risk_text, mitigant))

        if not pairs:
            self._text_box(slide, Inches(0.6), Inches(2), Inches(10), Inches(1),
                           "No significant risks identified in analysis.",
                           size=14, color=LIGHT_GRAY)
            self._footer(slide, snap.get("ticker", ""))
            return

        # Headers
        self._text_box(slide, Inches(0.6), Inches(1.2), Inches(5.5), Inches(0.35),
                       "RISK", size=10, bold=True, color=ACCENT_RED)
        self._text_box(slide, Inches(7), Inches(1.2), Inches(5.5), Inches(0.35),
                       "MITIGANT", size=10, bold=True, color=ACCENT_GREEN)

        for pi, (risk_text, mitigant) in enumerate(pairs):
            y = Inches(1.7) + pi * Inches(1.0)

            # Row background
            stripe = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE, Inches(0.5), y, Inches(12.33), Inches(0.9),
            )
            stripe.fill.solid()
            stripe.fill.fore_color.rgb = DARK_PANEL if pi % 2 == 0 else NAVY
            stripe.line.fill.background()

            # Risk indicator dot
            dot = slide.shapes.add_shape(
                MSO_SHAPE.OVAL, Inches(0.65), y + Inches(0.1), Inches(0.12), Inches(0.12),
            )
            dot.fill.solid()
            dot.fill.fore_color.rgb = ACCENT_RED
            dot.line.fill.background()

            # Risk text
            r_text = risk_text[:150] + "..." if len(risk_text) > 150 else risk_text
            self._text_box(slide, Inches(0.9), y + Inches(0.05), Inches(5.8), Inches(0.8),
                           r_text, size=11, color=LIGHT_GRAY)

            # Arrow
            self._text_box(slide, Inches(6.7), y + Inches(0.15), Inches(0.3), Inches(0.5),
                           "\u2192", size=16, color=MID_GRAY, align=PP_ALIGN.CENTER)

            # Mitigant indicator dot
            dot2 = slide.shapes.add_shape(
                MSO_SHAPE.OVAL, Inches(7.05), y + Inches(0.1), Inches(0.12), Inches(0.12),
            )
            dot2.fill.solid()
            dot2.fill.fore_color.rgb = ACCENT_GREEN
            dot2.line.fill.background()

            # Mitigant text
            m_text = mitigant[:150] + "..." if len(mitigant) > 150 else mitigant
            self._text_box(slide, Inches(7.3), y + Inches(0.05), Inches(5.4), Inches(0.8),
                           m_text, size=11, color=LIGHT_GRAY)

        self._footer(slide, snap.get("ticker", ""))

    # ------------------------------------------------------------------
    # SLIDE 12: Conclusion
    # ------------------------------------------------------------------

    def _add_conclusion_slide(self, analysis: dict):
        slide = self._blank_slide()

        snap = analysis.get("snapshot") or {}
        fund = analysis.get("fundamentals") or {}
        verdict = fund.get("verdict") or {}
        thesis = fund.get("investment_thesis") or {}
        forecast = analysis.get("forecast") or {}

        ticker = snap.get("ticker", analysis.get("ticker", ""))
        name = snap.get("name", ticker)
        label = verdict.get("label", "")
        upside = verdict.get("upside_pct")
        fair_mid = verdict.get("fair_value_mid")
        price = snap.get("price")

        is_long = label in ("undervalued", "fairly_valued")
        rec_text = "LONG" if is_long else "SHORT"
        rec_color = ACCENT_GREEN if is_long else ACCENT_RED

        # Title
        self._text_box(slide, Inches(0.8), Inches(0.8), Inches(11), Inches(0.7),
                       "Conclusion", size=32, bold=True, color=WHITE)

        # Badge
        badge = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(0.8), Inches(1.8), Inches(1.8), Inches(0.55),
        )
        badge.fill.solid()
        badge.fill.fore_color.rgb = rec_color
        badge.line.fill.background()
        tf = badge.text_frame
        p = tf.paragraphs[0]
        p.text = rec_text
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.font.name = "Calibri"
        p.alignment = PP_ALIGN.CENTER

        self._text_box(slide, Inches(3), Inches(1.8), Inches(8), Inches(0.55),
                       f"{name} ({ticker})  \u2022  Target: {_fmt_price(fair_mid)}  "
                       f"\u2022  Upside: {upside:+.1f}%" if upside else
                       f"{name} ({ticker})",
                       size=18, bold=True, color=WHITE)

        # Key points
        points = []
        strengths = thesis.get("strengths", [])
        catalysts = thesis.get("catalysts", [])
        if strengths:
            points.append(strengths[0])
        if catalysts:
            points.append(catalysts[0])
        reasoning = verdict.get("reasoning", "")
        if reasoning:
            points.append(reasoning)

        overall_dir = forecast.get("overall_direction", "")
        lt_outlook = forecast.get("long_term_outlook", "")
        if overall_dir:
            points.append(f"Forecast direction: {overall_dir}. Long-term outlook: {lt_outlook}.")

        for i, pt in enumerate(points[:4]):
            y = Inches(2.8) + i * Inches(0.75)
            self._text_box(slide, Inches(1.2), y, Inches(11), Inches(0.65),
                           f"{i + 1}.  {pt}", size=14, color=LIGHT_GRAY)

        # Bottom branding
        self._text_box(slide, Inches(0.8), Inches(5.8), Inches(12), Inches(0.5),
                       "Generated by AlphaEdge Analysis Platform",
                       size=12, color=ACCENT_BLUE, align=PP_ALIGN.CENTER)

        self._footer(slide, ticker)

    # ------------------------------------------------------------------
    # Chart generators (matplotlib -> PNG)
    # ------------------------------------------------------------------

    def _create_peer_chart(self, comps: dict, snap: dict) -> str | None:
        """Horizontal bar chart: target vs peer median for key metrics."""
        target = comps.get("target") or {}
        peers = comps.get("peers") or []
        if not target or not peers:
            return None

        metrics = ["pe", "ev_ebitda", "operating_margin", "roe"]
        labels = ["P/E", "EV/EBITDA", "Op Margin", "ROE"]

        target_vals = []
        peer_medians = []

        for m in metrics:
            tv = target.get(m)
            peer_vs = [p.get(m) for p in peers if p.get(m) is not None]
            if tv is None or not peer_vs:
                target_vals.append(0)
                peer_medians.append(0)
            else:
                try:
                    target_vals.append(float(tv))
                    peer_medians.append(float(np.median(peer_vs)))
                except (TypeError, ValueError):
                    target_vals.append(0)
                    peer_medians.append(0)

        # Filter out all-zero metrics
        valid = [(l, t, p) for l, t, p in zip(labels, target_vals, peer_medians)
                 if t != 0 or p != 0]
        if not valid:
            return None

        labels_f, target_f, peer_f = zip(*valid)

        with plt.rc_context(_MPL_RC):
            fig, ax = plt.subplots(figsize=(5.5, 3), dpi=150)
            y_pos = np.arange(len(labels_f))
            bar_h = 0.35

            ax.barh(y_pos - bar_h / 2, target_f, bar_h, label=snap.get("ticker", "Target"),
                    color="#6366F1", edgecolor="none")
            ax.barh(y_pos + bar_h / 2, peer_f, bar_h, label="Peer Median",
                    color="#475569", edgecolor="none")

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels_f)
            ax.legend(loc="lower right", fontsize=8, facecolor="#0B142B", edgecolor="#1E293B")
            ax.grid(axis="x", alpha=0.3)
            ax.set_title("Target vs Peer Median", fontsize=12, color="white", pad=10)
            fig.tight_layout()

            path = os.path.join(self._charts_dir, "peer_chart.png")
            fig.savefig(path, bbox_inches="tight", facecolor="#0B142B")
            plt.close(fig)
            return path

    def _create_fair_value_chart(self, price, low, mid, high) -> str | None:
        """Horizontal bar showing current price vs fair value range."""
        vals = [price, low, mid, high]
        if any(v is None for v in vals):
            return None
        try:
            price, low, mid, high = [float(v) for v in vals]
        except (TypeError, ValueError):
            return None

        with plt.rc_context(_MPL_RC):
            fig, ax = plt.subplots(figsize=(12, 1.8), dpi=150)

            all_vals = [price, low, mid, high]
            x_min = min(all_vals) * 0.85
            x_max = max(all_vals) * 1.15

            # Range bar
            ax.barh(0, high - low, left=low, height=0.4, color="#1E293B",
                    edgecolor="#6366F1", linewidth=1.5)

            # Mid marker
            ax.plot(mid, 0, marker="D", color="#6366F1", markersize=12, zorder=5)
            ax.annotate(f"Fair Value\n${mid:.2f}", (mid, 0.35),
                        ha="center", va="bottom", fontsize=9, color="#6366F1", fontweight="bold")

            # Current price marker
            ax.plot(price, 0, marker="v", color="#F59E0B", markersize=14, zorder=5)
            ax.annotate(f"Current\n${price:.2f}", (price, -0.35),
                        ha="center", va="top", fontsize=9, color="#F59E0B", fontweight="bold")

            # Bear / Bull labels
            ax.annotate(f"Bear: ${low:.2f}", (low, -0.35),
                        ha="center", va="top", fontsize=8, color="#EF4444")
            ax.annotate(f"Bull: ${high:.2f}", (high, -0.35),
                        ha="center", va="top", fontsize=8, color="#10B981")

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-0.8, 0.8)
            ax.set_yticks([])
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
            ax.set_title("Current Price vs Fair Value Range", fontsize=11, color="white", pad=8)
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()

            path = os.path.join(self._charts_dir, "fair_value.png")
            fig.savefig(path, bbox_inches="tight", facecolor="#0B142B")
            plt.close(fig)
            return path

    def _create_sensitivity_heatmap(self, dcf: dict, current_price) -> str | None:
        """5x5 WACC vs terminal growth heatmap."""
        grid = _sg(dcf, "sensitivity_grid", "implied_prices")
        if not grid or not isinstance(grid, dict):
            return None

        try:
            current_price = float(current_price) if current_price else None
        except (TypeError, ValueError):
            current_price = None

        wacc_keys = sorted(grid.keys())
        if not wacc_keys:
            return None

        first_tg_map = grid[wacc_keys[0]]
        if not isinstance(first_tg_map, dict):
            return None
        tg_keys = sorted(first_tg_map.keys())

        data = []
        for wk in wacc_keys:
            row = []
            tg_map = grid.get(wk, {})
            for tk in tg_keys:
                v = tg_map.get(tk)
                row.append(float(v) if v is not None else np.nan)
            data.append(row)

        arr = np.array(data)
        if arr.size == 0:
            return None

        with plt.rc_context(_MPL_RC):
            fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)

            # Color based on distance from current price
            if current_price:
                vmin = current_price * 0.3
                vmax = current_price * 2.0
                cmap = "RdYlGn"
            else:
                vmin, vmax = np.nanmin(arr), np.nanmax(arr)
                cmap = "YlGnBu"

            im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

            # Annotate cells
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    val = arr[i, j]
                    if np.isnan(val):
                        continue
                    text_color = "black" if val > (vmin + vmax) / 2 else "white"
                    ax.text(j, i, f"${val:.0f}", ha="center", va="center",
                            fontsize=8, color=text_color, fontweight="bold")

            # Format tick labels
            def _fmt_key(k):
                try:
                    return f"{float(k) * 100:.1f}%"
                except (TypeError, ValueError):
                    return str(k)

            ax.set_xticks(range(len(tg_keys)))
            ax.set_xticklabels([_fmt_key(k) for k in tg_keys], fontsize=8)
            ax.set_yticks(range(len(wacc_keys)))
            ax.set_yticklabels([_fmt_key(k) for k in wacc_keys], fontsize=8)

            ax.set_xlabel("Terminal Growth Rate", fontsize=9)
            ax.set_ylabel("WACC", fontsize=9)
            ax.set_title("DCF Sensitivity Analysis", fontsize=11, color="white", pad=10)

            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.ax.tick_params(labelsize=8)

            if current_price:
                cbar.ax.axhline(y=current_price, color="#F59E0B", linewidth=2, label="Current")

            fig.tight_layout()

            path = os.path.join(self._charts_dir, "sensitivity.png")
            fig.savefig(path, bbox_inches="tight", facecolor="#0B142B")
            plt.close(fig)
            return path

    def _create_range_chart(self, combined: dict, current_price) -> str | None:
        """Simple bull/base/bear horizontal bar."""
        low = combined.get("low")
        mid = combined.get("mid")
        high = combined.get("high")

        vals = [low, mid, high, current_price]
        if any(v is None for v in vals):
            return None

        try:
            low, mid, high, cp = [float(v) for v in vals]
        except (TypeError, ValueError):
            return None

        with plt.rc_context(_MPL_RC):
            fig, ax = plt.subplots(figsize=(5.5, 1.5), dpi=150)

            x_min = min(low, cp) * 0.9
            x_max = max(high, cp) * 1.1

            # Gradient bar segments
            ax.barh(0, mid - low, left=low, height=0.5, color="#EF4444", alpha=0.6, edgecolor="none")
            ax.barh(0, high - mid, left=mid, height=0.5, color="#10B981", alpha=0.6, edgecolor="none")

            # Markers
            ax.plot(cp, 0, marker="v", color="#F59E0B", markersize=14, zorder=5)
            ax.plot(mid, 0, marker="D", color="#6366F1", markersize=10, zorder=5)

            ax.annotate(f"Current ${cp:.0f}", (cp, 0.4), ha="center", fontsize=8,
                        color="#F59E0B", fontweight="bold")

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-0.6, 0.7)
            ax.set_yticks([])
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
            ax.set_title("Bull / Base / Bear Range", fontsize=10, color="white", pad=6)
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()

            path = os.path.join(self._charts_dir, "range.png")
            fig.savefig(path, bbox_inches="tight", facecolor="#0B142B")
            plt.close(fig)
            return path
