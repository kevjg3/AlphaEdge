"""Generate professional PPTX analysis decks from AlphaEdge results."""

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
CARD_BG = RGBColor(0x14, 0x22, 0x44)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
OFF_WHITE = RGBColor(0xE2, 0xE8, 0xF0)
LIGHT_GRAY = RGBColor(0xA0, 0xAE, 0xC4)
MID_GRAY = RGBColor(0x64, 0x74, 0x8B)
DIM_GRAY = RGBColor(0x47, 0x55, 0x69)
ACCENT_BLUE = RGBColor(0x63, 0x66, 0xF1)
ACCENT_BLUE_DIM = RGBColor(0x45, 0x48, 0xB0)
ACCENT_GREEN = RGBColor(0x10, 0xB9, 0x81)
ACCENT_GREEN_DIM = RGBColor(0x0D, 0x8A, 0x62)
ACCENT_RED = RGBColor(0xEF, 0x44, 0x44)
ACCENT_GOLD = RGBColor(0xF5, 0x9E, 0x0B)
ACCENT_CYAN = RGBColor(0x06, 0xB6, 0xD4)
BORDER_COLOR = RGBColor(0x1E, 0x29, 0x3B)
BORDER_LIGHT = RGBColor(0x33, 0x41, 0x55)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

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
        v = float(n)
        if abs(v) <= 1.0:
            return f"{v * 100:.{decimals}f}%"
        return f"{v:.{decimals}f}%"
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


def _pct_color(n: Any) -> RGBColor:
    """Return green/red based on sign."""
    if n is None:
        return LIGHT_GRAY
    try:
        v = float(n)
        return ACCENT_GREEN if v > 0 else (ACCENT_RED if v < 0 else LIGHT_GRAY)
    except (TypeError, ValueError):
        return LIGHT_GRAY


# ---------------------------------------------------------------------------
# PitchDeckGenerator
# ---------------------------------------------------------------------------

class PitchDeckGenerator:
    """Generate a PPTX analysis deck from AlphaEdge results."""

    def __init__(self) -> None:
        self._prs: Presentation | None = None
        self._charts_dir: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, analysis: dict) -> BytesIO:
        """Build the deck and return as in-memory bytes."""
        self._charts_dir = tempfile.mkdtemp(prefix="alphaedge_deck_")
        try:
            self._prs = Presentation()
            self._prs.slide_width = SLIDE_W
            self._prs.slide_height = SLIDE_H

            self._add_executive_summary(analysis)    # 1
            self._add_company_overview(analysis)      # 2
            self._add_financial_summary(analysis)     # 3
            self._add_valuation(analysis)             # 4
            self._add_peer_comparison(analysis)       # 5
            self._add_market_sentiment(analysis)      # 6
            self._add_risk_assessment(analysis)       # 7
            self._add_forecast_outlook(analysis)      # 8
            self._add_strengths_risks(analysis)       # 9
            self._add_conclusion(analysis)            # 10

            buf = BytesIO()
            self._prs.save(buf)
            buf.seek(0)
            return buf
        finally:
            shutil.rmtree(self._charts_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _blank_slide(self):
        layout = self._prs.slide_layouts[6]
        slide = self._prs.slides.add_slide(layout)
        bg = slide.background.fill
        bg.solid()
        bg.fore_color.rgb = NAVY
        return slide

    def _title_bar(self, slide, text: str, subtitle: str = ""):
        ln = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(0.45),
            Inches(0.08), Inches(0.35),
        )
        ln.fill.solid()
        ln.fill.fore_color.rgb = ACCENT_BLUE
        ln.line.fill.background()
        self._text_box(slide, Inches(0.9), Inches(0.35), Inches(10),
                       Inches(0.55), text, size=24, bold=True, color=WHITE)
        if subtitle:
            self._text_box(slide, Inches(0.9), Inches(0.78), Inches(10),
                           Inches(0.35), subtitle, size=11, color=MID_GRAY)

    def _text_box(self, slide, left, top, width, height, text: str, *,
                  size: int = 14, bold: bool = False, color=LIGHT_GRAY,
                  align=PP_ALIGN.LEFT, font_name: str = "Calibri",
                  line_spacing: float | None = None):
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
        if line_spacing:
            p.line_spacing = Pt(line_spacing)
        return txbox

    def _rich_text_box(self, slide, left, top, width, height,
                       paragraphs: list[dict]):
        txbox = slide.shapes.add_textbox(left, top, width, height)
        tf = txbox.text_frame
        tf.word_wrap = True
        for i, para in enumerate(paragraphs):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = para.get("text", "")
            p.font.size = Pt(para.get("size", 12))
            p.font.bold = para.get("bold", False)
            p.font.color.rgb = para.get("color", LIGHT_GRAY)
            p.font.name = para.get("font_name", "Calibri")
            p.alignment = para.get("align", PP_ALIGN.LEFT)
            if para.get("spacing_after"):
                p.space_after = Pt(para["spacing_after"])
            if para.get("spacing_before"):
                p.space_before = Pt(para["spacing_before"])
        return txbox

    def _panel(self, slide, left, top, width, height, *, fill=DARK_PANEL,
               border=BORDER_COLOR, radius=True):
        st = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
        box = slide.shapes.add_shape(st, left, top, width, height)
        box.fill.solid()
        box.fill.fore_color.rgb = fill
        box.line.color.rgb = border
        box.line.width = Pt(0.75)
        return box

    def _section_header(self, slide, left, top, text: str, width=Inches(12)):
        self._text_box(slide, left, top, width, Inches(0.25),
                       text.upper(), size=9, bold=True, color=MID_GRAY)
        ln = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, left, top + Inches(0.28),
            Inches(1.2), Pt(1.5),
        )
        ln.fill.solid()
        ln.fill.fore_color.rgb = ACCENT_BLUE_DIM
        ln.line.fill.background()

    def _metric_card(self, slide, left, top, label: str, value: str,
                     width=Inches(2.4), height=Inches(0.9), *,
                     value_color=WHITE, label_color=MID_GRAY,
                     subtitle: str = "", subtitle_color=DIM_GRAY):
        self._panel(slide, left, top, width, height)
        self._text_box(slide, left + Inches(0.15), top + Inches(0.08),
                       width - Inches(0.3), Inches(0.22),
                       label.upper(), size=8, bold=True, color=label_color)
        self._text_box(slide, left + Inches(0.15), top + Inches(0.3),
                       width - Inches(0.3), Inches(0.35),
                       value, size=18, bold=True, color=value_color)
        if subtitle:
            self._text_box(slide, left + Inches(0.15), top + Inches(0.65),
                           width - Inches(0.3), Inches(0.2),
                           subtitle, size=8, color=subtitle_color)

    def _large_metric_card(self, slide, left, top, label: str, value: str,
                           width=Inches(3.5), height=Inches(1.4), *,
                           value_color=WHITE, description: str = ""):
        self._panel(slide, left, top, width, height)
        self._text_box(slide, left + Inches(0.2), top + Inches(0.1),
                       width - Inches(0.4), Inches(0.25),
                       label.upper(), size=9, bold=True, color=MID_GRAY)
        self._text_box(slide, left + Inches(0.2), top + Inches(0.4),
                       width - Inches(0.4), Inches(0.55),
                       value, size=28, bold=True, color=value_color)
        if description:
            self._text_box(slide, left + Inches(0.2), top + Inches(1.0),
                           width - Inches(0.4), Inches(0.3),
                           description, size=9, color=LIGHT_GRAY)

    def _footer(self, slide, ticker: str = ""):
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

    def _add_picture(self, slide, path: str, left, top, width=None,
                     height=None):
        kwargs = {"left": left, "top": top}
        if width:
            kwargs["width"] = width
        if height:
            kwargs["height"] = height
        slide.shapes.add_picture(path, **kwargs)

    def _divider(self, slide, top):
        ln = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.6), top,
            Inches(12.13), Pt(1),
        )
        ln.fill.solid()
        ln.fill.fore_color.rgb = BORDER_COLOR
        ln.line.fill.background()

    def _data_row(self, slide, x, y, label: str, value: str, w_label=Inches(2.5),
                  w_value=Inches(1.5), *, value_color=WHITE, label_size=10,
                  value_size=12, bg: RGBColor | None = None):
        """Render a label-value data row, optionally with a background."""
        total_w = w_label + w_value + Inches(0.3)
        if bg:
            stripe = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE, x, y, total_w, Inches(0.42),
            )
            stripe.fill.solid()
            stripe.fill.fore_color.rgb = bg
            stripe.line.fill.background()
        self._text_box(slide, x + Inches(0.1), y + Inches(0.06),
                       w_label, Inches(0.3), label,
                       size=label_size, color=MID_GRAY)
        self._text_box(slide, x + w_label + Inches(0.1), y + Inches(0.06),
                       w_value, Inches(0.3), value,
                       size=value_size, bold=True, color=value_color,
                       align=PP_ALIGN.RIGHT)

    # ------------------------------------------------------------------
    # SLIDE 1 — Executive Summary
    # ------------------------------------------------------------------

    def _add_executive_summary(self, a: dict):
        slide = self._blank_slide()
        snap = a.get("snapshot") or {}
        fund = a.get("fundamentals") or {}
        verdict = fund.get("verdict") or {}
        thesis = fund.get("investment_thesis") or {}
        combined = fund.get("combined_range") or {}

        name = snap.get("name", a.get("ticker", ""))
        ticker = snap.get("ticker", a.get("ticker", ""))
        price = snap.get("price")
        label = verdict.get("label", "")
        upside = verdict.get("upside_pct")
        fair_mid = verdict.get("fair_value_mid")
        fair_low = combined.get("low")
        fair_high = combined.get("high")
        change_pct = snap.get("change_1d_pct")

        is_long = label in ("undervalued", "fairly_valued")
        rec_text = "BUY" if label == "undervalued" else ("HOLD" if label == "fairly_valued" else "SELL")
        rec_color = ACCENT_GREEN if is_long else ACCENT_RED

        # Verdict badge
        badge = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(0.8), Inches(0.7), Inches(1.6), Inches(0.6),
        )
        badge.fill.solid()
        badge.fill.fore_color.rgb = rec_color
        badge.line.fill.background()
        tf = badge.text_frame
        p = tf.paragraphs[0]
        p.text = rec_text
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.font.name = "Calibri"
        p.alignment = PP_ALIGN.CENTER
        tf.paragraphs[0].space_before = Pt(3)

        # Company + sector line
        sector = snap.get("sector", "")
        industry = snap.get("industry", "")
        sector_str = f"  |  {sector} \u2014 {industry}" if sector else ""
        self._text_box(slide, Inches(2.7), Inches(0.7), Inches(10), Inches(0.45),
                       f"{name} ({ticker}){sector_str}",
                       size=22, bold=True, color=WHITE)
        # Subtitle
        sub_parts = []
        if price is not None:
            chg = f" ({change_pct:+.2f}%)" if change_pct else ""
            sub_parts.append(f"Price: {_fmt_price(price)}{chg}")
        if fair_mid is not None:
            sub_parts.append(f"Fair Value: {_fmt_price(fair_mid)}")
        if upside is not None:
            sub_parts.append(f"Upside: {upside:+.1f}%")
        if sub_parts:
            self._text_box(slide, Inches(2.7), Inches(1.1), Inches(10), Inches(0.3),
                           "  \u2022  ".join(sub_parts), size=13, color=LIGHT_GRAY)

        # Key metrics row (5 cards)
        y = Inches(1.8)
        cw = Inches(2.35)
        cg = Inches(0.15)
        metrics = [
            ("Market Cap", _fmt_large(snap.get("market_cap")), WHITE),
            ("P/E Ratio", _fmt_ratio(snap.get("pe_ratio")), WHITE),
            ("Revenue", _fmt_large(snap.get("total_revenue")), WHITE),
            ("Operating Margin", _fmt_pct(snap.get("operating_margins")), WHITE),
            ("Free Cash Flow", _fmt_large(snap.get("free_cashflow")), WHITE),
        ]
        for i, (ml, mv, mc) in enumerate(metrics):
            self._metric_card(slide, Inches(0.8) + i * (cw + cg), y,
                              ml, mv, width=cw, height=Inches(0.85),
                              value_color=mc)

        # Investment thesis overview (full-width panel)
        thesis_text = thesis.get("investment_thesis", "")
        if thesis_text:
            display = thesis_text if len(thesis_text) <= 550 else thesis_text[:547] + "..."
            self._panel(slide, Inches(0.7), Inches(3.0), Inches(11.9), Inches(2.0),
                        fill=DARK_PANEL, border=BORDER_LIGHT)
            self._text_box(slide, Inches(0.9), Inches(3.1), Inches(2), Inches(0.25),
                           "INVESTMENT OVERVIEW", size=8, bold=True, color=ACCENT_BLUE)
            self._text_box(slide, Inches(0.9), Inches(3.4), Inches(11.5), Inches(1.5),
                           display, size=13, color=OFF_WHITE, line_spacing=19)

        # Fair value range chart (bottom)
        chart_path = self._create_fair_value_chart(price, fair_low, fair_mid, fair_high)
        if chart_path:
            self._add_picture(slide, chart_path, Inches(0.5), Inches(5.2),
                              width=Inches(12.3), height=Inches(1.6))

        self._footer(slide, ticker)

    # ------------------------------------------------------------------
    # SLIDE 2 — Company Overview
    # ------------------------------------------------------------------

    def _add_company_overview(self, a: dict):
        slide = self._blank_slide()
        self._title_bar(slide, "Company Overview")

        snap = a.get("snapshot") or {}
        thesis = _sg(a, "fundamentals", "investment_thesis") or {}
        fh = _sg(a, "fundamentals", "financial_health") or {}
        income = fh.get("income_summary") or {}

        # Left: description in panel
        overview = thesis.get("company_overview", snap.get("description", ""))
        if overview:
            disp = overview if len(overview) <= 950 else overview[:947] + "..."
            self._panel(slide, Inches(0.5), Inches(1.15), Inches(7.3), Inches(4.2),
                        fill=DARK_PANEL, border=BORDER_COLOR)
            self._text_box(slide, Inches(0.7), Inches(1.25), Inches(2), Inches(0.25),
                           "BUSINESS DESCRIPTION", size=8, bold=True, color=ACCENT_BLUE)
            self._text_box(slide, Inches(0.7), Inches(1.55), Inches(6.9), Inches(3.6),
                           disp, size=12, color=OFF_WHITE, line_spacing=17)

        # Right: stat grid
        x0 = Inches(8.1)
        y0 = Inches(1.15)
        cw = Inches(2.45)
        ch = Inches(0.95)
        gx = Inches(0.1)
        gy = Inches(0.1)

        stats = [
            ("Market Cap", _fmt_large(snap.get("market_cap")), WHITE),
            ("Revenue", _fmt_large(snap.get("total_revenue")), WHITE),
            ("Rev Growth", _fmt_pct(snap.get("revenue_growth")), _pct_color(snap.get("revenue_growth"))),
            ("Op Margin", _fmt_pct(snap.get("operating_margins")), WHITE),
            ("P/E Ratio", _fmt_ratio(snap.get("pe_ratio")), WHITE),
            ("Forward P/E", _fmt_ratio(snap.get("forward_pe")), WHITE),
            ("ROE", _fmt_pct(snap.get("return_on_equity")), WHITE),
            ("Div Yield", _fmt_pct(snap.get("dividend_yield")), WHITE),
            ("Beta", _fmt_ratio(snap.get("beta")), WHITE),
            ("Employees", f"{snap.get('employees', 0):,}" if snap.get("employees") else "\u2014", WHITE),
        ]
        for i, (lbl, val, vc) in enumerate(stats):
            row, col = divmod(i, 2)
            self._metric_card(slide, x0 + col * (cw + gx), y0 + row * (ch + gy),
                              lbl, val, width=cw, height=ch, value_color=vc)

        # Bottom: key strengths
        strengths = thesis.get("strengths", [])
        if strengths:
            self._divider(slide, Inches(5.55))
            self._section_header(slide, Inches(0.6), Inches(5.7), "Key Strengths")
            paras = [{"text": f"\u25B8  {s}", "size": 11, "color": OFF_WHITE,
                       "spacing_after": 6} for s in strengths[:3]]
            self._rich_text_box(slide, Inches(0.8), Inches(6.05), Inches(12), Inches(0.8),
                                paras)

        self._footer(slide, snap.get("ticker", ""))

    # ------------------------------------------------------------------
    # SLIDE 3 — Financial Summary
    # ------------------------------------------------------------------

    def _add_financial_summary(self, a: dict):
        slide = self._blank_slide()
        self._title_bar(slide, "Financial Summary",
                        subtitle="Key financial metrics from the latest filings")

        fh = _sg(a, "fundamentals", "financial_health") or {}
        income = fh.get("income_summary") or {}
        balance = fh.get("balance_summary") or {}
        cashflow = fh.get("cashflow_summary") or {}
        snap = a.get("snapshot") or {}

        col_w = Inches(3.8)
        col_gap = Inches(0.35)
        y_start = Inches(1.3)

        columns = [
            ("Income Statement", [
                ("Revenue", _fmt_large(income.get("revenue"))),
                ("Revenue Growth YoY", _fmt_pct(income.get("revenue_growth_yoy"))),
                ("Gross Margin", _fmt_pct(income.get("gross_margin"))),
                ("Operating Margin", _fmt_pct(income.get("operating_margin"))),
                ("Net Margin", _fmt_pct(income.get("net_margin"))),
                ("EBITDA", _fmt_large(income.get("ebitda"))),
                ("EPS", _fmt_price(income.get("eps"))),
            ]),
            ("Balance Sheet", [
                ("Total Assets", _fmt_large(balance.get("total_assets"))),
                ("Total Equity", _fmt_large(balance.get("total_equity"))),
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
                ("Book Value/Share", _fmt_price(balance.get("book_value_per_share"))),
            ]),
        ]

        for ci, (col_title, items) in enumerate(columns):
            x = Inches(0.6) + ci * (col_w + col_gap)
            hdr = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE, x, y_start, col_w, Inches(0.4),
            )
            hdr.fill.solid()
            hdr.fill.fore_color.rgb = ACCENT_BLUE
            hdr.line.fill.background()
            self._text_box(slide, x + Inches(0.15), y_start + Inches(0.02),
                           col_w - Inches(0.3), Inches(0.35),
                           col_title, size=11, bold=True, color=WHITE)

            for ri, (label, value) in enumerate(items):
                ry = y_start + Inches(0.5) + ri * Inches(0.65)
                bg = DARK_PANEL if ri % 2 == 0 else NAVY
                self._data_row(slide, x, ry, label, value,
                               w_label=Inches(2.1), w_value=Inches(1.4),
                               bg=bg)

        self._footer(slide, snap.get("ticker", ""))

    # ------------------------------------------------------------------
    # SLIDE 4 — Valuation Analysis
    # ------------------------------------------------------------------

    def _add_valuation(self, a: dict):
        slide = self._blank_slide()
        self._title_bar(slide, "Valuation Analysis",
                        subtitle="DCF and comparable company analysis")

        fund = a.get("fundamentals") or {}
        dcf = fund.get("dcf_valuation") or {}
        comps = fund.get("comps_valuation") or {}
        combined = fund.get("combined_range") or {}
        snap = a.get("snapshot") or {}
        price = snap.get("price")

        # --- DCF panel (left) ---
        pw = Inches(6.2)
        self._panel(slide, Inches(0.5), Inches(1.15), pw, Inches(2.4),
                    fill=DARK_PANEL, border=BORDER_COLOR)
        self._text_box(slide, Inches(0.7), Inches(1.25), Inches(3), Inches(0.25),
                       "DCF VALUATION", size=8, bold=True, color=ACCENT_BLUE)

        dcf_cards = [
            ("Implied Price", _fmt_price(dcf.get("implied_price")), ACCENT_GREEN),
            ("Enterprise Value", _fmt_large(dcf.get("enterprise_value")), WHITE),
            ("Equity Value", _fmt_large(dcf.get("equity_value")), WHITE),
        ]
        for di, (lbl, val, vc) in enumerate(dcf_cards):
            self._metric_card(slide, Inches(0.7) + di * Inches(1.85),
                              Inches(1.6), lbl, val,
                              width=Inches(1.75), height=Inches(0.8),
                              value_color=vc)

        assumptions = dcf.get("assumptions_used") or {}
        parts = []
        for key, lbl in [("wacc", "WACC"), ("terminal_growth_rate", "Terminal Growth"),
                         ("operating_margin", "Op Margin"), ("tax_rate", "Tax Rate")]:
            v = assumptions.get(key)
            if v:
                parts.append(f"{lbl}: {_fmt_pct(v)}")
        proj_y = assumptions.get("projection_years")
        if proj_y:
            parts.append(f"{proj_y}yr horizon")
        if parts:
            self._text_box(slide, Inches(0.7), Inches(2.6), Inches(5.8), Inches(0.4),
                           "   \u2022   ".join(parts), size=9, color=LIGHT_GRAY)

        dcf_upside = dcf.get("upside_downside_pct")
        if dcf_upside is not None:
            try:
                uv = float(dcf_upside)
                uc = ACCENT_GREEN if uv > 0 else ACCENT_RED
                self._text_box(slide, Inches(0.7), Inches(3.05), Inches(5.8), Inches(0.25),
                               f"DCF implies {uv:+.1f}% {'upside' if uv > 0 else 'downside'} from current price",
                               size=11, bold=True, color=uc)
            except (TypeError, ValueError):
                pass

        # --- Comps panel (right) ---
        cx = Inches(6.9)
        cw = Inches(5.9)
        self._panel(slide, cx, Inches(1.15), cw, Inches(2.4),
                    fill=DARK_PANEL, border=BORDER_COLOR)
        self._text_box(slide, cx + Inches(0.2), Inches(1.25), Inches(4), Inches(0.25),
                       "COMPARABLE COMPANIES", size=8, bold=True, color=ACCENT_BLUE)

        val_range = comps.get("valuation_range") or {}
        approaches = [
            ("P/E Approach", val_range.get("pe_approach") or {}),
            ("EV/EBITDA Approach", val_range.get("ev_ebitda_approach") or {}),
            ("EV/Revenue Approach", val_range.get("ev_revenue_approach") or {}),
        ]
        for ai, (approach_name, approach_data) in enumerate(approaches):
            ay = Inches(1.65) + ai * Inches(0.6)
            implied = approach_data.get("implied_value") or approach_data.get("implied_market_cap")
            upside = approach_data.get("implied_upside")

            self._text_box(slide, cx + Inches(0.2), ay, Inches(2.2), Inches(0.3),
                           approach_name, size=11, color=LIGHT_GRAY)
            self._text_box(slide, cx + Inches(2.5), ay, Inches(1.5), Inches(0.3),
                           _fmt_large(implied), size=12, bold=True, color=WHITE,
                           align=PP_ALIGN.RIGHT)
            if upside is not None:
                try:
                    uv = float(upside)
                    self._text_box(slide, cx + Inches(4.2), ay, Inches(1.2), Inches(0.3),
                                   f"{uv:+.1f}%", size=11, bold=True,
                                   color=ACCENT_GREEN if uv > 0 else ACCENT_RED,
                                   align=PP_ALIGN.RIGHT)
                except (TypeError, ValueError):
                    pass

        peer_count = comps.get("peer_count")
        if peer_count:
            self._text_box(slide, cx + Inches(0.2), Inches(3.25), Inches(5), Inches(0.2),
                           f"Based on {peer_count} comparable companies",
                           size=9, color=DIM_GRAY)

        # --- Bottom: Sensitivity + Range ---
        self._divider(slide, Inches(3.7))

        heatmap_path = self._create_sensitivity_heatmap(dcf, price)
        if heatmap_path:
            self._add_picture(slide, heatmap_path, Inches(0.3), Inches(3.9),
                              width=Inches(6.8), height=Inches(3.0))

        range_path = self._create_range_chart(combined, price)
        if range_path:
            self._add_picture(slide, range_path, Inches(7.2), Inches(3.9),
                              width=Inches(5.5), height=Inches(1.5))

        if combined:
            mw = combined.get("methodology_weights") or {}
            y_r = Inches(5.6)
            self._metric_card(slide, Inches(7.2), y_r, "Bear Case",
                              _fmt_price(combined.get("low")),
                              width=Inches(1.7), height=Inches(0.9),
                              value_color=ACCENT_RED, subtitle="Downside scenario")
            self._metric_card(slide, Inches(9.1), y_r, "Base Case",
                              _fmt_price(combined.get("mid")),
                              width=Inches(1.7), height=Inches(0.9),
                              value_color=ACCENT_BLUE, subtitle="Most likely")
            self._metric_card(slide, Inches(11.0), y_r, "Bull Case",
                              _fmt_price(combined.get("high")),
                              width=Inches(1.7), height=Inches(0.9),
                              value_color=ACCENT_GREEN, subtitle="Upside scenario")

            dcf_w = mw.get("dcf")
            comp_w = mw.get("comps")
            if dcf_w and comp_w:
                self._text_box(slide, Inches(7.2), Inches(6.55), Inches(5.5), Inches(0.2),
                               f"Blended: {dcf_w*100:.0f}% DCF + {comp_w*100:.0f}% Comps",
                               size=8, color=DIM_GRAY, align=PP_ALIGN.CENTER)

        self._footer(slide, snap.get("ticker", ""))

    # ------------------------------------------------------------------
    # SLIDE 5 — Peer Comparison
    # ------------------------------------------------------------------

    def _add_peer_comparison(self, a: dict):
        slide = self._blank_slide()
        self._title_bar(slide, "Peer Comparison",
                        subtitle="Key metrics relative to comparable companies")

        comps = _sg(a, "fundamentals", "comps_valuation") or {}
        snap = a.get("snapshot") or {}

        # Peer chart (left)
        chart_path = self._create_peer_chart(comps, snap)
        if chart_path:
            self._add_picture(slide, chart_path, Inches(0.3), Inches(1.1),
                              width=Inches(6.5), height=Inches(4.0))

        # Percentile rank cards (right)
        pct_rank = comps.get("percentile_rank") or {}
        if pct_rank:
            self._section_header(slide, Inches(7.2), Inches(1.1), "Percentile Rank vs Peers")
            bw = Inches(2.7)
            bh = Inches(0.9)
            bg = Inches(0.15)
            bx = Inches(7.2)
            by = Inches(1.55)
            for i, (metric, pctile) in enumerate(list(pct_rank.items())[:8]):
                try:
                    pval = float(pctile)
                except (TypeError, ValueError):
                    continue
                color = ACCENT_GREEN if pval >= 60 else (ACCENT_RED if pval <= 30 else WHITE)
                label_clean = metric.replace("_", " ").title()
                ctx = ("Top quartile" if pval >= 75 else
                       "Above median" if pval >= 50 else
                       "Below median" if pval >= 25 else "Bottom quartile")
                row, col = divmod(i, 2)
                self._metric_card(
                    slide, bx + col * (bw + bg), by + row * (bh + bg),
                    label_clean, f"{pval:.0f}th percentile",
                    width=bw, height=bh, value_color=color,
                    subtitle=ctx, subtitle_color=DIM_GRAY,
                )

        # Bottom: Peer table
        peers = comps.get("peers") or []
        target = comps.get("target") or {}
        if peers and target:
            self._divider(slide, Inches(5.3))
            self._section_header(slide, Inches(0.6), Inches(5.4), "Peer Summary")
            headers = ["Company", "Mkt Cap", "P/E", "EV/EBITDA", "Op Margin", "ROE"]
            hx = [Inches(0.6), Inches(3.5), Inches(5.5), Inches(7.2), Inches(9.2), Inches(11.2)]
            hw = [Inches(2.8), Inches(1.8), Inches(1.5), Inches(1.8), Inches(1.8), Inches(1.5)]
            for hi, h in enumerate(headers):
                self._text_box(slide, hx[hi], Inches(5.75), hw[hi], Inches(0.2),
                               h, size=8, bold=True, color=MID_GRAY)

            all_companies = [target] + peers[:4]
            for ci, comp in enumerate(all_companies):
                ry = Inches(6.0) + ci * Inches(0.35)
                is_target = ci == 0
                bg = DARK_PANEL if ci % 2 == 0 else NAVY
                stripe = slide.shapes.add_shape(
                    MSO_SHAPE.RECTANGLE, Inches(0.5), ry, Inches(12.33), Inches(0.32),
                )
                stripe.fill.solid()
                stripe.fill.fore_color.rgb = bg
                stripe.line.fill.background()

                txt_color = ACCENT_BLUE if is_target else OFF_WHITE
                name = comp.get("ticker", comp.get("name", ""))
                vals = [
                    name + (" \u2605" if is_target else ""),
                    _fmt_large(comp.get("market_cap")),
                    _fmt_ratio(comp.get("pe")),
                    _fmt_ratio(comp.get("ev_ebitda")),
                    _fmt_pct(comp.get("operating_margin")),
                    _fmt_pct(comp.get("roe")),
                ]
                for vi, v in enumerate(vals):
                    self._text_box(slide, hx[vi], ry + Inches(0.02), hw[vi], Inches(0.26),
                                   v, size=9, bold=is_target, color=txt_color)

        self._footer(slide, snap.get("ticker", ""))

    # ------------------------------------------------------------------
    # SLIDE 6 — Market Sentiment
    # ------------------------------------------------------------------

    def _add_market_sentiment(self, a: dict):
        slide = self._blank_slide()
        self._title_bar(slide, "Market Sentiment",
                        subtitle="News analysis and sentiment indicators")

        news = _sg(a, "news", "synthesis") or {}
        snap = a.get("snapshot") or {}

        sentiment = news.get("overall_sentiment", "")
        score = news.get("sentiment_score")
        momentum = news.get("news_momentum", "")
        narrative = news.get("narrative", "")
        agg = news.get("aggregate_sentiment") or {}
        pos_pct = agg.get("positive_pct")
        neg_pct = agg.get("negative_pct")
        neutral_pct = agg.get("neutral_pct")
        article_count = news.get("article_count")
        themes = news.get("key_themes") or []
        news_catalysts = news.get("catalysts") or []
        news_risks = news.get("risks") or []

        # Left panel: sentiment overview
        self._panel(slide, Inches(0.5), Inches(1.15), Inches(6.2), Inches(5.5),
                    fill=DARK_PANEL, border=BORDER_COLOR)
        self._text_box(slide, Inches(0.7), Inches(1.25), Inches(3), Inches(0.25),
                       "OVERALL SENTIMENT", size=8, bold=True, color=ACCENT_BLUE)

        if sentiment:
            s_color = ACCENT_GREEN if sentiment == "bullish" else (
                ACCENT_RED if sentiment == "bearish" else LIGHT_GRAY)
            score_str = f"  ({score:+.2f})" if score is not None else ""
            self._text_box(slide, Inches(0.7), Inches(1.6), Inches(5.5), Inches(0.5),
                           f"{sentiment.upper()}{score_str}",
                           size=26, bold=True, color=s_color)

        # Meta line
        meta = []
        if momentum:
            meta.append(f"Momentum: {momentum.capitalize()}")
        if article_count:
            meta.append(f"{article_count} articles analyzed")
        if meta:
            self._text_box(slide, Inches(0.7), Inches(2.15), Inches(5.5), Inches(0.3),
                           "  \u2022  ".join(meta), size=10, color=MID_GRAY)

        # Sentiment breakdown
        if pos_pct is not None:
            self._text_box(slide, Inches(0.7), Inches(2.55), Inches(2), Inches(0.25),
                           "SENTIMENT BREAKDOWN", size=8, bold=True, color=MID_GRAY)
            bars_y = Inches(2.85)
            bar_total_w = Inches(5.3)
            try:
                pw = float(pos_pct)
                nw = float(neg_pct or 0)
                uw = float(neutral_pct or 0)
            except (TypeError, ValueError):
                pw = nw = uw = 0.33
            total = pw + nw + uw or 1
            # Positive bar
            pos_w = max(Inches(0.1), Inches(5.3 * pw / total))
            b = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                       Inches(0.7), bars_y, pos_w, Inches(0.25))
            b.fill.solid(); b.fill.fore_color.rgb = ACCENT_GREEN; b.line.fill.background()
            # Neutral bar
            neu_w = max(Inches(0.05), Inches(5.3 * uw / total))
            b2 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                        Inches(0.7) + pos_w, bars_y, neu_w, Inches(0.25))
            b2.fill.solid(); b2.fill.fore_color.rgb = MID_GRAY; b2.line.fill.background()
            # Negative bar
            neg_w = max(Inches(0.05), Inches(5.3 * nw / total))
            b3 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                        Inches(0.7) + pos_w + neu_w, bars_y, neg_w, Inches(0.25))
            b3.fill.solid(); b3.fill.fore_color.rgb = ACCENT_RED; b3.line.fill.background()

            self._text_box(slide, Inches(0.7), bars_y + Inches(0.28), Inches(5.5), Inches(0.2),
                           f"\u2588 Positive {_fmt_pct(pos_pct)}   \u2588 Neutral {_fmt_pct(neutral_pct)}   \u2588 Negative {_fmt_pct(neg_pct)}",
                           size=8, color=LIGHT_GRAY)

        # Narrative
        if narrative:
            self._text_box(slide, Inches(0.7), Inches(3.45), Inches(1.5), Inches(0.25),
                           "NEWS NARRATIVE", size=8, bold=True, color=MID_GRAY)
            disp = narrative if len(narrative) <= 450 else narrative[:447] + "..."
            self._text_box(slide, Inches(0.7), Inches(3.75), Inches(5.8), Inches(1.8),
                           disp, size=12, color=OFF_WHITE, line_spacing=17)

        # Key themes
        if themes:
            self._text_box(slide, Inches(0.7), Inches(5.6), Inches(1.5), Inches(0.2),
                           "KEY THEMES", size=8, bold=True, color=MID_GRAY)
            self._text_box(slide, Inches(0.7), Inches(5.85), Inches(5.5), Inches(0.5),
                           "  \u2022  ".join(themes[:6]), size=10, color=LIGHT_GRAY)

        # Right panel: Catalysts & Risks
        rx = Inches(6.9)
        rw = Inches(5.9)

        # Catalysts panel
        self._panel(slide, rx, Inches(1.15), rw, Inches(2.6),
                    fill=DARK_PANEL, border=BORDER_COLOR)
        self._text_box(slide, rx + Inches(0.2), Inches(1.25), Inches(3), Inches(0.25),
                       "CATALYSTS", size=8, bold=True, color=ACCENT_GREEN)
        cat_items = []
        for nc in news_catalysts[:4]:
            if isinstance(nc, dict):
                event = nc.get("event", nc.get("headline", ""))
                conf = nc.get("confidence")
                conf_str = f" ({conf*100:.0f}%)" if conf else ""
                cat_items.append(f"{event}{conf_str}")
            elif isinstance(nc, str):
                cat_items.append(nc)
        if cat_items:
            paras = [{"text": f"\u25B8  {c}", "size": 11, "color": OFF_WHITE,
                       "spacing_after": 8} for c in cat_items]
            self._rich_text_box(slide, rx + Inches(0.2), Inches(1.6),
                                Inches(5.4), Inches(2.0), paras)

        # Risks panel
        self._panel(slide, rx, Inches(3.95), rw, Inches(2.7),
                    fill=DARK_PANEL, border=BORDER_COLOR)
        self._text_box(slide, rx + Inches(0.2), Inches(4.05), Inches(3), Inches(0.25),
                       "RISKS IDENTIFIED", size=8, bold=True, color=ACCENT_RED)
        risk_items = []
        for nr in news_risks[:4]:
            if isinstance(nr, dict):
                event = nr.get("event", nr.get("headline", ""))
                conf = nr.get("confidence")
                conf_str = f" ({conf*100:.0f}%)" if conf else ""
                risk_items.append(f"{event}{conf_str}")
            elif isinstance(nr, str):
                risk_items.append(nr)
        if risk_items:
            paras = [{"text": f"\u25B8  {r}", "size": 11, "color": OFF_WHITE,
                       "spacing_after": 8} for r in risk_items]
            self._rich_text_box(slide, rx + Inches(0.2), Inches(4.4),
                                Inches(5.4), Inches(2.0), paras)

        self._footer(slide, snap.get("ticker", ""))

    # ------------------------------------------------------------------
    # SLIDE 7 — Risk Assessment
    # ------------------------------------------------------------------

    def _add_risk_assessment(self, a: dict):
        slide = self._blank_slide()
        self._title_bar(slide, "Risk Assessment",
                        subtitle="Value at Risk, drawdown analysis, and historical stress tests")

        risk = a.get("risk") or {}
        var_data = risk.get("var") or {}
        dd_data = risk.get("drawdown") or {}
        scenarios = risk.get("scenarios") or []
        quant = a.get("quant") or {}
        perf = quant.get("performance") or {}
        snap = a.get("snapshot") or {}

        pw = Inches(3.9)
        pg = Inches(0.2)
        py = Inches(1.2)
        ph = Inches(2.6)

        # Panel 1: VaR
        self._panel(slide, Inches(0.5), py, pw, ph, fill=DARK_PANEL, border=BORDER_COLOR)
        self._text_box(slide, Inches(0.7), py + Inches(0.1), Inches(3), Inches(0.25),
                       "VALUE AT RISK (95%)", size=8, bold=True, color=ACCENT_BLUE)
        var_items = [
            ("Parametric VaR", var_data.get("parametric_var_pct")),
            ("Historical VaR", var_data.get("historical_var_pct")),
            ("Monte Carlo VaR", var_data.get("monte_carlo_var_pct")),
            ("CVaR (Exp. Shortfall)", var_data.get("expected_shortfall_pct")),
        ]
        for vi, (lbl, val) in enumerate(var_items):
            vy = py + Inches(0.5) + vi * Inches(0.5)
            self._data_row(slide, Inches(0.7), vy, lbl, _fmt_pct(val),
                           w_label=Inches(2.0), w_value=Inches(1.2),
                           value_color=ACCENT_RED)

        # Panel 2: Risk-Adjusted Performance
        p2x = Inches(0.5) + pw + pg
        self._panel(slide, p2x, py, pw, ph, fill=DARK_PANEL, border=BORDER_COLOR)
        self._text_box(slide, p2x + Inches(0.2), py + Inches(0.1), Inches(3), Inches(0.25),
                       "RISK-ADJUSTED RETURNS", size=8, bold=True, color=ACCENT_BLUE)
        risk_items = [
            ("Sharpe Ratio", _fmt_ratio(perf.get("sharpe_ratio"))),
            ("Sortino Ratio", _fmt_ratio(perf.get("sortino_ratio"))),
            ("Max Drawdown", _fmt_pct(perf.get("max_drawdown"))),
            ("Ann. Volatility", _fmt_pct(perf.get("annualized_vol"))),
        ]
        for vi, (lbl, val) in enumerate(risk_items):
            vy = py + Inches(0.5) + vi * Inches(0.5)
            self._data_row(slide, p2x + Inches(0.2), vy, lbl, val,
                           w_label=Inches(2.0), w_value=Inches(1.2),
                           value_color=WHITE)

        # Panel 3: Drawdown Stats
        p3x = Inches(0.5) + (pw + pg) * 2
        self._panel(slide, p3x, py, pw, ph, fill=DARK_PANEL, border=BORDER_COLOR)
        self._text_box(slide, p3x + Inches(0.2), py + Inches(0.1), Inches(3), Inches(0.25),
                       "DRAWDOWN ANALYSIS", size=8, bold=True, color=ACCENT_BLUE)
        dd_items = [
            ("Max Drawdown", _fmt_pct(dd_data.get("max_drawdown"))),
            ("Current Drawdown", _fmt_pct(dd_data.get("current_drawdown"))),
            ("# Drawdown Events", str(dd_data.get("n_drawdowns", "\u2014"))),
            ("Avg Recovery (days)", str(dd_data.get("average_recovery_days") or "\u2014")),
        ]
        for vi, (lbl, val) in enumerate(dd_items):
            vy = py + Inches(0.5) + vi * Inches(0.5)
            self._data_row(slide, p3x + Inches(0.2), vy, lbl, val,
                           w_label=Inches(2.0), w_value=Inches(1.2),
                           value_color=WHITE)

        # Bottom: Stress tests table
        self._divider(slide, Inches(4.1))
        self._section_header(slide, Inches(0.6), Inches(4.25), "Historical Stress Tests")

        if isinstance(scenarios, list) and scenarios:
            headers = ["Scenario", "Period", "Stock Return", "Benchmark", "Max Drawdown"]
            hx = [Inches(0.6), Inches(5.0), Inches(7.5), Inches(9.5), Inches(11.2)]
            hw = [Inches(4.2), Inches(2.3), Inches(1.8), Inches(1.5), Inches(1.6)]
            for hi, h in enumerate(headers):
                self._text_box(slide, hx[hi], Inches(4.65), hw[hi], Inches(0.2),
                               h, size=8, bold=True, color=MID_GRAY)

            for si, sc in enumerate(scenarios[:5]):
                if not isinstance(sc, dict):
                    continue
                sy = Inches(4.9) + si * Inches(0.38)
                bg = DARK_PANEL if si % 2 == 0 else NAVY
                stripe = slide.shapes.add_shape(
                    MSO_SHAPE.RECTANGLE, Inches(0.5), sy,
                    Inches(12.33), Inches(0.35),
                )
                stripe.fill.solid()
                stripe.fill.fore_color.rgb = bg
                stripe.line.fill.background()

                name = sc.get("description", sc.get("name", ""))
                stock_ret = sc.get("stock_return") or sc.get("beta_adjusted_return")
                bench_ret = sc.get("benchmark_return")
                mdd = sc.get("max_drawdown")
                period = sc.get("period", "")

                self._text_box(slide, hx[0], sy + Inches(0.03), hw[0], Inches(0.28),
                               str(name)[:45], size=10, color=OFF_WHITE)
                self._text_box(slide, hx[1], sy + Inches(0.03), hw[1], Inches(0.28),
                               str(period)[:22], size=9, color=DIM_GRAY)
                self._text_box(slide, hx[2], sy + Inches(0.03), hw[2], Inches(0.28),
                               _fmt_pct(stock_ret), size=10, bold=True,
                               color=_pct_color(stock_ret))
                self._text_box(slide, hx[3], sy + Inches(0.03), hw[3], Inches(0.28),
                               _fmt_pct(bench_ret), size=10, color=LIGHT_GRAY)
                self._text_box(slide, hx[4], sy + Inches(0.03), hw[4], Inches(0.28),
                               _fmt_pct(mdd), size=10, color=LIGHT_GRAY)

        self._footer(slide, snap.get("ticker", ""))

    # ------------------------------------------------------------------
    # SLIDE 8 — Forecast & Outlook
    # ------------------------------------------------------------------

    def _add_forecast_outlook(self, a: dict):
        slide = self._blank_slide()
        self._title_bar(slide, "Forecast & Outlook",
                        subtitle="Machine learning ensemble forecast and quantitative metrics")

        forecast = a.get("forecast") or {}
        quant = a.get("quant") or {}
        perf = quant.get("performance") or {}
        mc = quant.get("monte_carlo") or {}
        snap = a.get("snapshot") or {}

        overall_dir = forecast.get("overall_direction", "")
        lt_outlook = forecast.get("long_term_outlook", "")
        forecasts = forecast.get("forecasts") or {}

        # Top: direction + outlook
        if overall_dir:
            dir_color = ACCENT_GREEN if "bull" in overall_dir.lower() else (
                ACCENT_RED if "bear" in overall_dir.lower() else LIGHT_GRAY)
            self._text_box(slide, Inches(0.6), Inches(1.2), Inches(3), Inches(0.4),
                           f"Direction: {overall_dir.upper()}",
                           size=20, bold=True, color=dir_color)
        if lt_outlook:
            self._text_box(slide, Inches(0.6), Inches(1.7), Inches(8), Inches(0.3),
                           f"Long-term outlook: {lt_outlook}",
                           size=13, color=LIGHT_GRAY)

        # Forecast horizons
        horizon_keys = ["5D", "30D", "90D", "6M", "12M"]
        cards_shown = 0
        y_card = Inches(2.3)
        cw_h = Inches(2.35)
        cg_h = Inches(0.15)

        for hk in horizon_keys:
            h_data = forecasts.get(hk) or {}
            ep = h_data.get("ensemble_prediction") or {}
            pred_ret = ep.get("predicted_return")
            confidence = ep.get("confidence")
            direction = ep.get("direction", "")
            if pred_ret is None:
                continue

            desc_parts = []
            if direction:
                desc_parts.append(direction.capitalize())
            if confidence:
                desc_parts.append(f"{confidence*100:.0f}% conf")

            sign = "+" if pred_ret > 0 else ""
            vc = ACCENT_GREEN if pred_ret > 0 else ACCENT_RED
            self._large_metric_card(
                slide, Inches(0.6) + cards_shown * (cw_h + cg_h), y_card,
                f"{hk} Forecast", f"{sign}{pred_ret:.1f}%",
                width=cw_h, height=Inches(1.2), value_color=vc,
                description="  \u2022  ".join(desc_parts),
            )
            cards_shown += 1
            if cards_shown >= 5:
                break

        # Monte Carlo summary
        if mc:
            y_mc = Inches(3.8)
            self._divider(slide, y_mc - Inches(0.15))
            self._section_header(slide, Inches(0.6), y_mc, "Monte Carlo Simulation")
            mc_items = [
                ("Simulated Paths", str(mc.get("n_paths", "\u2014"))),
                ("Horizon", f"{mc.get('horizon_days', '\u2014')} days"),
                ("Mean Return", _fmt_pct(mc.get("mean_return"))),
                ("Median Return", _fmt_pct(mc.get("median_return"))),
                ("5th Percentile", _fmt_pct(mc.get("pct_5"))),
                ("95th Percentile", _fmt_pct(mc.get("pct_95"))),
                ("Prob. of Loss", _fmt_pct(mc.get("prob_loss"))),
            ]
            for mi, (lbl, val) in enumerate(mc_items):
                row, col = divmod(mi, 4)
                mx = Inches(0.6) + col * Inches(3.1)
                my = Inches(4.35) + row * Inches(0.45)
                self._data_row(slide, mx, my, lbl, val,
                               w_label=Inches(1.6), w_value=Inches(1.0),
                               value_color=WHITE)

        # Performance metrics
        if perf:
            y_perf = Inches(5.5)
            self._divider(slide, y_perf - Inches(0.15))
            self._section_header(slide, Inches(0.6), y_perf, "Performance Metrics")
            perf_items = [
                ("Ann. Return", _fmt_pct(perf.get("annualized_return")),
                 _pct_color(perf.get("annualized_return"))),
                ("Ann. Volatility", _fmt_pct(perf.get("annualized_vol")), WHITE),
                ("Sharpe Ratio", _fmt_ratio(perf.get("sharpe_ratio")), WHITE),
                ("Sortino Ratio", _fmt_ratio(perf.get("sortino_ratio")), WHITE),
                ("Max Drawdown", _fmt_pct(perf.get("max_drawdown")), ACCENT_RED),
            ]
            for pi, (lbl, val, vc) in enumerate(perf_items):
                self._metric_card(
                    slide, Inches(0.6) + pi * Inches(2.45), Inches(5.85),
                    lbl, val, width=Inches(2.3), height=Inches(0.85),
                    value_color=vc,
                )

        self._footer(slide, snap.get("ticker", ""))

    # ------------------------------------------------------------------
    # SLIDE 9 — Strengths & Risks
    # ------------------------------------------------------------------

    def _add_strengths_risks(self, a: dict):
        slide = self._blank_slide()
        self._title_bar(slide, "Strengths & Risks",
                        subtitle="Balanced view of the investment profile")

        thesis = _sg(a, "fundamentals", "investment_thesis") or {}
        snap = a.get("snapshot") or {}

        strengths = thesis.get("strengths", [])
        risks = thesis.get("risks", [])
        catalysts = thesis.get("catalysts", [])

        col_w = Inches(6.0)
        gutter = Inches(0.3)

        # Left: Strengths
        self._panel(slide, Inches(0.5), Inches(1.2), col_w, Inches(5.3),
                    fill=DARK_PANEL, border=BORDER_COLOR)
        # Green header bar
        hdr = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.2), col_w, Inches(0.4),
        )
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = ACCENT_GREEN_DIM
        hdr.line.fill.background()
        self._text_box(slide, Inches(0.7), Inches(1.23), Inches(5), Inches(0.35),
                       "STRENGTHS & CATALYSTS", size=11, bold=True, color=WHITE)

        all_pos = []
        for s in strengths[:4]:
            all_pos.append(("strength", s))
        for c in catalysts[:3]:
            all_pos.append(("catalyst", c))

        for pi, (ptype, text) in enumerate(all_pos[:6]):
            y = Inches(1.8) + pi * Inches(0.75)
            icon = "\u2713" if ptype == "strength" else "\u2605"
            icon_color = ACCENT_GREEN if ptype == "strength" else ACCENT_GOLD
            disp = text if len(text) <= 180 else text[:177] + "..."
            self._text_box(slide, Inches(0.7), y, Inches(0.3), Inches(0.3),
                           icon, size=14, bold=True, color=icon_color)
            tag = "STRENGTH" if ptype == "strength" else "CATALYST"
            self._text_box(slide, Inches(1.1), y - Inches(0.02), Inches(1.5), Inches(0.2),
                           tag, size=7, bold=True, color=icon_color)
            self._text_box(slide, Inches(1.1), y + Inches(0.2), Inches(5.2), Inches(0.5),
                           disp, size=11, color=OFF_WHITE, line_spacing=15)

        # Right: Risks
        rx = Inches(0.5) + col_w + gutter
        self._panel(slide, rx, Inches(1.2), col_w, Inches(5.3),
                    fill=DARK_PANEL, border=BORDER_COLOR)
        hdr2 = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, rx, Inches(1.2), col_w, Inches(0.4),
        )
        hdr2.fill.solid()
        hdr2.fill.fore_color.rgb = ACCENT_RED
        hdr2.line.fill.background()
        self._text_box(slide, rx + Inches(0.2), Inches(1.23), Inches(5), Inches(0.35),
                       "RISKS & CONCERNS", size=11, bold=True, color=WHITE)

        for ri, risk_text in enumerate(risks[:6]):
            y = Inches(1.8) + ri * Inches(0.75)
            disp = risk_text if len(risk_text) <= 180 else risk_text[:177] + "..."
            # Number badge
            num = slide.shapes.add_shape(
                MSO_SHAPE.OVAL, rx + Inches(0.15), y + Inches(0.05),
                Inches(0.25), Inches(0.25),
            )
            num.fill.solid()
            num.fill.fore_color.rgb = ACCENT_RED
            num.line.fill.background()
            self._text_box(slide, rx + Inches(0.15), y + Inches(0.05),
                           Inches(0.25), Inches(0.25),
                           str(ri + 1), size=9, bold=True, color=WHITE,
                           align=PP_ALIGN.CENTER)
            self._text_box(slide, rx + Inches(0.55), y, Inches(5.25), Inches(0.65),
                           disp, size=11, color=OFF_WHITE, line_spacing=15)

        self._footer(slide, snap.get("ticker", ""))

    # ------------------------------------------------------------------
    # SLIDE 10 — Conclusion
    # ------------------------------------------------------------------

    def _add_conclusion(self, a: dict):
        slide = self._blank_slide()

        snap = a.get("snapshot") or {}
        fund = a.get("fundamentals") or {}
        verdict = fund.get("verdict") or {}
        thesis = fund.get("investment_thesis") or {}
        combined = fund.get("combined_range") or {}
        forecast = a.get("forecast") or {}
        quant = a.get("quant") or {}
        perf = quant.get("performance") or {}

        ticker = snap.get("ticker", a.get("ticker", ""))
        name = snap.get("name", ticker)
        label = verdict.get("label", "")
        upside = verdict.get("upside_pct")
        fair_mid = verdict.get("fair_value_mid")
        fair_low = combined.get("low")
        fair_high = combined.get("high")
        price = snap.get("price")

        is_long = label in ("undervalued", "fairly_valued")
        rec_text = "BUY" if label == "undervalued" else ("HOLD" if label == "fairly_valued" else "SELL")
        rec_color = ACCENT_GREEN if is_long else ACCENT_RED

        self._text_box(slide, Inches(0.8), Inches(0.6), Inches(11), Inches(0.7),
                       "Summary & Conclusion", size=32, bold=True, color=WHITE)

        # Badge + company line
        badge = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(0.8), Inches(1.5), Inches(1.6), Inches(0.55),
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

        parts = [f"{name} ({ticker})"]
        if fair_mid:
            parts.append(f"Target: {_fmt_price(fair_mid)}")
        if upside:
            parts.append(f"Upside: {upside:+.1f}%")
        self._text_box(slide, Inches(2.6), Inches(1.5), Inches(10), Inches(0.55),
                       "  \u2022  ".join(parts), size=18, bold=True, color=WHITE)

        # Key metrics
        my = Inches(2.3)
        summary_metrics = []
        if price:
            summary_metrics.append(("Price", _fmt_price(price), WHITE))
        if fair_low and fair_high:
            summary_metrics.append(("Fair Value Range",
                                    f"{_fmt_price(fair_low)} \u2013 {_fmt_price(fair_high)}",
                                    ACCENT_BLUE))
        sharpe = perf.get("sharpe_ratio")
        if sharpe:
            summary_metrics.append(("Sharpe", _fmt_ratio(sharpe), WHITE))
        mcap = snap.get("market_cap")
        if mcap:
            summary_metrics.append(("Mkt Cap", _fmt_large(mcap), WHITE))

        for mi, (ml, mv, mc) in enumerate(summary_metrics[:4]):
            self._metric_card(slide, Inches(0.8) + mi * Inches(3.1), my,
                              ml, mv, width=Inches(2.9), height=Inches(0.85),
                              value_color=mc)

        # Key takeaways panel
        points = []
        reasoning = verdict.get("reasoning", "")
        if reasoning:
            points.append(reasoning)
        strengths = thesis.get("strengths", [])
        if strengths:
            points.append(strengths[0])
        catalysts = thesis.get("catalysts", [])
        if catalysts:
            points.append(f"Catalyst: {catalysts[0]}")
        risks = thesis.get("risks", [])
        if risks:
            points.append(f"Key risk: {risks[0]}")
        overall_dir = forecast.get("overall_direction", "")
        lt_outlook = forecast.get("long_term_outlook", "")
        if overall_dir:
            p_str = f"ML forecast: {overall_dir}"
            if lt_outlook:
                p_str += f"; long-term outlook: {lt_outlook}"
            points.append(p_str)

        if points:
            self._panel(slide, Inches(0.7), Inches(3.4), Inches(11.9), Inches(2.9),
                        fill=DARK_PANEL, border=BORDER_LIGHT)
            self._text_box(slide, Inches(0.9), Inches(3.5), Inches(3), Inches(0.25),
                           "KEY TAKEAWAYS", size=8, bold=True, color=ACCENT_BLUE)
            paras = []
            for idx, pt in enumerate(points[:5]):
                disp = pt if len(pt) <= 220 else pt[:217] + "..."
                paras.append({
                    "text": f"{idx + 1}.  {disp}",
                    "size": 13,
                    "color": OFF_WHITE,
                    "spacing_after": 10,
                })
            self._rich_text_box(slide, Inches(1.0), Inches(3.8),
                                Inches(11.4), Inches(2.4), paras)

        self._text_box(slide, Inches(0.8), Inches(6.35), Inches(12), Inches(0.4),
                       "Generated by AlphaEdge Analysis Platform",
                       size=12, bold=True, color=ACCENT_BLUE, align=PP_ALIGN.CENTER)

        self._footer(slide, ticker)

    # ------------------------------------------------------------------
    # Chart generators
    # ------------------------------------------------------------------

    def _create_peer_chart(self, comps: dict, snap: dict) -> str | None:
        target = comps.get("target") or {}
        peers = comps.get("peers") or []
        if not target or not peers:
            return None

        metrics = ["pe", "ev_ebitda", "operating_margin", "roe",
                    "revenue_growth", "gross_margin"]
        labels = ["P/E", "EV/EBITDA", "Op Margin", "ROE",
                   "Rev Growth", "Gross Margin"]

        target_vals, peer_medians, valid_labels = [], [], []
        for m, l in zip(metrics, labels):
            tv = target.get(m)
            pvs = [p.get(m) for p in peers if p.get(m) is not None]
            if tv is not None and pvs:
                try:
                    target_vals.append(float(tv))
                    peer_medians.append(float(np.median(pvs)))
                    valid_labels.append(l)
                except (TypeError, ValueError):
                    pass

        if not valid_labels:
            return None

        with plt.rc_context(_MPL_RC):
            fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=150)
            y_pos = np.arange(len(valid_labels))
            bar_h = 0.35
            ax.barh(y_pos - bar_h / 2, target_vals, bar_h,
                    label=snap.get("ticker", "Target"),
                    color="#6366F1", edgecolor="none")
            ax.barh(y_pos + bar_h / 2, peer_medians, bar_h,
                    label="Peer Median", color="#475569", edgecolor="none")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(valid_labels, fontsize=10)
            ax.legend(loc="lower right", fontsize=9, facecolor="#0B142B",
                      edgecolor="#1E293B", labelcolor="white")
            ax.grid(axis="x", alpha=0.3)
            ax.set_title("Target vs Peer Median", fontsize=13,
                         color="white", pad=12, fontweight="bold")
            fig.tight_layout()
            path = os.path.join(self._charts_dir, "peer_chart.png")
            fig.savefig(path, bbox_inches="tight", facecolor="#0B142B")
            plt.close(fig)
            return path

    def _create_fair_value_chart(self, price, low, mid, high) -> str | None:
        vals = [price, low, mid, high]
        if any(v is None for v in vals):
            return None
        try:
            price, low, mid, high = [float(v) for v in vals]
        except (TypeError, ValueError):
            return None

        with plt.rc_context(_MPL_RC):
            fig, ax = plt.subplots(figsize=(12.3, 1.4), dpi=150)
            all_v = [price, low, mid, high]
            x_min, x_max = min(all_v) * 0.82, max(all_v) * 1.18

            ax.barh(0, mid - low, left=low, height=0.45,
                    color="#EF4444", alpha=0.3, edgecolor="none")
            ax.barh(0, high - mid, left=mid, height=0.45,
                    color="#10B981", alpha=0.3, edgecolor="none")
            ax.barh(0, high - low, left=low, height=0.45,
                    color="none", edgecolor="#6366F1", linewidth=2)

            ax.plot(mid, 0, marker="D", color="#6366F1", markersize=14, zorder=5)
            ax.annotate(f"Fair Value\n${mid:.2f}", (mid, 0.4),
                        ha="center", va="bottom", fontsize=10,
                        color="#6366F1", fontweight="bold")
            ax.plot(price, 0, marker="v", color="#F59E0B", markersize=16, zorder=5)
            ax.annotate(f"Current\n${price:.2f}", (price, -0.4),
                        ha="center", va="top", fontsize=10,
                        color="#F59E0B", fontweight="bold")
            ax.annotate(f"Bear: ${low:.2f}", (low, -0.4),
                        ha="center", va="top", fontsize=9, color="#EF4444")
            ax.annotate(f"Bull: ${high:.2f}", (high, -0.4),
                        ha="center", va="top", fontsize=9, color="#10B981")

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-0.85, 0.85)
            ax.set_yticks([])
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
            ax.set_title("Current Price vs Fair Value Range", fontsize=12,
                         color="white", pad=10, fontweight="bold")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            path = os.path.join(self._charts_dir, "fair_value.png")
            fig.savefig(path, bbox_inches="tight", facecolor="#0B142B")
            plt.close(fig)
            return path

    def _create_sensitivity_heatmap(self, dcf: dict, current_price) -> str | None:
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
        first = grid[wacc_keys[0]]
        if not isinstance(first, dict):
            return None
        tg_keys = sorted(first.keys())

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
            fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=150)
            if current_price:
                vmin, vmax = current_price * 0.3, current_price * 2.0
                cmap = "RdYlGn"
            else:
                vmin, vmax = np.nanmin(arr), np.nanmax(arr)
                cmap = "YlGnBu"

            im = ax.imshow(arr, aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    val = arr[i, j]
                    if np.isnan(val):
                        continue
                    tc = "black" if val > (vmin + vmax) / 2 else "white"
                    ax.text(j, i, f"${val:.0f}", ha="center", va="center",
                            fontsize=9, color=tc, fontweight="bold")

            def _fk(k):
                try:
                    return f"{float(k)*100:.1f}%"
                except (TypeError, ValueError):
                    return str(k)

            ax.set_xticks(range(len(tg_keys)))
            ax.set_xticklabels([_fk(k) for k in tg_keys], fontsize=9)
            ax.set_yticks(range(len(wacc_keys)))
            ax.set_yticklabels([_fk(k) for k in wacc_keys], fontsize=9)
            ax.set_xlabel("Terminal Growth Rate", fontsize=10)
            ax.set_ylabel("WACC", fontsize=10)
            ax.set_title("DCF Sensitivity Analysis", fontsize=12,
                         color="white", pad=10, fontweight="bold")
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.ax.tick_params(labelsize=9)
            if current_price:
                cbar.ax.axhline(y=current_price, color="#F59E0B",
                                linewidth=2)
            fig.tight_layout()
            path = os.path.join(self._charts_dir, "sensitivity.png")
            fig.savefig(path, bbox_inches="tight", facecolor="#0B142B")
            plt.close(fig)
            return path

    def _create_range_chart(self, combined: dict, current_price) -> str | None:
        low, mid, high = combined.get("low"), combined.get("mid"), combined.get("high")
        vals = [low, mid, high, current_price]
        if any(v is None for v in vals):
            return None
        try:
            low, mid, high, cp = [float(v) for v in vals]
        except (TypeError, ValueError):
            return None

        with plt.rc_context(_MPL_RC):
            fig, ax = plt.subplots(figsize=(5.5, 1.3), dpi=150)
            x_min, x_max = min(low, cp) * 0.88, max(high, cp) * 1.12
            ax.barh(0, mid - low, left=low, height=0.5,
                    color="#EF4444", alpha=0.5, edgecolor="none")
            ax.barh(0, high - mid, left=mid, height=0.5,
                    color="#10B981", alpha=0.5, edgecolor="none")
            ax.plot(cp, 0, marker="v", color="#F59E0B",
                    markersize=15, zorder=5)
            ax.plot(mid, 0, marker="D", color="#6366F1",
                    markersize=11, zorder=5)
            ax.annotate(f"Current ${cp:.0f}", (cp, 0.42), ha="center",
                        fontsize=9, color="#F59E0B", fontweight="bold")
            ax.annotate(f"Base ${mid:.0f}", (mid, -0.42), ha="center",
                        va="top", fontsize=9, color="#6366F1",
                        fontweight="bold")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-0.65, 0.7)
            ax.set_yticks([])
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
            ax.set_title("Bull / Base / Bear Range", fontsize=11,
                         color="white", pad=8, fontweight="bold")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            path = os.path.join(self._charts_dir, "range.png")
            fig.savefig(path, bbox_inches="tight", facecolor="#0B142B")
            plt.close(fig)
            return path
