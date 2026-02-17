"""Investment thesis generator — synthesizes company data into IB-style analysis."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_investment_thesis(
    info: dict[str, Any],
    financial_health: dict[str, Any],
    comps_valuation: dict[str, Any],
    dcf_valuation: dict[str, Any],
    verdict: dict[str, Any],
    combined_range: dict[str, Any],
    current_price: float | None,
    news_sentiment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a structured investment thesis from all available data.

    Returns a dict with:
      - company_overview: str  (business description + key facts)
      - industry_analysis: str (sector positioning, competitive landscape)
      - investment_thesis: str (bull/bear case, catalysts, risks)
      - key_metrics: list[dict] (highlighted financial metrics with context)
      - strengths: list[str]
      - risks: list[str]
      - catalysts: list[str]
    """
    ticker = info.get("ticker", "")
    name = info.get("name", ticker)
    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")
    description = info.get("description", "")
    country = info.get("country", "")
    employees = info.get("employees")
    website = info.get("website", "")

    # ── Company Overview ──
    company_overview = _build_company_overview(
        name, ticker, sector, industry, description,
        country, employees, website, info,
    )

    # ── Industry Analysis ──
    industry_analysis = _build_industry_analysis(
        name, sector, industry, info, comps_valuation,
    )

    # ── Strengths, Risks, Catalysts ──
    strengths = _identify_strengths(info, financial_health)
    risks = _identify_risks(info, financial_health)
    catalysts = _identify_catalysts(info, financial_health, news_sentiment)

    # ── Key Highlighted Metrics ──
    key_metrics = _build_key_metrics(info, financial_health, dcf_valuation)

    # ── Investment Thesis ──
    investment_thesis = _build_investment_thesis(
        name, ticker, sector, info, financial_health,
        verdict, combined_range, current_price,
        strengths, risks, catalysts,
    )

    return {
        "company_overview": company_overview,
        "industry_analysis": industry_analysis,
        "investment_thesis": investment_thesis,
        "key_metrics": key_metrics,
        "strengths": strengths,
        "risks": risks,
        "catalysts": catalysts,
    }


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def _build_company_overview(
    name: str, ticker: str, sector: str, industry: str,
    description: str, country: str, employees: int | None,
    website: str, info: dict,
) -> str:
    """Build a concise company overview paragraph."""
    parts: list[str] = []

    if description:
        # Truncate to first 2 sentences for a clean overview
        sentences = description.replace("\n", " ").split(". ")
        overview = ". ".join(sentences[:3])
        if not overview.endswith("."):
            overview += "."
        parts.append(overview)
    else:
        parts.append(f"{name} ({ticker}) operates in the {industry} industry within the {sector} sector.")

    # Add key facts
    facts: list[str] = []
    if country:
        facts.append(f"headquartered in {country}")
    if employees and employees > 0:
        facts.append(f"approximately {employees:,} employees")

    market_cap = info.get("market_cap")
    if market_cap:
        if market_cap >= 200e9:
            cap_tier = "mega-cap"
        elif market_cap >= 10e9:
            cap_tier = "large-cap"
        elif market_cap >= 2e9:
            cap_tier = "mid-cap"
        elif market_cap >= 300e6:
            cap_tier = "small-cap"
        else:
            cap_tier = "micro-cap"
        facts.append(f"a {cap_tier} company")

    if facts:
        parts.append(f"The company is {', '.join(facts)}.")

    return " ".join(parts)


def _build_industry_analysis(
    name: str, sector: str, industry: str,
    info: dict, comps: dict,
) -> str:
    """Build industry positioning and competitive context."""
    parts: list[str] = []

    # Sector context
    sector_context = {
        "Technology": "The technology sector continues to be driven by digital transformation, cloud computing, AI adoption, and software-as-a-service business models.",
        "Healthcare": "The healthcare sector benefits from aging demographics, pharmaceutical innovation, and increasing healthcare spending globally.",
        "Financial Services": "The financial services sector is influenced by interest rate environments, regulatory changes, and the shift toward digital banking and fintech solutions.",
        "Consumer Cyclical": "The consumer cyclical sector is sensitive to economic cycles, consumer confidence, and discretionary spending trends.",
        "Consumer Defensive": "The consumer defensive sector offers relative stability, benefiting from inelastic demand for essential goods and steady dividend yields.",
        "Industrials": "The industrials sector is tied to economic growth, infrastructure investment, and manufacturing activity globally.",
        "Energy": "The energy sector is shaped by commodity price dynamics, the energy transition, and geopolitical supply factors.",
        "Communication Services": "The communication services sector spans traditional telecom, digital media, and entertainment, driven by content demand and advertising markets.",
        "Utilities": "The utilities sector provides regulated, stable cash flows with increasing growth opportunities from renewable energy and grid modernization.",
        "Real Estate": "The real estate sector offers income through REITs and is influenced by interest rates, occupancy trends, and property valuations.",
        "Basic Materials": "The basic materials sector is cyclical, driven by commodity prices, industrial demand, and supply chain dynamics.",
    }

    parts.append(f"{name} competes within the {industry} space in the broader {sector} sector.")
    parts.append(sector_context.get(sector, f"The {sector} sector presents both opportunities and challenges for companies in this space."))

    # Competitive positioning from metrics
    pe = info.get("trailing_pe")
    market_cap = info.get("market_cap")
    margins = info.get("operating_margins")
    growth = info.get("revenue_growth")

    positioning_points: list[str] = []
    if market_cap and market_cap > 100e9:
        positioning_points.append("market leadership position with significant scale advantages")
    elif market_cap and market_cap > 10e9:
        positioning_points.append("an established market position with meaningful competitive advantages")

    if margins is not None and margins > 0.20:
        positioning_points.append("strong pricing power reflected in above-average operating margins")
    elif margins is not None and margins > 0.10:
        positioning_points.append("healthy operating margins indicating competitive efficiency")

    if growth is not None and growth > 0.15:
        positioning_points.append("robust revenue growth outpacing industry averages")
    elif growth is not None and growth > 0.05:
        positioning_points.append("steady revenue growth demonstrating market share resilience")

    # Peer context
    peer_count = comps.get("peer_count", 0)
    if peer_count > 0:
        peers = comps.get("peers", [])
        peer_names = [p.get("ticker", "") for p in peers[:5] if isinstance(p, dict)]
        if peer_names:
            positioning_points.append(f"competing against peers including {', '.join(peer_names)}")

    if positioning_points:
        parts.append(f"The company demonstrates {'; '.join(positioning_points)}.")

    return " ".join(parts)


def _identify_strengths(info: dict, health: dict) -> list[str]:
    """Identify key strengths from financial data."""
    strengths: list[str] = []

    margins = info.get("operating_margins")
    if margins is not None:
        if margins > 0.25:
            strengths.append(f"Exceptional operating margins of {margins:.0%}, indicating strong pricing power and efficiency")
        elif margins > 0.15:
            strengths.append(f"Healthy operating margins of {margins:.0%}, above industry averages")

    growth = info.get("revenue_growth")
    if growth is not None and growth > 0.10:
        strengths.append(f"Strong revenue growth of {growth:.0%}, demonstrating market share gains")

    roe = info.get("return_on_equity")
    if roe is not None and roe > 0.15:
        strengths.append(f"High return on equity of {roe:.0%}, reflecting effective capital allocation")

    fcf = info.get("free_cashflow")
    revenue = info.get("total_revenue")
    if fcf and revenue and fcf > 0 and revenue > 0:
        fcf_margin = fcf / revenue
        if fcf_margin > 0.15:
            strengths.append(f"Robust free cash flow generation with {fcf_margin:.0%} FCF margin")

    current_ratio = info.get("current_ratio")
    if current_ratio is not None and current_ratio > 1.5:
        strengths.append(f"Strong balance sheet with current ratio of {current_ratio:.1f}x")

    div_yield = info.get("dividend_yield")
    if div_yield is not None and div_yield > 0.02:
        strengths.append(f"Attractive dividend yield of {div_yield:.1%}")

    market_cap = info.get("market_cap")
    if market_cap and market_cap > 100e9:
        strengths.append("Market-leading scale providing competitive moat and bargaining power")

    # From financial health
    income = health.get("income_summary", {})
    gross_margin = income.get("gross_margin")
    if gross_margin is not None and gross_margin > 0.50:
        strengths.append(f"High gross margins of {gross_margin:.0%} suggesting differentiated products or services")

    if not strengths:
        strengths.append("Established operations in a defined market segment")

    return strengths[:6]  # Cap at 6


def _identify_risks(info: dict, health: dict) -> list[str]:
    """Identify key risks from financial data."""
    risks: list[str] = []

    dte = info.get("debt_to_equity")
    if dte is not None and dte > 150:
        risks.append(f"Elevated leverage with debt-to-equity ratio of {dte:.0f}%, increasing financial risk")
    elif dte is not None and dte > 80:
        risks.append(f"Moderate leverage with debt-to-equity of {dte:.0f}%, requiring monitoring")

    growth = info.get("revenue_growth")
    if growth is not None and growth < 0:
        risks.append(f"Revenue declining at {growth:.0%}, signaling potential market share loss")
    elif growth is not None and growth < 0.02:
        risks.append("Sluggish revenue growth may indicate market saturation")

    margins = info.get("operating_margins")
    if margins is not None and margins < 0.05:
        risks.append(f"Thin operating margins of {margins:.0%} leave limited room for error")

    pe = info.get("trailing_pe")
    if pe is not None and pe > 50:
        risks.append(f"Premium valuation at {pe:.0f}x P/E requires continued growth execution")
    elif pe is not None and pe > 35:
        risks.append(f"Above-average P/E of {pe:.0f}x implies elevated market expectations")

    beta = info.get("beta")
    if beta is not None and beta > 1.5:
        risks.append(f"High beta of {beta:.2f} means amplified market volatility exposure")

    current_ratio = info.get("current_ratio")
    if current_ratio is not None and current_ratio < 1.0:
        risks.append(f"Current ratio below 1.0 ({current_ratio:.1f}x) may indicate near-term liquidity pressure")

    earnings_growth = info.get("earnings_growth")
    if earnings_growth is not None and earnings_growth < -0.10:
        risks.append(f"Earnings declining {earnings_growth:.0%}, raising profitability concerns")

    # Sector-specific risks
    sector = info.get("sector", "")
    sector_risks = {
        "Technology": "Technology disruption risk and competitive threats from well-funded peers",
        "Energy": "Commodity price volatility and regulatory pressures from energy transition",
        "Financial Services": "Interest rate sensitivity and regulatory compliance requirements",
        "Healthcare": "Regulatory and reimbursement risk, patent expiration exposure",
        "Consumer Cyclical": "Economic sensitivity and shifting consumer preferences",
    }
    if sector in sector_risks:
        risks.append(sector_risks[sector])

    if not risks:
        risks.append("General market and macroeconomic risks applicable to all equities")

    return risks[:6]


def _identify_catalysts(
    info: dict, health: dict,
    news_sentiment: dict | None,
) -> list[str]:
    """Identify potential catalysts."""
    catalysts: list[str] = []

    growth = info.get("revenue_growth")
    if growth is not None and growth > 0.15:
        catalysts.append("Accelerating revenue growth could drive multiple expansion")

    margins = info.get("operating_margins")
    gross_margins = info.get("gross_margins")
    if margins is not None and gross_margins is not None and gross_margins - margins > 0.20:
        catalysts.append("Significant operating leverage opportunity as gross-to-operating margin gap narrows")

    fcf = info.get("free_cashflow")
    if fcf and fcf > 0:
        catalysts.append("Strong free cash flow supports share buybacks, dividends, or strategic M&A")

    # Valuation-related
    pe = info.get("trailing_pe")
    forward_pe = info.get("forward_pe")
    if pe and forward_pe and forward_pe < pe * 0.85:
        catalysts.append("Forward P/E significantly below trailing P/E suggests expected earnings improvement")

    # Sentiment-related
    if news_sentiment:
        overall = news_sentiment.get("overall_sentiment", "")
        if overall in ("bullish", "very_bullish"):
            catalysts.append("Positive news sentiment may drive near-term investor interest")

    # Sector catalysts
    sector = info.get("sector", "")
    sector_catalysts = {
        "Technology": "AI and cloud adoption driving secular growth tailwinds",
        "Healthcare": "Aging demographics and innovation pipeline providing long-term demand",
        "Financial Services": "Interest rate normalization and digital transformation opportunities",
        "Industrials": "Infrastructure spending and reshoring trends benefiting domestic operations",
        "Energy": "Energy security focus and potential supply constraints supporting prices",
        "Consumer Defensive": "Pricing power and stable demand providing earnings visibility",
    }
    if sector in sector_catalysts:
        catalysts.append(sector_catalysts[sector])

    if not catalysts:
        catalysts.append("Potential for market re-rating on improved fundamental performance")

    return catalysts[:5]


def _build_key_metrics(
    info: dict, health: dict, dcf: dict,
) -> list[dict]:
    """Build highlighted key metrics with context."""
    metrics: list[dict] = []

    revenue = info.get("total_revenue")
    if revenue:
        growth = info.get("revenue_growth")
        context = f"YoY growth: {growth:.0%}" if growth is not None else ""
        metrics.append({"label": "Revenue", "value": _fmt_large(revenue), "context": context, "category": "growth"})

    ebitda = info.get("ebitda")
    if ebitda and ebitda > 0:
        margin = None
        if revenue and revenue > 0:
            margin = ebitda / revenue
        context = f"EBITDA margin: {margin:.0%}" if margin else ""
        metrics.append({"label": "EBITDA", "value": _fmt_large(ebitda), "context": context, "category": "profitability"})

    fcf = info.get("free_cashflow")
    if fcf is not None:
        context = "positive" if fcf > 0 else "negative"
        metrics.append({"label": "Free Cash Flow", "value": _fmt_large(fcf), "context": f"FCF is {context}", "category": "cash"})

    roe = info.get("return_on_equity")
    if roe is not None:
        quality = "excellent" if roe > 0.20 else "good" if roe > 0.12 else "below average"
        metrics.append({"label": "Return on Equity", "value": f"{roe:.1%}", "context": f"Capital efficiency: {quality}", "category": "returns"})

    dte = info.get("debt_to_equity")
    if dte is not None:
        level = "conservative" if dte < 50 else "moderate" if dte < 100 else "elevated"
        metrics.append({"label": "Debt/Equity", "value": f"{dte:.0f}%", "context": f"Leverage: {level}", "category": "risk"})

    pe = info.get("trailing_pe")
    if pe is not None and pe > 0:
        context = "premium" if pe > 30 else "reasonable" if pe > 15 else "value"
        metrics.append({"label": "P/E Ratio", "value": f"{pe:.1f}x", "context": f"Valuation: {context}", "category": "valuation"})

    return metrics[:6]


def _build_investment_thesis(
    name: str, ticker: str, sector: str, info: dict,
    health: dict, verdict: dict, combined_range: dict,
    current_price: float | None,
    strengths: list[str], risks: list[str],
    catalysts: list[str],
) -> str:
    """Build the main investment thesis paragraph."""
    parts: list[str] = []

    label = verdict.get("label", "insufficient_data")
    upside = verdict.get("upside_pct")
    fair_mid = verdict.get("fair_value_mid")

    # Opening statement based on verdict
    if label == "undervalued" and upside:
        parts.append(
            f"{name} appears undervalued based on our blended DCF and comparable company analysis, "
            f"with an estimated {upside:.0f}% upside to fair value of ${fair_mid:.2f}."
        )
    elif label == "overvalued" and upside:
        parts.append(
            f"{name} appears overvalued relative to our fundamental analysis, "
            f"trading {abs(upside):.0f}% above our estimated fair value of ${fair_mid:.2f}."
        )
    elif label == "fairly_valued":
        parts.append(
            f"{name} appears fairly valued near our estimated fair value of ${fair_mid:.2f}, "
            f"with limited near-term mispricing identified."
        )
    else:
        parts.append(f"{name} requires further analysis to establish a definitive valuation view.")

    # Growth and profitability characterization
    growth = info.get("revenue_growth")
    margins = info.get("operating_margins")
    if growth is not None and margins is not None:
        if growth > 0.10 and margins > 0.15:
            parts.append(
                "The company demonstrates a compelling growth-and-profitability profile, "
                "combining double-digit revenue growth with strong margins."
            )
        elif growth > 0.05 and margins > 0.10:
            parts.append(
                "The company shows steady growth with adequate profitability, "
                "positioning it for sustainable value creation."
            )
        elif growth < 0:
            parts.append(
                "Revenue contraction raises concerns about the company's competitive positioning, "
                "though margin stability could provide a floor for valuation."
            )
        else:
            parts.append(
                "Growth is modest but the business maintains operational discipline, "
                "suggesting a mature, cash-generative profile."
            )

    # Key strength
    if strengths:
        parts.append(f"Key strengths include: {strengths[0].lower()}.")

    # Key risk
    if risks:
        parts.append(f"Primary risks include: {risks[0].lower()}.")

    # Conclusion
    if label == "undervalued":
        parts.append(
            "We see an opportunity for capital appreciation as the market recognizes "
            "the company's intrinsic value. Investors should weigh the identified catalysts "
            "against the risk factors before establishing a position."
        )
    elif label == "overvalued":
        parts.append(
            "At current valuations, we see limited margin of safety. "
            "The stock may be pricing in optimistic growth assumptions that could "
            "prove difficult to sustain. A pullback could present a better entry point."
        )
    else:
        parts.append(
            "At current levels, the risk-reward appears balanced. "
            "Investors should monitor upcoming catalysts and fundamental "
            "trends for potential shifts in the investment case."
        )

    return " ".join(parts)


def _fmt_large(n: float | None) -> str:
    """Format a large number for display."""
    if n is None:
        return "N/A"
    absn = abs(n)
    sign = "-" if n < 0 else ""
    if absn >= 1e12:
        return f"{sign}${absn / 1e12:.1f}T"
    if absn >= 1e9:
        return f"{sign}${absn / 1e9:.1f}B"
    if absn >= 1e6:
        return f"{sign}${absn / 1e6:.0f}M"
    return f"{sign}${absn:,.0f}"
