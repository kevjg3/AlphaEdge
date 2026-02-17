"""Generate HTML and PDF reports from analysis results."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, BaseLoader

from alphaedge.config import settings

logger = logging.getLogger(__name__)

REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AlphaEdge Report — {{ ticker }}</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 24px; color: #1a1a2e; }
  h1 { border-bottom: 3px solid #4f46e5; padding-bottom: 8px; }
  h2 { color: #4f46e5; margin-top: 32px; }
  table { width: 100%; border-collapse: collapse; margin: 12px 0; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #e2e8f0; }
  th { background: #f8fafc; font-weight: 600; color: #475569; font-size: 0.85em; text-transform: uppercase; }
  td { font-size: 0.95em; }
  .positive { color: #16a34a; }
  .negative { color: #dc2626; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
  .badge-green { background: #dcfce7; color: #166534; }
  .badge-red { background: #fef2f2; color: #991b1b; }
  .badge-gray { background: #f1f5f9; color: #475569; }
  .section { margin: 24px 0; padding: 16px; background: #fafafa; border-radius: 8px; border: 1px solid #e2e8f0; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .stat-label { font-size: 0.75em; color: #94a3b8; text-transform: uppercase; }
  .stat-value { font-size: 1.1em; font-weight: 600; }
  .disclaimer { font-size: 0.75em; color: #94a3b8; margin-top: 40px; border-top: 1px solid #e2e8f0; padding-top: 12px; }
  @media print { body { max-width: 100%; } }
</style>
</head>
<body>
<h1>{{ ticker }} — Investment Analysis Report</h1>
<p style="color:#64748b;">Generated {{ generated_at }} | Run ID: {{ run_id }}</p>

{% if snapshot %}
<h2>Market Snapshot</h2>
<div class="section grid">
  <div><div class="stat-label">Price</div><div class="stat-value">${{ "%.2f"|format(snapshot.price or 0) }}</div></div>
  <div><div class="stat-label">1D Change</div><div class="stat-value {{ 'positive' if (snapshot.change_1d_pct or 0) >= 0 else 'negative' }}">{{ "%.2f"|format(snapshot.change_1d_pct or 0) }}%</div></div>
  <div><div class="stat-label">Market Cap</div><div class="stat-value">{{ fmt_large(snapshot.market_cap) }}</div></div>
  <div><div class="stat-label">P/E</div><div class="stat-value">{{ "%.1f"|format(snapshot.pe_ratio) if snapshot.pe_ratio else "—" }}</div></div>
  <div><div class="stat-label">Beta</div><div class="stat-value">{{ "%.2f"|format(snapshot.beta) if snapshot.beta else "—" }}</div></div>
  <div><div class="stat-label">Sector</div><div class="stat-value">{{ snapshot.sector or "—" }}</div></div>
  <div><div class="stat-label">52W High</div><div class="stat-value">${{ "%.2f"|format(snapshot.high_52w) if snapshot.high_52w else "—" }}</div></div>
  <div><div class="stat-label">52W Low</div><div class="stat-value">${{ "%.2f"|format(snapshot.low_52w) if snapshot.low_52w else "—" }}</div></div>
</div>
{% endif %}

{% if fundamentals %}
<h2>Fundamental Valuation</h2>
{% if fundamentals.verdict %}
<div class="section">
  <span class="badge {{ 'badge-green' if fundamentals.verdict.label == 'undervalued' else 'badge-red' if fundamentals.verdict.label == 'overvalued' else 'badge-gray' }}">
    {{ fundamentals.verdict.label|replace("_"," ")|title }}
  </span>
  {% if fundamentals.verdict.reasoning %}<p>{{ fundamentals.verdict.reasoning }}</p>{% endif %}
</div>
{% endif %}
{% if fundamentals.dcf_valuation and fundamentals.dcf_valuation.implied_price %}
<div class="section">
  <strong>DCF Implied Price:</strong> ${{ "%.2f"|format(fundamentals.dcf_valuation.implied_price) }}
  {% if fundamentals.dcf_valuation.upside_downside_pct is not none %}
    <span class="{{ 'positive' if fundamentals.dcf_valuation.upside_downside_pct > 0 else 'negative' }}">
      ({{ "%+.1f"|format(fundamentals.dcf_valuation.upside_downside_pct) }}%)
    </span>
  {% endif %}
</div>
{% endif %}
{% endif %}

{% if technicals %}
<h2>Technical Analysis</h2>
{% if technicals.regime %}
<div class="section">
  <strong>Current Regime:</strong>
  <span class="badge {{ 'badge-green' if technicals.regime.current_regime == 'low_vol' else 'badge-red' if technicals.regime.current_regime == 'high_vol' else 'badge-gray' }}">
    {{ technicals.regime.current_regime|replace("_"," ")|title }}
  </span>
</div>
{% endif %}
{% if technicals.indicators %}
<table>
  <tr><th>Indicator</th><th>Value</th></tr>
  {% for key in ["rsi_14","macd","macd_histogram","atr_14","adx_14","bb_pct_b","volume_ratio"] %}
  {% if technicals.indicators[key] is not none %}
  <tr><td>{{ key|replace("_"," ")|title }}</td><td>{{ "%.4f"|format(technicals.indicators[key]) }}</td></tr>
  {% endif %}
  {% endfor %}
</table>
{% endif %}
{% endif %}

{% if news %}
<h2>News & Sentiment</h2>
{% if news.synthesis %}
<div class="section">
  <strong>Sentiment:</strong>
  <span class="badge {{ 'badge-green' if news.synthesis.overall_sentiment == 'bullish' else 'badge-red' if news.synthesis.overall_sentiment == 'bearish' else 'badge-gray' }}">
    {{ news.synthesis.overall_sentiment|title }}
  </span>
  {% if news.synthesis.narrative %}<p>{{ news.synthesis.narrative }}</p>{% endif %}
</div>
{% endif %}
{% endif %}

{% if forecast %}
<h2>Forecast</h2>
<div class="section">
  <strong>Overall:</strong> {{ forecast.overall_direction|title }}
  | Short-term: {{ forecast.short_term_outlook|title }}
  | Long-term: {{ forecast.long_term_outlook|title }}
</div>
{% if forecast.forecasts %}
<table>
  <tr><th>Horizon</th><th>Predicted Return</th><th>Direction</th><th>Confidence</th></tr>
  {% for h, hdata in forecast.forecasts.items() %}
  {% set ep = hdata.ensemble_prediction if hdata.ensemble_prediction else {} %}
  <tr>
    <td>{{ h }}</td>
    <td class="{{ 'positive' if (ep.predicted_return or 0) > 0 else 'negative' }}">{{ "%.2f"|format(ep.predicted_return or 0) }}%</td>
    <td>{{ ep.direction or "—" }}</td>
    <td>{{ "%.0f"|format((ep.confidence or 0) * 100) }}%</td>
  </tr>
  {% endfor %}
</table>
{% endif %}
{% endif %}

{% if risk %}
<h2>Risk Analysis</h2>
{% if risk.var %}
<table>
  <tr><th>Method</th><th>VaR (95%)</th></tr>
  {% if risk.var.parametric_var_pct is not none %}<tr><td>Parametric</td><td>{{ "%.2f"|format(risk.var.parametric_var_pct * 100) }}%</td></tr>{% endif %}
  {% if risk.var.historical_var_pct is not none %}<tr><td>Historical</td><td>{{ "%.2f"|format(risk.var.historical_var_pct * 100) }}%</td></tr>{% endif %}
  {% if risk.var.monte_carlo_var_pct is not none %}<tr><td>Monte Carlo</td><td>{{ "%.2f"|format(risk.var.monte_carlo_var_pct * 100) }}%</td></tr>{% endif %}
  {% if risk.var.expected_shortfall_pct is not none %}<tr><td>CVaR (ES)</td><td>{{ "%.2f"|format(risk.var.expected_shortfall_pct * 100) }}%</td></tr>{% endif %}
</table>
{% endif %}
{% endif %}

{% if warnings %}
<h2>Warnings</h2>
<ul>
{% for w in warnings %}
<li style="color:#94a3b8;font-size:0.85em;">{{ w }}</li>
{% endfor %}
</ul>
{% endif %}

<div class="disclaimer">
  <strong>Disclaimer:</strong> This report is generated by AlphaEdge for informational purposes only.
  It does not constitute investment advice. Past performance is not indicative of future results.
  All data is sourced from public APIs and may contain errors or delays.
</div>
</body>
</html>
"""


def _fmt_large(n: Any) -> str:
    if n is None:
        return "—"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    if abs(n) >= 1e12:
        return f"${n / 1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"${n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n / 1e6:.2f}M"
    return f"${n:,.0f}"


class ReportGenerator:
    """Generate HTML reports from analysis results."""

    def __init__(self, output_dir: str | None = None):
        self._output_dir = Path(output_dir or settings.data_dir) / "reports"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._env = Environment(loader=BaseLoader())
        self._env.globals["fmt_large"] = _fmt_large
        self._template = self._env.from_string(REPORT_TEMPLATE)

    def generate_html(self, analysis: dict) -> str:
        """Render analysis dict to HTML string."""
        return self._template.render(
            ticker=analysis.get("ticker", ""),
            run_id=analysis.get("run_id", ""),
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            snapshot=analysis.get("snapshot"),
            fundamentals=analysis.get("fundamentals"),
            technicals=analysis.get("technicals"),
            news=analysis.get("news"),
            forecast=analysis.get("forecast"),
            risk=analysis.get("risk"),
            warnings=analysis.get("warnings", []),
        )

    def save_html(self, analysis: dict, filename: str | None = None) -> Path:
        """Save HTML report to file."""
        html = self.generate_html(analysis)
        ticker = analysis.get("ticker", "report")
        run_id = analysis.get("run_id", "")
        fname = filename or f"{ticker}_{run_id}.html"
        path = self._output_dir / fname
        path.write_text(html, encoding="utf-8")
        logger.info("HTML report saved to %s", path)
        return path

    def save_pdf(self, analysis: dict, filename: str | None = None) -> Path | None:
        """Save PDF report. Requires weasyprint."""
        try:
            from weasyprint import HTML as WeasyprintHTML

            html = self.generate_html(analysis)
            ticker = analysis.get("ticker", "report")
            run_id = analysis.get("run_id", "")
            fname = filename or f"{ticker}_{run_id}.pdf"
            path = self._output_dir / fname
            WeasyprintHTML(string=html).write_pdf(str(path))
            logger.info("PDF report saved to %s", path)
            return path
        except ImportError:
            logger.warning("weasyprint not installed — PDF generation unavailable")
            return None
        except Exception as e:
            logger.warning("PDF generation failed: %s", e)
            return None
