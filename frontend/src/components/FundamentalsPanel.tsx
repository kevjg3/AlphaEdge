"use client";

import { Card, Stat, Badge } from "./Card";
import { fmtNumber, fmtPercent, fmtLargeNumber } from "@/lib/format";

export default function FundamentalsPanel({ data }: { data: Record<string, any> | null }) {
  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-gray-500 animate-fade-in">
        <p className="text-sm">Fundamentals not available</p>
      </div>
    );
  }

  const health = data.financial_health || {};
  const income = health.income_summary || {};
  const balance = health.balance_summary || {};
  const cf = health.cashflow_summary || {};
  const comps = data.comps_valuation || {};
  const dcf = data.dcf_valuation || {};
  const verdict = data.verdict || {};
  const combined = data.combined_range || {};
  const thesis = data.investment_thesis || {};

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Verdict */}
      {verdict.label && (
        <Card glow>
          <div className="flex items-center gap-3 mb-2">
            <Badge
              text={verdict.label.replace(/_/g, " ")}
              variant={verdict.label === "undervalued" ? "positive" : verdict.label === "overvalued" ? "negative" : "neutral"}
              dot
            />
            {verdict.upside_pct != null && (
              <span className={`text-lg font-bold font-mono tabular-nums ${verdict.upside_pct >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                {fmtPercent(verdict.upside_pct)} to fair value
              </span>
            )}
          </div>
          {verdict.reasoning && <p className="text-sm text-gray-400 leading-relaxed">{verdict.reasoning}</p>}
          {combined.low != null && (
            <div className="flex items-center gap-4 mt-3 pt-3 border-t border-white/[0.06]">
              <span className="text-xs text-gray-500 uppercase tracking-wider">Fair Value Range</span>
              <div className="flex items-center gap-3 font-mono text-sm">
                <span className="text-red-400">${fmtNumber(combined.low)}</span>
                <span className="text-gray-600">&mdash;</span>
                <span className="text-white font-bold">${fmtNumber(combined.mid)}</span>
                <span className="text-gray-600">&mdash;</span>
                <span className="text-emerald-400">${fmtNumber(combined.high)}</span>
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Investment Thesis */}
      {thesis.investment_thesis && (
        <Card title="Investment Thesis">
          <p className="text-sm text-gray-300 leading-relaxed">{thesis.investment_thesis}</p>
        </Card>
      )}

      {/* Industry Analysis */}
      {thesis.industry_analysis && (
        <Card title="Industry Analysis">
          <p className="text-sm text-gray-300 leading-relaxed">{thesis.industry_analysis}</p>
        </Card>
      )}

      {/* Strengths, Risks & Catalysts */}
      {(thesis.strengths?.length > 0 || thesis.risks?.length > 0 || thesis.catalysts?.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Strengths */}
          {thesis.strengths?.length > 0 && (
            <Card title="Strengths">
              <ul className="space-y-2">
                {thesis.strengths.map((s: string, i: number) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <span className="text-emerald-400 mt-0.5 shrink-0">+</span>
                    <span className="text-gray-300">{s}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {/* Risks */}
          {thesis.risks?.length > 0 && (
            <Card title="Risks">
              <ul className="space-y-2">
                {thesis.risks.map((r: string, i: number) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <span className="text-red-400 mt-0.5 shrink-0">&minus;</span>
                    <span className="text-gray-300">{r}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {/* Catalysts */}
          {thesis.catalysts?.length > 0 && (
            <Card title="Catalysts">
              <ul className="space-y-2">
                {thesis.catalysts.map((c: string, i: number) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <span className="text-sky-400 mt-0.5 shrink-0">&#x2192;</span>
                    <span className="text-gray-300">{c}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      )}

      {/* Key Metrics (from thesis generator) */}
      {thesis.key_metrics?.length > 0 && (
        <Card title="Key Investment Metrics">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {thesis.key_metrics.map((m: any, i: number) => (
              <div key={i} className="space-y-1">
                <div className="text-xs text-gray-500 uppercase tracking-wider">{m.label}</div>
                <div className="text-lg font-bold text-white font-mono tabular-nums">{m.value}</div>
                {m.context && <div className="text-xs text-gray-400">{m.context}</div>}
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Income Statement */}
      <Card title="Income Statement">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Stat label="Revenue" value={fmtLargeNumber(income.revenue)} />
          <Stat label="Revenue Growth" value={income.revenue_growth_yoy != null ? fmtPercent(income.revenue_growth_yoy * 100) : "\u2014"} color={income.revenue_growth_yoy > 0 ? "text-emerald-400" : "text-red-400"} />
          <Stat label="Gross Margin" value={income.gross_margin != null ? fmtPercent(income.gross_margin * 100, 1) : "\u2014"} />
          <Stat label="Operating Margin" value={income.operating_margin != null ? fmtPercent(income.operating_margin * 100, 1) : "\u2014"} />
          <Stat label="Net Margin" value={income.net_margin != null ? fmtPercent(income.net_margin * 100, 1) : "\u2014"} />
          <Stat label="EBITDA" value={fmtLargeNumber(income.ebitda)} />
          <Stat label="EPS" value={income.eps != null ? `$${fmtNumber(income.eps)}` : "\u2014"} />
        </div>
      </Card>

      {/* Balance Sheet */}
      <Card title="Balance Sheet">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Stat label="Total Assets" value={fmtLargeNumber(balance.total_assets)} />
          <Stat label="Total Debt" value={fmtLargeNumber(balance.total_debt)} />
          <Stat label="Cash" value={fmtLargeNumber(balance.cash)} />
          <Stat label="Current Ratio" value={balance.current_ratio != null ? fmtNumber(balance.current_ratio, 1) : "\u2014"} />
          <Stat label="Debt / Equity" value={balance.debt_to_equity != null ? fmtNumber(balance.debt_to_equity, 2) : "\u2014"} />
          <Stat label="Book Value / Share" value={balance.book_value_per_share != null ? `$${fmtNumber(balance.book_value_per_share)}` : "\u2014"} />
        </div>
      </Card>

      {/* Cash Flow */}
      <Card title="Cash Flow">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Stat label="Operating CF" value={fmtLargeNumber(cf.operating_cf)} />
          <Stat label="Free Cash Flow" value={fmtLargeNumber(cf.free_cf)} />
          <Stat label="FCF Margin" value={cf.fcf_margin != null ? fmtPercent(cf.fcf_margin * 100, 1) : "\u2014"} />
          <Stat label="FCF Yield" value={cf.fcf_yield != null ? fmtPercent(cf.fcf_yield * 100, 1) : "\u2014"} />
        </div>
      </Card>

      {/* DCF */}
      {dcf.implied_price != null && (
        <Card title="DCF Valuation">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Stat label="Implied Price" value={`$${fmtNumber(dcf.implied_price)}`} color="text-brand-400" />
            <Stat label="Enterprise Value" value={fmtLargeNumber(dcf.enterprise_value)} />
            <Stat label="WACC" value={dcf.assumptions_used?.wacc != null ? fmtPercent(dcf.assumptions_used.wacc * 100, 1) : "\u2014"} />
            <Stat label="Upside/Downside" value={dcf.upside_downside_pct != null ? fmtPercent(dcf.upside_downside_pct) : "\u2014"}
              color={dcf.upside_downside_pct > 0 ? "text-emerald-400" : "text-red-400"} />
          </div>
        </Card>
      )}

      {/* Comps Summary */}
      {comps.peer_count > 0 && (
        <Card title={`Comparable Companies (${comps.peer_count} peers)`}>
          <div className="flex flex-wrap gap-2">
            {comps.peers?.slice(0, 8).map((p: any) => (
              <div key={p.ticker} className="px-3 py-1.5 bg-surface-overlay rounded-lg border border-white/[0.04] text-sm">
                <span className="text-white font-medium">{p.ticker}</span>
                {p.pe && <span className="text-gray-500 ml-2 font-mono text-xs">PE {fmtNumber(p.pe, 1)}</span>}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
