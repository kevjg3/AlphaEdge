"use client";

import { Card, Stat, Badge } from "./Card";
import { fmtNumber, fmtPercent, fmtLargeNumber } from "@/lib/format";
import type { SnapshotData } from "@/lib/api";

export default function SnapshotPanel({ data }: { data: SnapshotData }) {
  const changeColor = (data.change_1d_pct ?? 0) >= 0 ? "text-emerald-400" : "text-red-400";
  const changeBg = (data.change_1d_pct ?? 0) >= 0 ? "bg-emerald-500/10" : "bg-red-500/10";

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Hero Header */}
      <div className="bg-surface-raised border border-white/[0.06] rounded-2xl p-6 shadow-card">
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
          {/* Left: Ticker & Name */}
          <div>
            <div className="flex items-center gap-3 mb-1">
              <h2 className="text-3xl font-bold text-white tracking-tight">{data.ticker}</h2>
              {data.sector && <Badge text={data.sector} variant="info" />}
              {data.industry && <Badge text={data.industry} size="sm" />}
            </div>
            <span className="text-gray-400 text-sm">{data.name}</span>
            {data.country && (
              <span className="text-gray-500 text-xs ml-2">{data.country}</span>
            )}
            {data.website && (
              <div className="mt-1">
                <a href={data.website} target="_blank" rel="noopener noreferrer"
                  className="text-xs text-brand-400 hover:text-brand-300 transition-colors">
                  {data.website.replace(/^https?:\/\//, "").replace(/\/$/, "")}
                </a>
              </div>
            )}
          </div>

          {/* Right: Price */}
          <div className="text-right">
            <div className="text-4xl font-bold text-white tracking-tight font-mono tabular-nums">
              ${fmtNumber(data.price)}
            </div>
            <div className={`mt-1 inline-flex items-center gap-2 px-3 py-1 rounded-lg ${changeBg}`}>
              <span className={`text-lg font-semibold ${changeColor} tabular-nums`}>
                {fmtPercent(data.change_1d_pct)}
              </span>
              <span className={`text-sm ${changeColor} tabular-nums`}>
                ({data.change_1d != null ? (data.change_1d >= 0 ? "+" : "") + fmtNumber(data.change_1d) : "\u2014"})
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Company Overview */}
      {data.description && (
        <Card title="Company Overview">
          <p className="text-sm text-gray-300 leading-relaxed">
            {data.description.length > 500
              ? data.description.slice(0, 500).replace(/\s+\S*$/, "") + "..."
              : data.description
            }
          </p>
          <div className="flex flex-wrap gap-x-6 gap-y-2 mt-4 pt-3 border-t border-white/[0.04]">
            {data.employees != null && data.employees > 0 && (
              <div className="text-xs">
                <span className="text-gray-500">Employees</span>
                <span className="text-gray-300 ml-1.5 font-mono">{data.employees.toLocaleString()}</span>
              </div>
            )}
            {data.sector && (
              <div className="text-xs">
                <span className="text-gray-500">Sector</span>
                <span className="text-gray-300 ml-1.5">{data.sector}</span>
              </div>
            )}
            {data.industry && (
              <div className="text-xs">
                <span className="text-gray-500">Industry</span>
                <span className="text-gray-300 ml-1.5">{data.industry}</span>
              </div>
            )}
            {data.country && (
              <div className="text-xs">
                <span className="text-gray-500">Country</span>
                <span className="text-gray-300 ml-1.5">{data.country}</span>
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card compact>
          <Stat label="Market Cap" value={fmtLargeNumber(data.market_cap)} />
        </Card>
        <Card compact>
          <Stat
            label="P/E Ratio"
            value={data.pe_ratio != null ? fmtNumber(data.pe_ratio, 1) : "\u2014"}
            sub={data.forward_pe != null ? `Fwd: ${fmtNumber(data.forward_pe, 1)}x` : undefined}
            color={data.pe_ratio != null ? (data.pe_ratio < 20 ? "text-emerald-400" : data.pe_ratio > 40 ? "text-amber-400" : "text-white") : undefined}
          />
        </Card>
        <Card compact>
          <Stat
            label="Beta"
            value={data.beta != null ? fmtNumber(data.beta) : "\u2014"}
            color={data.beta != null ? (data.beta > 1.3 ? "text-red-400" : data.beta < 0.8 ? "text-emerald-400" : "text-white") : undefined}
          />
        </Card>
        <Card compact>
          <Stat label="Avg Volume" value={data.avg_volume != null ? data.avg_volume.toLocaleString() : "\u2014"} />
        </Card>
      </div>

      {/* Financial Highlights */}
      {(data.revenue_growth != null || data.operating_margins != null || data.return_on_equity != null || data.dividend_yield != null) && (
        <Card title="Financial Highlights">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {data.total_revenue != null && (
              <Stat
                label="Revenue"
                value={fmtLargeNumber(data.total_revenue)}
                sub={data.revenue_growth != null ? `Growth: ${fmtPercent(data.revenue_growth * 100, 1)}` : undefined}
              />
            )}
            {data.operating_margins != null && (
              <Stat
                label="Operating Margin"
                value={fmtPercent(data.operating_margins * 100, 1)}
                color={data.operating_margins > 0.15 ? "text-emerald-400" : data.operating_margins < 0.05 ? "text-red-400" : "text-white"}
              />
            )}
            {data.profit_margins != null && (
              <Stat
                label="Profit Margin"
                value={fmtPercent(data.profit_margins * 100, 1)}
                color={data.profit_margins > 0.10 ? "text-emerald-400" : data.profit_margins < 0 ? "text-red-400" : "text-white"}
              />
            )}
            {data.return_on_equity != null && (
              <Stat
                label="Return on Equity"
                value={fmtPercent(data.return_on_equity * 100, 1)}
                color={data.return_on_equity > 0.15 ? "text-emerald-400" : data.return_on_equity < 0.05 ? "text-amber-400" : "text-white"}
              />
            )}
            {data.debt_to_equity != null && (
              <Stat
                label="Debt / Equity"
                value={`${fmtNumber(data.debt_to_equity, 0)}%`}
                color={data.debt_to_equity > 100 ? "text-red-400" : data.debt_to_equity < 50 ? "text-emerald-400" : "text-white"}
              />
            )}
            {data.free_cashflow != null && (
              <Stat
                label="Free Cash Flow"
                value={fmtLargeNumber(data.free_cashflow)}
                color={data.free_cashflow > 0 ? "text-emerald-400" : "text-red-400"}
              />
            )}
            {data.dividend_yield != null && data.dividend_yield > 0 && (
              <Stat
                label="Dividend Yield"
                value={fmtPercent(data.dividend_yield * 100, 2)}
                color="text-sky-400"
              />
            )}
          </div>
        </Card>
      )}

      {/* 52W Range */}
      <Card title="52-Week Range">
        <div className="space-y-3">
          {data.low_52w != null && data.high_52w != null && data.price != null && (
            <div className="px-1">
              <div className="flex justify-between text-xs text-gray-500 mb-2">
                <span className="font-mono">${fmtNumber(data.low_52w)}</span>
                <span className="font-mono">${fmtNumber(data.high_52w)}</span>
              </div>
              <div className="relative h-2 bg-surface-overlay rounded-full overflow-hidden">
                <div
                  className="absolute inset-y-0 left-0 rounded-full"
                  style={{
                    width: `${Math.min(100, Math.max(0, ((data.price - data.low_52w) / (data.high_52w - data.low_52w)) * 100))}%`,
                    background: "linear-gradient(90deg, #ef4444, #f59e0b, #22c55e)",
                  }}
                />
                <div
                  className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow-float border-2 border-brand-500"
                  style={{
                    left: `${Math.min(97, Math.max(3, ((data.price - data.low_52w) / (data.high_52w - data.low_52w)) * 100))}%`,
                    transform: "translate(-50%, -50%)",
                  }}
                />
              </div>
              <div className="text-center text-xs text-gray-500 mt-1.5">
                Current: <span className="text-white font-mono">${fmtNumber(data.price)}</span>
              </div>
            </div>
          )}
          <div className="grid grid-cols-2 gap-3 pt-2 border-t border-gray-800/50">
            <Stat label="52W High" value={data.high_52w != null ? `$${fmtNumber(data.high_52w)}` : "\u2014"} size="sm" />
            <Stat label="52W Low" value={data.low_52w != null ? `$${fmtNumber(data.low_52w)}` : "\u2014"} size="sm" />
          </div>
        </div>
      </Card>
    </div>
  );
}
