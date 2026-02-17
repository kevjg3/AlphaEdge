"use client";

import { Card, Stat, Badge } from "./Card";
import { fmtNumber, fmtPercent } from "@/lib/format";

export default function RiskPanel({ data }: { data: Record<string, any> | null }) {
  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-gray-500 animate-fade-in">
        <p className="text-sm">Risk analysis not available</p>
      </div>
    );
  }

  const varData = data.var || {};
  const dd = data.drawdown || {};
  const scenarios = data.scenarios || [];
  const events = data.upcoming_events || [];

  return (
    <div className="space-y-6 animate-slide-up">
      {/* VaR */}
      {varData.parametric != null && (
        <Card title="Value at Risk (95%, 1-day)">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Stat label="Parametric VaR" value={fmtPercent(varData.parametric_var_pct || varData.parametric_pct)} color="text-red-400" />
            <Stat label="Historical VaR" value={fmtPercent(varData.historical_var_pct || varData.historical_pct)} color="text-red-400" />
            <Stat label="Monte Carlo VaR" value={fmtPercent(varData.monte_carlo_var_pct || varData.monte_carlo_pct)} color="text-red-400" />
            <Stat label="CVaR (ES)" value={fmtPercent(varData.cvar_pct)} color="text-red-400" />
          </div>
          <div className="mt-3 pt-3 border-t border-white/[0.06] text-xs text-gray-500 font-mono">
            Dollar VaR: Parametric ${fmtNumber(varData.parametric, 0)} | Historical ${fmtNumber(varData.historical, 0)} | MC ${fmtNumber(varData.monte_carlo, 0)}
          </div>
        </Card>
      )}

      {/* Drawdown */}
      {dd.stats && (
        <Card title="Drawdown Analysis">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <Stat label="Max Drawdown" value={fmtPercent(dd.stats.max_drawdown_pct)} color="text-red-400" />
            <Stat label="Avg Drawdown" value={fmtPercent(dd.stats.avg_drawdown_pct)} />
            <Stat label="Current DD" value={fmtPercent(dd.stats.current_drawdown_pct)} color={dd.stats.current_drawdown_pct < -5 ? "text-red-400" : "text-gray-300"} />
            <Stat label="Time Underwater" value={`${fmtNumber(dd.stats.pct_time_underwater, 0)}%`} />
          </div>

          {/* Top drawdown events */}
          {dd.events && dd.events.length > 0 && (
            <div className="border-t border-white/[0.06] pt-3">
              <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">Worst Drawdowns</div>
              {dd.events.slice(0, 5).map((e: any, i: number) => (
                <div key={i} className="data-row text-sm">
                  <span className="text-gray-400 font-mono text-xs">{e.start_date} \u2192 {e.trough_date || "ongoing"}</span>
                  <span className="text-red-400 font-mono tabular-nums">{fmtPercent(e.drawdown_pct)}</span>
                  <span className="text-gray-500 text-xs">{e.duration_days}d</span>
                  <span className="text-gray-600 text-xs">{e.recovery_days != null ? `${e.recovery_days}d rec.` : "no rec."}</span>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}

      {/* Stress Tests */}
      {scenarios.length > 0 && (
        <Card title="Historical Stress Tests">
          {scenarios.map((s: any, i: number) => (
            <div key={i} className="data-row">
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-300">{s.name || s.scenario}</span>
                {s.period && <span className="text-[10px] text-gray-600">{s.period}</span>}
              </div>
              <span className={`font-mono font-bold tabular-nums ${(s.ticker_return || s.return_pct || 0) < 0 ? "text-red-400" : "text-emerald-400"}`}>
                {fmtPercent(s.ticker_return || s.return_pct)}
              </span>
            </div>
          ))}
        </Card>
      )}

      {/* Upcoming Events */}
      {events.length > 0 && (
        <Card title="Upcoming Events">
          {events.map((e: any, i: number) => (
            <div key={i} className="data-row">
              <div className="flex items-center gap-2">
                <Badge text={e.type || "event"} variant="info" size="sm" />
                <span className="text-sm text-gray-300">{e.description || e.name || "\u2014"}</span>
              </div>
              <span className="text-xs text-gray-500 font-mono">{e.date || "\u2014"}</span>
            </div>
          ))}
        </Card>
      )}
    </div>
  );
}
