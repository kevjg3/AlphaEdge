"use client";

import { Card, Stat, Badge } from "./Card";
import { fmtNumber, fmtPercent } from "@/lib/format";

function HorizonCard({ label, data }: { label: string; data: any }) {
  const ensemble = data?.ensemble_prediction || {};
  const direction = ensemble.direction || "flat";
  const ret = ensemble.predicted_return;
  const decomposition = data?.decomposition || {};

  return (
    <Card className="space-y-3">
      <div className="flex justify-between items-center">
        <span className="text-sm font-bold text-white">{label}</span>
        <Badge
          text={direction}
          variant={direction === "up" ? "positive" : direction === "down" ? "negative" : "neutral"}
          dot
        />
      </div>
      <div className={`text-3xl font-bold font-mono tabular-nums ${ret > 0 ? "text-emerald-400" : ret < 0 ? "text-red-400" : "text-gray-300"}`}>
        {fmtPercent(ret)}
      </div>
      <div className="text-xs text-gray-500">
        Confidence: <span className="text-gray-400 font-mono tabular-nums">{fmtNumber((ensemble.confidence || 0) * 100, 0)}%</span>
      </div>

      {/* Decomposition: each model's weighted contribution */}
      {Object.keys(decomposition).length > 0 && (
        <div className="mt-1 pt-2 border-t border-white/[0.06]">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1.5">Contribution Breakdown</div>
          {Object.entries(decomposition).map(([name, d]: [string, any]) => {
            const wRet = d.weighted_return || 0;
            const maxBar = Math.max(
              ...Object.values(decomposition).map((v: any) => Math.abs(v.weighted_return || 0)),
              0.01
            );
            const barWidth = Math.min((Math.abs(wRet) / maxBar) * 100, 100);

            return (
              <div key={name} className="flex items-center gap-2 text-xs py-0.5">
                <span className="text-gray-400 w-20 truncate">{name}</span>
                <div className="flex-1 h-1.5 bg-surface-overlay rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${wRet >= 0 ? "bg-emerald-500" : "bg-red-500"}`}
                    style={{ width: `${barWidth}%` }}
                  />
                </div>
                <span className={`font-mono tabular-nums w-14 text-right ${wRet >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {wRet >= 0 ? "+" : ""}{wRet.toFixed(2)}%
                </span>
              </div>
            );
          })}
        </div>
      )}

      {/* Model weights */}
      {data?.model_weights && Object.keys(data.model_weights).length > 0 && (
        <div className="mt-1 pt-2 border-t border-white/[0.06]">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1.5">Model Weights</div>
          {Object.entries(data.model_weights).map(([name, w]: [string, any]) => (
            <div key={name} className="flex justify-between items-center text-xs py-0.5">
              <span className="text-gray-400">{name}</span>
              <div className="flex items-center gap-2">
                <div className="w-12 h-1 bg-surface-overlay rounded-full overflow-hidden">
                  <div className="h-full bg-brand-500 rounded-full" style={{ width: `${w * 100}%` }} />
                </div>
                <span className="text-gray-300 font-mono tabular-nums w-8 text-right">{fmtNumber(w * 100, 0)}%</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Individual model predictions */}
      {data?.individual_forecasts && (
        <div className="mt-1 pt-2 border-t border-white/[0.06]">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1.5">Individual Models</div>
          {Object.entries(data.individual_forecasts).map(([name, res]: [string, any]) => (
            <div key={name} className="flex justify-between text-xs py-0.5">
              <span className="text-gray-400">{name}</span>
              <span className={`font-mono tabular-nums ${res.predicted_return > 0 ? "text-emerald-400" : res.predicted_return < 0 ? "text-red-400" : "text-gray-300"}`}>
                {fmtPercent(res.predicted_return)}
              </span>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

export default function ForecastPanel({ data }: { data: Record<string, any> | null }) {
  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-gray-500 animate-fade-in">
        <p className="text-sm">Forecast not available</p>
      </div>
    );
  }

  const forecasts = data.forecasts || {};
  const horizons = ["1D", "1W", "1M", "3M", "12M"];
  const accuracy = data.accuracy || {};
  const accHorizons = accuracy.horizons || {};

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Overview */}
      <Card>
        <div className="flex flex-wrap items-center gap-6">
          <Stat
            label="Overall Direction"
            value={data.overall_direction || "\u2014"}
            color={data.overall_direction === "up" ? "text-emerald-400" : data.overall_direction === "down" ? "text-red-400" : "text-gray-300"}
          />
          <Stat label="Short-term" value={data.short_term_outlook || "\u2014"} />
          <Stat label="Long-term" value={data.long_term_outlook || "\u2014"} />
          <Stat label="Model Agreement" value={fmtNumber((data.model_agreement || 0) * 100, 0) + "%"} />
        </div>
      </Card>

      {/* Forecast Accuracy Tracking */}
      {accuracy.reliability_grade && (
        <Card title="Forecast Accuracy (Walk-Forward Backtest)">
          <div className="flex items-center gap-4 mb-4">
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">Reliability Grade:</span>
              <Badge
                text={`Grade ${accuracy.reliability_grade}`}
                variant={
                  accuracy.reliability_grade === "A" ? "positive" :
                  accuracy.reliability_grade === "B" ? "info" :
                  accuracy.reliability_grade === "C" ? "warning" : "negative"
                }
                dot
              />
            </div>
            <Stat
              label="Overall Directional Accuracy"
              value={`${fmtNumber((accuracy.overall_directional_accuracy || 0) * 100, 0)}%`}
              size="sm"
              color={accuracy.overall_directional_accuracy > 0.6 ? "text-emerald-400" : accuracy.overall_directional_accuracy > 0.5 ? "text-amber-400" : "text-red-400"}
            />
          </div>
          <div className="grid grid-cols-3 gap-3">
            {(["1W", "1M", "3M"] as const).map((h) => {
              const acc = accHorizons[h];
              if (!acc) return null;
              return (
                <div key={h} className="p-3 bg-surface-overlay rounded-xl border border-white/[0.04]">
                  <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">{h} Horizon</div>
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Direction Accuracy</span>
                      <span className={`font-mono tabular-nums ${acc.directional_accuracy > 0.6 ? "text-emerald-400" : acc.directional_accuracy > 0.5 ? "text-amber-400" : "text-red-400"}`}>
                        {fmtNumber(acc.directional_accuracy * 100, 0)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Mean Abs Error</span>
                      <span className="font-mono tabular-nums text-gray-300">{fmtNumber(acc.mean_abs_error_pct * 100, 1)}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">CI Coverage</span>
                      <span className="font-mono tabular-nums text-gray-300">{fmtNumber(acc.ci_coverage_rate * 100, 0)}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Evaluations</span>
                      <span className="font-mono tabular-nums text-gray-500">{acc.n_evaluations}</span>
                    </div>
                    {acc.recent_accuracy_5 != null && (
                      <div className="flex justify-between text-xs">
                        <span className="text-gray-400">Recent (last 5)</span>
                        <span className={`font-mono tabular-nums ${acc.recent_accuracy_5 > 0.6 ? "text-emerald-400" : "text-amber-400"}`}>
                          {fmtNumber(acc.recent_accuracy_5 * 100, 0)}%
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="text-[10px] text-gray-600 mt-2">
            Accuracy measured via walk-forward backtesting on historical data. Direction accuracy = % of correct up/down calls.
          </div>
        </Card>
      )}

      {/* Horizon Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-3">
        {horizons.map((h) => (
          forecasts[h] ? <HorizonCard key={h} label={h} data={forecasts[h]} /> : null
        ))}
      </div>
    </div>
  );
}
