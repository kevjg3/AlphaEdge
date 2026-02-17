"use client";

import { useState } from "react";
import { Card } from "./Card";
import PriceChart from "./charts/PriceChart";
import MonteCarloChart from "./charts/MonteCarloChart";
import ReturnDistribution from "./charts/ReturnDistribution";
import CorrelationHeatmap from "./charts/CorrelationHeatmap";
import RollingMetrics from "./charts/RollingMetrics";
import BacktestResults from "./charts/BacktestResults";
import PerformanceTable from "./charts/PerformanceTable";
import SeasonalityChart from "./charts/SeasonalityChart";

type SubTab = "overview" | "montecarlo" | "distribution" | "correlation" | "signals" | "seasonality";

const SUB_TABS: { id: SubTab; label: string }[] = [
  { id: "overview", label: "Performance" },
  { id: "montecarlo", label: "Monte Carlo" },
  { id: "distribution", label: "Returns" },
  { id: "correlation", label: "Correlation" },
  { id: "signals", label: "Signals" },
  { id: "seasonality", label: "Seasonality" },
];

export default function QuantPanel({ data }: { data: Record<string, any> | null }) {
  const [subTab, setSubTab] = useState<SubTab>("overview");

  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-gray-500 animate-fade-in">
        <p className="text-sm">Quantitative analysis not available</p>
      </div>
    );
  }

  const perf = data.performance || {};
  const returnAnalysis = data.return_analysis || {};
  const correlation = data.correlation || {};
  const mc = data.monte_carlo || {};
  const signals = data.signal_backtest || {};
  const priceSeries = data.price_series || [];
  const rolling = returnAnalysis.rolling_metrics || {};
  const seasonality = returnAnalysis.seasonality || {};

  return (
    <div className="space-y-5 animate-slide-up">
      {/* Sub-tabs */}
      <div className="flex gap-1 bg-surface-raised/50 p-1 rounded-lg border border-white/[0.04]">
        {SUB_TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setSubTab(tab.id)}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-200 ${
              subTab === tab.id
                ? "bg-brand-500/10 text-brand-400 shadow-glow-sm"
                : "text-gray-500 hover:text-gray-300 hover:bg-white/[0.03]"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* OVERVIEW */}
      {subTab === "overview" && (
        <div className="space-y-5 animate-fade-in">
          {priceSeries.length > 0 && (
            <Card title="Price History">
              <PriceChart data={priceSeries} />
            </Card>
          )}

          <Card title="Risk-Adjusted Performance">
            <PerformanceTable data={perf} />
          </Card>

          {Object.keys(rolling).length > 0 && (
            <Card title="Rolling Metrics">
              <RollingMetrics
                rollingVol21d={rolling.rolling_vol_21d || []}
                rollingVol63d={rolling.rolling_vol_63d || []}
                rollingSharpe63d={rolling.rolling_sharpe_63d || []}
                rollingSkewness63d={rolling.rolling_skewness_63d || []}
                cumulativeReturn={rolling.cumulative_return || []}
              />
            </Card>
          )}
        </div>
      )}

      {/* MONTE CARLO */}
      {subTab === "montecarlo" && (
        <div className="space-y-5 animate-fade-in">
          <Card title="Monte Carlo Simulation (252-Day, 2000 Paths)">
            <MonteCarloChart
              fanChart={mc.fan_chart || {}}
              currentPrice={mc.current_price || 0}
              samplePaths={mc.sample_paths}
            />
          </Card>

          {mc.terminal_stats && (
            <Card title="Terminal Distribution">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <StatBox label="Median Price" value={`$${mc.terminal_stats.median_price}`} />
                <StatBox label="Mean Price" value={`$${mc.terminal_stats.mean_price}`} />
                <StatBox label="5th Percentile" value={`$${mc.terminal_stats.worst_5pct_price}`} color="text-red-400" />
                <StatBox label="95th Percentile" value={`$${mc.terminal_stats.best_5pct_price}`} color="text-emerald-400" />
                <StatBox label="P(Positive)" value={`${(mc.terminal_stats.prob_positive * 100).toFixed(1)}%`} color={mc.terminal_stats.prob_positive > 0.5 ? "text-emerald-400" : "text-red-400"} />
                <StatBox label="P(Loss > 10%)" value={`${(mc.terminal_stats.prob_loss_gt_10pct * 100).toFixed(1)}%`} color="text-red-400" />
                <StatBox label="P(Gain > 10%)" value={`${(mc.terminal_stats.prob_gain_gt_10pct * 100).toFixed(1)}%`} color="text-emerald-400" />
                <StatBox label="P(Gain > 20%)" value={`${(mc.terminal_stats.prob_gain_gt_20pct * 100).toFixed(1)}%`} color="text-emerald-400" />
              </div>
            </Card>
          )}

          {mc.target_probabilities && (
            <Card title="Price Target Probabilities">
              <div className="grid grid-cols-5 gap-2">
                {Object.entries(mc.target_probabilities).map(([target, prob]: [string, any]) => (
                  <div key={target} className="text-center p-3 bg-surface-overlay rounded-xl border border-white/[0.04]">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider">{target}</div>
                    <div className="text-sm font-mono font-bold text-white tabular-nums mt-1">{(prob * 100).toFixed(1)}%</div>
                    <div className="h-1.5 bg-surface rounded-full mt-2 overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${Math.min(prob * 100, 100)}%`,
                          background: "linear-gradient(90deg, #6366f1, #818cf8)",
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          <div className="text-xs text-gray-600 px-1 font-mono">
            GBM simulation | {mc.n_paths} paths | {mc.horizon_days} trading days |
            Daily drift: {((mc.mu_daily || 0) * 100).toFixed(4)}% | Daily vol: {((mc.sigma_daily || 0) * 100).toFixed(2)}%
          </div>
        </div>
      )}

      {/* RETURN DISTRIBUTION */}
      {subTab === "distribution" && (
        <div className="space-y-5 animate-fade-in">
          {returnAnalysis.histogram && returnAnalysis.distribution && (
            <Card title="Daily Return Distribution">
              <ReturnDistribution
                histogram={returnAnalysis.histogram}
                stats={returnAnalysis.distribution}
              />
            </Card>
          )}

          {returnAnalysis.distribution?.percentiles && (
            <Card title="Return Percentiles">
              <div className="grid grid-cols-3 md:grid-cols-9 gap-2">
                {Object.entries(returnAnalysis.distribution.percentiles).map(([p, v]: [string, any]) => (
                  <div key={p} className="text-center p-2.5 bg-surface-overlay rounded-xl border border-white/[0.04]">
                    <div className="text-[10px] text-gray-500 uppercase">{p.toUpperCase()}</div>
                    <div className={`text-sm font-mono font-bold tabular-nums mt-0.5 ${v >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {(v * 100).toFixed(2)}%
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {returnAnalysis.distribution && (
            <Card title="Tail Risk Analysis">
              <div className="grid grid-cols-3 gap-3">
                <StatBox label="Left Tail Mean (5th pct)" value={`${(returnAnalysis.distribution.left_tail_mean * 100).toFixed(3)}%`} color="text-red-400" />
                <StatBox label="Right Tail Mean (95th pct)" value={`${(returnAnalysis.distribution.right_tail_mean * 100).toFixed(3)}%`} color="text-emerald-400" />
                <StatBox label="Tail Ratio (R/L)" value={returnAnalysis.distribution.tail_ratio?.toFixed(3)} color={returnAnalysis.distribution.tail_ratio > 1 ? "text-emerald-400" : "text-red-400"} />
              </div>
            </Card>
          )}
        </div>
      )}

      {/* CORRELATION */}
      {subTab === "correlation" && (
        <div className="space-y-5 animate-fade-in">
          {correlation.correlation_matrix && (
            <Card title="Cross-Asset Correlation Matrix">
              <CorrelationHeatmap
                matrix={correlation.correlation_matrix}
                labels={correlation.benchmark_labels || {}}
              />
            </Card>
          )}

          {correlation.rolling_correlation && correlation.rolling_correlation.length > 0 && (
            <Card title="Rolling 63d Correlation vs S&P 500">
              <RollingMetrics
                rollingVol21d={[]}
                rollingVol63d={[]}
                rollingSharpe63d={[]}
                rollingSkewness63d={[]}
                cumulativeReturn={correlation.rolling_correlation}
              />
            </Card>
          )}

          {correlation.beta_decomposition && (
            <Card title="Beta Decomposition (vs SPY)">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <StatBox label="Market Beta" value={correlation.beta_decomposition.beta?.toFixed(3)} />
                <StatBox label="Alpha (ann.)" value={`${(correlation.beta_decomposition.alpha_annualized * 100).toFixed(2)}%`}
                  color={correlation.beta_decomposition.alpha_annualized > 0 ? "text-emerald-400" : "text-red-400"} />
                <StatBox label="R-squared" value={correlation.beta_decomposition.r_squared?.toFixed(3)} />
                <StatBox label="Residual Vol (ann.)" value={`${(correlation.beta_decomposition.residual_vol_annualized * 100).toFixed(1)}%`} />
                <StatBox label="Tracking Error" value={`${(correlation.beta_decomposition.tracking_error * 100).toFixed(1)}%`} />
              </div>
            </Card>
          )}
        </div>
      )}

      {/* SIGNALS */}
      {subTab === "signals" && (
        <div className="space-y-5 animate-fade-in">
          <Card title="Technical Signal Backtesting">
            <BacktestResults signals={signals} />
          </Card>

          <div className="text-xs text-gray-600 px-1">
            Backtested on full lookback period. Forward returns measured at 1-day, 5-day, and 21-day horizons after signal.
            Win rate = % of signals followed by positive forward return. Profit factor = gross profit / gross loss.
          </div>
        </div>
      )}

      {/* SEASONALITY */}
      {subTab === "seasonality" && (
        <div className="space-y-5 animate-fade-in">
          {seasonality.monthly && (
            <Card title="Monthly Seasonality (Average Returns)">
              <SeasonalityChart monthly={seasonality.monthly} />
            </Card>
          )}

          {seasonality.monthly && (
            <Card title="Monthly Statistics">
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-white/[0.06]">
                      <th className="text-left p-2.5 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Month</th>
                      <th className="text-center p-2.5 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Avg Return</th>
                      <th className="text-center p-2.5 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Win Rate</th>
                      <th className="text-center p-2.5 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Samples</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(seasonality.monthly).map(([m, stats]: [string, any]) => {
                      const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
                      return (
                        <tr key={m} className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors">
                          <td className="p-2.5 text-gray-300 font-medium">{monthNames[parseInt(m) - 1]}</td>
                          <td className={`p-2.5 text-center font-mono tabular-nums ${stats.mean_return >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                            {(stats.mean_return * 100).toFixed(2)}%
                          </td>
                          <td className="p-2.5 text-center font-mono tabular-nums text-gray-300">{(stats.win_rate * 100).toFixed(0)}%</td>
                          <td className="p-2.5 text-center text-gray-500 font-mono tabular-nums">{stats.n_samples}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}

function StatBox({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="p-3.5 bg-surface-overlay rounded-xl border border-white/[0.04]">
      <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-sm font-bold font-mono tabular-nums ${color || "text-white"}`}>{value}</div>
    </div>
  );
}
