"use client";

import { Card, Badge } from "../Card";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  Cell,
} from "recharts";

// ── Types ──────────────────────────────────────────────────────────────

interface SeriesData {
  date: string;
  value: number;
}

interface MeanReversionData {
  hurst_exponent: number;
  hurst_regime: string;
  half_life_days: number | null;
  ou_theta: number | null;
  ou_mu: number | null;
  confidence: string;
}

interface GarchData {
  omega: number;
  alpha: number;
  beta: number;
  persistence: number;
  long_run_vol_annual: number;
  converged: boolean;
  forecast_5d_vol: number;
  forecast_10d_vol: number;
  forecast_21d_vol: number;
  realized_vol_21d: number;
  vol_ratio: number;
  conditional_vol_series: SeriesData[];
}

interface MomentumData {
  return_1m: number | null;
  return_3m: number | null;
  return_6m: number | null;
  return_12m: number | null;
  momentum_zscore_1m: number;
  momentum_zscore_3m: number;
  momentum_zscore_6m: number;
  momentum_zscore_12m: number;
  composite_momentum_zscore: number;
  acceleration_1m: number;
  acceleration_3m: number;
  momentum_regime: string;
  omega_ratios: Record<string, number>;
}

interface AlphaIntelData {
  liquidity: {
    amihud_illiquidity: number;
    avg_daily_volume: number;
    volume_trend_pct: number;
    relative_volume: number;
    bid_ask_spread_proxy: number;
    liquidity_score: string;
  };
  drawdown_duration: {
    max_drawdown_pct: number;
    max_drawdown_duration_days: number;
    avg_recovery_time_days: number;
    current_drawdown_pct: number;
    current_drawdown_days: number;
    n_drawdowns_gt_5pct: number;
    drawdown_periods: {
      start: string;
      trough: string;
      recovery: string | null;
      depth: number;
      duration_days: number;
      recovery_days: number | null;
    }[];
  };
  feature_importance: {
    features: { name: string; importance: number; direction: string }[];
    model_r2: number | null;
  };
}

interface Props {
  meanReversion: MeanReversionData;
  garchForecast: GarchData;
  momentum: MomentumData;
  alphaIntelligence: AlphaIntelData;
}

// ── Helpers ────────────────────────────────────────────────────────────

function StatBox({ label, value, color, sub }: { label: string; value: string; color?: string; sub?: string }) {
  return (
    <div className="p-3.5 bg-surface-overlay rounded-xl border border-white/[0.04]">
      <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-sm font-bold font-mono tabular-nums ${color || "text-white"}`}>{value}</div>
      {sub && <div className="text-[10px] text-gray-600 mt-0.5">{sub}</div>}
    </div>
  );
}

function pctColor(v: number): string {
  return v >= 0 ? "text-emerald-400" : "text-red-400";
}

function fmtPct(v: number | null | undefined, decimals = 1): string {
  if (v == null) return "N/A";
  return `${(v * 100).toFixed(decimals)}%`;
}

function fmtNum(v: number | null | undefined, decimals = 2): string {
  if (v == null) return "N/A";
  return v.toFixed(decimals);
}

function regimeBadgeVariant(regime: string): "positive" | "negative" | "neutral" | "warning" | "info" {
  if (regime === "mean_reverting") return "info";
  if (regime === "trending") return "positive";
  if (regime === "accelerating") return "positive";
  if (regime === "decelerating") return "negative";
  return "neutral";
}

function regimeLabel(regime: string): string {
  return regime
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

// ── Component ──────────────────────────────────────────────────────────

export default function AlphaSignalsDashboard({
  meanReversion,
  garchForecast,
  momentum,
  alphaIntelligence,
}: Props) {
  const mr = meanReversion || {} as MeanReversionData;
  const garch = garchForecast || {} as GarchData;
  const mom = momentum || {} as MomentumData;
  const ai = alphaIntelligence || {} as AlphaIntelData;
  const liq = ai.liquidity || {} as AlphaIntelData["liquidity"];
  const dd = ai.drawdown_duration || {} as AlphaIntelData["drawdown_duration"];
  const fi = ai.feature_importance || {} as AlphaIntelData["feature_importance"];

  return (
    <div className="space-y-5 animate-fade-in">
      {/* ── SECTION 1: Regime Dashboard ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Mean Reversion */}
        <Card title="Mean Reversion Detection">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] text-gray-500 uppercase tracking-wider">Hurst Exponent</div>
                <div className={`text-2xl font-bold font-mono tabular-nums ${
                  mr.hurst_exponent < 0.45
                    ? "text-sky-400"
                    : mr.hurst_exponent > 0.55
                    ? "text-emerald-400"
                    : "text-gray-300"
                }`}>
                  {fmtNum(mr.hurst_exponent, 4)}
                </div>
              </div>
              {mr.hurst_regime && (
                <Badge
                  text={regimeLabel(mr.hurst_regime)}
                  variant={regimeBadgeVariant(mr.hurst_regime)}
                  dot
                />
              )}
            </div>

            <div className="grid grid-cols-2 gap-2">
              <StatBox
                label="Half-Life"
                value={mr.half_life_days != null ? `${mr.half_life_days.toFixed(0)} days` : "N/A"}
                sub={mr.half_life_days != null ? "Mean reversion speed" : "Not mean-reverting"}
              />
              <StatBox
                label="Confidence"
                value={mr.confidence ? regimeLabel(mr.confidence) : "Low"}
                color={mr.confidence === "high" ? "text-emerald-400" : mr.confidence === "medium" ? "text-amber-400" : "text-gray-400"}
              />
            </div>

            <div className="text-[10px] text-gray-600 leading-relaxed">
              {mr.hurst_exponent < 0.45
                ? "H < 0.45 indicates mean-reverting behavior — price tends to revert to its mean. Consider mean-reversion strategies."
                : mr.hurst_exponent > 0.55
                ? "H > 0.55 indicates trending behavior — momentum strategies may be effective."
                : "H near 0.5 indicates a random walk — no strong trend or reversion signal."}
            </div>
          </div>
        </Card>

        {/* GARCH Volatility Forecast */}
        <Card title="Volatility Forecast (GARCH)">
          <div className="space-y-3">
            {!garch.converged && garch.forecast_5d_vol > 0 && (
              <div className="text-[10px] text-amber-400 bg-amber-500/10 rounded-lg px-2 py-1 border border-amber-500/20">
                GARCH did not converge — showing EWMA fallback
              </div>
            )}
            <div className="grid grid-cols-3 gap-2">
              <StatBox label="5-Day Vol" value={fmtPct(garch.forecast_5d_vol)} />
              <StatBox label="10-Day Vol" value={fmtPct(garch.forecast_10d_vol)} />
              <StatBox label="21-Day Vol" value={fmtPct(garch.forecast_21d_vol)} />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <StatBox
                label="Vol Ratio"
                value={fmtNum(garch.vol_ratio)}
                color={garch.vol_ratio > 1.1 ? "text-red-400" : garch.vol_ratio < 0.9 ? "text-emerald-400" : "text-gray-300"}
                sub={garch.vol_ratio > 1.1 ? "Vol expansion expected" : garch.vol_ratio < 0.9 ? "Vol compression expected" : "Stable"}
              />
              <StatBox
                label="Persistence"
                value={fmtNum(garch.persistence, 4)}
                sub={garch.persistence > 0.95 ? "Highly persistent" : garch.persistence > 0.85 ? "Persistent" : "Low persistence"}
              />
            </div>

            {/* Conditional Vol Chart */}
            {garch.conditional_vol_series && garch.conditional_vol_series.length > 5 && (
              <div className="h-[120px] w-full">
                <ResponsiveContainer>
                  <LineChart data={garch.conditional_vol_series} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="date" tick={false} axisLine={{ stroke: "#334155" }} />
                    <YAxis
                      tick={{ fill: "#64748b", fontSize: 9 }}
                      tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                      width={40}
                    />
                    <Tooltip
                      contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, fontSize: 10 }}
                      labelStyle={{ color: "#94a3b8" }}
                      formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, "Ann. Vol"]}
                    />
                    {garch.long_run_vol_annual > 0 && (
                      <ReferenceLine y={garch.long_run_vol_annual} stroke="#475569" strokeDasharray="4 4" strokeWidth={1} />
                    )}
                    <Line type="monotone" dataKey="value" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </Card>

        {/* Momentum */}
        <Card title="Momentum Scoring">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] text-gray-500 uppercase tracking-wider">Composite Z-Score</div>
                <div className={`text-2xl font-bold font-mono tabular-nums ${
                  mom.composite_momentum_zscore > 0.5 ? "text-emerald-400" :
                  mom.composite_momentum_zscore < -0.5 ? "text-red-400" : "text-gray-300"
                }`}>
                  {fmtNum(mom.composite_momentum_zscore, 3)}
                </div>
              </div>
              {mom.momentum_regime && (
                <Badge
                  text={regimeLabel(mom.momentum_regime)}
                  variant={regimeBadgeVariant(mom.momentum_regime)}
                  dot
                />
              )}
            </div>

            {/* Timeframe returns */}
            <div className="grid grid-cols-4 gap-1.5">
              {[
                { label: "1M", val: mom.return_1m },
                { label: "3M", val: mom.return_3m },
                { label: "6M", val: mom.return_6m },
                { label: "12M", val: mom.return_12m },
              ].map(({ label, val }) => (
                <div key={label} className="text-center p-2 bg-surface-overlay rounded-lg border border-white/[0.04]">
                  <div className="text-[9px] text-gray-500">{label}</div>
                  <div className={`text-xs font-mono font-bold tabular-nums mt-0.5 ${val != null ? pctColor(val) : "text-gray-500"}`}>
                    {fmtPct(val)}
                  </div>
                </div>
              ))}
            </div>

            {/* Z-scores */}
            <div className="grid grid-cols-4 gap-1.5">
              {[
                { label: "Z-1M", val: mom.momentum_zscore_1m },
                { label: "Z-3M", val: mom.momentum_zscore_3m },
                { label: "Z-6M", val: mom.momentum_zscore_6m },
                { label: "Z-12M", val: mom.momentum_zscore_12m },
              ].map(({ label, val }) => (
                <div key={label} className="text-center p-2 bg-surface-overlay rounded-lg border border-white/[0.04]">
                  <div className="text-[9px] text-gray-500">{label}</div>
                  <div className={`text-xs font-mono font-bold tabular-nums mt-0.5 ${pctColor(val)}`}>
                    {fmtNum(val, 2)}
                  </div>
                </div>
              ))}
            </div>

            <div className="text-[10px] text-gray-600 leading-relaxed">
              {mom.composite_momentum_zscore > 1
                ? "Strong positive momentum — returns are well above historical norms."
                : mom.composite_momentum_zscore < -1
                ? "Strong negative momentum — returns are well below historical norms."
                : mom.momentum_regime === "accelerating"
                ? "Momentum is accelerating — trend is strengthening."
                : mom.momentum_regime === "decelerating"
                ? "Momentum is decelerating — trend may be weakening."
                : "Momentum is near historical averages."}
            </div>
          </div>
        </Card>
      </div>

      {/* ── SECTION 2: Risk Intelligence ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Liquidity */}
        <Card title="Liquidity Analysis">
          <div className="space-y-3">
            <div className="flex items-center gap-2 mb-2">
              <Badge
                text={liq.liquidity_score ? regimeLabel(liq.liquidity_score) + " Liquidity" : "Unknown"}
                variant={liq.liquidity_score === "high" ? "positive" : liq.liquidity_score === "low" ? "negative" : "neutral"}
                dot
              />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <StatBox
                label="Avg Daily Volume"
                value={liq.avg_daily_volume ? (liq.avg_daily_volume / 1e6).toFixed(1) + "M" : "N/A"}
              />
              <StatBox
                label="Relative Volume"
                value={fmtNum(liq.relative_volume)}
                color={liq.relative_volume > 1.5 ? "text-amber-400" : liq.relative_volume < 0.5 ? "text-red-400" : "text-gray-300"}
                sub={liq.relative_volume > 1.5 ? "Above average" : liq.relative_volume < 0.5 ? "Below average" : "Normal range"}
              />
              <StatBox
                label="Volume Trend"
                value={fmtPct(liq.volume_trend_pct)}
                color={pctColor(liq.volume_trend_pct || 0)}
                sub="21d vs 63d avg"
              />
              <StatBox
                label="Spread Proxy"
                value={liq.bid_ask_spread_proxy ? (liq.bid_ask_spread_proxy * 10000).toFixed(1) + " bps" : "N/A"}
                sub="OHLC range-based"
              />
            </div>
            <StatBox
              label="Amihud Illiquidity"
              value={liq.amihud_illiquidity ? liq.amihud_illiquidity.toExponential(2) : "N/A"}
              sub="Lower = more liquid"
            />
          </div>
        </Card>

        {/* Drawdown Duration */}
        <Card title="Drawdown Duration Analysis">
          <div className="space-y-3">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              <StatBox
                label="Max Drawdown"
                value={fmtPct(dd.max_drawdown_pct)}
                color="text-red-400"
              />
              <StatBox
                label="Max DD Duration"
                value={dd.max_drawdown_duration_days ? `${dd.max_drawdown_duration_days} days` : "N/A"}
              />
              <StatBox
                label="Avg Recovery"
                value={dd.avg_recovery_time_days ? `${dd.avg_recovery_time_days.toFixed(0)} days` : "N/A"}
              />
              <StatBox
                label="Current DD"
                value={fmtPct(dd.current_drawdown_pct)}
                color={dd.current_drawdown_pct < -0.05 ? "text-red-400" : dd.current_drawdown_pct < -0.01 ? "text-amber-400" : "text-emerald-400"}
              />
              <StatBox
                label="Days in DD"
                value={dd.current_drawdown_days ? `${dd.current_drawdown_days}` : "0"}
                color={dd.current_drawdown_days > 60 ? "text-red-400" : "text-gray-300"}
              />
              <StatBox
                label="DDs > 5%"
                value={`${dd.n_drawdowns_gt_5pct || 0}`}
              />
            </div>

            {/* Top drawdown periods table */}
            {dd.drawdown_periods && dd.drawdown_periods.length > 0 && (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-white/[0.06]">
                      <th className="text-left p-2 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Start</th>
                      <th className="text-center p-2 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Depth</th>
                      <th className="text-center p-2 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Duration</th>
                      <th className="text-center p-2 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Recovery</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dd.drawdown_periods.map((p, i) => (
                      <tr key={i} className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors">
                        <td className="p-2 text-gray-300 font-mono">{p.start}</td>
                        <td className="p-2 text-center font-mono text-red-400">{(p.depth * 100).toFixed(1)}%</td>
                        <td className="p-2 text-center font-mono text-gray-300">{p.duration_days}d</td>
                        <td className="p-2 text-center font-mono text-gray-400">
                          {p.recovery_days != null ? `${p.recovery_days}d` : "Ongoing"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* ── SECTION 3: Omega Ratio + Feature Importance ── */}

      {/* Omega Ratio */}
      {mom.omega_ratios && Object.keys(mom.omega_ratios).length > 0 && (
        <Card title="Omega Ratio (Gain/Loss Probability)">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {Object.entries(mom.omega_ratios).map(([threshold, value]) => {
              const isInfinite = !isFinite(value);
              const displayVal = isInfinite ? "Inf" : value.toFixed(3);
              const pct = isInfinite ? 100 : Math.min(value / 2 * 100, 100);

              return (
                <div key={threshold} className="text-center p-3 bg-surface-overlay rounded-xl border border-white/[0.04]">
                  <div className="text-[10px] text-gray-500 uppercase tracking-wider">{threshold}</div>
                  <div className={`text-sm font-mono font-bold tabular-nums mt-1 ${
                    value > 1.2 ? "text-emerald-400" : value < 0.8 ? "text-red-400" : "text-white"
                  }`}>
                    {displayVal}
                  </div>
                  <div className="h-1.5 bg-surface rounded-full mt-2 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${pct}%`,
                        background: value > 1 ? "linear-gradient(90deg, #10b981, #34d399)" : "linear-gradient(90deg, #ef4444, #f87171)",
                      }}
                    />
                  </div>
                  <div className="text-[9px] text-gray-600 mt-1">
                    {value > 1 ? "Gain-biased" : value < 1 ? "Loss-biased" : "Balanced"}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="text-[10px] text-gray-600 mt-2">
            Omega &gt; 1 means probability-weighted gains exceed losses at that threshold. Higher is better.
          </div>
        </Card>
      )}

      {/* Feature Importance */}
      {fi.features && fi.features.length > 0 && (
        <Card title="Predictive Feature Importance (XGBoost)">
          <div className="space-y-3">
            <div className="flex items-center gap-4 mb-1">
              {fi.model_r2 != null && (
                <StatBox
                  label="Model R-squared"
                  value={fmtNum(fi.model_r2, 4)}
                  sub="Test set (21-day forward returns)"
                  color={fi.model_r2 > 0.05 ? "text-emerald-400" : fi.model_r2 > 0 ? "text-amber-400" : "text-red-400"}
                />
              )}
            </div>

            <div className="h-[220px] w-full">
              <ResponsiveContainer>
                <BarChart
                  data={fi.features.slice(0, 8)}
                  layout="vertical"
                  margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fill: "#64748b", fontSize: 10 }}
                    tickFormatter={(v: number) => v.toFixed(4)}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={{ fill: "#94a3b8", fontSize: 10 }}
                    width={110}
                  />
                  <Tooltip
                    contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, fontSize: 11 }}
                    formatter={(v: number, _: string, item: { payload?: { direction?: string } }) => [
                      `${v.toFixed(6)} (${item?.payload?.direction || ""})`,
                      "Importance",
                    ]}
                  />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {fi.features.slice(0, 8).map((f, i) => (
                      <Cell
                        key={i}
                        fill={f.direction === "positive" ? "#10b981" : "#ef4444"}
                        fillOpacity={0.7}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="flex items-center gap-4 text-[10px] text-gray-500">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-sm bg-emerald-500/70" />
                Positive predictor
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-sm bg-red-500/70" />
                Negative predictor
              </span>
            </div>

            <div className="text-[10px] text-gray-600 leading-relaxed">
              Feature importance measured via permutation importance on held-out test data.
              Shows which factors most predict 21-day forward returns. Low R² is expected — markets are hard to predict.
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}
