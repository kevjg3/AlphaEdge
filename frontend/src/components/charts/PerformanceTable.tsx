"use client";

interface Props {
  data: {
    sharpe_ratio: number;
    sortino_ratio: number;
    calmar_ratio: number;
    information_ratio: number | null;
    annualized_vol: number;
    downside_vol: number;
    total_return: number;
    annualized_return: number;
    daily_win_rate: number;
    weekly_win_rate: number;
    monthly_win_rate: number;
    best_day: number;
    worst_day: number;
    best_month: number;
    worst_month: number;
    avg_positive_day: number;
    avg_negative_day: number;
    profit_factor: number;
    skewness: number;
    kurtosis: number;
    period_returns: Record<string, number>;
  };
}

function Metric({ label, value, color, suffix }: { label: string; value: string; color?: string; suffix?: string }) {
  return (
    <div className="p-3 bg-surface-overlay rounded-xl border border-white/[0.04]">
      <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-lg font-bold font-mono tabular-nums ${color || "text-white"}`}>
        {value}
        {suffix && <span className="text-xs text-gray-500 ml-0.5">{suffix}</span>}
      </div>
    </div>
  );
}

function fmtRatio(v: number | null): string {
  if (v == null) return "\u2014";
  return v.toFixed(2);
}

function fmtPct(v: number): string {
  return `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`;
}

function ratioColor(v: number | null, good: number, bad: number): string {
  if (v == null) return "text-gray-300";
  if (v >= good) return "text-emerald-400";
  if (v <= bad) return "text-red-400";
  return "text-amber-400";
}

export default function PerformanceTable({ data }: Props) {
  if (!data) return null;

  return (
    <div className="space-y-3">
      {/* Risk-Adjusted Ratios */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <Metric label="Sharpe Ratio" value={fmtRatio(data.sharpe_ratio)} color={ratioColor(data.sharpe_ratio, 1.0, 0)} />
        <Metric label="Sortino Ratio" value={fmtRatio(data.sortino_ratio)} color={ratioColor(data.sortino_ratio, 1.5, 0)} />
        <Metric label="Calmar Ratio" value={fmtRatio(data.calmar_ratio)} color={ratioColor(data.calmar_ratio, 1.0, 0.3)} />
        <Metric label="Info Ratio" value={fmtRatio(data.information_ratio)} color={ratioColor(data.information_ratio, 0.5, -0.5)} />
      </div>

      {/* Returns */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <Metric label="Total Return" value={fmtPct(data.total_return)} color={data.total_return >= 0 ? "text-emerald-400" : "text-red-400"} />
        <Metric label="Annualized Return" value={fmtPct(data.annualized_return)} color={data.annualized_return >= 0 ? "text-emerald-400" : "text-red-400"} />
        <Metric label="Ann. Volatility" value={`${(data.annualized_vol * 100).toFixed(1)}%`} color="text-amber-400" />
        <Metric label="Downside Vol" value={`${(data.downside_vol * 100).toFixed(3)}%`} color="text-amber-400" />
      </div>

      {/* Period Returns */}
      {data.period_returns && Object.keys(data.period_returns).length > 0 && (
        <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
          {Object.entries(data.period_returns).map(([period, ret]) => (
            <Metric
              key={period}
              label={period}
              value={fmtPct(ret)}
              color={ret >= 0 ? "text-emerald-400" : "text-red-400"}
            />
          ))}
        </div>
      )}

      {/* Win Rates & Extremes */}
      <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
        <Metric label="Daily Win Rate" value={`${(data.daily_win_rate * 100).toFixed(0)}%`} color={data.daily_win_rate > 0.52 ? "text-emerald-400" : "text-gray-300"} />
        <Metric label="Weekly Win Rate" value={`${(data.weekly_win_rate * 100).toFixed(0)}%`} />
        <Metric label="Monthly Win Rate" value={`${(data.monthly_win_rate * 100).toFixed(0)}%`} />
        <Metric label="Best Day" value={fmtPct(data.best_day)} color="text-emerald-400" />
        <Metric label="Worst Day" value={fmtPct(data.worst_day)} color="text-red-400" />
        <Metric label="Profit Factor" value={data.profit_factor.toFixed(2)} color={ratioColor(data.profit_factor, 1.5, 1)} />
      </div>
    </div>
  );
}
