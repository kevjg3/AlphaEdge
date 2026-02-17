"use client";

import { Badge } from "../Card";

interface SignalResult {
  signal_name: string;
  total_signals: number;
  win_rate: number;
  avg_return_1d: number;
  avg_return_5d: number;
  avg_return_21d: number;
  best_return_5d: number;
  worst_return_5d: number;
  profit_factor: number;
  signal_dates: { date: string; price: number }[];
}

interface Props {
  signals: Record<string, SignalResult>;
}

function fmtPct(v: number): string {
  return `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`;
}

export default function BacktestResults({ signals }: Props) {
  if (!signals || Object.keys(signals).length === 0) {
    return <div className="text-gray-500 text-sm py-4 text-center">No signals detected in the lookback period.</div>;
  }

  const entries = Object.entries(signals).sort(
    ([, a], [, b]) => b.avg_return_5d - a.avg_return_5d
  );

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-white/[0.06]">
              <th className="text-left p-2.5 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Signal</th>
              <th className="text-center p-2.5 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Count</th>
              <th className="text-center p-2.5 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Win Rate</th>
              <th className="text-center p-2.5 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Avg 1D</th>
              <th className="text-center p-2.5 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Avg 5D</th>
              <th className="text-center p-2.5 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Avg 21D</th>
              <th className="text-center p-2.5 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Best 5D</th>
              <th className="text-center p-2.5 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Worst 5D</th>
              <th className="text-center p-2.5 text-gray-500 font-medium uppercase tracking-wider text-[10px]">Profit Fac.</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([key, sig]) => (
              <tr key={key} className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors">
                <td className="p-2.5">
                  <span className="text-gray-200 font-medium">{sig.signal_name}</span>
                </td>
                <td className="p-2.5 text-center text-gray-400 font-mono tabular-nums">{sig.total_signals}</td>
                <td className="p-2.5 text-center">
                  <span className={`font-mono tabular-nums ${sig.win_rate >= 0.55 ? "text-emerald-400" : sig.win_rate < 0.45 ? "text-red-400" : "text-gray-300"}`}>
                    {(sig.win_rate * 100).toFixed(0)}%
                  </span>
                </td>
                <td className={`p-2.5 text-center font-mono tabular-nums ${sig.avg_return_1d >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {fmtPct(sig.avg_return_1d)}
                </td>
                <td className={`p-2.5 text-center font-mono tabular-nums ${sig.avg_return_5d >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {fmtPct(sig.avg_return_5d)}
                </td>
                <td className={`p-2.5 text-center font-mono tabular-nums ${sig.avg_return_21d >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {fmtPct(sig.avg_return_21d)}
                </td>
                <td className="p-2.5 text-center font-mono tabular-nums text-emerald-400">{fmtPct(sig.best_return_5d)}</td>
                <td className="p-2.5 text-center font-mono tabular-nums text-red-400">{fmtPct(sig.worst_return_5d)}</td>
                <td className="p-2.5 text-center">
                  <span className={`font-mono tabular-nums ${sig.profit_factor >= 1.5 ? "text-emerald-400" : sig.profit_factor < 1 ? "text-red-400" : "text-gray-300"}`}>
                    {sig.profit_factor.toFixed(2)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex gap-4 text-[10px] text-gray-600 pt-1">
        <span>Win Rate: % of signals followed by positive return</span>
        <span>Profit Factor: gross wins / gross losses</span>
      </div>
    </div>
  );
}
