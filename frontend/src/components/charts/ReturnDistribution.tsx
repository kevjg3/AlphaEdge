"use client";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from "recharts";

interface Props {
  histogram: { bins: { x: number; count: number; pct: number }[]; normal_overlay?: { x: number; y: number }[] };
  stats: {
    mean: number;
    std: number;
    skewness: number;
    kurtosis: number;
    is_normal: boolean;
    jarque_bera_pval: number;
  };
}

export default function ReturnDistribution({ histogram, stats }: Props) {
  if (!histogram?.bins) return null;

  return (
    <div className="space-y-3">
      <div className="h-[280px] w-full">
        <ResponsiveContainer>
          <BarChart data={histogram.bins} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="x"
              tick={{ fill: "#64748b", fontSize: 9 }}
              tickFormatter={(v: number) => `${(v * 100).toFixed(1)}%`}
              interval="preserveStartEnd"
              minTickGap={40}
            />
            <YAxis
              tick={{ fill: "#64748b", fontSize: 10 }}
              width={40}
            />
            <Tooltip
              contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, fontSize: 11 }}
              formatter={(v: number, name: string) => {
                if (name === "count") return [v, "Observations"];
                return [`${v.toFixed(1)}%`, "% of Days"];
              }}
              labelFormatter={(v: number) => `Return: ${(v * 100).toFixed(2)}%`}
            />
            <ReferenceLine x={0} stroke="#475569" strokeWidth={1} />
            <ReferenceLine x={stats.mean} stroke="#f59e0b" strokeWidth={1} strokeDasharray="4 4" />
            <Bar dataKey="count" fill="#6366f1" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-center">
        <div>
          <div className="text-xs text-gray-500">Mean</div>
          <div className="text-sm font-mono text-white">{(stats.mean * 100).toFixed(3)}%</div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Std Dev</div>
          <div className="text-sm font-mono text-white">{(stats.std * 100).toFixed(3)}%</div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Skewness</div>
          <div className={`text-sm font-mono ${stats.skewness < -0.5 ? "text-red-400" : stats.skewness > 0.5 ? "text-green-400" : "text-white"}`}>
            {stats.skewness.toFixed(3)}
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Kurtosis</div>
          <div className={`text-sm font-mono ${stats.kurtosis > 3 ? "text-yellow-400" : "text-white"}`}>
            {stats.kurtosis.toFixed(3)}
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Normality</div>
          <div className={`text-sm font-mono ${stats.is_normal ? "text-green-400" : "text-red-400"}`}>
            {stats.is_normal ? "Normal" : "Fat Tails"} <span className="text-gray-500 text-xs">(p={stats.jarque_bera_pval.toFixed(3)})</span>
          </div>
        </div>
      </div>
    </div>
  );
}
