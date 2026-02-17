"use client";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
  ReferenceLine,
} from "recharts";

interface Props {
  monthly: Record<number, { mean_return: number; win_rate: number; n_samples: number }>;
}

const MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export default function SeasonalityChart({ monthly }: Props) {
  if (!monthly || Object.keys(monthly).length === 0) return null;

  const data = Object.entries(monthly).map(([m, stats]) => ({
    month: MONTH_LABELS[parseInt(m) - 1] || m,
    return: stats.mean_return,
    winRate: stats.win_rate,
    samples: stats.n_samples,
  }));

  return (
    <div className="h-[250px] w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="month" tick={{ fill: "#64748b", fontSize: 11 }} />
          <YAxis
            tick={{ fill: "#64748b", fontSize: 10 }}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            width={45}
          />
          <Tooltip
            contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, fontSize: 11 }}
            formatter={(v: number, name: string) => {
              if (name === "return") return [`${(v * 100).toFixed(2)}%`, "Avg Return"];
              return [v, name];
            }}
          />
          <ReferenceLine y={0} stroke="#475569" />
          <Bar dataKey="return" radius={[3, 3, 0, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.return >= 0 ? "#22c55e" : "#ef4444"} fillOpacity={0.7} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
