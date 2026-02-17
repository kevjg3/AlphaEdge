"use client";

import { useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from "recharts";

interface SeriesData {
  date: string;
  value: number;
}

interface Props {
  rollingVol21d: SeriesData[];
  rollingVol63d: SeriesData[];
  rollingSharpe63d: SeriesData[];
  rollingSkewness63d: SeriesData[];
  cumulativeReturn: SeriesData[];
}

type MetricKey = "vol" | "sharpe" | "cumreturn" | "skewness";

const METRIC_CONFIG: Record<MetricKey, { label: string; color: string; format: (v: number) => string; refLine?: number }> = {
  vol: { label: "Rolling Volatility (ann.)", color: "#f59e0b", format: (v) => `${(v * 100).toFixed(1)}%` },
  sharpe: { label: "Rolling Sharpe (63d)", color: "#22d3ee", format: (v) => v.toFixed(2), refLine: 0 },
  cumreturn: { label: "Cumulative Return", color: "#10b981", format: (v) => `${(v * 100).toFixed(1)}%`, refLine: 0 },
  skewness: { label: "Rolling Skewness (63d)", color: "#f472b6", format: (v) => v.toFixed(3), refLine: 0 },
};

export default function RollingMetrics({ rollingVol21d, rollingVol63d, rollingSharpe63d, rollingSkewness63d, cumulativeReturn }: Props) {
  const [metric, setMetric] = useState<MetricKey>("cumreturn");

  const dataMap: Record<MetricKey, SeriesData[]> = {
    vol: rollingVol63d || [],
    sharpe: rollingSharpe63d || [],
    cumreturn: cumulativeReturn || [],
    skewness: rollingSkewness63d || [],
  };

  const config = METRIC_CONFIG[metric];
  const data = dataMap[metric];

  if (!data || data.length === 0) return null;

  return (
    <div className="space-y-3">
      {/* Metric toggle buttons */}
      <div className="flex gap-1.5 flex-wrap">
        {(Object.keys(METRIC_CONFIG) as MetricKey[]).map((key) => (
          <button
            key={key}
            onClick={() => setMetric(key)}
            className={`px-3 py-1 text-xs rounded-full transition-colors ${
              metric === key
                ? "bg-gray-700 text-white"
                : "bg-gray-900 text-gray-400 hover:text-gray-200 border border-gray-800"
            }`}
          >
            {METRIC_CONFIG[key].label}
          </button>
        ))}
      </div>

      <div className="h-[280px] w-full">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="date"
              tick={{ fill: "#64748b", fontSize: 10 }}
              tickFormatter={(v) => v.slice(5)}
              interval="preserveStartEnd"
              minTickGap={60}
            />
            <YAxis
              tick={{ fill: "#64748b", fontSize: 10 }}
              tickFormatter={(v: number) => config.format(v)}
              width={55}
            />
            <Tooltip
              contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, fontSize: 11 }}
              labelStyle={{ color: "#94a3b8" }}
              formatter={(v: number) => [config.format(v), config.label]}
            />
            {config.refLine !== undefined && (
              <ReferenceLine y={config.refLine} stroke="#475569" strokeWidth={1} strokeDasharray="4 4" />
            )}
            <Line type="monotone" dataKey="value" stroke={config.color} strokeWidth={1.5} dot={false} />

            {/* Overlay 21d vol when showing vol */}
            {metric === "vol" && rollingVol21d && rollingVol21d.length > 0 && (
              <Line
                data={rollingVol21d}
                type="monotone"
                dataKey="value"
                stroke="#fbbf24"
                strokeWidth={1}
                strokeOpacity={0.4}
                dot={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
