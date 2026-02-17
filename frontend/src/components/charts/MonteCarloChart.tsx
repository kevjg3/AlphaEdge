"use client";

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ComposedChart,
} from "recharts";

interface Props {
  fanChart: Record<string, { day: number; price: number }[]>;
  currentPrice: number;
  samplePaths?: { day: number; price: number }[][];
}

export default function MonteCarloChart({ fanChart, currentPrice, samplePaths }: Props) {
  if (!fanChart || !fanChart.p50) return null;

  // Merge all percentiles into a single data array keyed by day
  const dayMap = new Map<number, Record<string, number>>();

  for (const [key, pts] of Object.entries(fanChart)) {
    for (const pt of pts) {
      if (!dayMap.has(pt.day)) dayMap.set(pt.day, { day: pt.day });
      dayMap.get(pt.day)![key] = pt.price;
    }
  }

  // Add sample paths
  if (samplePaths) {
    samplePaths.forEach((path, i) => {
      for (const pt of path) {
        if (!dayMap.has(pt.day)) dayMap.set(pt.day, { day: pt.day });
        dayMap.get(pt.day)![`path${i}`] = pt.price;
      }
    });
  }

  const data = Array.from(dayMap.values()).sort((a, b) => a.day - b.day);

  return (
    <div className="h-[350px] w-full">
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
          <defs>
            <linearGradient id="mc95" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#6366f1" stopOpacity={0.08} />
              <stop offset="100%" stopColor="#6366f1" stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="mc75" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#6366f1" stopOpacity={0.15} />
              <stop offset="100%" stopColor="#6366f1" stopOpacity={0.05} />
            </linearGradient>
            <linearGradient id="mc50" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#6366f1" stopOpacity={0.25} />
              <stop offset="100%" stopColor="#6366f1" stopOpacity={0.1} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="day"
            tick={{ fill: "#64748b", fontSize: 10 }}
            tickFormatter={(d) => `${d}d`}
            interval="preserveStartEnd"
            minTickGap={40}
          />
          <YAxis
            domain={["auto", "auto"]}
            tick={{ fill: "#64748b", fontSize: 10 }}
            tickFormatter={(v: number) => `$${v.toFixed(0)}`}
            width={60}
          />
          <Tooltip
            contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, fontSize: 11 }}
            labelStyle={{ color: "#94a3b8" }}
            labelFormatter={(d) => `Day ${d}`}
            formatter={(v: number, name: string) => [`$${v.toFixed(2)}`, name.replace("p", "P")]}
          />

          {/* 5-95 band */}
          <Area type="monotone" dataKey="p95" stroke="none" fill="url(#mc95)" stackId="band1" />
          <Area type="monotone" dataKey="p5" stroke="none" fill="transparent" stackId="band1" />

          {/* 25-75 band */}
          <Area type="monotone" dataKey="p75" stroke="none" fill="url(#mc75)" />
          <Area type="monotone" dataKey="p25" stroke="none" fill="transparent" />

          {/* Median */}
          <Line type="monotone" dataKey="p50" stroke="#a78bfa" strokeWidth={2} dot={false} />

          {/* 10/90 lines */}
          <Line type="monotone" dataKey="p90" stroke="#818cf8" strokeWidth={1} strokeDasharray="4 4" dot={false} />
          <Line type="monotone" dataKey="p10" stroke="#818cf8" strokeWidth={1} strokeDasharray="4 4" dot={false} />

          {/* Sample paths */}
          {samplePaths?.map((_, i) => (
            <Line
              key={`path${i}`}
              type="monotone"
              dataKey={`path${i}`}
              stroke="#475569"
              strokeWidth={0.5}
              dot={false}
              strokeOpacity={0.4}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
