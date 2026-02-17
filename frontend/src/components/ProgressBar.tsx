"use client";

const STEP_LABELS: Record<string, string> = {
  pending: "Initializing",
  data_ingestion: "Fetching market data",
  fundamentals: "Analyzing fundamentals",
  technicals: "Computing technical indicators",
  news: "Processing news & sentiment",
  forecasting: "Running ML forecasts",
  risk: "Calculating risk metrics",
  quantitative: "Running quantitative analysis",
  completed: "Analysis complete",
};

export default function ProgressBar({ progress, step }: { progress: number; step: string }) {
  const pct = Math.round(progress * 100);
  const label = STEP_LABELS[step] || step.replace(/_/g, " ");

  return (
    <div className="mb-6 bg-surface-raised border border-white/[0.06] rounded-2xl p-5 shadow-card animate-pulse-glow">
      <div className="flex justify-between items-center mb-3">
        <div className="flex items-center gap-2.5">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-brand-500" />
          </span>
          <span className="text-sm font-medium text-gray-300">{label}</span>
        </div>
        <span className="text-sm font-mono font-bold text-brand-400 tabular-nums">{pct}%</span>
      </div>

      {/* Track */}
      <div className="h-2 bg-surface-overlay rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out relative"
          style={{
            width: `${pct}%`,
            background: "linear-gradient(90deg, #6366f1, #818cf8)",
          }}
        >
          <div
            className="absolute inset-0 animate-progress-stripe opacity-30"
            style={{
              backgroundImage:
                "repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(255,255,255,0.15) 10px, rgba(255,255,255,0.15) 20px)",
              backgroundSize: "40px 40px",
            }}
          />
        </div>
      </div>

      {/* Step indicators */}
      <div className="flex justify-between mt-2.5">
        {Object.entries(STEP_LABELS).slice(1, -1).map(([key, name]) => {
          const isActive = step === key;
          const isDone = Object.keys(STEP_LABELS).indexOf(step) > Object.keys(STEP_LABELS).indexOf(key);
          return (
            <div key={key} className="flex items-center gap-1">
              <span className={`w-1 h-1 rounded-full ${isDone ? "bg-brand-500" : isActive ? "bg-brand-400" : "bg-gray-700"}`} />
              <span className={`text-[10px] hidden md:inline ${isDone ? "text-gray-400" : isActive ? "text-brand-400" : "text-gray-600"}`}>
                {name.split(" ").slice(-1)[0]}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
