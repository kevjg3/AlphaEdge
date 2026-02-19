"use client";

import { useState, useCallback, useRef } from "react";
import { api, FullAnalysis } from "@/lib/api";
import SnapshotPanel from "@/components/SnapshotPanel";
import FundamentalsPanel from "@/components/FundamentalsPanel";
import TechnicalsPanel from "@/components/TechnicalsPanel";
import NewsPanel from "@/components/NewsPanel";
import ForecastPanel from "@/components/ForecastPanel";
import RiskPanel from "@/components/RiskPanel";
import QuantPanel from "@/components/QuantPanel";

const TABS = [
  { id: "snapshot", label: "Snapshot", icon: "\u{1F4CA}" },
  { id: "fundamentals", label: "Valuation", icon: "\u{1F4B0}" },
  { id: "technicals", label: "Technicals", icon: "\u{1F4C8}" },
  { id: "news", label: "News", icon: "\u{1F4F0}" },
  { id: "forecast", label: "Forecast", icon: "\u{1F52E}" },
  { id: "risk", label: "Risk", icon: "\u26A1" },
  { id: "quant", label: "Quant", icon: "\u{1F9EA}" },
];

const STEPS = [
  { key: "fetching", label: "Fetching data" },
  { key: "fundamentals", label: "Fundamentals" },
  { key: "technicals", label: "Technicals" },
  { key: "news", label: "News & NLP" },
  { key: "forecast", label: "Forecasting" },
  { key: "risk", label: "Risk analysis" },
  { key: "quant", label: "Quant metrics" },
  { key: "done", label: "Done" },
];

/** Map SSE step string to our STEPS index. */
function stepIndex(step: string): number {
  const s = step.toLowerCase();
  if (s.includes("fetch") || s.includes("data")) return 0;
  if (s.includes("fundamental") || s.includes("valuation")) return 1;
  if (s.includes("technical")) return 2;
  if (s.includes("news") || s.includes("sentiment") || s.includes("nlp")) return 3;
  if (s.includes("forecast")) return 4;
  if (s.includes("risk")) return 5;
  if (s.includes("quant") || s.includes("backtest") || s.includes("monte")) return 6;
  if (s.includes("done") || s.includes("complete")) return 7;
  return -1;
}

export default function Home() {
  const [ticker, setTicker] = useState("");
  const [activeTab, setActiveTab] = useState("snapshot");

  // Streaming analysis state
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<FullAnalysis | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState("");
  const [progress, setProgress] = useState(0);
  const [activeStepIdx, setActiveStepIdx] = useState(-1);

  const abortRef = useRef(false);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      const t = ticker.trim().toUpperCase();
      if (!t || isRunning) return;

      setActiveTab("snapshot");
      setIsRunning(true);
      setResult(undefined);
      setError(null);
      setCurrentStep("Starting analysis...");
      setProgress(0);
      setActiveStepIdx(-1);
      abortRef.current = false;

      api
        .runAnalysisStream(
          t,
          (step, pct) => {
            if (abortRef.current) return;
            setCurrentStep(step);
            setProgress(pct);
            setActiveStepIdx(stepIndex(step));
          },
        )
        .then((data) => {
          if (abortRef.current) return;
          setResult(data);
          setProgress(100);
          setActiveStepIdx(STEPS.length - 1);
          setCurrentStep("Done");
        })
        .catch((err) => {
          if (abortRef.current) return;
          setError(err?.message || "Analysis failed");
        })
        .finally(() => {
          setIsRunning(false);
        });
    },
    [ticker, isRunning],
  );

  /** Percentage to display (0–100). Backend sends 0.0–1.0. */
  const pctDisplay = Math.min(Math.round(progress * 100), 100);

  return (
    <div className="min-h-screen bg-surface">
      {/* Header */}
      <header className="sticky top-0 z-50 glass shadow-header">
        <div className="max-w-7xl mx-auto px-6 py-3.5 flex items-center gap-8">
          {/* Logo */}
          <div className="flex items-center gap-2.5 shrink-0">
            <div className="w-8 h-8 rounded-lg bg-brand-gradient flex items-center justify-center shadow-glow-sm">
              <span className="text-white font-bold text-sm">A</span>
            </div>
            <div>
              <h1 className="text-base font-bold text-white tracking-tight leading-none">
                Alpha<span className="text-gradient">Edge</span>
              </h1>
              <span className="text-[10px] text-gray-500 font-medium tracking-wider uppercase">Analysis Platform</span>
            </div>
          </div>

          {/* Search */}
          <form onSubmit={handleSubmit} className="flex gap-2.5 flex-1 max-w-lg">
            <div className="relative flex-1">
              <div className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-500">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
                </svg>
              </div>
              <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                placeholder="Search ticker..."
                className="w-full bg-surface-overlay border border-white/[0.06] rounded-xl
                  pl-10 pr-4 py-2.5 text-sm text-white placeholder:text-gray-500
                  focus:outline-none focus:border-brand-500/50 focus:shadow-glow-sm
                  transition-all duration-200"
              />
            </div>
            <button
              type="submit"
              disabled={isRunning}
              className="bg-brand-gradient hover:opacity-90 disabled:opacity-40
                text-white px-6 py-2.5 rounded-xl text-sm font-semibold
                shadow-glow-sm hover:shadow-glow
                transition-all duration-200 whitespace-nowrap"
            >
              {isRunning ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>
                  Analyzing...
                </span>
              ) : "Analyze"}
            </button>
          </form>

          {/* Right side status + pitch deck download */}
          {result && !isRunning && (
            <div className="hidden lg:flex items-center gap-3">
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                Analysis complete
              </div>
              <button
                onClick={() => api.downloadPitchDeck(result.run_id, result.ticker)}
                className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg
                  bg-brand-gradient text-white text-xs font-semibold
                  hover:opacity-90 shadow-glow-sm hover:shadow-glow
                  transition-all duration-200"
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                     stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"
                     strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                Pitch Deck
              </button>
            </div>
          )}
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-6">
        {/* Loading State — real progress tracking */}
        {isRunning && (
          <div className="animate-slide-down">
            <div className="bg-surface-raised border border-white/[0.06] rounded-2xl p-6 mb-5">
              <div className="flex items-center gap-4 mb-4">
                <div className="relative">
                  <div className="w-10 h-10 rounded-xl bg-brand-gradient flex items-center justify-center shadow-glow-sm">
                    <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <h3 className="text-sm font-semibold text-white">Running Full Analysis</h3>
                    <span className="text-xs font-mono text-brand-400 tabular-nums">
                      {pctDisplay}%
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 mt-0.5 truncate">
                    {currentStep || "Initializing..."}
                  </p>
                </div>
              </div>

              {/* Progress bar */}
              <div className="w-full h-1.5 bg-white/[0.06] rounded-full overflow-hidden mb-4">
                <div
                  className="h-full bg-brand-gradient rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${pctDisplay}%` }}
                />
              </div>

              {/* Step indicators */}
              <div className="grid grid-cols-4 md:grid-cols-8 gap-2">
                {STEPS.map((step, i) => {
                  const isCompleted = i < activeStepIdx;
                  const isActive = i === activeStepIdx;
                  const isPending = i > activeStepIdx;

                  return (
                    <div key={step.key} className="flex flex-col items-center gap-1.5">
                      <div
                        className={`w-2.5 h-2.5 rounded-full transition-all duration-300 ${
                          isCompleted
                            ? "bg-emerald-400 shadow-[0_0_6px_rgba(16,185,129,0.5)]"
                            : isActive
                              ? "bg-brand-400 shadow-[0_0_8px_rgba(99,102,241,0.6)] animate-pulse"
                              : "bg-white/[0.08]"
                        }`}
                      />
                      <span
                        className={`text-[9px] text-center leading-tight transition-colors duration-300 ${
                          isCompleted
                            ? "text-emerald-400/80"
                            : isActive
                              ? "text-brand-300 font-medium"
                              : "text-gray-600"
                        }`}
                      >
                        {step.label}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* Error States */}
        {error && !isRunning && (
          <div className="bg-red-500/10 border border-red-500/20 rounded-2xl p-4 mb-5 text-red-300 text-sm animate-slide-down flex items-start gap-3">
            <span className="text-red-400 mt-0.5">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
            </span>
            <span>Analysis failed: {error}</span>
          </div>
        )}
        {result?.status === "failed" && (
          <div className="bg-red-500/10 border border-red-500/20 rounded-2xl p-4 mb-5 text-red-300 text-sm animate-slide-down flex items-start gap-3">
            <span className="text-red-400 mt-0.5">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
            </span>
            <span>Analysis completed with errors. {result.warnings?.join("; ")}</span>
          </div>
        )}

        {/* Warnings */}
        {result && result.warnings && result.warnings.length > 0 && result.status !== "failed" && (
          <div className="bg-amber-500/10 border border-amber-500/20 rounded-2xl p-3.5 mb-5 text-amber-300 text-xs animate-fade-in">
            {result.warnings.map((w, i) => <div key={i}>{w}</div>)}
          </div>
        )}

        {/* Tab Navigation */}
        {result && !isRunning && (
          <>
            <nav className="flex gap-1 mb-6 bg-surface-raised/50 p-1.5 rounded-xl border border-white/[0.04] animate-fade-in">
              {TABS.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`tab-btn flex items-center gap-1.5 ${
                    activeTab === tab.id ? "tab-btn-active" : ""
                  }`}
                >
                  <span className="text-xs">{tab.icon}</span>
                  {tab.label}
                </button>
              ))}
            </nav>

            {/* Tab Content */}
            <div className="min-h-[400px] animate-fade-in">
              {activeTab === "snapshot" && <SnapshotPanel data={result.snapshot} />}
              {activeTab === "fundamentals" && <FundamentalsPanel data={result.fundamentals} />}
              {activeTab === "technicals" && <TechnicalsPanel data={result.technicals} />}
              {activeTab === "news" && <NewsPanel data={result.news} />}
              {activeTab === "forecast" && <ForecastPanel data={result.forecast} />}
              {activeTab === "risk" && <RiskPanel data={result.risk} />}
              {activeTab === "quant" && <QuantPanel data={result.quant as Record<string, any> | null} />}
            </div>
          </>
        )}

        {/* Empty State */}
        {!isRunning && !result && !error && (
          <div className="flex flex-col items-center justify-center py-24 animate-fade-in">
            <div className="w-16 h-16 rounded-2xl bg-brand-gradient flex items-center justify-center shadow-glow mb-6">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
            </div>
            <h2 className="text-xl font-semibold text-white mb-2">Investment Analysis Platform</h2>
            <p className="text-sm text-gray-500 max-w-md text-center leading-relaxed">
              Enter a ticker symbol to run institutional-grade analysis including
              valuation, technicals, ML forecasting, sentiment analysis, and quantitative risk metrics.
            </p>
            <div className="flex gap-2 mt-6">
              {["AAPL", "MSFT", "GOOGL", "NVDA"].map((t) => (
                <button
                  key={t}
                  onClick={() => { setTicker(t); }}
                  className="px-3 py-1.5 rounded-lg bg-surface-raised border border-white/[0.06]
                    text-xs font-mono text-gray-400 hover:text-white hover:border-brand-500/30
                    transition-all duration-200 hover:shadow-glow-sm"
                >
                  {t}
                </button>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
