const BASE = "/api/v1";

// For long-running requests, call the backend directly to avoid Vercel proxy timeout
const DIRECT_BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL || "";

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

/** Fetch directly from the backend (bypasses Vercel rewrite proxy). */
async function fetchBackendJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const base = DIRECT_BACKEND ? `${DIRECT_BACKEND}/api/v1` : BASE;
  const res = await fetch(`${base}${url}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

export interface SnapshotData {
  ticker: string;
  name: string;
  price: number | null;
  change_1d: number | null;
  change_1d_pct: number | null;
  market_cap: number | null;
  pe_ratio: number | null;
  beta: number | null;
  sector: string;
  industry: string;
  high_52w: number | null;
  low_52w: number | null;
  avg_volume: number | null;
  description: string;
  country: string;
  employees: number | null;
  website: string;
  dividend_yield: number | null;
  forward_pe: number | null;
  revenue_growth: number | null;
  operating_margins: number | null;
  profit_margins: number | null;
  return_on_equity: number | null;
  debt_to_equity: number | null;
  free_cashflow: number | null;
  total_revenue: number | null;
  warnings: string[];
}

export interface AnalysisStatus {
  run_id: string;
  ticker: string;
  status: "pending" | "running" | "completed" | "failed";
  started_at: string | null;
  completed_at: string | null;
  progress: number;
  current_step: string;
  warnings: string[];
}

export interface FullAnalysis {
  run_id: string;
  ticker: string;
  status: string;
  snapshot: SnapshotData;
  fundamentals: Record<string, unknown> | null;
  technicals: Record<string, unknown> | null;
  news: Record<string, unknown> | null;
  forecast: Record<string, unknown> | null;
  risk: Record<string, unknown> | null;
  quant: Record<string, unknown> | null;
  warnings: string[];
  attribution: Record<string, unknown>[];
}

export const api = {
  health: () => fetchJSON<{ status: string; version: string; services: Record<string, boolean> }>("/health"),

  snapshot: (ticker: string) => fetchJSON<SnapshotData>(`/snapshot/${ticker}`),

  startAnalysis: (ticker: string, seed = 42) =>
    fetchJSON<AnalysisStatus>("/analysis/run", {
      method: "POST",
      body: JSON.stringify({ ticker, seed }),
    }),

  /** Run analysis synchronously — calls backend directly to avoid proxy timeout. */
  runAnalysisSync: (ticker: string, seed = 42) =>
    fetchBackendJSON<FullAnalysis>("/analysis/sync", {
      method: "POST",
      body: JSON.stringify({ ticker, seed }),
    }),

  getStatus: (runId: string) => fetchJSON<AnalysisStatus>(`/analysis/status/${runId}`),

  getResult: (runId: string) => fetchJSON<FullAnalysis>(`/analysis/result/${runId}`),

  listRuns: () => fetchJSON<AnalysisStatus[]>("/analysis/runs"),

  /** Download a PPTX pitch deck for a completed analysis run. */
  downloadPitchDeck: async (runId: string, ticker: string) => {
    const base = DIRECT_BACKEND ? `${DIRECT_BACKEND}/api/v1` : BASE;
    const res = await fetch(`${base}/analysis/pitch-deck/${runId}`);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`Pitch deck download failed: ${res.status} ${text}`);
    }
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${ticker}_pitch_deck.pptx`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  },
};
