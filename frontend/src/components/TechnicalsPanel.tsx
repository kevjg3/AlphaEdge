"use client";

import { Card, Stat, Badge } from "./Card";
import { fmtNumber } from "@/lib/format";

function IndicatorRow({ label, value, signal }: { label: string; value: string; signal?: "bullish" | "bearish" | "neutral" }) {
  return (
    <div className="data-row">
      <span className="text-sm text-gray-400">{label}</span>
      <div className="flex items-center gap-2">
        <span className="text-sm font-mono text-white tabular-nums">{value}</span>
        {signal && (
          <Badge text={signal} variant={signal === "bullish" ? "positive" : signal === "bearish" ? "negative" : "neutral"} size="sm" />
        )}
      </div>
    </div>
  );
}

export default function TechnicalsPanel({ data }: { data: Record<string, any> | null }) {
  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-gray-500 animate-fade-in">
        <p className="text-sm">Technicals not available</p>
      </div>
    );
  }

  const ind = data.indicators || {};
  const regime = data.regime || {};
  const sr = data.support_resistance || {};
  const factors = data.factor_exposures || {};

  const rsiSignal = ind.rsi_14 > 70 ? "bearish" : ind.rsi_14 < 30 ? "bullish" : "neutral";
  const macdSignal = ind.macd_histogram > 0 ? "bullish" : ind.macd_histogram < 0 ? "bearish" : "neutral";

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Regime */}
      {regime.current_regime && (
        <Card>
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-500 uppercase tracking-wider">Market Regime</span>
            <Badge
              text={regime.current_regime.replace(/_/g, " ")}
              variant={regime.current_regime === "low_vol" ? "positive" : regime.current_regime === "high_vol" ? "negative" : "neutral"}
              dot
            />
            <span className="text-sm font-mono text-gray-400 tabular-nums">
              {fmtNumber(regime.regime_probability * 100, 0)}% confidence
            </span>
          </div>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Moving Averages */}
        <Card title="Moving Averages">
          <IndicatorRow label="SMA 20" value={`$${fmtNumber(ind.sma_20)}`} signal={ind.above_sma_20 ? "bullish" : "bearish"} />
          <IndicatorRow label="SMA 50" value={`$${fmtNumber(ind.sma_50)}`} signal={ind.above_sma_50 ? "bullish" : "bearish"} />
          <IndicatorRow label="SMA 200" value={`$${fmtNumber(ind.sma_200)}`} signal={ind.above_sma_200 ? "bullish" : "bearish"} />
          <IndicatorRow label="EMA 12" value={`$${fmtNumber(ind.ema_12)}`} />
          <IndicatorRow label="Golden Cross" value={ind.golden_cross ? "Yes" : "No"} signal={ind.golden_cross ? "bullish" : "neutral"} />
        </Card>

        {/* Oscillators */}
        <Card title="Oscillators">
          <IndicatorRow label="RSI (14)" value={fmtNumber(ind.rsi_14, 1)} signal={rsiSignal as any} />
          <IndicatorRow label="MACD" value={fmtNumber(ind.macd, 4)} signal={macdSignal as any} />
          <IndicatorRow label="MACD Histogram" value={fmtNumber(ind.macd_histogram, 4)} />
          <IndicatorRow label="Stoch %K" value={fmtNumber(ind.stoch_k, 1)} />
          <IndicatorRow label="Stoch %D" value={fmtNumber(ind.stoch_d, 1)} />
          <IndicatorRow label="ADX" value={fmtNumber(ind.adx_14, 1)} />
        </Card>

        {/* Volatility */}
        <Card title="Volatility & Volume">
          <IndicatorRow label="ATR (14)" value={`$${fmtNumber(ind.atr_14)}`} />
          <IndicatorRow label="BB Upper" value={`$${fmtNumber(ind.bb_upper)}`} />
          <IndicatorRow label="BB Lower" value={`$${fmtNumber(ind.bb_lower)}`} />
          <IndicatorRow label="BB %B" value={fmtNumber(ind.bb_pct_b, 3)} />
          <IndicatorRow label="Volume Ratio" value={fmtNumber(ind.volume_ratio, 2)} signal={ind.volume_ratio > 1.5 ? "bullish" : ind.volume_ratio < 0.5 ? "bearish" : "neutral"} />
        </Card>

        {/* Support / Resistance */}
        <Card title="Support & Resistance">
          {sr.nearest_support && <IndicatorRow label="Nearest Support" value={`$${fmtNumber(sr.nearest_support)}`} />}
          {sr.nearest_resistance && <IndicatorRow label="Nearest Resistance" value={`$${fmtNumber(sr.nearest_resistance)}`} />}
          {sr.pivot_points && (
            <>
              <IndicatorRow label="Pivot" value={`$${fmtNumber(sr.pivot_points.pivot)}`} />
              <IndicatorRow label="R1 / S1" value={`$${fmtNumber(sr.pivot_points.r1)} / $${fmtNumber(sr.pivot_points.s1)}`} />
              <IndicatorRow label="R2 / S2" value={`$${fmtNumber(sr.pivot_points.r2)} / $${fmtNumber(sr.pivot_points.s2)}`} />
            </>
          )}
          {sr.levels && (
            <div className="mt-2 text-xs text-gray-500">{sr.levels.length} S/R levels detected via clustering</div>
          )}
        </Card>
      </div>

      {/* Factor Exposures */}
      {factors.exposures && factors.exposures.length > 0 && (
        <Card title="Factor Exposures">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
            <Stat label="R-squared" value={fmtNumber(factors.r_squared, 3)} />
            <Stat label="Alpha (ann.)" value={fmtNumber(factors.alpha_annualized * 100, 2) + "%"} color={factors.alpha_annualized > 0 ? "text-emerald-400" : "text-red-400"} />
            <Stat label="Residual Vol" value={fmtNumber(factors.residual_vol * 100, 1) + "%"} />
          </div>
          <div className="border-t border-white/[0.06] pt-3">
            {factors.exposures.map((e: any) => (
              <IndicatorRow key={e.factor_name} label={e.factor_name} value={fmtNumber(e.beta, 3)}
                signal={e.p_value < 0.05 ? (e.beta > 0 ? "bullish" : "bearish") : "neutral"} />
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
