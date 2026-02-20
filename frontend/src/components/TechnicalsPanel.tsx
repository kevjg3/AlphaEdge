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

function signalLabel(s: string): string {
  return s
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
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
  const confluence = data.confluence || {};
  const ichimoku = data.ichimoku || {};
  const volProfile = data.volume_profile || {};
  const divergence = data.divergence || {};
  const relStrength = data.relative_strength || {};

  const rsiSignal = ind.rsi_14 > 70 ? "bearish" : ind.rsi_14 < 30 ? "bullish" : "neutral";
  const macdSignal = ind.macd_histogram > 0 ? "bullish" : ind.macd_histogram < 0 ? "bearish" : "neutral";

  return (
    <div className="space-y-6 animate-slide-up">
      {/* ── Signal Confluence (top — most actionable) ── */}
      {confluence.verdict && (
        <Card>
          <div className="flex flex-col md:flex-row items-start md:items-center gap-4">
            <div className="flex items-center gap-3">
              <span className="text-xs text-gray-500 uppercase tracking-wider">Signal Confluence</span>
              <Badge
                text={confluence.verdict}
                variant={
                  confluence.verdict === "Strong Buy" || confluence.verdict === "Buy"
                    ? "positive"
                    : confluence.verdict === "Strong Sell" || confluence.verdict === "Sell"
                    ? "negative"
                    : "neutral"
                }
                dot
              />
            </div>
            <div className="flex items-center gap-3 text-sm">
              <span className="text-emerald-400 font-mono font-bold">{confluence.bullish_count}</span>
              <span className="text-gray-500">bullish</span>
              <span className="text-gray-600">|</span>
              <span className="text-red-400 font-mono font-bold">{confluence.bearish_count}</span>
              <span className="text-gray-500">bearish</span>
              <span className="text-gray-600">|</span>
              <span className="text-gray-400 font-mono font-bold">{confluence.neutral_count}</span>
              <span className="text-gray-500">neutral</span>
            </div>
            <div className="flex items-center gap-2 ml-auto">
              <div className="w-24 h-2 bg-surface-overlay rounded-full overflow-hidden flex">
                <div
                  className="h-full bg-emerald-500 rounded-l-full"
                  style={{ width: `${(confluence.bullish_count / confluence.total_signals) * 100}%` }}
                />
                <div
                  className="h-full bg-red-500 rounded-r-full"
                  style={{ width: `${(confluence.bearish_count / confluence.total_signals) * 100}%` }}
                />
              </div>
              <span className="text-xs font-mono text-gray-400">{fmtNumber(confluence.score_pct, 0)}%</span>
            </div>
          </div>

          {/* Signal breakdown */}
          {confluence.signals && confluence.signals.length > 0 && (
            <div className="mt-3 pt-3 border-t border-white/[0.06] grid grid-cols-2 md:grid-cols-3 gap-1">
              {confluence.signals.map((s: any) => (
                <div key={s.name} className="flex items-center justify-between text-xs py-1 px-2 rounded-md hover:bg-white/[0.02]">
                  <span className="text-gray-400">{s.name}</span>
                  <Badge text={s.signal} variant={s.signal === "bullish" ? "positive" : s.signal === "bearish" ? "negative" : "neutral"} size="sm" />
                </div>
              ))}
            </div>
          )}
        </Card>
      )}

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

        {/* Ichimoku Cloud */}
        {ichimoku.tenkan_sen && (
          <Card title="Ichimoku Cloud">
            <IndicatorRow label="Tenkan-sen (9)" value={`$${fmtNumber(ichimoku.tenkan_sen)}`} />
            <IndicatorRow label="Kijun-sen (26)" value={`$${fmtNumber(ichimoku.kijun_sen)}`} />
            <IndicatorRow
              label="TK Cross"
              value={signalLabel(ichimoku.tk_cross)}
              signal={ichimoku.tk_cross === "bullish" ? "bullish" : ichimoku.tk_cross === "bearish" ? "bearish" : "neutral"}
            />
            <IndicatorRow label="Senkou Span A" value={`$${fmtNumber(ichimoku.senkou_span_a)}`} />
            <IndicatorRow label="Senkou Span B" value={`$${fmtNumber(ichimoku.senkou_span_b)}`} />
            <IndicatorRow
              label="Price vs Cloud"
              value={signalLabel(ichimoku.price_vs_cloud)}
              signal={ichimoku.price_vs_cloud === "above" ? "bullish" : ichimoku.price_vs_cloud === "below" ? "bearish" : "neutral"}
            />
            <IndicatorRow
              label="Cloud Color"
              value={signalLabel(ichimoku.cloud_color)}
              signal={ichimoku.cloud_color === "green" ? "bullish" : "bearish"}
            />
            <div className="mt-2 flex items-center gap-2">
              <span className="text-xs text-gray-500">Overall:</span>
              <Badge
                text={signalLabel(ichimoku.overall_signal)}
                variant={
                  ichimoku.overall_signal?.includes("bullish") ? "positive" :
                  ichimoku.overall_signal?.includes("bearish") ? "negative" : "neutral"
                }
                dot
              />
            </div>
          </Card>
        )}

        {/* Volume Profile & VWAP */}
        {(volProfile.poc_price || volProfile.vwap) && (
          <Card title="Volume Profile & VWAP">
            {volProfile.poc_price && (
              <>
                <IndicatorRow label="Point of Control" value={`$${fmtNumber(volProfile.poc_price)}`}
                  signal={volProfile.price_vs_poc === "above" ? "bullish" : volProfile.price_vs_poc === "below" ? "bearish" : "neutral"}
                />
                <IndicatorRow label="Value Area High" value={`$${fmtNumber(volProfile.value_area_high)}`} />
                <IndicatorRow label="Value Area Low" value={`$${fmtNumber(volProfile.value_area_low)}`} />
                <IndicatorRow
                  label="Price vs VA"
                  value={signalLabel(volProfile.price_vs_value_area)}
                  signal={volProfile.price_vs_value_area === "above" ? "bullish" : volProfile.price_vs_value_area === "below" ? "bearish" : "neutral"}
                />
              </>
            )}
            {volProfile.vwap && (
              <>
                <div className="mt-2 pt-2 border-t border-white/[0.04]" />
                <IndicatorRow label="VWAP" value={`$${fmtNumber(volProfile.vwap)}`} />
                <IndicatorRow label="VWAP +1σ / -1σ" value={`$${fmtNumber(volProfile.vwap_upper_1)} / $${fmtNumber(volProfile.vwap_lower_1)}`} />
                <IndicatorRow label="VWAP +2σ / -2σ" value={`$${fmtNumber(volProfile.vwap_upper_2)} / $${fmtNumber(volProfile.vwap_lower_2)}`} />
              </>
            )}
          </Card>
        )}
      </div>

      {/* Divergence Detection */}
      {divergence.rsi_divergence && (
        <Card title="Divergence Detection">
          {divergence.has_any_divergence ? (
            <div className="space-y-3">
              {divergence.rsi_divergence.type !== "none" && (
                <div className="flex items-start gap-3">
                  <Badge
                    text={`RSI ${signalLabel(divergence.rsi_divergence.type)}`}
                    variant={divergence.rsi_divergence.type === "bullish" ? "positive" : "negative"}
                    dot
                  />
                  <div>
                    <div className="text-xs text-gray-400">{divergence.rsi_divergence.description}</div>
                    <div className="text-[10px] text-gray-600 mt-0.5">Strength: {signalLabel(divergence.rsi_divergence.strength)}</div>
                  </div>
                </div>
              )}
              {divergence.macd_divergence.type !== "none" && (
                <div className="flex items-start gap-3">
                  <Badge
                    text={`MACD ${signalLabel(divergence.macd_divergence.type)}`}
                    variant={divergence.macd_divergence.type === "bullish" ? "positive" : "negative"}
                    dot
                  />
                  <div>
                    <div className="text-xs text-gray-400">{divergence.macd_divergence.description}</div>
                    <div className="text-[10px] text-gray-600 mt-0.5">Strength: {signalLabel(divergence.macd_divergence.strength)}</div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-gray-500">No RSI or MACD divergence detected in recent price action</div>
          )}
        </Card>
      )}

      {/* Relative Strength vs Sector */}
      {relStrength.sector && (
        <Card title={`Relative Strength vs ${relStrength.sector} (${relStrength.sector_etf})`}>
          <div className="flex items-center gap-2 mb-3">
            <Badge
              text={signalLabel(relStrength.rs_trend)}
              variant={relStrength.rs_trend === "outperforming" ? "positive" : relStrength.rs_trend === "underperforming" ? "negative" : "neutral"}
              dot
            />
            {relStrength.outperformance_streak_days > 0 && (
              <span className="text-xs text-gray-500">
                <span className="text-emerald-400 font-mono">{relStrength.outperformance_streak_days}</span>-day outperformance streak
              </span>
            )}
          </div>
          <div className="grid grid-cols-3 gap-3">
            {relStrength.relative_return_1m != null && (
              <Stat
                label="1M Relative"
                value={`${(relStrength.relative_return_1m * 100).toFixed(1)}%`}
                color={relStrength.relative_return_1m > 0 ? "text-emerald-400" : "text-red-400"}
              />
            )}
            {relStrength.relative_return_3m != null && (
              <Stat
                label="3M Relative"
                value={`${(relStrength.relative_return_3m * 100).toFixed(1)}%`}
                color={relStrength.relative_return_3m > 0 ? "text-emerald-400" : "text-red-400"}
              />
            )}
            {relStrength.relative_return_6m != null && (
              <Stat
                label="6M Relative"
                value={`${(relStrength.relative_return_6m * 100).toFixed(1)}%`}
                color={relStrength.relative_return_6m > 0 ? "text-emerald-400" : "text-red-400"}
              />
            )}
          </div>
        </Card>
      )}

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
