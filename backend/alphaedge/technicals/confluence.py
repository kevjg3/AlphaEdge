"""Technical signal confluence scorer — aggregate bullish/bearish signal count."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ConfluenceScorer:
    """Count how many technical indicators align bullish vs bearish."""

    def score(self, indicators: dict, sr: dict) -> dict:
        """Evaluate 12 signals and produce a confluence scorecard.

        Args:
            indicators: Output of TechnicalIndicators.compute_all()
            sr: Output of SupportResistance.detect()
        """
        if not indicators:
            return {}

        signals: list[dict] = []
        price = indicators.get("price")
        if price is None:
            return {}

        # 1. Price vs SMA 20
        sma20 = indicators.get("sma_20")
        if sma20 is not None:
            sig = "bullish" if price > sma20 else "bearish"
            signals.append({"name": "Price vs SMA 20", "value": f"${sma20:.2f}", "signal": sig})

        # 2. Price vs SMA 50
        sma50 = indicators.get("sma_50")
        if sma50 is not None:
            sig = "bullish" if price > sma50 else "bearish"
            signals.append({"name": "Price vs SMA 50", "value": f"${sma50:.2f}", "signal": sig})

        # 3. Price vs SMA 200
        sma200 = indicators.get("sma_200")
        if sma200 is not None:
            sig = "bullish" if price > sma200 else "bearish"
            signals.append({"name": "Price vs SMA 200", "value": f"${sma200:.2f}", "signal": sig})

        # 4. Golden Cross
        gc = indicators.get("golden_cross")
        if gc is not None:
            sig = "bullish" if gc else "bearish"
            signals.append({"name": "Golden Cross", "value": "Yes" if gc else "No", "signal": sig})

        # 5. RSI (14)
        rsi = indicators.get("rsi_14")
        if rsi is not None:
            if rsi < 30:
                sig = "bullish"
                val = f"{rsi:.1f} (Oversold)"
            elif rsi > 70:
                sig = "bearish"
                val = f"{rsi:.1f} (Overbought)"
            else:
                sig = "neutral"
                val = f"{rsi:.1f}"
            signals.append({"name": "RSI (14)", "value": val, "signal": sig})

        # 6. MACD Histogram
        macd_hist = indicators.get("macd_histogram")
        if macd_hist is not None:
            sig = "bullish" if macd_hist > 0 else "bearish"
            signals.append({"name": "MACD Histogram", "value": f"{macd_hist:.4f}", "signal": sig})

        # 7. Stochastic
        stoch_k = indicators.get("stoch_k")
        stoch_d = indicators.get("stoch_d")
        if stoch_k is not None and stoch_d is not None:
            if stoch_k > stoch_d and stoch_k < 80:
                sig = "bullish"
            elif stoch_k < stoch_d or stoch_k > 80:
                sig = "bearish"
            else:
                sig = "neutral"
            signals.append({"name": "Stochastic %K/%D", "value": f"{stoch_k:.1f}/{stoch_d:.1f}", "signal": sig})

        # 8. Bollinger %B
        bb_pct_b = indicators.get("bb_pct_b")
        if bb_pct_b is not None:
            if bb_pct_b < 0.2:
                sig = "bullish"  # oversold
                val = f"{bb_pct_b:.3f} (Oversold)"
            elif bb_pct_b > 0.8:
                sig = "bearish"  # overbought
                val = f"{bb_pct_b:.3f} (Overbought)"
            else:
                sig = "neutral"
                val = f"{bb_pct_b:.3f}"
            signals.append({"name": "Bollinger %B", "value": val, "signal": sig})

        # 9. Volume Ratio
        vol_ratio = indicators.get("volume_ratio")
        if vol_ratio is not None:
            if vol_ratio > 1.0:
                sig = "bullish"
            elif vol_ratio < 0.7:
                sig = "bearish"
            else:
                sig = "neutral"
            signals.append({"name": "Volume Ratio", "value": f"{vol_ratio:.2f}x", "signal": sig})

        # 10. ADX Trend Strength
        adx = indicators.get("adx_14")
        if adx is not None:
            if adx > 25:
                sig = "bullish"  # strong trend exists
                val = f"{adx:.1f} (Strong)"
            else:
                sig = "neutral"
                val = f"{adx:.1f} (Weak)"
            signals.append({"name": "ADX Trend", "value": val, "signal": sig})

        # 11. Near Support
        nearest_support = sr.get("nearest_support") if sr else None
        if nearest_support and price:
            dist = (price - nearest_support) / price
            if dist < 0.02 and dist > 0:
                sig = "bullish"  # near support = potential bounce
                val = f"${nearest_support:.2f} ({dist:.1%} away)"
            elif dist < 0:
                sig = "bearish"  # below support = broken
                val = f"${nearest_support:.2f} (Broken)"
            else:
                sig = "neutral"
                val = f"${nearest_support:.2f}"
            signals.append({"name": "Support Level", "value": val, "signal": sig})

        # 12. Price vs VWAP
        vwap = indicators.get("vwap")
        if vwap is not None:
            sig = "bullish" if price > vwap else "bearish"
            signals.append({"name": "Price vs VWAP", "value": f"${vwap:.2f}", "signal": sig})

        # --- Aggregate ---
        bullish = sum(1 for s in signals if s["signal"] == "bullish")
        bearish = sum(1 for s in signals if s["signal"] == "bearish")
        neutral = sum(1 for s in signals if s["signal"] == "neutral")
        total = len(signals)

        active = bullish + bearish
        score_pct = (bullish / active * 100) if active > 0 else 50.0

        if score_pct > 75:
            verdict = "Strong Buy"
        elif score_pct > 60:
            verdict = "Buy"
        elif score_pct > 40:
            verdict = "Neutral"
        elif score_pct > 25:
            verdict = "Sell"
        else:
            verdict = "Strong Sell"

        return {
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "total_signals": total,
            "score_pct": round(score_pct, 1),
            "verdict": verdict,
            "signals": signals,
        }
