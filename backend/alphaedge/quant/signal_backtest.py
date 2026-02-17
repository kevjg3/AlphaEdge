"""Simple backtesting of technical signals: RSI, MACD crossover, golden cross."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """Result of a single signal backtest."""
    signal_name: str
    total_signals: int
    win_rate: float
    avg_return_1d: float
    avg_return_5d: float
    avg_return_21d: float
    best_return_5d: float
    worst_return_5d: float
    avg_return_win: float
    avg_return_loss: float
    profit_factor: float
    signal_dates: list[dict] = field(default_factory=list)  # last 10 signals

    def to_dict(self) -> dict:
        return {
            "signal_name": self.signal_name,
            "total_signals": self.total_signals,
            "win_rate": round(self.win_rate, 4),
            "avg_return_1d": round(self.avg_return_1d, 4),
            "avg_return_5d": round(self.avg_return_5d, 4),
            "avg_return_21d": round(self.avg_return_21d, 4),
            "best_return_5d": round(self.best_return_5d, 4),
            "worst_return_5d": round(self.worst_return_5d, 4),
            "avg_return_win": round(self.avg_return_win, 4),
            "avg_return_loss": round(self.avg_return_loss, 4),
            "profit_factor": round(self.profit_factor, 4),
            "signal_dates": self.signal_dates[-10:],
        }


class SignalBacktester:
    """Backtest simple technical signals on historical data."""

    def backtest_all(self, hist_df: pd.DataFrame) -> dict:
        """Run all signal backtests and return results."""
        if len(hist_df) < 60:
            return {"error": "Insufficient data for backtesting"}

        close = hist_df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        results = {}

        # RSI oversold (< 30) → buy signal
        try:
            rsi = self._rsi(close, 14)
            rsi_buy = rsi < 30
            r = self._evaluate_signal(close, rsi_buy, "RSI Oversold (<30) Buy")
            if r:
                results["rsi_oversold"] = r.to_dict()
        except Exception as e:
            logger.debug("RSI backtest failed: %s", e)

        # RSI overbought (> 70) → sell signal (measure negative return)
        try:
            rsi_sell = rsi > 70
            r = self._evaluate_signal(close, rsi_sell, "RSI Overbought (>70) Sell", invert=True)
            if r:
                results["rsi_overbought"] = r.to_dict()
        except Exception as e:
            logger.debug("RSI overbought backtest failed: %s", e)

        # MACD bullish crossover
        try:
            fast_ema = close.ewm(span=12, adjust=False).mean()
            slow_ema = close.ewm(span=26, adjust=False).mean()
            macd = fast_ema - slow_ema
            signal_line = macd.ewm(span=9, adjust=False).mean()
            macd_cross = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
            r = self._evaluate_signal(close, macd_cross, "MACD Bullish Crossover")
            if r:
                results["macd_bullish"] = r.to_dict()
        except Exception as e:
            logger.debug("MACD backtest failed: %s", e)

        # MACD bearish crossover
        try:
            macd_bear = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
            r = self._evaluate_signal(close, macd_bear, "MACD Bearish Crossover", invert=True)
            if r:
                results["macd_bearish"] = r.to_dict()
        except Exception as e:
            logger.debug("MACD bearish backtest failed: %s", e)

        # Golden Cross (SMA 50 crosses above SMA 200)
        try:
            sma50 = close.rolling(50).mean()
            sma200 = close.rolling(200).mean()
            golden = (sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1))
            r = self._evaluate_signal(close, golden, "Golden Cross (50/200)")
            if r:
                results["golden_cross"] = r.to_dict()
        except Exception as e:
            logger.debug("Golden cross backtest failed: %s", e)

        # Death Cross
        try:
            death = (sma50 < sma200) & (sma50.shift(1) >= sma200.shift(1))
            r = self._evaluate_signal(close, death, "Death Cross (50/200)", invert=True)
            if r:
                results["death_cross"] = r.to_dict()
        except Exception as e:
            logger.debug("Death cross backtest failed: %s", e)

        # Bollinger Band squeeze (bandwidth < 10th percentile) → volatility expansion
        try:
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            bb_width = (2 * std20 / sma20) * 100
            threshold = bb_width.quantile(0.10)
            squeeze = (bb_width < threshold) & (bb_width.shift(1) >= threshold)
            r = self._evaluate_signal(close, squeeze, "Bollinger Squeeze")
            if r:
                results["bb_squeeze"] = r.to_dict()
        except Exception as e:
            logger.debug("BB squeeze backtest failed: %s", e)

        # Mean Reversion: price drops > 2 std below 20-day SMA
        try:
            z_score = (close - sma20) / std20
            mean_rev = z_score < -2
            r = self._evaluate_signal(close, mean_rev, "Mean Reversion (Z < -2)")
            if r:
                results["mean_reversion"] = r.to_dict()
        except Exception as e:
            logger.debug("Mean reversion backtest failed: %s", e)

        return results

    def _evaluate_signal(
        self,
        close: pd.Series,
        signals: pd.Series,
        name: str,
        invert: bool = False,
    ) -> SignalResult | None:
        """Evaluate forward returns after signal dates."""
        signal_idx = signals[signals].index
        if len(signal_idx) < 3:
            return None

        returns_1d = []
        returns_5d = []
        returns_21d = []
        signal_dates = []

        for date in signal_idx:
            pos = close.index.get_loc(date)
            if isinstance(pos, slice):
                pos = pos.start

            # 1-day forward return
            if pos + 1 < len(close):
                r1 = float(close.iloc[pos + 1] / close.iloc[pos] - 1)
                returns_1d.append(-r1 if invert else r1)

            # 5-day forward return
            if pos + 5 < len(close):
                r5 = float(close.iloc[pos + 5] / close.iloc[pos] - 1)
                returns_5d.append(-r5 if invert else r5)

            # 21-day forward return
            if pos + 21 < len(close):
                r21 = float(close.iloc[pos + 21] / close.iloc[pos] - 1)
                returns_21d.append(-r21 if invert else r21)

            signal_dates.append({
                "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                "price": round(float(close.iloc[pos]), 2),
            })

        if not returns_5d:
            return None

        r5 = np.array(returns_5d)
        wins = r5[r5 > 0]
        losses = r5[r5 <= 0]

        return SignalResult(
            signal_name=name,
            total_signals=len(signal_idx),
            win_rate=float(np.mean(r5 > 0)),
            avg_return_1d=float(np.mean(returns_1d)) if returns_1d else 0.0,
            avg_return_5d=float(np.mean(returns_5d)),
            avg_return_21d=float(np.mean(returns_21d)) if returns_21d else 0.0,
            best_return_5d=float(np.max(r5)),
            worst_return_5d=float(np.min(r5)),
            avg_return_win=float(np.mean(wins)) if len(wins) > 0 else 0.0,
            avg_return_loss=float(np.mean(losses)) if len(losses) > 0 else 0.0,
            profit_factor=float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else 0.0,
            signal_dates=signal_dates[-10:],
        )

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
