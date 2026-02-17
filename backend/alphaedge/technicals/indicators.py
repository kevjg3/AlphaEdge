"""Technical indicator calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Compute standard technical indicators from OHLCV data."""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        fast_ema = series.ewm(span=fast, adjust=False).mean()
        slow_ema = series.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return {"macd": macd_line, "signal": signal_line, "histogram": macd_line - signal_line}

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> dict:
        middle = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        bandwidth = (upper - lower) / middle
        pct_b = (series - lower) / (upper - lower)
        return {"upper": upper, "middle": middle, "lower": lower, "bandwidth": bandwidth, "pct_b": pct_b}

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff())
        return (volume * direction).fillna(0).cumsum()

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> dict:
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        d = k.rolling(d_period).mean()
        return {"k": k, "d": d}

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr_val = TechnicalIndicators.atr(high, low, close, period)
        plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(period).mean() / atr_val.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(period).mean() / atr_val.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.rolling(period).mean()

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        typical = (high + low + close) / 3
        cum_vol = volume.cumsum()
        return (typical * volume).cumsum() / cum_vol.replace(0, np.nan)

    @classmethod
    def compute_all(cls, df: pd.DataFrame) -> dict:
        """Compute all indicators from OHLCV DataFrame."""
        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

        rsi_val = cls.rsi(c)
        macd_data = cls.macd(c)
        bb = cls.bollinger_bands(c)
        stoch = cls.stochastic(h, l, c)

        def _last(s: pd.Series):
            if len(s) == 0:
                return None
            val = s.iloc[-1]
            return round(float(val), 6) if pd.notna(val) else None

        sma20 = _last(cls.sma(c, 20))
        sma50 = _last(cls.sma(c, 50))
        sma200 = _last(cls.sma(c, 200))
        price = _last(c)
        vol_sma20 = _last(cls.sma(v, 20))

        return {
            "price": price,
            "sma_20": sma20,
            "sma_50": sma50,
            "sma_200": sma200,
            "ema_12": _last(cls.ema(c, 12)),
            "ema_26": _last(cls.ema(c, 26)),
            "rsi_14": _last(rsi_val),
            "macd": _last(macd_data["macd"]),
            "macd_signal": _last(macd_data["signal"]),
            "macd_histogram": _last(macd_data["histogram"]),
            "bb_upper": _last(bb["upper"]),
            "bb_lower": _last(bb["lower"]),
            "bb_bandwidth": _last(bb["bandwidth"]),
            "bb_pct_b": _last(bb["pct_b"]),
            "atr_14": _last(cls.atr(h, l, c, 14)),
            "adx_14": _last(cls.adx(h, l, c, 14)),
            "stoch_k": _last(stoch["k"]),
            "stoch_d": _last(stoch["d"]),
            "obv": _last(cls.obv(c, v)),
            "vwap": _last(cls.vwap(h, l, c, v)),
            "volume_sma_20": vol_sma20,
            "volume_ratio": round(_last(v) / vol_sma20, 4) if _last(v) and vol_sma20 else None,
            "above_sma_20": price > sma20 if price and sma20 else None,
            "above_sma_50": price > sma50 if price and sma50 else None,
            "above_sma_200": price > sma200 if price and sma200 else None,
            "golden_cross": sma50 > sma200 if sma50 and sma200 else None,
        }
