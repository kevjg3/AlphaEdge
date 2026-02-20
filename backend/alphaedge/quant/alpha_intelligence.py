"""Alpha intelligence: liquidity analysis, drawdown duration, and feature importance."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class AlphaIntelligenceResult:
    """Result of alpha intelligence analysis."""

    # --- Liquidity ---
    amihud_illiquidity: float = 0.0
    avg_daily_volume: float = 0.0
    volume_trend_pct: float = 0.0  # 21d avg vs 63d avg change
    relative_volume: float = 0.0  # recent 5d avg / 63d avg
    bid_ask_spread_proxy: float = 0.0  # from OHLC
    liquidity_score: str = "medium"  # "high" | "medium" | "low"

    # --- Drawdown Duration ---
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    avg_recovery_time_days: float = 0.0
    current_drawdown_pct: float = 0.0
    current_drawdown_days: int = 0
    n_drawdowns_gt_5pct: int = 0
    drawdown_periods: list[dict] = field(default_factory=list)  # top 5 deepest

    # --- Feature Importance ---
    feature_importance: list[dict] = field(default_factory=list)
    model_r2: float | None = None

    def to_dict(self) -> dict:
        return {
            "liquidity": {
                "amihud_illiquidity": round(self.amihud_illiquidity, 8),
                "avg_daily_volume": round(self.avg_daily_volume, 0),
                "volume_trend_pct": round(self.volume_trend_pct, 4),
                "relative_volume": round(self.relative_volume, 4),
                "bid_ask_spread_proxy": round(self.bid_ask_spread_proxy, 6),
                "liquidity_score": self.liquidity_score,
            },
            "drawdown_duration": {
                "max_drawdown_pct": round(self.max_drawdown_pct, 4),
                "max_drawdown_duration_days": self.max_drawdown_duration_days,
                "avg_recovery_time_days": round(self.avg_recovery_time_days, 1),
                "current_drawdown_pct": round(self.current_drawdown_pct, 4),
                "current_drawdown_days": self.current_drawdown_days,
                "n_drawdowns_gt_5pct": self.n_drawdowns_gt_5pct,
                "drawdown_periods": self.drawdown_periods[:5],
            },
            "feature_importance": {
                "features": self.feature_importance,
                "model_r2": round(self.model_r2, 4) if self.model_r2 is not None else None,
            },
        }


class AlphaIntelligenceAnalyzer:
    """Liquidity analysis, drawdown duration, and XGBoost feature importance."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def analyze(
        self, prices: pd.Series, hist_df: pd.DataFrame
    ) -> AlphaIntelligenceResult:
        """Run all alpha intelligence analyses."""
        if len(prices) < 100:
            return AlphaIntelligenceResult()

        result = AlphaIntelligenceResult()

        # --- Liquidity ---
        try:
            self._liquidity(result, prices, hist_df)
        except Exception as e:
            logger.warning("Liquidity analysis failed: %s", e)

        # --- Drawdown Duration ---
        try:
            self._drawdown_duration(result, prices)
        except Exception as e:
            logger.warning("Drawdown duration analysis failed: %s", e)

        # --- Feature Importance ---
        try:
            self._feature_importance(result, prices, hist_df)
        except Exception as e:
            logger.warning("Feature importance analysis failed: %s", e)

        return result

    # ── Liquidity ──────────────────────────────────────────────────────

    @staticmethod
    def _liquidity(
        result: AlphaIntelligenceResult,
        prices: pd.Series,
        hist_df: pd.DataFrame,
    ) -> None:
        """Compute liquidity metrics from OHLCV data."""
        daily_ret = prices.pct_change().dropna()

        # Amihud illiquidity ratio
        if "Volume" in hist_df.columns:
            volume = hist_df["Volume"].reindex(daily_ret.index)
            dollar_volume = prices.reindex(daily_ret.index) * volume
            # Avoid division by zero
            mask = dollar_volume > 0
            if mask.sum() > 0:
                amihud = (daily_ret.abs()[mask] / dollar_volume[mask]).mean()
                result.amihud_illiquidity = float(amihud) if np.isfinite(amihud) else 0.0

            # Volume metrics
            vol_series = hist_df["Volume"].dropna()
            if len(vol_series) > 0:
                result.avg_daily_volume = float(vol_series.mean())

                if len(vol_series) >= 63:
                    avg_21 = vol_series.iloc[-21:].mean()
                    avg_63 = vol_series.iloc[-63:].mean()
                    if avg_63 > 0:
                        result.volume_trend_pct = float(avg_21 / avg_63 - 1)

                if len(vol_series) >= 63:
                    avg_5 = vol_series.iloc[-5:].mean()
                    avg_63 = vol_series.iloc[-63:].mean()
                    if avg_63 > 0:
                        result.relative_volume = float(avg_5 / avg_63)

        # Bid-ask spread proxy from OHLC (High-Low range method)
        if "High" in hist_df.columns and "Low" in hist_df.columns:
            high = hist_df["High"].iloc[-21:]
            low = hist_df["Low"].iloc[-21:]
            spread = 2 * (high - low) / (high + low)
            spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
            if len(spread) > 0:
                result.bid_ask_spread_proxy = float(spread.mean())

        # Liquidity score
        if result.amihud_illiquidity > 0:
            # Low Amihud = high liquidity
            if result.amihud_illiquidity < 1e-8 and result.relative_volume > 0.8:
                result.liquidity_score = "high"
            elif result.amihud_illiquidity > 1e-6 or result.relative_volume < 0.5:
                result.liquidity_score = "low"
            else:
                result.liquidity_score = "medium"

    # ── Drawdown Duration ─────────────────────────────────────────────

    @staticmethod
    def _drawdown_duration(result: AlphaIntelligenceResult, prices: pd.Series) -> None:
        """Analyze drawdown periods: depth, duration, and recovery times."""
        running_max = prices.cummax()
        underwater = prices / running_max - 1  # negative when in drawdown

        result.max_drawdown_pct = float(underwater.min())

        # Current drawdown
        result.current_drawdown_pct = float(underwater.iloc[-1])
        if underwater.iloc[-1] < -0.001:
            # Count days since last peak
            peak_idx = running_max[running_max == running_max.iloc[-1]].index[0]
            result.current_drawdown_days = len(prices.loc[peak_idx:]) - 1

        # Identify individual drawdown periods
        in_drawdown = underwater < -0.005  # 0.5% threshold to start a drawdown
        periods = []
        start = None

        for i, (date, is_dd) in enumerate(in_drawdown.items()):
            if is_dd and start is None:
                start = date
            elif not is_dd and start is not None:
                # Drawdown ended — find trough
                segment = underwater.loc[start:date]
                trough_date = segment.idxmin()
                depth = float(segment.min())
                duration = (date - start).days if hasattr(date, "days") else i

                # Try to compute duration from index positions
                start_pos = prices.index.get_loc(start)
                end_pos = prices.index.get_loc(date)
                trough_pos = prices.index.get_loc(trough_date)
                duration = end_pos - start_pos
                recovery_days = end_pos - trough_pos

                periods.append({
                    "start": start.strftime("%Y-%m-%d") if hasattr(start, "strftime") else str(start),
                    "trough": trough_date.strftime("%Y-%m-%d") if hasattr(trough_date, "strftime") else str(trough_date),
                    "recovery": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                    "depth": round(depth, 4),
                    "duration_days": duration,
                    "recovery_days": recovery_days,
                })
                start = None

        # Handle ongoing drawdown (no recovery yet)
        if start is not None:
            segment = underwater.loc[start:]
            trough_date = segment.idxmin()
            depth = float(segment.min())
            start_pos = prices.index.get_loc(start)
            end_pos = len(prices) - 1
            duration = end_pos - start_pos

            periods.append({
                "start": start.strftime("%Y-%m-%d") if hasattr(start, "strftime") else str(start),
                "trough": trough_date.strftime("%Y-%m-%d") if hasattr(trough_date, "strftime") else str(trough_date),
                "recovery": None,
                "depth": round(depth, 4),
                "duration_days": duration,
                "recovery_days": None,
            })

        # Sort by depth (most severe first)
        periods.sort(key=lambda p: p["depth"])

        # Filter significant drawdowns (>5%)
        significant = [p for p in periods if p["depth"] < -0.05]
        result.n_drawdowns_gt_5pct = len(significant)

        # Max drawdown duration
        if periods:
            result.max_drawdown_duration_days = max(p["duration_days"] for p in periods)

        # Average recovery time (only for recovered drawdowns > 5%)
        recovery_times = [
            p["recovery_days"]
            for p in significant
            if p["recovery_days"] is not None and p["recovery_days"] > 0
        ]
        if recovery_times:
            result.avg_recovery_time_days = float(np.mean(recovery_times))

        result.drawdown_periods = periods[:5]

    # ── Feature Importance ────────────────────────────────────────────

    def _feature_importance(
        self,
        result: AlphaIntelligenceResult,
        prices: pd.Series,
        hist_df: pd.DataFrame,
    ) -> None:
        """Compute feature importance for predicting forward 21-day returns using XGBoost."""
        try:
            from xgboost import XGBRegressor
            from sklearn.inspection import permutation_importance
        except ImportError:
            logger.warning("XGBoost or scikit-learn not available for feature importance")
            return

        if len(prices) < 300:
            return  # Need enough data for train/test split

        # --- Build feature matrix ---
        features = pd.DataFrame(index=prices.index)

        # Momentum features
        features["momentum_21d"] = prices.pct_change(21)
        features["momentum_63d"] = prices.pct_change(63)

        # Volatility features
        log_ret = np.log(prices / prices.shift(1))
        features["vol_21d"] = log_ret.rolling(21).std() * np.sqrt(TRADING_DAYS)
        features["vol_63d"] = log_ret.rolling(63).std() * np.sqrt(TRADING_DAYS)

        # Volume features
        if "Volume" in hist_df.columns:
            vol = hist_df["Volume"].reindex(prices.index)
            vol_mean = vol.rolling(63).mean()
            vol_std = vol.rolling(63).std()
            features["volume_zscore"] = (vol - vol_mean) / vol_std.replace(0, np.nan)

        # RSI(14)
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features["rsi_14"] = 100 - 100 / (1 + rs)

        # SMA ratio
        features["sma_ratio_5_21"] = prices.rolling(5).mean() / prices.rolling(21).mean()

        # Mean reversion proxy (price vs 63d SMA)
        features["price_vs_sma63"] = prices / prices.rolling(63).mean() - 1

        # Target: forward 21-day return
        target = prices.pct_change(21).shift(-21)

        # Align and drop NaN
        df = features.copy()
        df["target"] = target
        df = df.dropna()

        if len(df) < 200:
            return

        feature_names = [c for c in df.columns if c != "target"]
        X = df[feature_names].values
        y = df["target"].values

        # Train/test split (80/20, chronological)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if len(X_test) < 20:
            return

        # Train XGBoost
        model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=self.seed,
            verbosity=0,
        )
        model.fit(X_train, y_train)

        # R-squared on test set
        from sklearn.metrics import r2_score

        y_pred = model.predict(X_test)
        result.model_r2 = float(r2_score(y_test, y_pred))

        # Permutation importance on test set
        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=5,
            random_state=self.seed,
            scoring="r2",
        )

        # Compute direction: correlation between each feature and forward return
        correlations = {}
        for i, name in enumerate(feature_names):
            corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
            correlations[name] = "positive" if corr > 0 else "negative"

        # Build sorted importance list
        importances = []
        for i, name in enumerate(feature_names):
            imp = float(perm.importances_mean[i])
            if imp > 0:  # only include features with positive importance
                importances.append({
                    "name": self._feature_display_name(name),
                    "importance": round(imp, 6),
                    "direction": correlations.get(name, "unknown"),
                })

        importances.sort(key=lambda x: x["importance"], reverse=True)
        result.feature_importance = importances

    @staticmethod
    def _feature_display_name(col: str) -> str:
        """Convert column name to human-readable display name."""
        mapping = {
            "momentum_21d": "21-Day Momentum",
            "momentum_63d": "63-Day Momentum",
            "vol_21d": "21-Day Volatility",
            "vol_63d": "63-Day Volatility",
            "volume_zscore": "Volume Z-Score",
            "rsi_14": "RSI (14)",
            "sma_ratio_5_21": "SMA 5/21 Ratio",
            "price_vs_sma63": "Price vs SMA-63",
        }
        return mapping.get(col, col)
