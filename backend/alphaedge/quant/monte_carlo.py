"""Monte Carlo simulation with percentile fan charts and probability targets."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """GBM Monte Carlo simulation returning percentile paths for visualization."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def simulate(
        self,
        prices: pd.Series,
        horizon_days: int = 252,
        n_paths: int = 2000,
        targets: list[float] | None = None,
    ) -> dict:
        """Run Monte Carlo simulation and return percentile paths + probability analysis.

        Parameters
        ----------
        prices : historical price series
        horizon_days : forecast horizon in trading days
        n_paths : number of simulation paths
        targets : list of price levels to compute probabilities for
        """
        if len(prices) < 30:
            return {"error": "Insufficient data"}

        log_returns = np.log(prices / prices.shift(1)).dropna()
        current_price = float(prices.iloc[-1])
        mu = float(log_returns.mean())
        sigma = float(log_returns.std())

        rng = np.random.default_rng(self.seed)

        # Generate paths
        dt = 1.0  # daily
        Z = rng.standard_normal((n_paths, horizon_days))
        daily_log_ret = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        cum_log_ret = np.cumsum(daily_log_ret, axis=1)
        paths = current_price * np.exp(cum_log_ret)

        # Insert current price as day 0
        paths = np.column_stack([np.full(n_paths, current_price), paths])

        # ── Percentile fan (for chart) ──
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        fan = {}
        for p in percentiles:
            pvals = np.percentile(paths, p, axis=0)
            # Downsample to ~100 points for JSON
            n_pts = paths.shape[1]
            if n_pts > 100:
                step = max(1, n_pts // 100)
                indices = list(range(0, n_pts, step))
                if indices[-1] != n_pts - 1:
                    indices.append(n_pts - 1)
            else:
                indices = list(range(n_pts))

            fan[f"p{p}"] = [
                {"day": int(i), "price": round(float(pvals[i]), 2)}
                for i in indices
            ]

        # ── Terminal distribution ──
        terminal_prices = paths[:, -1]
        terminal_returns = (terminal_prices / current_price - 1) * 100

        terminal_stats = {
            "mean_price": round(float(np.mean(terminal_prices)), 2),
            "median_price": round(float(np.median(terminal_prices)), 2),
            "std_price": round(float(np.std(terminal_prices)), 2),
            "mean_return_pct": round(float(np.mean(terminal_returns)), 2),
            "median_return_pct": round(float(np.median(terminal_returns)), 2),
            "prob_positive": round(float(np.mean(terminal_returns > 0)), 4),
            "prob_loss_gt_10pct": round(float(np.mean(terminal_returns < -10)), 4),
            "prob_loss_gt_20pct": round(float(np.mean(terminal_returns < -20)), 4),
            "prob_gain_gt_10pct": round(float(np.mean(terminal_returns > 10)), 4),
            "prob_gain_gt_20pct": round(float(np.mean(terminal_returns > 20)), 4),
            "worst_5pct_price": round(float(np.percentile(terminal_prices, 5)), 2),
            "best_5pct_price": round(float(np.percentile(terminal_prices, 95)), 2),
        }

        # ── Terminal return histogram ──
        hist_counts, hist_edges = np.histogram(terminal_returns, bins=40)
        terminal_histogram = [
            {
                "x": round(float((hist_edges[i] + hist_edges[i + 1]) / 2), 2),
                "count": int(hist_counts[i]),
                "pct": round(float(hist_counts[i] / n_paths * 100), 2),
            }
            for i in range(len(hist_counts))
        ]

        # ── Price targets ──
        target_probs = {}
        if targets:
            for t in targets:
                prob = float(np.mean(terminal_prices >= t))
                target_probs[str(round(t, 2))] = round(prob, 4)
        else:
            # Auto-generate targets: -20%, -10%, 0%, +10%, +20%
            for pct in [-20, -10, 0, 10, 20]:
                target = current_price * (1 + pct / 100)
                prob = float(np.mean(terminal_prices >= target))
                target_probs[f"{'+'if pct>0 else ''}{pct}%"] = round(prob, 4)

        # ── Sample paths (5 representative) for chart overlay ──
        sample_indices = np.linspace(0, n_paths - 1, 5, dtype=int)
        sample_paths = []
        for idx in sample_indices:
            path = paths[idx]
            n_pts = len(path)
            if n_pts > 100:
                step = max(1, n_pts // 100)
                indices = list(range(0, n_pts, step))
                if indices[-1] != n_pts - 1:
                    indices.append(n_pts - 1)
            else:
                indices = list(range(n_pts))

            sample_paths.append([
                {"day": int(i), "price": round(float(path[i]), 2)}
                for i in indices
            ])

        return {
            "current_price": current_price,
            "horizon_days": horizon_days,
            "n_paths": n_paths,
            "mu_daily": round(mu, 6),
            "sigma_daily": round(sigma, 6),
            "fan_chart": fan,
            "terminal_stats": terminal_stats,
            "terminal_histogram": terminal_histogram,
            "target_probabilities": target_probs,
            "sample_paths": sample_paths,
        }
