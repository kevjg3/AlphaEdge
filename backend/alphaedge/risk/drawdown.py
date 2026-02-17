"""Drawdown analysis: max drawdown, top-N drawdowns, recovery times."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DrawdownEvent:
    """A single drawdown event."""
    start_date: datetime
    trough_date: datetime
    end_date: Optional[datetime]  # None if still in drawdown
    magnitude: float  # peak-to-trough percentage (negative)
    duration_days: int  # start to trough
    recovery_days: Optional[int]  # trough to end (None if not recovered)
    peak_price: float
    trough_price: float

    def to_dict(self) -> dict:
        return {
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "trough_date": self.trough_date.isoformat() if self.trough_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "magnitude": round(self.magnitude, 4),
            "duration_days": self.duration_days,
            "recovery_days": self.recovery_days,
            "peak_price": round(self.peak_price, 2),
            "trough_price": round(self.trough_price, 2),
        }


@dataclass
class DrawdownStats:
    """Full drawdown analysis results."""
    max_drawdown: float
    max_drawdown_start: Optional[datetime] = None
    max_drawdown_trough: Optional[datetime] = None
    max_drawdown_end: Optional[datetime] = None
    current_drawdown: float = 0.0
    current_drawdown_start: Optional[datetime] = None
    n_drawdowns: int = 0
    top_drawdowns: list[DrawdownEvent] = field(default_factory=list)
    average_recovery_days: Optional[float] = None
    underwater_series: Optional[pd.Series] = None

    def to_dict(self) -> dict:
        return {
            "max_drawdown": round(self.max_drawdown, 4),
            "max_drawdown_start": self.max_drawdown_start.isoformat() if self.max_drawdown_start else None,
            "max_drawdown_trough": self.max_drawdown_trough.isoformat() if self.max_drawdown_trough else None,
            "max_drawdown_end": self.max_drawdown_end.isoformat() if self.max_drawdown_end else None,
            "current_drawdown": round(self.current_drawdown, 4),
            "n_drawdowns": self.n_drawdowns,
            "top_drawdowns": [d.to_dict() for d in self.top_drawdowns],
            "average_recovery_days": round(self.average_recovery_days, 1) if self.average_recovery_days else None,
        }


class DrawdownAnalyzer:
    """Analyze drawdowns in a price series."""

    def analyze(self, prices: pd.Series, n_worst: int = 5, threshold: float = 0.05) -> DrawdownStats:
        """Full drawdown analysis.

        Parameters
        ----------
        prices : price series (index should be datetime-like)
        n_worst : number of worst drawdowns to return
        threshold : minimum drawdown magnitude to count as an event
        """
        if len(prices) < 2:
            return DrawdownStats(max_drawdown=0.0)

        # Compute running maximum and drawdown series
        running_max = prices.expanding().max()
        underwater = (prices - running_max) / running_max  # negative values

        # Current drawdown
        current_dd = float(underwater.iloc[-1])

        # Find drawdown events
        events = self._find_events(prices, underwater, threshold)

        # Sort by magnitude (most negative first)
        events.sort(key=lambda e: e.magnitude)
        top_events = events[:n_worst]

        # Max drawdown
        max_dd = float(underwater.min())
        max_dd_idx = underwater.idxmin()

        # Find start and end of max drawdown
        max_dd_start = None
        max_dd_end = None
        if not events:
            max_dd_start = None
        else:
            worst = events[0] if events else None
            if worst:
                max_dd_start = worst.start_date
                max_dd_end = worst.end_date

        # Average recovery time
        recovery_times = [e.recovery_days for e in events if e.recovery_days is not None]
        avg_recovery = float(np.mean(recovery_times)) if recovery_times else None

        # Current drawdown start
        current_dd_start = None
        if current_dd < -0.001:
            # Walk backwards to find when current drawdown started
            peak_idx = prices.loc[:prices.index[-1]].idxmax()
            current_dd_start = peak_idx

        return DrawdownStats(
            max_drawdown=max_dd,
            max_drawdown_start=max_dd_start,
            max_drawdown_trough=max_dd_idx if isinstance(max_dd_idx, datetime) else None,
            max_drawdown_end=max_dd_end,
            current_drawdown=current_dd,
            current_drawdown_start=current_dd_start,
            n_drawdowns=len(events),
            top_drawdowns=top_events,
            average_recovery_days=avg_recovery,
            underwater_series=underwater,
        )

    def _find_events(
        self, prices: pd.Series, underwater: pd.Series, threshold: float,
    ) -> list[DrawdownEvent]:
        """Identify distinct drawdown events."""
        events: list[DrawdownEvent] = []
        in_drawdown = False
        peak_date = None
        peak_price = 0.0
        trough_date = None
        trough_price = float("inf")
        trough_dd = 0.0

        for i, (date, dd) in enumerate(underwater.items()):
            price = prices.iloc[i]

            if dd < -threshold and not in_drawdown:
                # Start of new drawdown
                in_drawdown = True
                # Peak was the most recent high
                recent = prices.iloc[:i + 1]
                peak_idx = recent.idxmax()
                peak_date = peak_idx
                peak_price = float(recent.max())
                trough_date = date
                trough_price = price
                trough_dd = dd

            elif in_drawdown:
                if dd < trough_dd:
                    # New trough
                    trough_date = date
                    trough_price = price
                    trough_dd = dd

                if dd >= -0.001:
                    # Recovery — drawdown event complete
                    if peak_date is not None and trough_date is not None:
                        duration = (trough_date - peak_date).days if hasattr(trough_date, 'days') else 0
                        try:
                            duration = (pd.Timestamp(trough_date) - pd.Timestamp(peak_date)).days
                        except Exception:
                            duration = 0
                        try:
                            recovery = (pd.Timestamp(date) - pd.Timestamp(trough_date)).days
                        except Exception:
                            recovery = 0

                        events.append(DrawdownEvent(
                            start_date=peak_date,
                            trough_date=trough_date,
                            end_date=date,
                            magnitude=trough_dd,
                            duration_days=max(duration, 0),
                            recovery_days=max(recovery, 0),
                            peak_price=peak_price,
                            trough_price=trough_price,
                        ))

                    in_drawdown = False
                    trough_dd = 0.0

        # If still in drawdown at end
        if in_drawdown and peak_date is not None and trough_date is not None:
            try:
                duration = (pd.Timestamp(trough_date) - pd.Timestamp(peak_date)).days
            except Exception:
                duration = 0
            events.append(DrawdownEvent(
                start_date=peak_date,
                trough_date=trough_date,
                end_date=None,
                magnitude=trough_dd,
                duration_days=max(duration, 0),
                recovery_days=None,
                peak_price=peak_price,
                trough_price=trough_price,
            ))

        return events
