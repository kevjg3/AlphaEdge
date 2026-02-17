"""Upcoming risk events: earnings, ex-dividend, economic dates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import pandas as pd


@dataclass
class CalendarEvent:
    """A single calendar event."""
    date: datetime
    event_type: str  # "earnings", "ex_dividend", "options_expiry"
    description: str
    importance: str = "medium"  # low, medium, high

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat() if self.date else None,
            "event_type": self.event_type,
            "description": self.description,
            "importance": self.importance,
        }


class EventCalendar:
    """Fetch upcoming risk events for a ticker."""

    def __init__(self, yf_source=None):
        self._yf_source = yf_source

    def get_upcoming_events(
        self, ticker: str, days_ahead: int = 60,
    ) -> list[CalendarEvent]:
        """Get upcoming earnings, dividends, and other events."""
        events: list[CalendarEvent] = []
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days_ahead)

        if not self._yf_source:
            return events

        # Earnings dates
        earnings_res = self._yf_source.get_earnings_dates(ticker)
        if earnings_res.success and not earnings_res.data.empty:
            for date_idx in earnings_res.data.index:
                try:
                    dt = pd.Timestamp(date_idx).to_pydatetime()
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if now <= dt <= cutoff:
                        events.append(CalendarEvent(
                            date=dt,
                            event_type="earnings",
                            description=f"{ticker} earnings report",
                            importance="high",
                        ))
                except Exception:
                    continue

        # Calendar (ex-dividend, etc.)
        cal_res = self._yf_source.get_calendar(ticker)
        if cal_res.success and cal_res.data:
            cal = cal_res.data
            if isinstance(cal, dict):
                for key in ["Ex-Dividend Date", "Dividend Date"]:
                    if key in cal:
                        try:
                            dt = pd.Timestamp(cal[key]).to_pydatetime()
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            if now <= dt <= cutoff:
                                events.append(CalendarEvent(
                                    date=dt,
                                    event_type="ex_dividend",
                                    description=f"{ticker} {key.lower()}",
                                    importance="medium",
                                ))
                        except Exception:
                            continue

        # Monthly options expiry (third Friday of each month)
        for month_offset in range(3):
            exp_date = self._next_options_expiry(now, month_offset)
            if exp_date <= cutoff:
                events.append(CalendarEvent(
                    date=exp_date,
                    event_type="options_expiry",
                    description="Monthly options expiration",
                    importance="low",
                ))

        events.sort(key=lambda e: e.date)
        return events

    @staticmethod
    def _next_options_expiry(from_date: datetime, month_offset: int = 0) -> datetime:
        """Calculate third Friday of a given month."""
        import calendar
        year = from_date.year
        month = from_date.month + month_offset
        while month > 12:
            month -= 12
            year += 1

        cal = calendar.monthcalendar(year, month)
        # Third Friday: find all Fridays (index 4), take third one
        fridays = [week[4] for week in cal if week[4] != 0]
        third_friday = fridays[2] if len(fridays) >= 3 else fridays[-1]
        return datetime(year, month, third_friday, 16, 0, tzinfo=timezone.utc)
