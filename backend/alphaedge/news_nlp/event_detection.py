"""Detect significant financial events from news articles."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectedEvent:
    event_type: str
    headline: str
    confidence: float
    source: str
    date: Optional[str]
    impact: str  # "positive", "negative", "neutral", "unknown"

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "headline": self.headline[:200],
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "date": self.date,
            "impact": self.impact,
        }


class EventDetector:
    """Rule-based + pattern matching event detection from news headlines."""

    EVENT_PATTERNS: dict[str, list[str]] = {
        "earnings": [
            r"earnings", r"revenue\s+(?:beat|miss)", r"quarterly\s+results",
            r"\bEPS\b", r"beat\s+estimates", r"missed\s+expectations",
            r"profit\s+report", r"income\s+report",
        ],
        "merger_acquisition": [
            r"acqui(?:re|sition)", r"\bmerger\b", r"buyout", r"takeover",
            r"deal\s+to\s+buy", r"purchase\s+agreement",
        ],
        "executive_change": [
            r"\bCEO\b", r"\bCFO\b", r"\bCTO\b", r"appoint", r"resign",
            r"step(?:s|ped)?\s+down", r"leadership\s+change", r"board\s+of\s+directors",
        ],
        "regulatory": [
            r"\bSEC\b", r"\bFDA\b", r"\bFTC\b", r"regulat",
            r"compliance", r"investigation", r"subpoena", r"antitrust",
            r"approv(?:al|ed)", r"cleared\s+by",
        ],
        "product_launch": [
            r"launch(?:es|ed)?", r"new\s+product", r"unveil", r"introduc",
            r"releas(?:e|ed|ing)", r"rollout",
        ],
        "lawsuit": [
            r"lawsuit", r"litigation", r"(?:is\s+)?su(?:ed|ing)", r"settlement",
            r"class\s+action", r"legal\s+(?:action|battle)",
        ],
        "dividend": [
            r"dividend", r"payout", r"distribution", r"buyback",
            r"repurchase\s+program", r"share\s+repurchase",
        ],
        "guidance": [
            r"guidance", r"outlook", r"(?:raised|lowered)\s+guidance",
            r"full[\s-]year", r"revised\s+forecast",
        ],
        "analyst_rating": [
            r"upgrade[ds]?", r"downgrade[ds]?", r"price\s+target",
            r"overweight", r"underweight", r"buy\s+rating", r"sell\s+rating",
            r"analyst",
        ],
    }

    IMPACT_POSITIVE = {
        "beat", "raise", "raised", "upgrade", "upgraded", "approve", "approved",
        "growth", "record", "surge", "above", "outperform", "buy", "strong",
        "positive", "cleared", "launch", "breakout",
    }
    IMPACT_NEGATIVE = {
        "miss", "missed", "lower", "lowered", "downgrade", "downgraded",
        "reject", "rejected", "decline", "below", "warning", "cut",
        "lawsuit", "fraud", "scandal", "sell", "weak", "plunge",
    }

    def detect_events(self, articles: list) -> list[DetectedEvent]:
        """Detect events from list of NewsArticle objects."""
        events: list[DetectedEvent] = []

        for article in articles:
            title = getattr(article, "title", "") or ""
            snippet = getattr(article, "snippet", "") or ""
            combined = f"{title} {snippet}"
            source = getattr(article, "source", "") or ""
            pub = getattr(article, "published_at", None)
            date_str = pub if isinstance(pub, str) else (pub.isoformat() if pub else None)

            best_type = None
            best_score = 0.0

            for event_type, patterns in self.EVENT_PATTERNS.items():
                match_count = 0
                for pattern in patterns:
                    if re.search(pattern, combined, re.IGNORECASE):
                        match_count += 1

                if match_count > 0:
                    score = min(match_count / 3.0, 1.0)
                    if score > best_score:
                        best_score = score
                        best_type = event_type

            if best_type is not None:
                impact = self._determine_impact(combined)
                events.append(DetectedEvent(
                    event_type=best_type,
                    headline=title,
                    confidence=best_score,
                    source=source,
                    date=date_str,
                    impact=impact,
                ))

        events.sort(key=lambda e: e.confidence, reverse=True)
        return events

    def _determine_impact(self, text: str) -> str:
        tokens = set(text.lower().split())
        pos = len(tokens & self.IMPACT_POSITIVE)
        neg = len(tokens & self.IMPACT_NEGATIVE)
        if pos > neg:
            return "positive"
        elif neg > pos:
            return "negative"
        elif pos == neg and pos > 0:
            return "neutral"
        return "unknown"

    def summarize_events(self, events: list[DetectedEvent]) -> dict:
        """Summarize detected events."""
        if not events:
            return {
                "total_events": 0,
                "by_type": {},
                "by_impact": {},
                "key_events": [],
                "event_timeline": [],
            }

        by_type: dict[str, int] = {}
        by_impact: dict[str, int] = {}
        for e in events:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
            by_impact[e.impact] = by_impact.get(e.impact, 0) + 1

        key_events = [e.to_dict() for e in events[:5]]

        timeline = sorted(
            [e.to_dict() for e in events if e.date],
            key=lambda x: x["date"],
            reverse=True,
        )

        return {
            "total_events": len(events),
            "by_type": by_type,
            "by_impact": by_impact,
            "key_events": key_events,
            "event_timeline": timeline[:10],
        }
