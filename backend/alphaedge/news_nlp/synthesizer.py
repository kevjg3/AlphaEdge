"""Synthesize news analysis into coherent narrative."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from alphaedge.news_nlp.sentiment import SentimentAnalyzer, AggregateSentiment
from alphaedge.news_nlp.event_detection import EventDetector, DetectedEvent

logger = logging.getLogger(__name__)


@dataclass
class NewsSynthesis:
    overall_sentiment: str  # "bullish", "bearish", "mixed", "neutral"
    sentiment_score: float  # -1 to 1
    key_themes: list[str]
    narrative: str
    catalysts: list[dict]
    risks: list[dict]
    news_momentum: str  # "improving", "deteriorating", "stable"
    article_count: int
    event_summary: dict = field(default_factory=dict)
    aggregate_sentiment: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_sentiment": self.overall_sentiment,
            "sentiment_score": round(self.sentiment_score, 4),
            "key_themes": self.key_themes,
            "narrative": self.narrative,
            "catalysts": self.catalysts,
            "risks": self.risks,
            "news_momentum": self.news_momentum,
            "article_count": self.article_count,
            "event_summary": self.event_summary,
            "aggregate_sentiment": self.aggregate_sentiment,
            "warnings": self.warnings,
        }


class NewsSynthesizer:
    """Combine sentiment analysis and event detection into actionable synthesis."""

    def __init__(self, sentiment_analyzer: SentimentAnalyzer, event_detector: EventDetector):
        self.sentiment = sentiment_analyzer
        self.events = event_detector

    def synthesize(self, articles: list, ticker: str) -> NewsSynthesis:
        """Full news synthesis pipeline."""
        warnings: list[str] = []

        if not articles:
            return NewsSynthesis(
                overall_sentiment="neutral",
                sentiment_score=0.0,
                key_themes=[],
                narrative=f"No recent news found for {ticker}.",
                catalysts=[],
                risks=[],
                news_momentum="stable",
                article_count=0,
                warnings=["No articles available for analysis"],
            )

        # 1. Sentiment analysis
        agg_sentiment = self.sentiment.analyze_articles(articles)

        # 2. Event detection
        detected_events = self.events.detect_events(articles)
        event_summary = self.events.summarize_events(detected_events)

        # 3. Key themes
        key_themes = self._extract_themes(event_summary)

        # 4. News momentum
        momentum = self._compute_momentum(articles, agg_sentiment)

        # 5. Catalysts and risks
        catalysts = self._identify_catalysts(detected_events)
        risks = self._identify_risks(detected_events)

        # 6. Narrative
        narrative = self._generate_narrative(ticker, agg_sentiment, detected_events, momentum)

        # Map sentiment to market sentiment — require strong signal for
        # bullish/bearish labels to avoid false confidence
        score = agg_sentiment.overall_score
        if score > 0.30:
            overall = "bullish"
        elif score < -0.30:
            overall = "bearish"
        elif abs(score) <= 0.10:
            overall = "neutral"
        else:
            overall = "mixed"

        return NewsSynthesis(
            overall_sentiment=overall,
            sentiment_score=score,
            key_themes=key_themes,
            narrative=narrative,
            catalysts=catalysts,
            risks=risks,
            news_momentum=momentum,
            article_count=len(articles),
            event_summary=event_summary,
            aggregate_sentiment=agg_sentiment.to_dict(),
            warnings=warnings,
        )

    def _extract_themes(self, event_summary: dict) -> list[str]:
        """Extract top themes from event types."""
        by_type = event_summary.get("by_type", {})
        if not by_type:
            return ["general_market"]

        sorted_types = sorted(by_type.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_types[:5]]

    def _compute_momentum(self, articles: list, agg: AggregateSentiment) -> str:
        """Compare recent vs older article sentiment to determine momentum."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=3)

        recent = []
        older = []
        for article in articles:
            pub = getattr(article, "published_at", None)
            if pub is None:
                recent.append(article)
                continue
            if not hasattr(pub, "tzinfo") or pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if pub >= cutoff:
                recent.append(article)
            else:
                older.append(article)

        if not recent or not older:
            return "stable"

        recent_sent = self.sentiment.analyze_articles(recent)
        older_sent = self.sentiment.analyze_articles(older)

        diff = recent_sent.overall_score - older_sent.overall_score
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "deteriorating"
        return "stable"

    def _identify_catalysts(self, events: list[DetectedEvent]) -> list[dict]:
        """Identify potential positive catalysts."""
        catalysts = []
        for e in events:
            if e.impact == "positive" and e.confidence >= 0.3:
                catalysts.append({
                    "event": e.event_type,
                    "headline": e.headline[:150],
                    "confidence": round(e.confidence, 3),
                })
        return catalysts[:5]

    def _identify_risks(self, events: list[DetectedEvent]) -> list[dict]:
        """Identify potential risks."""
        risks = []
        for e in events:
            if e.impact == "negative" and e.confidence >= 0.3:
                risks.append({
                    "event": e.event_type,
                    "headline": e.headline[:150],
                    "confidence": round(e.confidence, 3),
                })
        return risks[:5]

    def _generate_narrative(
        self,
        ticker: str,
        agg: AggregateSentiment,
        events: list[DetectedEvent],
        momentum: str,
    ) -> str:
        """Generate a 2-3 sentence narrative summary."""
        parts = []

        # Sentiment intro — aligned with the bullish/bearish thresholds above
        score = agg.overall_score
        if score > 0.30:
            parts.append(f"News sentiment for {ticker} is predominantly positive ({agg.positive_pct:.0%} positive articles).")
        elif score < -0.30:
            parts.append(f"News sentiment for {ticker} is predominantly negative ({agg.negative_pct:.0%} negative articles).")
        elif abs(score) <= 0.10:
            parts.append(f"News sentiment for {ticker} is neutral ({agg.neutral_pct:.0%} neutral articles).")
        else:
            parts.append(f"News sentiment for {ticker} is mixed ({agg.positive_pct:.0%} positive, {agg.negative_pct:.0%} negative).")

        # Key events
        if events:
            top_types = {}
            for e in events[:5]:
                top_types[e.event_type] = top_types.get(e.event_type, 0) + 1
            themes = ", ".join(t.replace("_", " ") for t in top_types.keys())
            parts.append(f"Key themes include {themes}.")

        # Momentum
        if momentum == "improving":
            parts.append("Sentiment momentum is improving with more recent coverage trending positive.")
        elif momentum == "deteriorating":
            parts.append("Sentiment momentum is deteriorating with recent coverage trending more negative.")
        else:
            parts.append("Sentiment has been relatively stable in recent coverage.")

        return " ".join(parts)
