"""Financial sentiment analysis using FinBERT with keyword fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    text: str
    positive: float
    negative: float
    neutral: float
    label: str  # "positive", "negative", "neutral"
    confidence: float

    def to_dict(self) -> dict:
        return {
            "text": self.text[:200],
            "positive": round(self.positive, 4),
            "negative": round(self.negative, 4),
            "neutral": round(self.neutral, 4),
            "label": self.label,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class AggregateSentiment:
    overall_label: str
    overall_score: float  # -1 to 1
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    num_articles: int
    article_sentiments: list[SentimentScore] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_label": self.overall_label,
            "overall_score": round(self.overall_score, 4),
            "positive_pct": round(self.positive_pct, 4),
            "negative_pct": round(self.negative_pct, 4),
            "neutral_pct": round(self.neutral_pct, 4),
            "num_articles": self.num_articles,
            "article_sentiments": [s.to_dict() for s in self.article_sentiments[:20]],
            "warnings": self.warnings,
        }


class SentimentAnalyzer:
    """Lazy-loaded FinBERT for financial sentiment with keyword fallback."""

    MODEL_NAME = "ProsusAI/finbert"
    _pipeline = None

    def __init__(self):
        pass

    @classmethod
    def _load_model(cls):
        if cls._pipeline is not None:
            return
        try:
            from transformers import pipeline

            cls._pipeline = pipeline(
                "sentiment-analysis",
                model=cls.MODEL_NAME,
                tokenizer=cls.MODEL_NAME,
                top_k=None,
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning("Failed to load FinBERT: %s — using keyword fallback", e)
            cls._pipeline = "fallback"

    def analyze_text(self, text: str) -> SentimentScore:
        """Analyze a single text."""
        self._load_model()
        if self._pipeline == "fallback" or self._pipeline is None:
            return self._keyword_fallback(text)

        try:
            results = self._pipeline(text[:512])[0]
            scores = {r["label"]: r["score"] for r in results}
            pos = scores.get("positive", 0)
            neg = scores.get("negative", 0)
            neu = scores.get("neutral", 0)
            label = max(scores, key=scores.get)
            confidence = max(pos, neg, neu)
            return SentimentScore(
                text=text, positive=pos, negative=neg,
                neutral=neu, label=label, confidence=confidence,
            )
        except Exception as e:
            logger.warning("FinBERT inference failed: %s", e)
            return self._keyword_fallback(text)

    def analyze_articles(self, articles: list) -> AggregateSentiment:
        """Analyze list of NewsArticle objects."""
        if not articles:
            return AggregateSentiment(
                overall_label="neutral",
                overall_score=0.0,
                positive_pct=0.0,
                negative_pct=0.0,
                neutral_pct=1.0,
                num_articles=0,
                warnings=["No articles to analyze"],
            )

        sentiments: list[SentimentScore] = []
        for article in articles:
            text = getattr(article, "title", "") or ""
            snippet = getattr(article, "snippet", "") or ""
            combined = f"{text}. {snippet}".strip()
            if combined:
                sentiments.append(self.analyze_text(combined))

        if not sentiments:
            return AggregateSentiment(
                overall_label="neutral",
                overall_score=0.0,
                positive_pct=0.0,
                negative_pct=0.0,
                neutral_pct=1.0,
                num_articles=0,
                warnings=["No analyzable text in articles"],
            )

        pos_count = sum(1 for s in sentiments if s.label == "positive")
        neg_count = sum(1 for s in sentiments if s.label == "negative")
        neu_count = sum(1 for s in sentiments if s.label == "neutral")
        n = len(sentiments)

        # Weighted score: use confidence-weighted average for more robust aggregation
        total_conf = sum(s.confidence for s in sentiments) or 1.0
        avg_score = sum((s.positive - s.negative) * s.confidence for s in sentiments) / total_conf

        # Require clear majority for non-neutral label (threshold 0.25)
        if avg_score > 0.25:
            overall_label = "positive"
        elif avg_score < -0.25:
            overall_label = "negative"
        else:
            overall_label = "neutral"

        return AggregateSentiment(
            overall_label=overall_label,
            overall_score=avg_score,
            positive_pct=pos_count / n,
            negative_pct=neg_count / n,
            neutral_pct=neu_count / n,
            num_articles=n,
            article_sentiments=sentiments,
        )

    @staticmethod
    def _keyword_fallback(text: str) -> SentimentScore:
        # Strong positive words only (removed generic "growth", "strong", "record"
        # which appear in neutral financial reporting)
        positive_words = {
            "surge", "beat", "upgrade", "bullish", "rally", "outperform",
            "buy", "exceed", "soar", "breakout", "upbeat", "optimistic",
        }
        # Expanded negative words that commonly signal bearish financial sentiment
        negative_words = {
            "crash", "loss", "miss", "downgrade", "bearish", "decline", "weak",
            "fall", "drop", "underperform", "sell", "plunge", "slump",
            "cut", "layoff", "bankruptcy", "fraud", "scandal", "recession",
            "warning", "concern", "fear", "uncertainty", "disappointing",
            "slowdown", "contraction", "default", "overvalued", "debt",
        }
        tokens = set(text.lower().split())
        pos_count = len(tokens & positive_words)
        neg_count = len(tokens & negative_words)
        total = pos_count + neg_count
        if total == 0:
            # No sentiment keywords found — return neutral
            return SentimentScore(
                text=text, positive=0.0, negative=0.0,
                neutral=1.0, label="neutral", confidence=0.5,
            )
        pos_score = pos_count / total
        neg_score = neg_count / total
        neu_score = max(0, 1 - pos_score - neg_score)
        # Require clear margin for non-neutral label
        if pos_score > neg_score + 0.15:
            label = "positive"
        elif neg_score > pos_score + 0.15:
            label = "negative"
        else:
            label = "neutral"
        return SentimentScore(
            text=text, positive=pos_score, negative=neg_score,
            neutral=neu_score, label=label,
            confidence=max(pos_score, neg_score, neu_score),
        )
