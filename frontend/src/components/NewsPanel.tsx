"use client";

import { Card, Badge } from "./Card";
import { fmtNumber } from "@/lib/format";

export default function NewsPanel({ data }: { data: Record<string, any> | null }) {
  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-gray-500 animate-fade-in">
        <p className="text-sm">News analysis not available</p>
      </div>
    );
  }

  const synthesis = data.synthesis || {};
  const articles = data.articles || [];
  const sentiment = synthesis.aggregate_sentiment || {};

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Sentiment Overview */}
      <Card>
        <div className="flex flex-wrap items-center gap-3 mb-3">
          <span className="text-xs text-gray-500 uppercase tracking-wider">Sentiment</span>
          <Badge
            text={synthesis.overall_sentiment || "neutral"}
            variant={
              synthesis.overall_sentiment === "bullish" ? "positive" :
              synthesis.overall_sentiment === "bearish" ? "negative" :
              "neutral"
            }
            dot
          />
          {synthesis.news_momentum && (
            <Badge
              text={`Momentum: ${synthesis.news_momentum}`}
              variant={
                synthesis.news_momentum === "improving" ? "positive" :
                synthesis.news_momentum === "deteriorating" ? "negative" :
                "neutral"
              }
            />
          )}
          <span className="text-xs text-gray-500 font-mono tabular-nums">{synthesis.article_count} articles</span>
        </div>
        {synthesis.narrative && (
          <p className="text-sm text-gray-300 leading-relaxed">{synthesis.narrative}</p>
        )}
      </Card>

      {/* Sentiment Breakdown */}
      {sentiment.overall_score != null && (
        <Card title="Sentiment Breakdown">
          <div className="flex gap-6 items-center">
            <div className="shrink-0">
              <div className="text-xs text-gray-500 mb-1">Score</div>
              <div className={`text-3xl font-bold font-mono tabular-nums ${sentiment.overall_score > 0 ? "text-emerald-400" : sentiment.overall_score < 0 ? "text-red-400" : "text-gray-300"}`}>
                {fmtNumber(sentiment.overall_score, 3)}
              </div>
            </div>
            <div className="flex-1 min-w-0">
              <div className="w-full bg-surface-overlay rounded-full h-3 flex overflow-hidden">
                <div className="bg-emerald-500 h-full transition-all" style={{ width: `${(sentiment.positive_pct || 0) * 100}%` }} />
                <div className="bg-gray-600 h-full transition-all" style={{ width: `${(sentiment.neutral_pct || 0) * 100}%` }} />
                <div className="bg-red-500 h-full transition-all" style={{ width: `${(sentiment.negative_pct || 0) * 100}%` }} />
              </div>
              <div className="flex justify-between text-xs mt-1.5">
                <span className="text-emerald-400 font-mono tabular-nums">{((sentiment.positive_pct || 0) * 100).toFixed(0)}% pos</span>
                <span className="text-gray-500 font-mono tabular-nums">{((sentiment.neutral_pct || 0) * 100).toFixed(0)}% neu</span>
                <span className="text-red-400 font-mono tabular-nums">{((sentiment.negative_pct || 0) * 100).toFixed(0)}% neg</span>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Key Themes */}
      {synthesis.key_themes && synthesis.key_themes.length > 0 && (
        <Card title="Key Themes">
          <div className="flex gap-2 flex-wrap">
            {synthesis.key_themes.map((theme: string, i: number) => (
              <Badge key={i} text={theme.replace(/_/g, " ")} variant="info" />
            ))}
          </div>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Catalysts */}
        {synthesis.catalysts && synthesis.catalysts.length > 0 && (
          <Card title="Potential Catalysts">
            {synthesis.catalysts.map((c: any, i: number) => (
              <div key={i} className="py-2.5 border-b border-white/[0.04] last:border-0">
                <Badge text={c.event} variant="positive" size="sm" />
                <p className="text-sm text-gray-300 mt-1.5 leading-relaxed">{c.headline}</p>
              </div>
            ))}
          </Card>
        )}

        {/* Risks */}
        {synthesis.risks && synthesis.risks.length > 0 && (
          <Card title="Identified Risks">
            {synthesis.risks.map((r: any, i: number) => (
              <div key={i} className="py-2.5 border-b border-white/[0.04] last:border-0">
                <Badge text={r.event} variant="negative" size="sm" />
                <p className="text-sm text-gray-300 mt-1.5 leading-relaxed">{r.headline}</p>
              </div>
            ))}
          </Card>
        )}
      </div>

      {/* Recent Articles */}
      {articles.length > 0 && (
        <Card title={`Recent Articles (${articles.length})`}>
          <div className="space-y-1 max-h-96 overflow-y-auto fade-bottom">
            {articles.map((article: any, i: number) => (
              <div key={i} className="data-row group/article">
                <div className="min-w-0 flex-1">
                  <a
                    href={article.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-brand-400 hover:text-brand-300 transition-colors line-clamp-1"
                  >
                    {article.title}
                  </a>
                  <div className="text-xs text-gray-600 mt-0.5">{article.source}</div>
                </div>
                <svg className="w-4 h-4 text-gray-700 group-hover/article:text-brand-400 transition-colors shrink-0 ml-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" /><polyline points="15 3 21 3 21 9" /><line x1="10" y1="14" x2="21" y2="3" />
                </svg>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
