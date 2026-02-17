"""Optional LLM hook for enhanced news synthesis."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    provider: str = "openai"  # "openai", "anthropic", "ollama"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.3


class LLMHook:
    """Optional LLM integration for enhanced synthesis.

    All features degrade gracefully when no LLM is configured.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client = None
        self._available = False
        self._init_client()

    def _init_client(self):
        """Try to initialize LLM client. Silently fail if not available."""
        # Check environment for API keys
        api_key = self.config.api_key

        if self.config.provider == "openai":
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if api_key:
                try:
                    import openai

                    self._client = openai.OpenAI(
                        api_key=api_key,
                        base_url=self.config.base_url,
                    )
                    self._available = True
                    logger.info("OpenAI LLM hook initialized")
                except Exception as e:
                    logger.debug("OpenAI init failed: %s", e)

        elif self.config.provider == "anthropic":
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                try:
                    import anthropic

                    self._client = anthropic.Anthropic(api_key=api_key)
                    self._available = True
                    logger.info("Anthropic LLM hook initialized")
                except Exception as e:
                    logger.debug("Anthropic init failed: %s", e)

        elif self.config.provider == "ollama":
            base_url = self.config.base_url or "http://localhost:11434"
            try:
                import httpx

                resp = httpx.get(f"{base_url}/api/tags", timeout=3)
                if resp.status_code == 200:
                    self._client = base_url
                    self._available = True
                    logger.info("Ollama LLM hook initialized")
            except Exception as e:
                logger.debug("Ollama init failed: %s", e)

    @property
    def available(self) -> bool:
        return self._available

    def enhance_synthesis(
        self,
        ticker: str,
        articles_summary: str,
        sentiment_data: dict,
        events_data: dict,
    ) -> Optional[dict]:
        """Ask LLM for enhanced analysis. Returns None if unavailable."""
        if not self._available or self._client is None:
            return None

        prompt = (
            f"You are a financial analyst. Analyze the following news data for {ticker} "
            f"and provide a concise investment-relevant synthesis.\n\n"
            f"Sentiment: {json.dumps(sentiment_data, default=str)[:500]}\n\n"
            f"Key Events: {json.dumps(events_data, default=str)[:500]}\n\n"
            f"Articles Summary: {articles_summary[:500]}\n\n"
            f"Provide: 1) A 2-sentence enhanced narrative, 2) Top 3 key insights, "
            f"3) A brief risk assessment. Format as JSON with keys: "
            f"enhanced_narrative, key_insights (list), risk_assessment."
        )

        try:
            if self.config.provider == "openai":
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                text = response.choices[0].message.content

            elif self.config.provider == "anthropic":
                response = self._client.messages.create(
                    model=self.config.model or "claude-sonnet-4-5-20250929",
                    max_tokens=self.config.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text

            elif self.config.provider == "ollama":
                import httpx

                resp = httpx.post(
                    f"{self._client}/api/generate",
                    json={
                        "model": self.config.model or "llama3.2",
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=60,
                )
                text = resp.json().get("response", "")
            else:
                return None

            # Try to parse JSON from response
            try:
                # Find JSON in response
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    result = json.loads(text[start:end])
                    result["confidence"] = 0.7
                    return result
            except json.JSONDecodeError:
                pass

            return {
                "enhanced_narrative": text[:500],
                "key_insights": [],
                "risk_assessment": "",
                "confidence": 0.5,
            }

        except Exception as e:
            logger.warning("LLM enhancement failed: %s", e)
            return None

    def generate_question(self, ticker: str, context: dict) -> Optional[str]:
        """Generate a key question an analyst should ask. Returns None if unavailable."""
        if not self._available:
            return None

        prompt = (
            f"Given the analysis of {ticker} with the following context:\n"
            f"{json.dumps(context, default=str)[:300]}\n\n"
            f"What is the single most important question an analyst should investigate? "
            f"Respond with just the question."
        )

        try:
            if self.config.provider == "openai":
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.5,
                )
                return response.choices[0].message.content.strip()
            elif self.config.provider == "anthropic":
                response = self._client.messages.create(
                    model=self.config.model or "claude-sonnet-4-5-20250929",
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text.strip()
        except Exception as e:
            logger.debug("LLM question generation failed: %s", e)

        return None
