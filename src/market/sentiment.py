"""Geopolitical sentiment analysis via Claude LLM."""

from __future__ import annotations

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from src.models.schemas import SentimentScores


class SentimentAnalyzer:
    """Use Claude to assess geopolitical/news sentiment for each asset."""

    def __init__(self, llm: ChatAnthropic | None = None):
        self.llm = llm or ChatAnthropic(
            model="claude-sonnet-4-20250514", temperature=0.0
        )

    def analyze(
        self,
        tickers: list[str],
        concerns: list[str],
    ) -> SentimentScores:
        """
        Ask Claude to evaluate current geopolitical sentiment for each ticker.
        Returns scores in [-1.0, 1.0].
        """
        concerns_text = ", ".join(concerns) if concerns else "general market conditions"

        prompt = f"""You are a geopolitical risk analyst. For each of the following
tickers, assess the current sentiment impact considering: {concerns_text}.

Tickers: {', '.join(tickers)}

For each ticker, provide:
1. A sentiment score from -1.0 (very negative) to 1.0 (very positive)
2. A one-line rationale

Also provide an overall market sentiment score.

Respond in this exact JSON format with no other text:
{{
  "scores": {{"TICKER": 0.0}},
  "rationale": {{"TICKER": "one-line reason"}},
  "overall_market_sentiment": 0.0,
  "source_summary": "brief summary of key factors considered"
}}"""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(content)
            return SentimentScores(**data)
        except Exception:
            return SentimentScores(
                scores={t: 0.0 for t in tickers},
                rationale={t: "Unable to assess" for t in tickers},
                overall_market_sentiment=0.0,
                source_summary="Sentiment analysis unavailable; using neutral.",
            )
