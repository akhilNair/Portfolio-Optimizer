"""LangGraph AgentState definition for the portfolio optimization pipeline."""

from __future__ import annotations

from operator import add
from typing import Annotated, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage

from src.models.schemas import (
    AnalystRequest,
    BlackLittermanOutput,
    HistoricalReturnsData,
    PortfolioWeights,
    RealTimeMarketSnapshot,
    SentimentScores,
    StructuredNote,
)


def replace_value(existing, new):
    """Reducer: latest value wins."""
    return new


class AgentState(TypedDict):
    # Chat history (append-only)
    messages: Annotated[Sequence[BaseMessage], add]

    # Phase tracking
    current_step: Annotated[str, replace_value]
    error: Annotated[Optional[str], replace_value]

    # Parsed analyst intent
    analyst_request: Annotated[Optional[AnalystRequest], replace_value]

    # Step 1 artifacts
    matched_notes: Annotated[Optional[list[StructuredNote]], replace_value]
    candidate_tickers: Annotated[Optional[list[str]], replace_value]
    historical_returns: Annotated[Optional[HistoricalReturnsData], replace_value]
    step1_weights: Annotated[Optional[PortfolioWeights], replace_value]

    # Step 2 artifacts
    market_snapshot: Annotated[Optional[RealTimeMarketSnapshot], replace_value]
    sentiment_scores: Annotated[Optional[SentimentScores], replace_value]
    step2_weights: Annotated[Optional[PortfolioWeights], replace_value]

    # Step 3 artifacts
    final_result: Annotated[Optional[BlackLittermanOutput], replace_value]
