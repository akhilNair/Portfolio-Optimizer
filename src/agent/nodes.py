"""All LangGraph node functions for the portfolio optimization pipeline."""

from __future__ import annotations

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.state import AgentState
from src.market.data_fetcher import MarketDataFetcher
from src.market.sentiment import SentimentAnalyzer
from src.models.schemas import (
    AnalystRequest,
    OptimizationObjective,
    PortfolioWeights,
)
from src.optimization.black_litterman import BlackLittermanBlender
from src.optimization.historical import HistoricalOptimizer
from src.optimization.realtime import RealTimeOptimizer
from src.pdf.retriever import NoteRetriever

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.0)


def _map_risk_to_objective(risk_tolerance: str) -> OptimizationObjective:
    mapping = {
        "conservative": OptimizationObjective.MIN_VARIANCE,
        "moderate": OptimizationObjective.MAX_SHARPE,
        "aggressive": OptimizationObjective.MAX_RETURN,
    }
    return mapping.get(risk_tolerance, OptimizationObjective.MAX_SHARPE)


# ── Node 1: Parse the analyst's natural-language request ─────────────


def parse_request_node(state: AgentState) -> dict:
    """Use Claude to extract structured AnalystRequest from the latest user message."""
    last_message = state["messages"][-1]

    prompt = f"""You are a structured-note analyst assistant.
Extract the following fields from the user request as JSON.
Return ONLY valid JSON, no other text.

Fields:
- desired_payoff_type (one of: autocall, barrier_reverse_convertible, phoenix,
  capital_protected, vanilla_basket, range_accrual, or null)
- desired_underlyings (list of stock ticker symbols, or null)
- sector_preferences (list of sector names, or null)
- min_coupon (float as decimal e.g. 0.08 for 8%, or null)
- max_maturity_years (float, or null)
- risk_tolerance ("conservative", "moderate", or "aggressive")
- target_notional (float, or null)
- geopolitical_concerns (list of strings, or null)
- additional_constraints (free text string, or null)

User request: {last_message.content}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(content)
        request = AnalystRequest(**data)
        return {
            "analyst_request": request,
            "current_step": "parse_request",
            "error": None,
            "messages": [
                AIMessage(
                    content=(
                        f"Understood. Searching for "
                        f"{request.desired_payoff_type or 'any'} notes "
                        f"with underlyings: {request.desired_underlyings or 'any'}."
                    )
                )
            ],
        }
    except Exception as e:
        return {
            "error": f"Could not parse request: {e}",
            "current_step": "parse_request",
            "messages": [
                AIMessage(
                    content=(
                        "I could not fully understand your request. "
                        "Could you clarify the payoff type, desired "
                        "underlyings, and risk tolerance?"
                    )
                )
            ],
        }


# ── Node 2: Clarify ─────────────────────────────────────────────────


def clarify_node(state: AgentState) -> dict:
    """No-op node: waits for the next user message."""
    return {
        "current_step": "clarify",
        "error": None,
    }


# ── Node 3: Retrieve matching structured notes ─────────────────────


def retrieve_notes_node(state: AgentState) -> dict:
    """Search the FAISS index for notes matching the analyst request."""
    request = state["analyst_request"]
    retriever = NoteRetriever()

    matched_notes = retriever.search(
        payoff_type=request.desired_payoff_type,
        tickers=request.desired_underlyings,
        sectors=request.sector_preferences,
        min_coupon=request.min_coupon,
        max_maturity=request.max_maturity_years,
        top_k=10,
    )

    all_tickers = list(
        {asset.ticker for note in matched_notes for asset in note.underlying_basket}
    )

    return {
        "matched_notes": matched_notes,
        "candidate_tickers": all_tickers,
        "current_step": "retrieve_notes",
        "messages": [
            AIMessage(
                content=(
                    f"Found {len(matched_notes)} relevant notes "
                    f"spanning {len(all_tickers)} unique assets: {all_tickers}"
                )
            )
        ],
    }


# ── Node 4: Expand search ──────────────────────────────────────────


def expand_search_node(state: AgentState) -> dict:
    """When no PDF matches found, use Claude to suggest a ticker basket."""
    request = state["analyst_request"]

    prompt = f"""Given the analyst wants a {request.desired_payoff_type} note
in sectors {request.sector_preferences}, suggest 5-8 liquid US equity
tickers that would form a good underlying basket.
Return as a JSON array of strings only, no other text."""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        tickers = json.loads(content)
    except Exception:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    return {
        "candidate_tickers": tickers,
        "matched_notes": [],
        "current_step": "expand_search",
        "messages": [
            AIMessage(
                content=f"No exact note matches. Using AI-suggested basket: {tickers}"
            )
        ],
    }


# ── Node 5: Step 1 — Historical Optimization ──────────────────────


def step1_optimize_node(state: AgentState) -> dict:
    """Fetch historical data and run CVXPY optimization."""
    tickers = state["candidate_tickers"]
    request = state["analyst_request"]

    fetcher = MarketDataFetcher()
    hist_data = fetcher.get_historical_returns(tickers, lookback_days=252)

    optimizer = HistoricalOptimizer()
    objective = _map_risk_to_objective(request.risk_tolerance)
    weights = optimizer.optimize(
        tickers=tickers,
        mean_returns=hist_data.mean_returns,
        cov_matrix=hist_data.cov_matrix,
        objective=objective,
    )

    # Format weights for display
    sorted_w = sorted(weights.weights.items(), key=lambda x: -x[1])
    weight_str = ", ".join(f"{t}: {w:.1%}" for t, w in sorted_w[:5])

    return {
        "historical_returns": hist_data,
        "step1_weights": weights,
        "current_step": "step1",
        "messages": [
            AIMessage(
                content=(
                    f"**Step 1 Complete** (Historical, {objective.value})\n"
                    f"Top weights: {weight_str}\n"
                    f"Sharpe: {weights.sharpe_ratio:.3f} | "
                    f"Return: {weights.expected_return:.2%} | "
                    f"Vol: {weights.expected_volatility:.2%}"
                )
            )
        ],
    }


# ── Node 6: Step 2 — Real-Time Optimization ───────────────────────


def step2_optimize_node(state: AgentState) -> dict:
    """Fetch recent prices, get sentiment, run optimization."""
    tickers = state["candidate_tickers"]
    request = state["analyst_request"]

    fetcher = MarketDataFetcher()
    snapshot = fetcher.get_realtime_snapshot(tickers, window_days=30)

    analyzer = SentimentAnalyzer(llm=llm)
    sentiments = analyzer.analyze(
        tickers=tickers,
        concerns=request.geopolitical_concerns or [],
    )

    optimizer = RealTimeOptimizer()
    objective = _map_risk_to_objective(request.risk_tolerance)
    weights = optimizer.optimize(
        tickers=tickers,
        snapshot=snapshot,
        sentiment=sentiments,
        objective=objective,
    )

    sorted_w = sorted(weights.weights.items(), key=lambda x: -x[1])
    weight_str = ", ".join(f"{t}: {w:.1%}" for t, w in sorted_w[:5])

    sentiment_str = ", ".join(
        f"{t}: {s:+.2f}" for t, s in sentiments.scores.items()
    )

    return {
        "market_snapshot": snapshot,
        "sentiment_scores": sentiments,
        "step2_weights": weights,
        "current_step": "step2",
        "messages": [
            AIMessage(
                content=(
                    f"**Step 2 Complete** (Real-time + Sentiment)\n"
                    f"Sentiment: {sentiment_str}\n"
                    f"Top weights: {weight_str}\n"
                    f"Sharpe: {weights.sharpe_ratio:.3f} | "
                    f"Return: {weights.expected_return:.2%} | "
                    f"Vol: {weights.expected_volatility:.2%}"
                )
            )
        ],
    }


# ── Node 7: Step 3 — Black-Litterman Blend ────────────────────────


def step3_blend_node(state: AgentState) -> dict:
    """Blend Step 1 and Step 2 results via Black-Litterman."""
    blender = BlackLittermanBlender()
    result = blender.blend(
        step1_weights=state["step1_weights"],
        step2_weights=state["step2_weights"],
        cov_matrix=state["historical_returns"].cov_matrix,
        tickers=state["candidate_tickers"],
    )

    sorted_w = sorted(result.posterior_weights.weights.items(), key=lambda x: -x[1])
    weight_str = ", ".join(f"{t}: {w:.1%}" for t, w in sorted_w[:5])

    return {
        "final_result": result,
        "current_step": "step3",
        "messages": [
            AIMessage(
                content=(
                    f"**Step 3 Complete** (Black-Litterman Blend)\n"
                    f"Blended weights: {weight_str}\n"
                    f"Sharpe: {result.posterior_weights.sharpe_ratio:.3f}"
                )
            )
        ],
    }


# ── Node 8: Present Results ────────────────────────────────────────


def present_results_node(state: AgentState) -> dict:
    """Format and present the final portfolio recommendation."""
    result = state["final_result"]
    step1 = state["step1_weights"]
    step2 = state["step2_weights"]

    # Build comparison table
    rows = []
    for ticker in sorted(result.posterior_weights.weights.keys()):
        w1 = step1.weights.get(ticker, 0)
        w2 = step2.weights.get(ticker, 0)
        wf = result.posterior_weights.weights.get(ticker, 0)
        rows.append(f"| {ticker} | {w1:.2%} | {w2:.2%} | {wf:.2%} |")

    table = "\n".join(rows)

    summary = f"""
## Final Portfolio Recommendation

| Ticker | Historical | Real-time | **Blended** |
|--------|-----------|-----------|------------|
{table}

### Performance Metrics
| Metric | Historical | Real-time | **Blended** |
|--------|-----------|-----------|------------|
| Expected Return | {step1.expected_return:.2%} | {step2.expected_return:.2%} | {result.posterior_weights.expected_return:.2%} |
| Volatility | {step1.expected_volatility:.2%} | {step2.expected_volatility:.2%} | {result.posterior_weights.expected_volatility:.2%} |
| Sharpe Ratio | {step1.sharpe_ratio:.3f} | {step2.sharpe_ratio:.3f} | {result.posterior_weights.sharpe_ratio:.3f} |

### Diagnostics
- Blending method: Black-Litterman
- Historical confidence: {result.blend_diagnostics.get('step1_confidence', 'N/A')}
- Real-time confidence: {result.blend_diagnostics.get('step2_confidence', 'N/A')}
- Tracking error vs. prior: {result.blend_diagnostics.get('tracking_error_vs_prior', 0):.4f}
"""

    return {
        "current_step": "done",
        "messages": [AIMessage(content=summary)],
    }
