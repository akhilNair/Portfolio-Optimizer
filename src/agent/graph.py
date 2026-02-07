"""LangGraph StateGraph definition for the portfolio optimization pipeline."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agent import nodes
from src.agent.state import AgentState


def should_clarify(state: AgentState) -> str:
    """Route after parse_request: if parsing failed, ask for clarification."""
    if state.get("error") or state.get("analyst_request") is None:
        return "clarify"
    return "retrieve_notes"


def has_matches(state: AgentState) -> str:
    """Route after retrieve_notes: if no notes matched, expand search."""
    matched = state.get("matched_notes")
    if not matched or len(matched) == 0:
        return "expand_search"
    return "step1_optimize"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("parse_request", nodes.parse_request_node)
    graph.add_node("clarify", nodes.clarify_node)
    graph.add_node("retrieve_notes", nodes.retrieve_notes_node)
    graph.add_node("expand_search", nodes.expand_search_node)
    graph.add_node("step1_optimize", nodes.step1_optimize_node)
    graph.add_node("step2_optimize", nodes.step2_optimize_node)
    graph.add_node("step3_blend", nodes.step3_blend_node)
    graph.add_node("present_results", nodes.present_results_node)

    # Entry
    graph.add_edge(START, "parse_request")

    # Conditional: parse success or clarify
    graph.add_conditional_edges(
        "parse_request",
        should_clarify,
        {"clarify": "clarify", "retrieve_notes": "retrieve_notes"},
    )

    # Clarify loops back to parse
    graph.add_edge("clarify", "parse_request")

    # Conditional: notes found or expand search
    graph.add_conditional_edges(
        "retrieve_notes",
        has_matches,
        {"expand_search": "expand_search", "step1_optimize": "step1_optimize"},
    )

    # Expand search feeds into step1
    graph.add_edge("expand_search", "step1_optimize")

    # Linear pipeline: step1 -> step2 -> step3 -> present -> END
    graph.add_edge("step1_optimize", "step2_optimize")
    graph.add_edge("step2_optimize", "step3_blend")
    graph.add_edge("step3_blend", "present_results")
    graph.add_edge("present_results", END)

    return graph


def get_compiled_graph():
    """Return the compiled, ready-to-invoke graph."""
    return build_graph().compile()
