"""Tests for the LangGraph agent graph structure."""

import pytest

from src.agent.graph import build_graph, should_clarify, has_matches


class TestGraphRouting:
    def test_should_clarify_on_error(self):
        state = {"error": "parse failed", "analyst_request": None}
        assert should_clarify(state) == "clarify"

    def test_should_not_clarify_on_success(self):
        state = {"error": None, "analyst_request": object()}
        assert should_clarify(state) == "retrieve_notes"

    def test_has_matches_with_notes(self):
        state = {"matched_notes": ["note1", "note2"]}
        assert has_matches(state) == "step1_optimize"

    def test_has_matches_empty(self):
        state = {"matched_notes": []}
        assert has_matches(state) == "expand_search"

    def test_has_matches_none(self):
        state = {"matched_notes": None}
        assert has_matches(state) == "expand_search"

    def test_graph_compiles(self):
        graph = build_graph()
        compiled = graph.compile()
        assert compiled is not None
