"""Streamlit chat interface for the portfolio optimization agent."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

from src.agent.graph import get_compiled_graph  # noqa: E402
from src.ui.components import render_pipeline_status, render_portfolio_chart  # noqa: E402

# Page config
st.set_page_config(
    page_title="Portfolio Optimization Assistant",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.title("Portfolio Optimization Assistant")
st.caption(
    "Structured Note Analysis | Historical + Real-Time Optimization | Black-Litterman Blending"
)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {}
if "graph" not in st.session_state:
    st.session_state.graph = get_compiled_graph()

# Sidebar
with st.sidebar:
    st.header("Configuration")
    risk_free_rate = st.slider("Risk-Free Rate", 0.0, 0.10, 0.05, 0.01)
    max_weight = st.slider("Max Single-Asset Weight", 0.10, 1.0, 0.40, 0.05)

    st.divider()
    render_pipeline_status(st.session_state.agent_state)

    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.agent_state = {}
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input
if user_input := st.chat_input("Describe the structured note you are looking for..."):
    user_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the agent graph
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            input_state = {
                "messages": st.session_state.messages,
                "current_step": "parse_request",
            }

            # Stream through graph nodes
            for event in st.session_state.graph.stream(input_state):
                for node_name, node_output in event.items():
                    if "messages" in node_output:
                        for msg in node_output["messages"]:
                            st.markdown(msg.content)
                            st.session_state.messages.append(msg)

                    # Update agent state for sidebar tracking
                    for key, value in node_output.items():
                        if key != "messages":
                            st.session_state.agent_state[key] = value

        # Show portfolio chart if final result is available
        final = st.session_state.agent_state.get("final_result")
        if final:
            render_portfolio_chart(final)
