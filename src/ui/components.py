"""Reusable Streamlit widgets for the portfolio optimization UI."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.models.schemas import BlackLittermanOutput


PIPELINE_STEPS = [
    ("parse_request", "Parse Request"),
    ("clarify", "Clarify"),
    ("retrieve_notes", "Retrieve Notes"),
    ("expand_search", "Expand Search"),
    ("step1", "Step 1: Historical Optimization"),
    ("step2", "Step 2: Real-Time Optimization"),
    ("step3", "Step 3: Black-Litterman Blend"),
    ("done", "Complete"),
]


def render_pipeline_status(agent_state: dict) -> None:
    """Render the pipeline progress in the sidebar."""
    st.header("Pipeline Status")

    if not agent_state:
        st.write("Waiting for input...")
        return

    current = agent_state.get("current_step", "")
    passed = False

    for step_key, label in PIPELINE_STEPS:
        if step_key == current:
            st.write(f"**> {label}**")
            passed = True
        elif not passed:
            st.write(f"~~{label}~~")
        else:
            st.write(f"  {label}")


def render_portfolio_chart(result: BlackLittermanOutput) -> None:
    """Render the final portfolio weights as a bar chart."""
    weights = result.posterior_weights.weights

    df = pd.DataFrame(
        {"Ticker": list(weights.keys()), "Weight": list(weights.values())}
    ).sort_values("Weight", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Portfolio Weights")
        st.dataframe(
            df.style.format({"Weight": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.subheader("Allocation")
        st.bar_chart(df.set_index("Ticker")["Weight"])
