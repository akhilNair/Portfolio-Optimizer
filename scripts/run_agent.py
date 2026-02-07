#!/usr/bin/env python3
"""CLI script to run the portfolio optimization agent without Streamlit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from src.agent.graph import get_compiled_graph


def main():
    graph = get_compiled_graph()

    print("Portfolio Optimization Agent (CLI Mode)")
    print("Type your structured note requirements. Type 'quit' to exit.\n")

    messages = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))

        input_state = {
            "messages": messages,
            "current_step": "parse_request",
        }

        print("\n--- Running pipeline ---")
        for event in graph.stream(input_state):
            for node_name, node_output in event.items():
                print(f"\n[{node_name}]")
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        print(msg.content)
                        messages.append(msg)

        print("\n--- Pipeline complete ---\n")


if __name__ == "__main__":
    main()
