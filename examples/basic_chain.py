"""
Example 1 — Prompt Chaining (Chapter 1)
========================================
Shows how to use ChainNode directly without the full agent graph.
Each step's output is injected as context into the next step.
"""
import os
from dotenv import load_dotenv

from agent_infra.patterns.chaining import ChainNode, ChainStep, build_chain_subgraph
from agent_infra.core.state import AgentState
from langchain_core.messages import HumanMessage

load_dotenv()


def main() -> None:
    # Define a 3-step research chain
    steps = [
        ChainStep(
            name="outline",
            system_prompt=(
                "Create a structured outline for a blog post on the given topic. "
                "Return 4-6 section headings with one-line descriptions."
            ),
        ),
        ChainStep(
            name="draft",
            system_prompt=(
                "Using the outline below, write a full blog post draft. "
                "Each section should be 2-3 paragraphs."
            ),
            depends_on=["outline"],
        ),
        ChainStep(
            name="seo_title",
            system_prompt=(
                "Given the draft below, generate 3 SEO-optimised title options. "
                "Each title must be under 60 characters."
            ),
            depends_on=["draft"],
        ),
    ]

    chain = build_chain_subgraph(steps)

    # Simulate the AgentState that would normally come from LangGraph
    from agent_infra.agent import EMPTY_STATE
    state: AgentState = {
        **EMPTY_STATE,  # type: ignore[arg-type]
        "messages": [HumanMessage(content="The future of AI agents in software engineering")],
    }

    result = chain(state)
    ctx = result["chain_context"]

    print("=== OUTLINE ===")
    print(ctx["outline"])
    print("\n=== SEO TITLES ===")
    print(ctx["seo_title"])


if __name__ == "__main__":
    main()
