"""
Example 2 — Full Agent (all patterns combined)
===============================================
Demonstrates the complete agent graph:
  Routing → Chaining or Planning → Reflection → Memory
"""
import os
from dotenv import load_dotenv

from agent_infra.agent import build_agent, run

load_dotenv()


def demo(agent, label: str, message: str, thread: str) -> None:
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"User: {message}")
    print("-" * 60)
    response = run(agent, message, thread_id=thread)
    print(f"Agent: {response}")


def main() -> None:
    agent = build_agent()

    # Route: "chain" — simple Q&A
    demo(
        agent,
        label="Chain route — simple question",
        message="Explain the difference between LangChain and LangGraph in 3 sentences.",
        thread="thread-001",
    )

    # Route: "plan" — multi-step task
    demo(
        agent,
        label="Plan route — multi-step task",
        message=(
            "Research and summarise the top 3 agentic design patterns for "
            "production AI systems. For each pattern, list its main use case "
            "and one concrete implementation tip."
        ),
        thread="thread-002",
    )

    # Memory persists across turns in the same thread
    demo(
        agent,
        label="Memory — follow-up in same thread",
        message="Based on what you just explained, which pattern would you recommend for a customer-support bot?",
        thread="thread-002",
    )


if __name__ == "__main__":
    main()
