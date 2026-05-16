"""
Chapter 1 — Prompt Chaining
============================
Each step's output is injected as context into the next step's prompt.
The chain runs as a sub-sequence of nodes inside the main LangGraph.

Pattern:
  Input → Step1 → Step2 → ... → StepN → Output

Key insight from the book:
  Use structured output (JSON/XML) between steps to avoid parsing errors
  in downstream prompts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent_infra.core.llm import get_llm
from agent_infra.core.state import AgentState


@dataclass
class ChainStep:
    name: str
    system_prompt: str
    # Names of previous steps whose outputs to inject as context
    depends_on: list[str] = field(default_factory=list)


class ChainNode:
    """
    Executes a sequence of ChainSteps, threading outputs through
    chain_context so each step can read prior results.
    """

    def __init__(self, steps: list[ChainStep], model: str = "claude-sonnet-4-6") -> None:
        self.steps = steps
        self.llm = get_llm(model=model)
        self._parser = StrOutputParser()

    def __call__(self, state: AgentState) -> dict[str, Any]:
        user_input = _extract_user_input(state)
        context: dict[str, str] = dict(state.get("chain_context") or {})
        # ✅ 把上轮 Reflection 的 critique 注入 context，让每个步骤都能"知道上次错哪了"
        critique = state.get("reflection") or ""

        for step in self.steps:
            context[step.name] = self._run_step(step, user_input, context, critique)

        final_output = context[self.steps[-1].name]
        return {
            "chain_context": context,
            "messages": [AIMessage(content=final_output)],
        }

    def _run_step(
        self,
        step: ChainStep,
        user_input: str,
        context: dict[str, str],
        critique: str = "",
    ) -> str:
        prior = "\n\n".join(
            f"[{dep}]\n{context[dep]}"
            for dep in step.depends_on
            if dep in context
        )
        human_text = user_input
        if prior:
            human_text = f"{prior}\n\n---\nOriginal request: {user_input}"
        # ✅ 注入 critique：告诉模型上次具体哪里不对，引导定向修复而非盲目重试
        if critique:
            human_text = f"[上轮审查反馈，本次必须修复]\n{critique}\n\n---\n{human_text}"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=step.system_prompt),
            HumanMessage(content=human_text),
        ])
        chain = prompt | self.llm | self._parser
        return chain.invoke({})


def build_chain_subgraph(steps: list[ChainStep]) -> ChainNode:
    """Convenience factory — returns a callable node ready for graph.add_node."""
    return ChainNode(steps)


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_user_input(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""
