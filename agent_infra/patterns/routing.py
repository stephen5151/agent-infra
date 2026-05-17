"""
Chapter 2 — Routing
====================
A classifier LLM labels the user intent, and a conditional edge in
LangGraph directs execution to the appropriate specialist node.

Pattern:
  Input → Router → [specialist_A | specialist_B | specialist_C | ...]

Key insight from the book:
  Keep route labels small and well-defined. Overlap between categories
  causes the classifier to produce unreliable labels.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agent_infra.core.llm import cached_system, get_llm
from agent_infra.core.state import AgentState


class RouteDecision(BaseModel):
    route: str = Field(description="The selected route label")
    confidence: float = Field(ge=0.0, le=1.0, description="Classifier confidence 0-1")
    reasoning: str = Field(description="One-sentence explanation of the routing choice")


class RouterNode:
    """
    Classifies the latest user message into one of the registered routes.

    Args:
        routes: Dict mapping label → description, e.g.
                {"code": "coding / debugging task",
                 "research": "information retrieval or summarisation",
                 "math": "arithmetic or algebraic calculation"}
        default_route: Fallback when confidence is below the threshold.
        confidence_threshold: Minimum confidence to accept a classification.
        keyword_map: Optional dict mapping route label → list of keyword patterns.
                     When provided, matching keywords short-circuit the LLM call.
    """

    def __init__(
        self,
        routes: dict[str, str],
        default_route: str = "general",
        confidence_threshold: float = 0.6,
        model: str = "claude-haiku-4-5-20251001",
        keyword_map: dict[str, list[str]] | None = None,
    ) -> None:
        self.routes = routes
        self.default_route = default_route
        self.threshold = confidence_threshold
        self.llm = get_llm(model=model).with_structured_output(RouteDecision)
        # Compile keyword patterns once at init time
        self._kw: list[tuple[str, re.Pattern[str]]] = [
            (label, re.compile("|".join(re.escape(kw) for kw in kws), re.IGNORECASE))
            for label, kws in (keyword_map or {}).items()
        ]

    def __call__(self, state: AgentState) -> dict[str, Any]:
        user_input = _latest_human_message(state)

        # Fast path: keyword matching — no LLM call needed
        for label, pattern in self._kw:
            if pattern.search(user_input):
                return {"route": label}

        # Slow path: LLM classification for ambiguous inputs
        route_descriptions = "\n".join(
            f'  • "{label}": {desc}' for label, desc in self.routes.items()
        )
        system = (
            "You are a routing classifier. "
            "Given the user's message, select the single best route from the list below "
            "and return a JSON object with keys: route, confidence, reasoning.\n\n"
            f"Available routes:\n{route_descriptions}\n\n"
            f"If none match well, use route='{self.default_route}'."
        )
        prompt = ChatPromptTemplate.from_messages([
            cached_system(system),
            HumanMessage(content=user_input),
        ])
        decision: RouteDecision = (prompt | self.llm).invoke({})

        chosen = (
            decision.route
            if decision.confidence >= self.threshold
            else self.default_route
        )
        return {"route": chosen}


def route_edge(state: AgentState) -> str:
    """
    Conditional edge function for StateGraph.add_conditional_edges.

    Returns the current route value so LangGraph can branch to the
    matching node name.
    """
    return state.get("route") or "general"


# ── helpers ───────────────────────────────────────────────────────────────────

def _latest_human_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""
