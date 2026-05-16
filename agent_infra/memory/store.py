"""
Chapter 8 — Memory Management
================================
Two memory tiers:

  Short-term  — working_memory list in AgentState (in-graph, per-run)
  Long-term   — LangGraph InMemoryStore keyed by (namespace, key)
                Swap InMemoryStore for langgraph.store.postgres.AsyncPostgresStore
                or any other langgraph.store backend for persistence.

Pattern:
  Every turn: recall relevant memories → inject → generate → store new facts

Key insight from the book:
  Retrieval must happen *before* generation so memories can influence the
  model's reasoning, not just be logged afterwards.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.store.memory import InMemoryStore

from agent_infra.core.llm import get_llm
from agent_infra.core.state import AgentState


class MemoryStore:
    """
    Thin wrapper around LangGraph's InMemoryStore.

    Namespace strategy: (user_id, "memories") per user session.
    Each entry: {"content": str, "created_at": ISO-8601, "source": str}
    """

    def __init__(self, store: InMemoryStore | None = None) -> None:
        self._store = store or InMemoryStore()

    # ── write ──────────────────────────────────────────────────────────────

    def save(self, content: str, namespace: tuple[str, ...] = ("default", "memories"), source: str = "agent") -> str:
        key = hashlib.md5(content.encode()).hexdigest()[:12]
        self._store.put(
            namespace,
            key,
            {
                "content": content,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "source": source,
            },
        )
        return key

    # ── read ───────────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        namespace: tuple[str, ...] = ("default", "memories"),
        top_k: int = 5,
    ) -> list[str]:
        """
        Simple keyword search over stored memories.

        For production: replace with vector similarity search using
        langgraph.store backends that support embeddings.
        """
        results = self._store.search(namespace, query=query, limit=top_k)
        return [item.value["content"] for item in results]

    # ── list all ───────────────────────────────────────────────────────────

    def list_all(self, namespace: tuple[str, ...] = ("default", "memories")) -> list[dict[str, Any]]:
        items = self._store.search(namespace, query="", limit=100)
        return [item.value for item in items]


class MemoryNode:
    """
    LangGraph node that:
      1. Recalls relevant memories and populates working_memory
      2. After generation, extracts and saves new facts

    Use it in two positions in the graph:
      - BEFORE generation: recall
      - AFTER generation:  consolidate
    """

    def __init__(
        self,
        store: MemoryStore,
        namespace: tuple[str, ...] = ("default", "memories"),
        top_k: int = 5,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self.store = store
        self.namespace = namespace
        self.top_k = top_k
        self.llm = get_llm(model=model)
        self._parser = StrOutputParser()

    def recall(self, state: AgentState) -> dict[str, Any]:
        """Pre-generation: fetch relevant memories into working_memory."""
        query = _latest_human_message(state)
        memories = self.store.recall(query, namespace=self.namespace, top_k=self.top_k)
        return {"working_memory": memories}

    def consolidate(self, state: AgentState) -> dict[str, Any]:
        """Post-generation: extract new facts from the latest exchange and save them."""
        human = _latest_human_message(state)
        ai = _latest_ai_message(state)
        if not human or not ai:
            return {}

        system = (
            "Extract any new, reusable facts or user preferences from this exchange. "
            "Return a JSON array of strings, one fact per item. "
            "Return [] if nothing worth remembering."
        )
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system),
            HumanMessage(content=f"User: {human}\nAssistant: {ai}"),
        ])
        raw = (prompt | self.llm | self._parser).invoke({})

        try:
            facts: list[str] = json.loads(raw)
        except json.JSONDecodeError:
            facts = []

        for fact in facts:
            self.store.save(fact, namespace=self.namespace, source="consolidation")

        return {}


# ── helpers ───────────────────────────────────────────────────────────────────

def _latest_human_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _latest_ai_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""
