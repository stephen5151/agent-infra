"""
Chapter 6 — Planning
======================
The agent first decomposes a complex goal into an ordered list of steps,
then executes each step sequentially (with tool access), updating the plan
as it learns from intermediate results.

Pattern:
  Input → Planner → [Executor → ToolNode]* → Synthesiser → Output

Key insight from the book:
  Plans should be *dynamic* — the executor can add, remove, or reorder
  remaining steps based on what it discovers during execution.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agent_infra.core.llm import get_llm, get_structured_llm
from agent_infra.core.state import AgentState, PlanStep, StepResult
from agent_infra.core.tools import TOOLS


# ── Pydantic schema for structured plan output ────────────────────────────────

class Plan(BaseModel):
    steps: list[str] = Field(description="Ordered list of concrete subtask descriptions")
    reasoning: str = Field(description="One-sentence explanation of this plan")


# ── Nodes ─────────────────────────────────────────────────────────────────────

class PlannerNode:
    """
    Generates an initial execution plan from the user's request.
    Stores it in state as a list of PlanStep dicts.
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.llm = get_structured_llm(Plan, model=model)

    def __call__(self, state: AgentState) -> dict[str, Any]:
        user_input = _latest_human_message(state)
        system = (
            "You are an expert task planner. "
            "Break the user's request into 2-6 concrete, ordered subtasks. "
            "Each subtask must be independently executable and produce a clear output. "
            "Return JSON with keys: steps (list[str]), reasoning (str)."
        )
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system),
            HumanMessage(content=user_input),
        ])
        plan: Plan = (prompt | self.llm).invoke({})

        plan_steps: list[PlanStep] = [
            {"step": i, "task": task, "status": "pending", "result": None}
            for i, task in enumerate(plan.steps)
        ]
        return {"plan": plan_steps, "current_step": 0}


class ExecutorNode:
    """
    Executes the current plan step.  Uses the LLM with bound tools so
    it can invoke external functions if needed.  Marks the step done
    and advances the pointer.
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.llm = get_llm(model=model).bind_tools(TOOLS)
        self.parser = StrOutputParser()

    def __call__(self, state: AgentState) -> dict[str, Any]:
        plan: list[PlanStep] = list(state.get("plan") or [])
        idx = state.get("current_step") or 0

        if idx >= len(plan):
            return {"finished": True}

        current = plan[idx]
        prior_results = _format_prior_results(state.get("step_results") or [])
        working_mem = "\n".join(state.get("working_memory") or [])

        system_parts = ["你是专注精确的执行者，完成分配的子任务。"]
        if working_mem:
            system_parts.append(f"\n工作记忆:\n{working_mem}")
        if prior_results:
            system_parts.append(f"\n前序步骤结果:\n{prior_results}")
        # ✅ 注入上轮 Reflection critique，让 executor 知道上次错在哪里并定向修复
        critique = state.get("reflection") or ""
        if critique:
            system_parts.append(f"\n[上轮审查反馈，本次必须修复]\n{critique}")

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="\n".join(system_parts)),
            HumanMessage(content=f"Subtask {idx + 1}: {current['task']}"),
        ])
        response = (prompt | self.llm).invoke({})

        # Update the plan step in-place
        updated_plan = list(plan)
        updated_plan[idx] = {**current, "status": "done", "result": str(response.content)}

        step_results = list(state.get("step_results") or [])
        step_results.append({
            "step": idx,
            "output": str(response.content),
            "tool_calls": [tc.model_dump() for tc in (response.tool_calls or [])],
        })

        return {
            "plan": updated_plan,
            "current_step": idx + 1,
            "step_results": step_results,
            "messages": [response],
        }


def should_continue_plan(state: AgentState) -> str:
    """
    Conditional edge after ExecutorNode.

    Returns:
        "tools"   — executor requested a tool call
        "execute" — more plan steps remain
        "end"     — plan fully executed
    """
    messages = state.get("messages") or []
    if messages:
        last = messages[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"

    plan = state.get("plan") or []
    idx = state.get("current_step") or 0
    if idx < len(plan):
        return "execute"
    return "end"


# ── helpers ───────────────────────────────────────────────────────────────────

def _latest_human_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _format_prior_results(results: list[StepResult]) -> str:
    if not results:
        return ""
    lines = [f"Step {r['step'] + 1}: {r['output']}" for r in results]
    return "\n".join(lines)
