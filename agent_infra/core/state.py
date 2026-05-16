"""
Shared state definitions for the agent graph.

State flows through every node in the LangGraph — nodes read from it and
return partial updates that are merged back in.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class PlanStep(TypedDict):
    step: int
    task: str
    status: str          # "pending" | "in_progress" | "done" | "failed"
    result: str | None


class StepResult(TypedDict):
    step: int
    output: str
    tool_calls: list[dict[str, Any]]


class AgentState(TypedDict):
    # ── conversation ──────────────────────────────────────────────────────────
    # add_messages reducer: new messages are appended, not replaced
    messages: Annotated[list[BaseMessage], add_messages]

    # ── planning (Chapter 6) ──────────────────────────────────────────────────
    plan: list[PlanStep]
    current_step: int

    # ── chaining (Chapter 1) ──────────────────────────────────────────────────
    # Intermediate outputs passed between chain steps
    chain_context: dict[str, Any]

    # ── reflection (Chapter 4) ────────────────────────────────────────────────
    reflection: str | None
    reflection_count: int

    # ── routing (Chapter 2) ───────────────────────────────────────────────────
    route: str | None   # resolved intent label

    # ── tool results (Chapter 5) ──────────────────────────────────────────────
    step_results: list[StepResult]

    # ── memory (Chapter 8) ────────────────────────────────────────────────────
    # Short-term working memory injected into prompts
    working_memory: list[str]

    # ── RAG 检索 (Chapter 14) ─────────────────────────────────────────────────
    rag_context: list[str]          # 检索到的相关文档片段

    # ── 并行化 (Chapter 3) ────────────────────────────────────────────────────
    parallel_tasks: list[str]       # 待并行执行的子任务列表
    # operator.add reducer: 并行 worker 各自返回单元素列表，reducer 负责合并
    parallel_results: Annotated[list[dict[str, Any]], operator.add]

    # ── Human-in-the-Loop (Chapter 13) ───────────────────────────────────────
    hitl_payload: dict[str, Any] | None     # 呈现给人类审核的数据
    hitl_response: dict[str, Any] | None    # 人类的审核回应
    requires_human: bool                    # 是否触发了 HITL

    # ── 多 Agent 协作 (Chapter 7) ─────────────────────────────────────────────
    active_agent: str | None        # 当前执行中的 Agent 名称
    agent_handoffs: list[str]       # Agent 交接历史

    # ── 异常恢复 (Chapter 12) ─────────────────────────────────────────────────
    retry_count: int
    last_error: str | None
    recovery_strategy: str | None   # "retry" | "fallback" | "escalate"

    # ── Guardrails (Chapter 18) ───────────────────────────────────────────────
    guard_input_passed: bool        # 输入安全检查是否通过
    guard_output_passed: bool       # 输出安全检查是否通过
    guard_reason: str | None        # 拦截原因

    # ── 代码执行 & 文件系统 ───────────────────────────────────────────────────
    execution_result: dict[str, Any] | None   # Sandbox 最近一次执行结果
    tool_errors: list[str]                    # lint/test/compile 的真实错误行
    file_changes: list[dict[str, Any]]        # 待落盘的 FileDiff 列表（供 HITL 展示）
    reflection_source: str | None             # "tool" | "llm" — 本轮 critique 来源

    # ── control ───────────────────────────────────────────────────────────────
    error: str | None
    finished: bool
