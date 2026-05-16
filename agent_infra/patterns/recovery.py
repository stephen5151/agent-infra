"""
Chapter 12 — Exception Handling & Recovery
============================================
捕获执行错误，根据策略自动恢复：重试、降级或上报人工。

Pattern:
  执行节点 → 异常 → RecoveryNode → [retry | fallback | escalate]

书中 Ch.12 的三层恢复策略:
  1. Retry       — 同一操作重试（指数退避），适合临时性错误
  2. Fallback    — 换一个更简单/更保守的策略，适合能力边界问题
  3. Escalate    — 超出自动处理能力，转给人工（触发 HITL）

实现原则:
  - 错误信息注入提示词，让模型"知道上次错哪了"
  - 重试有硬性上限，防止无限循环
  - 降级策略主动降低输出要求（如从结构化 JSON 降为纯文本）
"""
from __future__ import annotations

import time
from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent_infra.core.llm import get_llm
from agent_infra.core.state import AgentState


RecoveryStrategy = Literal["retry", "fallback", "escalate"]


# ── 核心恢复节点 ──────────────────────────────────────────────────────────────

class RecoveryNode:
    """
    根据当前错误和重试次数决定恢复策略。

    图中位置: 任何可能出错的节点 → recovery → [retry_target | fallback | hitl]

    Args:
        max_retries:       自动重试上限（超过后 escalate）
        fallback_at:       达到此重试次数后切换为 fallback 策略
        model:             用于生成 fallback 回复的模型
    """

    def __init__(
        self,
        max_retries: int = 3,
        fallback_at: int = 2,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self.max_retries = max_retries
        self.fallback_at = fallback_at
        self.llm = get_llm(model=model)
        self._parser = StrOutputParser()

    def __call__(self, state: AgentState) -> dict[str, Any]:
        count = state.get("retry_count") or 0
        error = state.get("last_error") or "未知错误"

        if count >= self.max_retries:
            strategy: RecoveryStrategy = "escalate"
        elif count >= self.fallback_at:
            strategy = "fallback"
        else:
            strategy = "retry"

        result: dict[str, Any] = {
            "retry_count": count + 1,
            "recovery_strategy": strategy,
        }

        if strategy == "fallback":
            fallback_response = self._generate_fallback(state, error)
            result["messages"] = [AIMessage(content=fallback_response)]
            result["finished"] = True

        return result

    def _generate_fallback(self, state: AgentState, error: str) -> str:
        """降级策略：用更简单的提示词重新生成，放弃结构化格式要求。"""
        user_input = _latest_human_message(state)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "之前的处理遇到了问题，请用最简单直接的方式回答用户的问题。"
                f"参考错误信息: {error}"
            )),
            HumanMessage(content=user_input),
        ])
        return (prompt | self.llm | self._parser).invoke({})


def recovery_edge(state: AgentState) -> str:
    """
    Recovery 后的路由:
      retry    → 返回原来出错的节点重试
      fallback → 直接结束（RecoveryNode 已生成降级回复）
      escalate → 转 HITL 节点由人工处理
    """
    strategy = state.get("recovery_strategy") or "escalate"
    if strategy == "retry":
        return "retry"
    if strategy == "fallback":
        return "end"
    return "escalate"   # → hitl 节点


# ── 装饰器：为任意节点函数包裹自动错误捕获 ────────────────────────────────────

def with_recovery(node_fn: Callable, error_field: str = "last_error") -> Callable:
    """
    将节点函数包裹在 try/except 中。
    出错时把错误信息写入 state[error_field]，让 recovery 节点接管。

    用法:
        graph.add_node("executor", with_recovery(ExecutorNode()))
    """
    def wrapped(state: AgentState) -> dict[str, Any]:
        try:
            return node_fn(state)
        except Exception as exc:
            return {error_field: str(exc), "last_error": str(exc)}
    return wrapped


# ── 指数退避重试工具 ──────────────────────────────────────────────────────────

def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Any:
    """
    对普通函数（非 LangGraph 节点）做指数退避重试。
    适用于工具调用、外部 API 请求等。
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _latest_human_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""
