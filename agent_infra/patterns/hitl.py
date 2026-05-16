"""
Chapter 13 — Human-in-the-Loop (HITL)
========================================
在关键决策点暂停执行，等待人类审核或修正，再继续。

Pattern:
  生成草稿 → interrupt() 暂停 → 人类审核 → resume(response) → 执行/修改/拒绝

LangGraph 实现:
  langgraph.types.interrupt(value) 将当前状态持久化并抛出中断信号。
  外部调用者通过 graph.invoke(None, config, command=Command(resume=...)) 恢复。

书中 Ch.13 的三种触发策略:
  1. 总是触发 — 高风险操作（发送邮件、写入数据库）
  2. 条件触发 — 置信度低时触发（reflector 评分低于阈值）
  3. 抽样触发 — 按比例随机监控（质量审计）

Cursor 对标:
  Accept / Reject 代码差异就是 HITL 的典型应用。
"""
from __future__ import annotations

import random
from typing import Any, Literal

from langgraph.types import interrupt

from agent_infra.core.state import AgentState


# ── 触发策略 ──────────────────────────────────────────────────────────────────

TriggerMode = Literal["always", "conditional", "sampling"]


class HITLNode:
    """
    Human-in-the-Loop 节点。

    interrupt() 会将 hitl_payload 序列化并暂停图执行。
    外部系统（UI / CLI）展示内容给人类，收到回应后调用:

        graph.invoke(
            None,
            config={"configurable": {"thread_id": "..."}},
            command=Command(resume={"approved": True, "feedback": "LGTM"}),
        )

    Args:
        mode:               触发模式 (always | conditional | sampling)
        confidence_field:   conditional 模式下读取 state 中的置信分字段
        confidence_threshold: 低于此值才触发 HITL
        sampling_rate:      sampling 模式下触发概率 (0-1)
        operation_label:    呈现给人类的操作描述
    """

    def __init__(
        self,
        mode: TriggerMode = "conditional",
        confidence_field: str = "reflection_count",
        confidence_threshold: float = 0.6,
        sampling_rate: float = 0.2,
        operation_label: str = "AI 生成内容审核",
    ) -> None:
        self.mode = mode
        self.conf_field = confidence_field
        self.conf_threshold = confidence_threshold
        self.sampling_rate = sampling_rate
        self.label = operation_label

    def __call__(self, state: AgentState) -> dict[str, Any]:
        if not self._should_trigger(state):
            return {"requires_human": False}

        # 构造呈现给人类的 payload
        last_ai = _latest_ai_content(state)
        payload: dict[str, Any] = {
            "operation": self.label,
            "draft": last_ai,
            "plan": state.get("plan"),
            "reflection": state.get("reflection"),
        }

        # ⚡ 暂停执行，等待人类回应
        # 回应结构建议: {"approved": bool, "feedback": str, "edited_content": str | None}
        human_response: dict[str, Any] = interrupt(payload)

        return {
            "hitl_payload": payload,
            "hitl_response": human_response,
            "requires_human": True,
            # 如果人类提供了修改内容，写入 chain_context
            "chain_context": (
                {**state.get("chain_context", {}),
                 "human_edit": human_response.get("edited_content")}
                if human_response.get("edited_content")
                else state.get("chain_context", {})
            ),
        }

    def _should_trigger(self, state: AgentState) -> bool:
        if self.mode == "always":
            return True
        if self.mode == "sampling":
            return random.random() < self.sampling_rate
        # conditional: 反思次数多说明质量不稳定，触发审核
        if self.mode == "conditional":
            count = state.get("reflection_count") or 0
            return count >= 1  # 经过反思仍需人工确认
        return False


def hitl_edge(state: AgentState) -> Literal["approved", "rejected", "skip"]:
    """
    HITL 后的条件边。

    - approved  → 继续正常流程
    - rejected  → 回到生成节点重新生成
    - skip      → HITL 未触发，直接跳过
    """
    if not state.get("requires_human"):
        return "skip"
    response = state.get("hitl_response") or {}
    return "approved" if response.get("approved", True) else "rejected"


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _latest_ai_content(state: AgentState) -> str:
    from langchain_core.messages import AIMessage
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return ""
