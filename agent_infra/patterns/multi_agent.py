"""
Chapter 7 — Multi-Agent Collaboration
========================================
Supervisor 负责任务分派，各 Specialist Agent 专注各自领域，
通过 handoff 机制交接控制权。

Pattern:
  Supervisor → 选择 Agent → Specialist 执行 → 返回 Supervisor → 继续/结束

书中 Ch.7 的两种协作模式:
  1. Supervisor 模式  — 中央调度，本文件实现
  2. Peer-to-Peer 模式 — Agent 间直接交接（可通过 Send 扩展）

Cursor 对标:
  Composer 中的不同 agent（代码生成、解释、终端执行）之间的切换。
"""
from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agent_infra.core.llm import get_llm
from agent_infra.core.state import AgentState
from agent_infra.core.tools import TOOLS


# ── 路由决策 schema ────────────────────────────────────────────────────────────

class AgentDecision(BaseModel):
    next_agent: str = Field(description="下一个应当接管的 Agent 名称，或 'FINISH' 表示任务完成")
    reasoning: str = Field(description="选择该 Agent 的理由")
    subtask: str = Field(description="需要该 Agent 完成的具体子任务描述")


# ── Supervisor ────────────────────────────────────────────────────────────────

class SupervisorNode:
    """
    中央调度节点。

    读取当前对话和已完成的 agent 历史，决定下一步调用哪个 specialist。
    返回 "FINISH" 时图退出。

    Args:
        agents: dict，键为 agent 名称，值为该 agent 的能力描述
    """

    FINISH = "FINISH"

    def __init__(
        self,
        agents: dict[str, str],
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self.agents = agents
        self.llm = get_llm(model=model).with_structured_output(AgentDecision)

    def __call__(self, state: AgentState) -> dict[str, Any]:
        agent_list = "\n".join(
            f"  • {name}: {desc}" for name, desc in self.agents.items()
        )
        handoffs = state.get("agent_handoffs") or []
        history = " → ".join(handoffs) if handoffs else "（无）"

        system = (
            "你是一个多 Agent 系统的 Supervisor。\n"
            f"可用的 Agent:\n{agent_list}\n"
            f"  • FINISH: 任务已完成，不需要更多 Agent\n\n"
            f"已完成的 Agent 交接历史: {history}\n\n"
            "根据当前进展，选择下一个最合适的 Agent 或 FINISH。"
        )
        messages = [
            SystemMessage(content=system),
            *_recent_messages(state, n=10),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        decision: AgentDecision = (prompt | self.llm).invoke({})

        handoffs_updated = handoffs + [decision.next_agent]
        return {
            "active_agent": decision.next_agent,
            "agent_handoffs": handoffs_updated,
            # 将 supervisor 的子任务描述注入 chain_context，供 specialist 读取
            "chain_context": {
                **state.get("chain_context", {}),
                "current_subtask": decision.subtask,
            },
            "finished": decision.next_agent == self.FINISH,
        }


def supervisor_edge(state: AgentState) -> str:
    """
    条件边：根据 active_agent 路由到对应的 specialist 节点，
    或在 FINISH 时路由到 'end'。
    """
    agent = state.get("active_agent") or SupervisorNode.FINISH
    if agent == SupervisorNode.FINISH or state.get("finished"):
        return "end"
    return agent


# ── Specialist Agent 基类 ─────────────────────────────────────────────────────

class SpecialistNode:
    """
    通用 Specialist 节点。

    从 chain_context["current_subtask"] 读取当前子任务，
    使用自己的系统提示和可选工具执行，再把控制权还给 Supervisor。

    继承此类并重写 system_prompt 来创建特定领域的 specialist。
    """

    system_prompt: str = "你是一个全能助手，尽力完成分配的子任务。"

    def __init__(
        self,
        name: str,
        system_prompt: str | None = None,
        use_tools: bool = False,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self.name = name
        if system_prompt:
            self.system_prompt = system_prompt
        llm = get_llm(model=model)
        self.llm = llm.bind_tools(TOOLS) if use_tools else llm
        self._parser = StrOutputParser()

    def __call__(self, state: AgentState) -> dict[str, Any]:
        subtask = state.get("chain_context", {}).get("current_subtask", "")
        rag_ctx = "\n".join(state.get("rag_context") or [])

        system = self.system_prompt
        if rag_ctx:
            system += f"\n\n参考文档:\n{rag_ctx}"

        messages = [
            SystemMessage(content=system),
            *_recent_messages(state, n=6),
        ]
        if subtask:
            messages.append(HumanMessage(content=f"[当前子任务] {subtask}"))

        prompt = ChatPromptTemplate.from_messages(messages)
        response = (prompt | self.llm).invoke({})

        if hasattr(response, "content"):
            content = str(response.content)
        else:
            content = str(response)

        return {
            "messages": [AIMessage(content=content, name=self.name)],
            "active_agent": None,
        }


# ── 预置常用 Specialist ───────────────────────────────────────────────────────

def make_researcher(model: str = "claude-sonnet-4-6") -> SpecialistNode:
    return SpecialistNode(
        name="researcher",
        system_prompt=(
            "你是资深研究员，擅长信息整合与深度分析。"
            "给出有据可查、逻辑严谨的研究结论。"
        ),
        use_tools=True,
        model=model,
    )


def make_writer(model: str = "claude-sonnet-4-6") -> SpecialistNode:
    return SpecialistNode(
        name="writer",
        system_prompt=(
            "你是专业内容写作者，擅长将复杂信息转化为清晰流畅的文字。"
            "风格简洁有力，避免行话堆砌。"
        ),
        model=model,
    )


def make_critic(model: str = "claude-sonnet-4-6") -> SpecialistNode:
    return SpecialistNode(
        name="critic",
        system_prompt=(
            "你是严格的内容审核者。找出逻辑漏洞、事实错误和表达不清晰之处，"
            "给出具体可执行的改进建议。"
        ),
        model=model,
    )


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _recent_messages(state: AgentState, n: int = 10) -> list:
    return list(state.get("messages") or [])[-n:]
