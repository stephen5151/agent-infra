"""
Perspective Engine — 多视角并行分析
======================================
当用户提出一个议题时，不返回单一结论，而是：

  1. PerspectivePlannerNode   — 读取议题，智能生成 3-5 个差异化视角
  2. PerspectiveDispatchNode  — Send API 扇出，每个视角独立并行执行
  3. PerspectiveWorkerNode    — 深度投入单一视角，给出真实立场（不两边倒）
  4. PerspectiveSynthesisNode — 结构化展示：各方观点 → 共识 → 分歧 → 综合思考

核心设计原则：
  - Worker 阶段：每个 agent 只代表一个视角，必须有立场，不允许"另一方面……"式骑墙
  - Synthesis 阶段：才做整合，明确标出共识点与核心张力，保留观点分歧，不做虚假统一
  - 视角类型动态生成：根据议题域（商业/技术/伦理/社会/科学）选择最有认知价值的透镜
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Send
from pydantic import BaseModel, Field

from agent_infra.core.llm import get_llm, get_structured_llm
from agent_infra.core.state import AgentState
from agent_infra.patterns.parallelization import WorkerState

logger = logging.getLogger(__name__)


# ── 视角规格 ──────────────────────────────────────────────────────────────────

class PerspectiveSpec(BaseModel):
    name: str = Field(description="视角名称，如 '经济学家' / '伦理学家' / '一线工程师'")
    emoji: str = Field(description="代表这个视角的 emoji，用于展示区分")
    lens: str = Field(description="这个视角的认知框架核心，一句话")
    core_question: str = Field(description="这个视角最关心的核心问题")
    likely_stance: str = Field(description="这个视角对议题的预设倾向，不要中立")


class _PerspectivePlan(BaseModel):
    topic_summary: str = Field(description="议题的核心争议点，一句话")
    domain: str = Field(description="议题领域：business / tech / ethics / social / science / history / mixed")
    perspectives: list[PerspectiveSpec] = Field(
        description="3-5 个差异化视角，必须覆盖对立面，不能全是相似立场"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 节点
# ═══════════════════════════════════════════════════════════════════════════════

class PerspectivePlannerNode:
    """
    读取用户议题，智能生成 3-5 个最有认知价值的视角。

    选角原则：
      - 覆盖议题的主要利益相关方
      - 必须包含至少一个反直觉或批判性视角
      - 视角之间要有真实张力，不能都得出相同结论
      - 根据领域选择最合适的透镜（经济/伦理/历史/技术/心理学等）
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.llm = get_structured_llm(_PerspectivePlan, model=model)

    def __call__(self, state: AgentState) -> dict[str, Any]:
        user_message = _latest_human_message(state)
        working_mem = "\n".join(state.get("working_memory") or [])

        system = (
            "你是一位跨学科思想家，擅长从多个角度解构复杂议题。\n\n"
            "你的任务：为用户的议题设计 3-5 个最有认知价值的分析视角。\n\n"
            "选角要求：\n"
            "  1. 每个视角必须有独特的认知框架，不是同一立场的变体\n"
            "  2. 必须覆盖主流与反主流，不能全是'支持方'\n"
            "  3. likely_stance 必须明确表态，禁止'两方面都有道理'式骑墙\n"
            "  4. 选角要匹配议题领域：\n"
            "     - 商业议题 → 投资方/用户/竞争者/监管者/员工\n"
            "     - 技术议题 → 工程师/产品经理/伦理学家/历史学家/普通用户\n"
            "     - 社会议题 → 政策制定者/受影响群体/学者/反对派/国际视角\n"
            "     - 哲学议题 → 理性主义/经验主义/实用主义/存在主义/东方哲学\n"
            "  5. 至少一个视角要挑战用户的隐含假设"
        )
        human = f"议题：{user_message}"
        if working_mem:
            human = f"对话背景：\n{working_mem}\n\n议题：{user_message}"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        plan: _PerspectivePlan = (prompt | self.llm).invoke({})
        logger.info(
            f"[PerspectivePlanner] topic='{plan.topic_summary}', "
            f"domain={plan.domain}, perspectives={len(plan.perspectives)}"
        )

        # 将每个视角序列化为 task 字符串，供 dispatch 扇出
        tasks = [
            json.dumps({
                "user_message": user_message,
                "topic_summary": plan.topic_summary,
                "perspective": p.model_dump(),
            }, ensure_ascii=False)
            for p in plan.perspectives
        ]

        return {
            "parallel_tasks": tasks,
            "parallel_results": [],          # 清空上轮残留
            "chain_context": {
                **state.get("chain_context", {}),
                "topic_summary": plan.topic_summary,
                "perspective_domain": plan.domain,
            },
        }


class PerspectiveDispatchNode:
    """
    扇出节点：为每个视角发出独立的 Send，路由到 perspective_worker。
    与 ParallelDispatchNode 的区别：目标节点是 perspective_worker 而非 parallel_worker。
    """

    def __call__(self, state: AgentState) -> list[Send]:
        tasks: list[str] = state.get("parallel_tasks") or []
        if not tasks:
            return []
        return [
            Send("perspective_worker", {**state, "worker_task": task, "worker_result": ""})
            for task in tasks
        ]


class PerspectiveWorkerNode:
    """
    深度投入单一视角的 Worker。

    关键设计：
      - 完全代入该视角的认知框架和价值观
      - 必须有立场，必须给出真实论点
      - 不允许主动为其他视角辩护
      - 遇到议题的弱点也要承认，但从本视角出发解释
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.llm = get_llm(model=model)
        self._parser = StrOutputParser()

    def __call__(self, state: WorkerState) -> dict[str, Any]:
        raw_task = state.get("worker_task", "{}")
        try:
            task = json.loads(raw_task)
        except json.JSONDecodeError:
            task = {"user_message": raw_task, "perspective": {"name": "通用视角", "lens": "", "core_question": raw_task, "likely_stance": ""}}

        user_message = task.get("user_message", "")
        topic_summary = task.get("topic_summary", user_message)
        p = task.get("perspective", {})

        perspective_name = p.get("name", "分析者")
        lens = p.get("lens", "")
        core_question = p.get("core_question", "")
        stance = p.get("likely_stance", "")
        emoji = p.get("emoji", "🔵")

        system = (
            f"你现在完全代入 **{perspective_name}** 的角色和思维方式。\n\n"
            f"你的认知框架：{lens}\n"
            f"你最关心的问题：{core_question}\n"
            f"你的基本立场倾向：{stance}\n\n"
            "回应规则：\n"
            "  1. 完全从这个视角出发，使用这个领域的语言和逻辑\n"
            "  2. 必须给出明确立场，不允许以'两面都有道理'绕过\n"
            "  3. 提出 2-3 个支撑你立场的核心论点，每个论点要有具体依据\n"
            "  4. 承认本视角的局限性（一句话即可），但不要因此动摇立场\n"
            "  5. 语气直接、有说服力，像真正持有这个观点的人在说话\n"
            "  6. 篇幅：300-500 字，不要超过"
        )
        human = f"议题：{topic_summary}\n\n原始问题：{user_message}"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        result = (prompt | self.llm | self._parser).invoke({})
        logger.debug(f"[PerspectiveWorker] {perspective_name} responded ({len(result)} chars)")

        # 在结果前加上视角标题，供 synthesis 节点识别
        labeled_result = f"{emoji} **{perspective_name}**\n\n{result}"

        return {
            "parallel_results": [{"perspective": perspective_name, "emoji": emoji, "result": labeled_result}],
            "messages": [AIMessage(content=labeled_result, name=perspective_name)],
        }


class PerspectiveSynthesisNode:
    """
    多视角结构化整合。

    输出结构：
      1. 各视角观点（直接引用 worker 输出）
      2. 跨视角共识点（所有人都认同的）
      3. 核心分歧（真正的张力所在，不做假统一）
      4. 综合思考（提供认知框架，不代替用户下结论）
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.llm = get_llm(model=model)
        self._parser = StrOutputParser()

    def __call__(self, state: AgentState) -> dict[str, Any]:
        results: list[dict] = state.get("parallel_results") or []
        if not results:
            return {}

        topic_summary = state.get("chain_context", {}).get("topic_summary", "")
        user_message = _latest_human_message(state)

        # 构建各视角摘要
        perspectives_block = "\n\n---\n\n".join(r["result"] for r in results)
        perspective_names = [r.get("perspective", "") for r in results]

        system = (
            "你是多元思维的整合者。你的任务不是给出'正确答案'，\n"
            "而是帮助用户看清这个议题的真实复杂性。\n\n"
            "整合规则：\n"
            "  1. **各视角观点**：直接呈现，不改变立场，不稀释观点\n"
            "  2. **共识点**：只列真正跨视角认同的，不要把分歧写成共识\n"
            "  3. **核心分歧**：明确指出哪些分歧是价值观差异（不可调和），\n"
            "                    哪些是事实判断差异（可以被证据改变）\n"
            "  4. **综合思考**：提供思考框架，帮用户决定哪些因素对自己最重要，\n"
            "                    但不要代替用户下结论\n\n"
            "格式要求：\n"
            "  - 使用 Markdown，各视角用 ### 分隔\n"
            "  - 共识/分歧/综合各用独立的 ## 标题\n"
            "  - 分歧部分要区分'价值观分歧'和'事实分歧'\n"
            "  - 不要在最后给出'建议选择哪个视角'这类画蛇添足的结尾"
        )
        human = (
            f"原始议题：{user_message}\n"
            f"核心争议：{topic_summary}\n"
            f"参与视角：{', '.join(perspective_names)}\n\n"
            f"各视角分析：\n\n{perspectives_block}"
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        synthesis = (prompt | self.llm | self._parser).invoke({})
        logger.info(f"[PerspectiveSynthesis] synthesized {len(results)} perspectives")

        return {"messages": [AIMessage(content=synthesis)]}


# ── 条件边 ────────────────────────────────────────────────────────────────────

def perspective_dispatch_edge(state: AgentState) -> list[Send] | str:
    """
    条件边：有视角任务则扇出到 perspective_worker，否则跳过到 perspective_synthesis。
    """
    node = PerspectiveDispatchNode()
    sends = node(state)
    return sends if sends else "perspective_synthesis"


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _latest_human_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""
