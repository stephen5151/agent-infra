"""
Chapter 3 — Parallelization
==============================
将独立子任务分发给多个 worker 节点并行执行，再汇总结果。

Pattern:
  任务列表 → [Send API 扇出] → worker × N → 聚合节点 → 继续

LangGraph 实现方式:
  使用 langgraph.types.Send 做动态扇出，每个 Send 携带独立的任务状态，
  worker 节点并行运行，结果通过 reducer 合并回主状态。

适用场景 (书中 Ch.3):
  - 多角度分析同一问题（乐观 / 悲观 / 中立视角同时生成）
  - 批量处理独立文档（对每份文档做摘要后合并）
  - 并发调用多个工具（搜索 + 计算 + 数据库查询同时进行）
"""
from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Send

from agent_infra.core.llm import get_llm
from agent_infra.core.state import AgentState


# ── 单个 worker 的状态（扇出时传递） ─────────────────────────────────────────

class WorkerState(AgentState):
    """Send 传给 worker 的最小状态，包含当前子任务描述。"""
    worker_task: str
    worker_result: str


# ── 节点 ──────────────────────────────────────────────────────────────────────

class ParallelDispatchNode:
    """
    扇出节点：读取 state["parallel_tasks"]，为每个任务发出一个 Send。

    图中用法:
        graph.add_node("dispatch", ParallelDispatchNode())
        graph.add_conditional_edges("dispatch", dispatch_edge, ["worker"])
        graph.add_node("worker", ParallelWorkerNode())
        graph.add_edge("worker", "aggregate")
    """

    def __call__(self, state: AgentState) -> list[Send]:
        tasks: list[str] = state.get("parallel_tasks") or []
        if not tasks:
            # 没有任务时退化为空跳转，由后续 aggregate 收尾
            return []
        return [
            Send("parallel_worker", {**state, "worker_task": task, "worker_result": ""})
            for task in tasks
        ]


class ParallelWorkerNode:
    """
    执行单个子任务，将结果写入 worker_result。
    LangGraph 会在所有 worker 完成后把结果通过 reducer 合并。
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.llm = get_llm(model=model)
        self._parser = StrOutputParser()

    def __call__(self, state: WorkerState) -> dict[str, Any]:
        task = state.get("worker_task", "")
        context = "\n".join(state.get("working_memory") or [])
        rag_ctx = "\n".join(state.get("rag_context") or [])

        system_parts = ["你是专注高效的执行者，完成分配给你的单一子任务。"]
        if context:
            system_parts.append(f"\n工作记忆:\n{context}")
        if rag_ctx:
            system_parts.append(f"\n参考文档:\n{rag_ctx}")

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="\n".join(system_parts)),
            HumanMessage(content=task),
        ])
        result = (prompt | self.llm | self._parser).invoke({})

        # 注意：parallel_results 使用 list append reducer（见下方）
        return {
            "parallel_results": [{"task": task, "result": result}],
            "messages": [AIMessage(content=result)],
        }


class AggregateNode:
    """
    扇入节点：汇总所有 worker 的结果，合成最终答案。
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.llm = get_llm(model=model)
        self._parser = StrOutputParser()

    def __call__(self, state: AgentState) -> dict[str, Any]:
        results: list[dict] = state.get("parallel_results") or []
        if not results:
            return {}

        parts = "\n\n".join(
            f"### 子任务: {r['task']}\n{r['result']}" for r in results
        )
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "你负责将多个并行子任务的结果整合成一份连贯、完整的最终答案。"
                "去除重复内容，保持逻辑顺序。"
            )),
            HumanMessage(content=f"各子任务结果如下:\n\n{parts}"),
        ])
        summary = (prompt | self.llm | self._parser).invoke({})
        return {"messages": [AIMessage(content=summary)]}


# ── 辅助边函数 ────────────────────────────────────────────────────────────────

def dispatch_edge(state: AgentState) -> list[Send] | str:
    """
    条件边：有并行任务则扇出，否则跳过。

    用法:
        graph.add_conditional_edges(
            "dispatch",
            dispatch_edge,
            {"parallel_worker": "parallel_worker", "__end__": "aggregate"},
        )
    """
    dispatcher = ParallelDispatchNode()
    sends = dispatcher(state)
    return sends if sends else "aggregate"
