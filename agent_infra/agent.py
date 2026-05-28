"""
Full Agent Graph — 完整 Agent 图
==================================
集成所有 Agentic Design Patterns + 真实执行环境。

完整执行拓扑:

  START
    │
    ▼
  input_guard ──[blocked]──────────────────────────────────► END
    │[continue]
    ▼
  memory_recall → rag_retrieve → router
                                   │
              ┌────────────────────┼─────────────────┐
              ▼                    ▼                  ▼
           [chain]             [planner]          [dispatch]
              │                    │                  │(Send×N)
              ▼               [executor]◄──[tools]    ▼
          code_check               │[end]       [parallel_worker]
              │                    ▼                  │
              ▼                code_check         [aggregate]
           [reflect]◄──────────────┘                  │
              │                                        ▼
              │[should_reflect]                    code_check
              ▼                                        │
           [hitl]◄──────────────────────────────────[reflect]
         ┌────┴──────┐
     [rejected]  [approved/skip]
         │            │
         ▼            ▼
      [chain]    [output_guard]
                 ┌──────┴──────┐
              [pass]         [fail]
                 │              │
       [memory_consolidate]  [recovery]
                 │           ┌──┼──────┐
                END       [retry] [end] [escalate→hitl]

路由标签:
  "chain"    — 单轮问答 / 内容生成
  "plan"     — 多步骤任务（规划 + 工具）
  "parallel" — 独立子任务并行处理
  "general"  — 兜底，等同 chain
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from agent_infra.core.state import AgentState
from agent_infra.core.tools import get_tool_node
from agent_infra.memory.store import MemoryNode, MemoryStore
from agent_infra.patterns.chaining import ChainNode, ChainStep
from agent_infra.patterns.project_builder import FileApplyNode, ProjectBuilderNode
from agent_infra.patterns.perspective_engine import (
    PerspectiveDispatchNode,
    PerspectivePlannerNode,
    PerspectiveSynthesisNode,
    PerspectiveWorkerNode,
    perspective_dispatch_edge,
)
from agent_infra.patterns.code_check import CodeCheckNode
from agent_infra.patterns.guardrails import InputGuardNode, OutputGuardNode, guard_edge
from agent_infra.patterns.hitl import HITLNode, TriggerMode, hitl_edge
from agent_infra.patterns.multi_agent import SpecialistNode, SupervisorNode, supervisor_edge
from agent_infra.patterns.parallelization import (
    AggregateNode, ParallelDispatchNode, ParallelWorkerNode, dispatch_edge,
)
from agent_infra.patterns.planning import ExecutorNode, PlannerNode, should_continue_plan
from agent_infra.patterns.rag import RAGNode, RAGStore
from agent_infra.patterns.recovery import RecoveryNode, recovery_edge, with_recovery
from agent_infra.patterns.reflection import CodeReflectionNode, ReflectionNode, should_reflect
from agent_infra.patterns.routing import RouterNode, route_edge


# ── 默认路由配置 ───────────────────────────────────────────────────────────────

DEFAULT_ROUTES = {
    "chain":    "单轮问答、摘要、短内容生成",
    "plan":     "复杂多步骤任务，需要规划、研究或工具调用",
    "parallel": "可拆分为多个独立子任务的批量处理请求",
    "project":  "创建新项目、生成完整应用代码、脚手架构建、从零搭建系统",
    "debate":   "多视角分析、辩证思考、需要从不同角度理解的议题、争议性话题、利弊权衡",
}

# ── 默认 Chain 步骤 ────────────────────────────────────────────────────────────

DEFAULT_CHAIN_STEPS = [
    ChainStep(
        name="analyse",
        system_prompt="分析用户请求：核心问题是什么，有哪些约束，期望输出格式是什么。",
    ),
    ChainStep(
        name="draft",
        system_prompt="根据以下分析，生成准确、结构清晰的回答。",
        depends_on=["analyse"],
    ),
    ChainStep(
        name="polish",
        system_prompt="审查草稿，修正错误，提升表达清晰度，输出最终答案（不要包含分析过程）。",
        depends_on=["draft"],
    ),
]

# ── 初始状态默认值（覆盖所有 AgentState 字段）────────────────────────────────

EMPTY_STATE: dict[str, Any] = {
    # conversation
    "messages": [],
    # planning
    "plan": [],
    "current_step": 0,
    # chaining
    "chain_context": {},
    # reflection
    "reflection": None,
    "reflection_count": 0,
    "reflection_source": None,
    # routing
    "route": None,
    # tool results
    "step_results": [],
    # memory
    "working_memory": [],
    # rag
    "rag_context": [],
    # parallelization
    "parallel_tasks": [],
    "parallel_results": [],
    # hitl
    "hitl_payload": None,
    "hitl_response": None,
    "requires_human": False,
    # multi-agent
    "active_agent": None,
    "agent_handoffs": [],
    # recovery
    "retry_count": 0,
    "last_error": None,
    "recovery_strategy": None,
    # guardrails
    "guard_input_passed": True,
    "guard_output_passed": True,
    "guard_reason": None,
    # execution & filesystem
    "execution_result": None,
    "tool_errors": [],
    "file_changes": [],
    # control
    "error": None,
    "finished": False,
}


# ── 图构建 ────────────────────────────────────────────────────────────────────

def build_agent(
    # 内容配置
    routes: dict[str, str] | None = None,
    chain_steps: list[ChainStep] | None = None,
    allowed_topics: list[str] | None = None,
    # RAG
    rag_store: RAGStore | None = None,
    # 反思
    reflection_criteria: str | None = None,
    max_reflection_iterations: int = 2,
    code_mode: bool = False,          # True → 使用 CodeReflectionNode + CodeCheckNode
    run_code_in_check: bool = False,  # True → code_check 节点真实执行代码
    # HITL
    hitl_mode: TriggerMode = "conditional",
    # 恢复
    max_retries: int = 3,
    # 模型
    model: str = "claude-sonnet-4-6",
    # 记忆
    memory_namespace: tuple[str, ...] = ("default", "memories"),
    # LangGraph 基础设施
    checkpointer: Any = None,
) -> Any:
    """
    构建并编译完整 Agent 图。

    Args:
        routes:                   路由标签 → 描述
        chain_steps:              Prompt Chaining 步骤
        allowed_topics:           InputGuard 白名单主题（空=不限）
        rag_store:                预填充 RAGStore
        reflection_criteria:      自定义质量评审标准
        max_reflection_iterations:反思最大循环次数
        code_mode:                True 时启用代码专用反思 + CodeCheckNode
        run_code_in_check:        True 时 CodeCheckNode 真实执行代码拿 stderr
        hitl_mode:                HITL 触发模式 (always|conditional|sampling)
        max_retries:              异常恢复最大重试次数
        model:                    Claude 模型 ID
        memory_namespace:         长期记忆命名空间
        checkpointer:             LangGraph checkpointer（默认内存）
    """
    # ── 实例化各组件 ──────────────────────────────────────────────────────────
    lt_store = InMemoryStore()
    memory_store = MemoryStore(lt_store)
    memory_node = MemoryNode(store=memory_store, namespace=memory_namespace, model=model)

    input_guard = InputGuardNode(allowed_topics=allowed_topics, model=model)
    output_guard = OutputGuardNode(model=model)
    rag_node = RAGNode(rag_store or RAGStore())
    router = RouterNode(routes=routes or DEFAULT_ROUTES, default_route="chain", model=model)
    chain_node = ChainNode(steps=chain_steps or DEFAULT_CHAIN_STEPS, model=model)
    planner = PlannerNode(model=model)
    executor = with_recovery(ExecutorNode(model=model))
    tool_node = get_tool_node()

    # code_check: 生成后立即做静态分析 + 可选执行，把真实错误写入 tool_errors
    code_check = CodeCheckNode(run_code=run_code_in_check, run_lint=True)

    # reflection: code_mode 时使用 CodeReflectionNode（额外做语法检查）
    reflect_kwargs: dict[str, Any] = {
        "model": model,
        "max_iterations": max_reflection_iterations,
        "code_mode": code_mode,
    }
    if reflection_criteria:
        reflect_kwargs["criteria"] = reflection_criteria
    reflector = CodeReflectionNode(**reflect_kwargs) if code_mode else ReflectionNode(**reflect_kwargs)

    hitl_node = HITLNode(mode=hitl_mode)
    recovery_node = RecoveryNode(max_retries=max_retries, model=model)
    dispatch_node = ParallelDispatchNode()
    parallel_worker = ParallelWorkerNode(model=model)
    aggregate_node = AggregateNode(model=model)
    project_builder = ProjectBuilderNode(model=model)
    file_apply = FileApplyNode()
    perspective_planner   = PerspectivePlannerNode(model=model)
    perspective_dispatch  = PerspectiveDispatchNode()
    perspective_worker    = PerspectiveWorkerNode(model=model)
    perspective_synthesis = PerspectiveSynthesisNode(model=model)
    cp = checkpointer or MemorySaver()

    # ── 构建图 ────────────────────────────────────────────────────────────────
    g = StateGraph(AgentState)

    # ── 节点注册 ──────────────────────────────────────────────────────────────
    g.add_node("input_guard",        input_guard)
    g.add_node("memory_recall",      memory_node.recall)
    g.add_node("rag_retrieve",       rag_node)
    g.add_node("router",             router)
    g.add_node("chain",              chain_node)
    g.add_node("planner",            planner)
    g.add_node("executor",           executor)
    g.add_node("tools",              tool_node)
    g.add_node("dispatch",           dispatch_node)        # parallel 扇出入口
    g.add_node("parallel_worker",    parallel_worker)
    g.add_node("aggregate",          aggregate_node)
    g.add_node("project_builder",       project_builder)       # 项目创建矩阵 agent
    g.add_node("file_apply",            file_apply)            # HITL 批准后落盘
    g.add_node("perspective_planner",   perspective_planner)   # 多视角：规划视角
    g.add_node("perspective_dispatch",  perspective_dispatch)   # 多视角：扇出
    g.add_node("perspective_worker",    perspective_worker)     # 多视角：单视角 worker
    g.add_node("perspective_synthesis", perspective_synthesis)  # 多视角：整合
    g.add_node("code_check",            code_check)             # 生成后质量门
    g.add_node("reflect",            reflector)
    g.add_node("hitl",               hitl_node)
    g.add_node("output_guard",       output_guard)
    g.add_node("recovery",           recovery_node)
    g.add_node("memory_consolidate", memory_node.consolidate)

    # ── 边定义 ────────────────────────────────────────────────────────────────

    # 入口链
    g.add_edge(START, "input_guard")
    g.add_conditional_edges(
        "input_guard", guard_edge,
        {"blocked": END, "continue": "memory_recall"},
    )
    g.add_edge("memory_recall", "rag_retrieve")
    g.add_edge("rag_retrieve", "router")

    # 路由分叉（单一出口，不重复注册）
    g.add_conditional_edges(
        "router", route_edge,
        {
            "chain":    "chain",
            "plan":     "planner",
            "parallel": "dispatch",
            "project":  "project_builder",
            "debate":   "perspective_planner",
            "general":  "chain",
        },
    )

    # chain 路径: chain → code_check → reflect
    g.add_edge("chain", "code_check")

    # plan 路径: planner → executor ⇆ tools → code_check → reflect
    g.add_edge("planner", "executor")
    g.add_conditional_edges(
        "executor", should_continue_plan,
        {"tools": "tools", "execute": "executor", "end": "code_check"},
    )
    g.add_edge("tools", "executor")

    # parallel 路径: dispatch →(Send)→ parallel_worker → aggregate → code_check → reflect
    g.add_conditional_edges(
        "dispatch", dispatch_edge,
        {"parallel_worker": "parallel_worker", "aggregate": "aggregate"},
    )
    g.add_edge("parallel_worker", "aggregate")
    g.add_edge("aggregate", "code_check")

    # project 路径: project_builder → code_check（lint 生成文件）→ reflect → hitl → file_apply
    g.add_edge("project_builder", "code_check")

    # debate 路径: perspective_planner → dispatch →(Send×N)→ worker → synthesis → reflect → hitl
    g.add_edge("perspective_planner", "perspective_dispatch")
    g.add_conditional_edges(
        "perspective_dispatch", perspective_dispatch_edge,
        {"perspective_worker": "perspective_worker", "perspective_synthesis": "perspective_synthesis"},
    )
    g.add_edge("perspective_worker",    "perspective_synthesis")
    g.add_edge("perspective_synthesis", "reflect")

    # code_check → reflect（所有生成路径的汇合点）
    g.add_edge("code_check", "reflect")

    # 反思循环: reflect → [hitl | 回到生成]
    g.add_conditional_edges(
        "reflect", should_reflect,
        {"reflect": "hitl", "end": "hitl"},
    )

    # HITL 分叉
    # project 路径：approved → file_apply（写盘）→ output_guard
    # 其他路径：approved → output_guard（file_apply 无 file_changes 时透传）
    g.add_conditional_edges(
        "hitl", hitl_edge,
        {"approved": "file_apply", "rejected": "chain", "skip": "output_guard"},
    )
    g.add_edge("file_apply", "output_guard")

    # 输出护栏
    g.add_conditional_edges(
        "output_guard",
        lambda s: "pass" if s.get("guard_output_passed", True) else "fail",
        {"pass": "memory_consolidate", "fail": "recovery"},
    )

    # 异常恢复
    g.add_conditional_edges(
        "recovery", recovery_edge,
        {"retry": "executor", "end": "memory_consolidate", "escalate": "hitl"},
    )

    g.add_edge("memory_consolidate", END)

    return g.compile(checkpointer=cp, store=lt_store)


# ── 同步运行器 ────────────────────────────────────────────────────────────────

def run(
    agent: Any,
    user_message: str,
    thread_id: str = "default",
    extra_state: dict[str, Any] | None = None,
) -> str:
    """同步调用 agent，返回最终文本回复。"""
    config = {"configurable": {"thread_id": thread_id}}
    state = {**EMPTY_STATE, "messages": [HumanMessage(content=user_message)], **(extra_state or {})}
    result = agent.invoke(state, config=config)
    return _extract_last_ai(result)


# ── 流式运行器 ────────────────────────────────────────────────────────────────

def stream(
    agent: Any,
    user_message: str,
    thread_id: str = "default",
    extra_state: dict[str, Any] | None = None,
) -> Iterator[str]:
    """
    同步流式输出，逐 token 产出，适合 CLI / SSE。

    用法:
        for token in stream(agent, "写一个冒泡排序"):
            print(token, end="", flush=True)
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = {**EMPTY_STATE, "messages": [HumanMessage(content=user_message)], **(extra_state or {})}
    for chunk, _ in agent.stream(state, config=config, stream_mode="messages"):
        if isinstance(chunk, AIMessage) and chunk.content:
            yield str(chunk.content)


async def astream(
    agent: Any,
    user_message: str,
    thread_id: str = "default",
    extra_state: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """
    异步流式输出，适合 FastAPI WebSocket / Server-Sent Events。

    用法:
        async for token in astream(agent, "写一个冒泡排序"):
            await websocket.send_text(token)
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = {**EMPTY_STATE, "messages": [HumanMessage(content=user_message)], **(extra_state or {})}
    async for chunk, _ in agent.astream(state, config=config, stream_mode="messages"):
        if isinstance(chunk, AIMessage) and chunk.content:
            yield str(chunk.content)


# ── HITL 恢复接口 ─────────────────────────────────────────────────────────────

def resume_after_hitl(
    agent: Any,
    thread_id: str,
    approved: bool,
    feedback: str = "",
    edited_content: str | None = None,
) -> str:
    """
    人类审核后恢复图执行。

    Args:
        approved:       是否批准当前输出
        feedback:       文字反馈（拒绝时说明原因）
        edited_content: 人类直接修改后的内容（可选，写入 chain_context）
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        None,
        config=config,
        command=Command(resume={
            "approved": approved,
            "feedback": feedback,
            "edited_content": edited_content,
        }),
    )
    return _extract_last_ai(result)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _extract_last_ai(result: dict[str, Any]) -> str:
    for msg in reversed(result.get("messages") or []):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return result.get("guard_reason") or ""
