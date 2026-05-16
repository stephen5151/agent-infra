"""
Chapter 4 — Reflection（升级版：真实错误优先）
================================================
反思循环的核心升级：

  旧版：LLM 评审 LLM 输出 → 容易放水 / 幻觉
  新版：真实工具错误 → 注入 critique（编译器不会幻觉）
        无工具错误时 → 降级为 LLM 自我评审

优先级:
  1. state["tool_errors"]   — lint/test/compile 的真实 stderr（最可靠）
  2. state["execution_result"]["stderr"] — 运行时崩溃信息（可靠）
  3. LLM 自我评审            — 兜底，用于逻辑/内容质量检查

这是与 Cursor / Claude Code 对齐的核心机制：
  让编译器、测试框架、linter 驱动修复循环，而非 LLM 猜测哪里有问题。
"""
from __future__ import annotations

import re
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agent_infra.core.llm import get_llm
from agent_infra.core.state import AgentState


_DEFAULT_CRITERIA = """
1. 事实准确 — 无幻觉，逻辑自洽。
2. 完整性   — 覆盖问题的所有部分。
3. 清晰度   — 条理清晰，易于理解。
4. 简洁性   — 无冗余重复。
""".strip()

_CODE_CRITERIA = """
1. 语法正确  — 代码可解析，无语法错误。
2. 逻辑正确  — 实现符合需求，边界情况处理完整。
3. 无运行错误 — 执行后 exit_code == 0。
4. 代码风格  — 变量命名清晰，无明显代码坏味道。
""".strip()


class ReflectionVerdict(BaseModel):
    passed: bool = Field(description="输出是否达到质量标准")
    critique: str = Field(description="具体、可执行的改进建议")
    score: int = Field(ge=1, le=10, description="整体质量评分 1-10")
    source: str = Field(description="评审来源: tool | llm", default="llm")


class ReflectionNode:
    """
    双轨反思节点。

    Track 1 — 工具错误驱动（真实 stderr/lint 输出）
      有 tool_errors 或 execution_result 失败时直接使用，跳过 LLM 评审。

    Track 2 — LLM 自我评审（兜底）
      无工具错误时由 LLM 对照 criteria 打分，生成 critique。

    两种 critique 都写入 state["reflection"]，
    生成节点（ChainNode / ExecutorNode）统一读取并注入下轮提示词。
    """

    def __init__(
        self,
        criteria: str = _DEFAULT_CRITERIA,
        code_mode: bool = False,
        max_iterations: int = 3,
        pass_threshold: int = 7,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self.criteria = _CODE_CRITERIA if code_mode else criteria
        self.max_iterations = max_iterations
        self.pass_threshold = pass_threshold
        self.llm = get_llm(model=model).with_structured_output(ReflectionVerdict)

    def __call__(self, state: AgentState) -> dict[str, Any]:
        count = (state.get("reflection_count") or 0) + 1

        # ── Track 1: 工具错误优先 ─────────────────────────────────────────────
        tool_critique = self._collect_tool_errors(state)
        if tool_critique:
            return {
                "reflection": tool_critique,
                "reflection_count": count,
                "reflection_source": "tool",
                "finished": False,
            }

        # ── Track 2: LLM 自我评审（兜底）─────────────────────────────────────
        verdict = self._llm_review(state)
        passed = verdict.passed and verdict.score >= self.pass_threshold
        return {
            "reflection": verdict.critique if not passed else None,
            "reflection_count": count,
            "reflection_source": "llm",
            "finished": passed,
        }

    # ── Track 1 实现 ──────────────────────────────────────────────────────────

    def _collect_tool_errors(self, state: AgentState) -> str:
        """
        从 state 中收集所有真实工具错误，合并为一条 critique 字符串。
        有任何真实错误就走此路径，跳过 LLM 评审。
        """
        parts: list[str] = []

        # lint / test / compile 错误
        tool_errors: list[str] = state.get("tool_errors") or []
        if tool_errors:
            parts.append("[静态分析 / 测试错误]\n" + "\n".join(tool_errors[:25]))

        # 运行时错误
        exec_result: dict | None = state.get("execution_result")
        if exec_result and not exec_result.get("success", True):
            stderr = exec_result.get("stderr", "").strip()
            if exec_result.get("timed_out"):
                parts.append("[运行超时] 代码未在限定时间内完成，检查死循环或性能问题。")
            elif stderr:
                # 提取最有信息量的最后 20 行
                error_lines = [l for l in stderr.splitlines() if l.strip()][-20:]
                parts.append("[运行时错误]\n" + "\n".join(error_lines))

        return "\n\n".join(parts)

    # ── Track 2 实现 ──────────────────────────────────────────────────────────

    def _llm_review(self, state: AgentState) -> ReflectionVerdict:
        latest_output = _latest_ai_message(state)
        user_request = _latest_human_message(state)

        system = (
            "你是严格的质量审核员。根据以下标准评审 AI 回复：\n\n"
            f"{self.criteria}\n\n"
            "返回 JSON: passed(bool), critique(str), score(1-10), source='llm'"
        )
        human = f"用户请求:\n{user_request}\n\nAI 回复:\n{latest_output}"
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        return (prompt | self.llm).invoke({})


# ── 代码专用 Reflection（code_mode=True 的快捷方式）───────────────────────────

class CodeReflectionNode(ReflectionNode):
    """
    专为代码生成场景设计的反思节点。
    自动提取 AI 消息中的代码块做语法检查，再走双轨反思。
    """

    def __call__(self, state: AgentState) -> dict[str, Any]:
        # 先做轻量语法检查，结果写入 tool_errors
        updated_state = dict(state)
        syntax_errors = self._quick_syntax_check(state)
        if syntax_errors:
            existing = list(state.get("tool_errors") or [])
            updated_state["tool_errors"] = existing + syntax_errors

        return super().__call__(updated_state)  # type: ignore[arg-type]

    def _quick_syntax_check(self, state: AgentState) -> list[str]:
        """从最新 AI 消息提取代码块并做 AST 语法检查。"""
        import ast
        content = _latest_ai_message(state)
        errors: list[str] = []
        for block in _extract_code_blocks(content, "python"):
            try:
                ast.parse(block)
            except SyntaxError as e:
                errors.append(f"语法错误 第{e.lineno}行: {e.msg} → {e.text or ''}")
        return errors


# ── 条件边 ────────────────────────────────────────────────────────────────────

def should_reflect(state: AgentState) -> Literal["reflect", "end"]:
    """
    反思前的门控边：硬上限保护 + 快速通过判断。

    - finished=True  → 直接 end
    - 超过最大次数   → 强制 end
    - 否则          → 进入 reflect 节点
    """
    if state.get("finished"):
        return "end"
    count = state.get("reflection_count") or 0
    if count >= 3:
        return "end"
    return "reflect"


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _latest_ai_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""


def _latest_human_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _extract_code_blocks(text: str, language: str = "") -> list[str]:
    """从 Markdown 代码块中提取代码。"""
    pattern = rf"```{language}\n(.*?)```" if language else r"```(?:\w+)?\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)
