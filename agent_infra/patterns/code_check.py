"""
CodeCheckNode — 代码生成后的自动质量门
=========================================
衔接「生成」与「反思」两个节点：

  [ChainNode / ExecutorNode 生成代码]
          ↓
  [CodeCheckNode]  ← 本节点
    1. 从 AI 消息中提取代码块
    2. 语法检查（AST，无需外部工具）
    3. Lint（ruff/flake8/pyflakes，自动降级）
    4. 可选：在沙箱中真实执行
    5. 将所有错误写入 state["tool_errors"]
          ↓
  [ReflectionNode]  读 tool_errors → 直接用真实错误作 critique

这是"编译器不会幻觉"原则的执行层。
无工具错误时 ReflectionNode 降级为 LLM 自我评审。
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from agent_infra.core.state import AgentState
from agent_infra.execution.code_tools import CodeToolRunner
from agent_infra.execution.sandbox import Sandbox


class CodeCheckNode:
    """
    代码质量自动检查节点。

    Args:
        run_code:    是否在沙箱中真实执行提取到的代码（默认 False，仅静态检查）
        run_lint:    是否运行 lint 检查（默认 True）
        language:    默认代码语言（用于无 fence 标记的代码块）
        timeout_s:   沙箱执行超时秒数
    """

    def __init__(
        self,
        run_code: bool = False,
        run_lint: bool = True,
        language: str = "python",
        timeout_s: int = 15,
    ) -> None:
        self.run_code = run_code
        self.run_lint = run_lint
        self.language = language
        self._sandbox = Sandbox(timeout_s=timeout_s)
        self._runner = CodeToolRunner(sandbox=self._sandbox)

    def __call__(self, state: AgentState) -> dict[str, Any]:
        content = _latest_ai_content(state)
        blocks = _extract_code_blocks(content)

        if not blocks:
            # 没有代码块，不做检查，清空旧错误
            return {"tool_errors": [], "execution_result": None}

        all_errors: list[str] = []
        last_exec: dict | None = None

        for lang, code in blocks:
            effective_lang = lang or self.language

            # 1. 语法检查（始终执行，零外部依赖）
            syntax = self._runner.syntax_check(code, effective_lang)
            if not syntax.passed:
                all_errors.extend(syntax.errors)
                continue  # 语法错了就不继续后续检查

            # 2. Lint（写入临时文件再检查）
            if self.run_lint and effective_lang == "python":
                tmp_path = self._write_temp(code, ".py")
                lint_result = self._runner.lint(tmp_path)
                if not lint_result.passed:
                    all_errors.extend(lint_result.errors[:15])

            # 3. 真实执行（可选）
            if self.run_code:
                exec_result = self._sandbox.run_code(code, effective_lang)
                last_exec = exec_result.to_dict()
                if not exec_result.success:
                    error_lines = [
                        l for l in exec_result.stderr.splitlines() if l.strip()
                    ][-15:]
                    all_errors.extend(error_lines)

        return {
            "tool_errors": all_errors,
            "execution_result": last_exec,
        }

    def _write_temp(self, code: str, suffix: str) -> str:
        with tempfile.NamedTemporaryFile(
            suffix=suffix, mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            return f.name


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _latest_ai_content(state: AgentState) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return ""


def _extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """
    提取所有 Markdown 代码块，返回 [(language, code)] 列表。
    无 fence 标记时 language 为空字符串。
    """
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(lang.lower(), code.strip()) for lang, code in matches if code.strip()]
