"""
工具定义 — Chapter 5: Tool Use / Function Calling
===================================================
所有工具通过 @tool 装饰器注册，ToolNode 自动发现并执行。

分三层：
  通用工具   — 时间、计算、搜索（始终可用）
  执行工具   — 代码运行、命令执行（需要 Sandbox）
  文件工具   — 读写文件、生成 diff（需要 FileSystem）
  代码质量   — lint、类型检查、测试（需要 CodeToolRunner）
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

# 延迟导入：避免循环依赖，工具被调用时才实例化
_sandbox = None
_filesystem = None
_code_runner = None


def _get_sandbox():
    global _sandbox
    if _sandbox is None:
        from agent_infra.execution.sandbox import Sandbox
        _sandbox = Sandbox()
    return _sandbox


def _get_fs(root: str = "."):
    from agent_infra.execution.filesystem import FileSystem
    return FileSystem(root=root)


def _get_runner():
    global _code_runner
    if _code_runner is None:
        from agent_infra.execution.code_tools import CodeToolRunner
        _code_runner = CodeToolRunner(sandbox=_get_sandbox())
    return _code_runner


# ══════════════════════════════════════════════════════════════════════════════
# 通用工具
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_current_time() -> str:
    """返回当前 UTC 时间（ISO-8601 格式）。"""
    return datetime.now(timezone.utc).isoformat()


@tool
def calculate(expression: str) -> str:
    """
    安全地计算简单数学表达式。

    Args:
        expression: Python 算术表达式，如 "2 ** 10 / 3"
    """
    allowed = set("0123456789+-*/().,% ")
    if not all(c in allowed for c in expression):
        return "Error: 包含不允许的字符"
    try:
        return str(eval(expression, {"__builtins__": {}}))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    执行网络搜索（DuckDuckGo，无需 API Key）。

    Args:
        query:       搜索关键词
        max_results: 最多返回结果数（默认 5）

    Returns:
        JSON，包含 query / results 列表，每项含 title / url / snippet
    """
    import urllib.parse
    import urllib.request

    url = (
        "https://api.duckduckgo.com/?q="
        + urllib.parse.quote(query)
        + "&format=json&no_redirect=1&no_html=1&skip_disambig=1"
    )
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "agent-infra/1.0 (personal AI assistant)"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return json.dumps({"query": query, "error": str(e), "results": []})

    results = []

    # 直接答案
    abstract = data.get("AbstractText", "")
    if abstract:
        results.append({
            "title": data.get("Heading", ""),
            "url": data.get("AbstractURL", ""),
            "snippet": abstract[:300],
        })

    # 相关主题
    for topic in data.get("RelatedTopics", [])[:max_results]:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({
                "title": topic.get("Text", "")[:80],
                "url": topic.get("FirstURL", ""),
                "snippet": topic.get("Text", "")[:200],
            })
        elif isinstance(topic, dict) and topic.get("Topics"):
            for sub in topic["Topics"][:2]:
                if sub.get("Text"):
                    results.append({
                        "title": sub.get("Text", "")[:80],
                        "url": sub.get("FirstURL", ""),
                        "snippet": sub.get("Text", "")[:200],
                    })

    results = results[:max_results]
    if not results:
        results = [{"title": "无结果", "url": "", "snippet": f"DuckDuckGo 未找到 '{query}' 的相关内容"}]

    return json.dumps({"query": query, "results": results}, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# 代码执行工具
# ══════════════════════════════════════════════════════════════════════════════

@tool
def execute_code(code: str, language: str = "python") -> str:
    """
    在沙箱中执行代码，返回 stdout/stderr 和退出码。

    Args:
        code:     要执行的代码字符串
        language: "python" | "shell" | "node"

    Returns:
        JSON 格式的执行结果，包含 success / stdout / stderr / exit_code
    """
    result = _get_sandbox().run_code(code, language)
    return json.dumps(result.to_dict(), ensure_ascii=False)


@tool
def execute_command(command: str) -> str:
    """
    执行 shell 命令（用于 build / install / migrate 等操作）。

    Args:
        command: Shell 命令字符串

    Returns:
        JSON 格式结果，包含 stdout / stderr / exit_code
    """
    result = _get_sandbox().run_command(command)
    return json.dumps(result.to_dict(), ensure_ascii=False)


@tool
def write_and_run(code: str, filename: str = "main.py") -> str:
    """
    将代码写入文件后执行，适合包含多模块导入的场景。

    Args:
        code:     完整的代码内容
        filename: 文件名（在沙箱工作目录中）
    """
    result = _get_sandbox().write_and_run(code, filename)
    return json.dumps(result.to_dict(), ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# 文件系统工具
# ══════════════════════════════════════════════════════════════════════════════

@tool
def read_file(path: str) -> str:
    """
    读取文件内容。

    Args:
        path: 相对于工作目录的文件路径
    """
    try:
        return _get_fs().read(path)
    except FileNotFoundError:
        return f"Error: 文件不存在 — {path}"
    except PermissionError as e:
        return f"Error: 权限拒绝 — {e}"


@tool
def list_files(directory: str = ".", pattern: str = "**/*") -> str:
    """
    列出目录中的文件。

    Args:
        directory: 目标目录路径
        pattern:   glob 模式，默认列出所有文件
    """
    try:
        files = _get_fs(directory).list_files(pattern)
        return json.dumps(files, ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str, auto_apply: bool = False) -> str:
    """
    生成文件写入 diff（不立即落盘）。auto_apply=True 时直接写入。

    Args:
        path:       目标文件路径
        content:    新的文件内容
        auto_apply: True 时立即写入磁盘；False 时仅返回 diff 供审核
    """
    try:
        fs = _get_fs()
        diff = fs.prepare_write(path, content)
        if auto_apply:
            fs.apply(diff)
            return json.dumps({"applied": True, "path": path,
                               "lines_added": diff.lines_added,
                               "lines_removed": diff.lines_removed})
        return json.dumps(diff.to_dict(), ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


@tool
def apply_diff(path: str, unified_diff: str) -> str:
    """
    将 unified diff 字符串应用到文件（先生成 diff 预览，不自动落盘）。

    Args:
        path:         目标文件路径
        unified_diff: unified diff 格式的字符串
    """
    try:
        fs = _get_fs()
        diff = fs.prepare_patch(path, unified_diff)
        return json.dumps(diff.to_dict(), ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# 代码质量工具
# ══════════════════════════════════════════════════════════════════════════════

@tool
def lint_code(path: str) -> str:
    """
    对文件运行静态分析（ruff / flake8 / pyflakes / ast，自动选择可用工具）。

    Args:
        path: 要检查的文件路径（支持 .py / .js / .ts）

    Returns:
        JSON，包含 passed / tool / errors 列表
    """
    result = _get_runner().lint(path)
    return json.dumps({
        "tool": result.tool,
        "passed": result.passed,
        "errors": result.errors[:30],
        "summary": result.error_summary,
    }, ensure_ascii=False)


@tool
def type_check(path: str) -> str:
    """
    运行类型检查（mypy / pyright，自动选择可用工具）。

    Args:
        path: Python 文件路径
    """
    result = _get_runner().type_check(path)
    return json.dumps({
        "tool": result.tool,
        "passed": result.passed,
        "errors": result.errors[:30],
    }, ensure_ascii=False)


@tool
def run_tests(test_path: str = ".") -> str:
    """
    运行测试套件（pytest / unittest，自动选择）。

    Args:
        test_path: 测试目录或文件路径，默认当前目录
    """
    result = _get_runner().run_tests(test_path)
    return json.dumps({
        "tool": result.tool,
        "passed": result.passed,
        "errors": result.errors[:30],
        "output": result.output[:3000],
    }, ensure_ascii=False)


@tool
def syntax_check(code: str, language: str = "python") -> str:
    """
    快速语法检查（无需外部工具，纯 AST 解析）。

    Args:
        code:     代码字符串
        language: "python" | "javascript" | "typescript"
    """
    result = _get_runner().syntax_check(code, language)
    return json.dumps({
        "passed": result.passed,
        "errors": result.errors,
    }, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# 知识库工具
# ══════════════════════════════════════════════════════════════════════════════

@tool
def search_knowledge_base(query: str, n_results: int = 5) -> str:
    """
    搜索 Obsidian 语义知识库（向量检索，需先运行 jarvis index）。

    Args:
        query:     自然语言检索词
        n_results: 最多返回结果数（默认 5）

    Returns:
        JSON，包含 query / results 列表，每项含 source / header / score / text
    """
    try:
        from agent_infra.memory.semantic import get_semantic_memory
        sem = get_semantic_memory()
        results = sem.search(query, n_results=n_results)
        if not results:
            return json.dumps({"query": query, "results": [],
                               "note": "知识库为空，请先运行 jarvis index"})
        return json.dumps({"query": query, "results": results}, ensure_ascii=False)
    except ImportError:
        return json.dumps({"query": query, "results": [],
                           "note": "向量检索未安装，运行: pip install chromadb sentence-transformers"})
    except Exception as e:
        return json.dumps({"query": query, "results": [], "error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# 工具集合
# ══════════════════════════════════════════════════════════════════════════════

# 通用工具：所有场景都加载
GENERAL_TOOLS = [get_current_time, calculate, web_search, search_knowledge_base]

# 执行工具：代码生成/调试场景
EXECUTION_TOOLS = [execute_code, execute_command, write_and_run]

# 文件工具：文件编辑场景
FILE_TOOLS = [read_file, list_files, write_file, apply_diff]

# 质量工具：代码质量检查场景
QUALITY_TOOLS = [lint_code, type_check, run_tests, syntax_check]

# 全量工具列表
TOOLS = GENERAL_TOOLS + EXECUTION_TOOLS + FILE_TOOLS + QUALITY_TOOLS


def get_tool_node(tools: list | None = None) -> ToolNode:
    """返回包含指定工具集的 ToolNode，默认使用全量工具。"""
    return ToolNode(tools or TOOLS)


def get_tool_node_for(mode: str) -> ToolNode:
    """
    按场景返回精简工具集，避免工具过多干扰模型选择。

    mode: "general" | "code" | "file" | "quality" | "all"
    """
    mapping: dict[str, list] = {
        "general": GENERAL_TOOLS,
        "code":    GENERAL_TOOLS + EXECUTION_TOOLS,
        "file":    GENERAL_TOOLS + FILE_TOOLS,
        "quality": QUALITY_TOOLS,
        "all":     TOOLS,
    }
    return ToolNode(mapping.get(mode, TOOLS))
