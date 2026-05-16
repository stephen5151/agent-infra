"""
代码质量工具运行器
===================
自动检测并运行 lint / type-check / test 工具，
将真实错误输出转化为结构化的 ToolCheckResult。

这是"编译器不会幻觉"原则的实现：
  LLM 生成代码 → 工具检查 → 真实错误 → 注入 Reflection prompt → LLM 定向修复

工具优先级（自动降级）:
  Lint:        ruff → flake8 → pyflakes → ast.parse（兜底）
  Type-check:  mypy → pyright
  Test:        pytest → unittest
  Format-check:ruff format → black
  JS/TS:       eslint → tsc
"""
from __future__ import annotations

import ast
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from agent_infra.execution.sandbox import Sandbox


@dataclass
class ToolCheckResult:
    tool: str           # 工具名称
    passed: bool
    output: str         # 完整输出
    errors: list[str]   # 解析出的错误行（供 Reflection 使用）
    warnings: list[str] = field(default_factory=list)

    @property
    def error_summary(self) -> str:
        """紧凑的错误摘要，直接注入 Reflection prompt。"""
        if self.passed:
            return ""
        lines = self.errors[:20]   # 最多 20 条
        header = f"[{self.tool}] 发现 {len(self.errors)} 个问题:"
        return header + "\n" + "\n".join(lines)


class CodeToolRunner:
    """
    代码工具运行器。自动选择本地已安装的工具，不强制依赖任何特定工具。

    Args:
        sandbox:  Sandbox 实例（用于执行子进程）
        work_dir: 代码工作目录
    """

    def __init__(self, sandbox: Sandbox | None = None, work_dir: str | Path = ".") -> None:
        self._sb = sandbox or Sandbox(work_dir=work_dir)
        self.work_dir = Path(work_dir)

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def lint(self, path: str | Path) -> ToolCheckResult:
        """静态分析：找出语法错误、未使用变量、风格问题等。"""
        p = Path(path)
        if p.suffix == ".py":
            return self._lint_python(p)
        if p.suffix in (".js", ".ts"):
            return self._lint_js(p)
        return ToolCheckResult("none", True, "不支持的文件类型", [])

    def type_check(self, path: str | Path) -> ToolCheckResult:
        """类型检查：发现类型不匹配、未定义变量等。"""
        p = Path(path)
        if p.suffix == ".py":
            return self._typecheck_python(p)
        return ToolCheckResult("none", True, "类型检查仅支持 Python", [])

    def run_tests(self, test_path: str | Path | None = None) -> ToolCheckResult:
        """运行测试套件，返回真实的测试失败信息。"""
        target = str(test_path or self.work_dir)
        if shutil.which("pytest"):
            return self._run_pytest(target)
        return self._run_unittest(target)

    def syntax_check(self, code: str, language: str = "python") -> ToolCheckResult:
        """无需任何外部工具的最快检查：纯 AST 语法解析。"""
        if language == "python":
            return self._ast_check(code)
        # JS/TS: 用 node --check
        if shutil.which("node") and language in ("javascript", "typescript"):
            result = self._sb.run_code(f"require('vm').Script('{code.replace(chr(39), chr(34))}')", "node")
            ok = result.success
            return ToolCheckResult("node-syntax", ok, result.stderr, [] if ok else [result.stderr])
        return ToolCheckResult("none", True, "无法检查语法", [])

    def check_all(self, path: str | Path) -> list[ToolCheckResult]:
        """运行所有可用检查，返回结果列表。"""
        results = [self.syntax_check(Path(path).read_text(), _lang(Path(path)))]
        results.append(self.lint(path))
        if Path(path).suffix == ".py":
            results.append(self.type_check(path))
        return [r for r in results if r.tool != "none"]

    # ── Python 工具 ───────────────────────────────────────────────────────────

    def _lint_python(self, path: Path) -> ToolCheckResult:
        # 优先级：ruff → flake8 → pyflakes → ast（兜底）
        for tool_name, cmd in [
            ("ruff",     f"ruff check --output-format=text {path}"),
            ("flake8",   f"flake8 {path}"),
            ("pyflakes", f"pyflakes {path}"),
        ]:
            if shutil.which(tool_name):
                res = self._sb.run_command(cmd)
                errors = [l for l in res.stderr.splitlines() if l.strip() and "error" in l.lower()] \
                       + [l for l in res.stdout.splitlines() if l.strip()]
                return ToolCheckResult(tool_name, res.success, res.stdout + res.stderr, errors)

        # 兜底：ast.parse
        return self._ast_check(path.read_text())

    def _typecheck_python(self, path: Path) -> ToolCheckResult:
        for tool_name, cmd in [
            ("mypy",    f"mypy {path} --ignore-missing-imports"),
            ("pyright", f"pyright {path}"),
        ]:
            if shutil.which(tool_name):
                res = self._sb.run_command(cmd)
                errors = [l for l in res.stdout.splitlines() if "error:" in l]
                return ToolCheckResult(tool_name, res.success, res.stdout, errors)
        return ToolCheckResult("none", True, "未安装类型检查工具", [])

    def _ast_check(self, code: str) -> ToolCheckResult:
        try:
            ast.parse(code)
            return ToolCheckResult("ast", True, "语法正确", [])
        except SyntaxError as e:
            msg = f"第 {e.lineno} 行: {e.msg}  →  {e.text or ''}"
            return ToolCheckResult("ast", False, msg, [msg])

    def _run_pytest(self, target: str) -> ToolCheckResult:
        res = self._sb.run_command(f"pytest {target} -v --tb=short 2>&1")
        output = res.stdout + res.stderr
        errors = [l for l in output.splitlines() if "FAILED" in l or "ERROR" in l]
        passed = res.success and "failed" not in output.lower()
        return ToolCheckResult("pytest", passed, output, errors)

    def _run_unittest(self, target: str) -> ToolCheckResult:
        res = self._sb.run_command(f"python3 -m unittest discover {target} 2>&1")
        output = res.stdout + res.stderr
        errors = [l for l in output.splitlines() if "FAIL:" in l or "ERROR:" in l]
        return ToolCheckResult("unittest", res.success, output, errors)

    # ── JS/TS 工具 ────────────────────────────────────────────────────────────

    def _lint_js(self, path: Path) -> ToolCheckResult:
        for tool_name, cmd in [
            ("eslint", f"eslint {path}"),
            ("tsc",    f"tsc --noEmit {path}"),
        ]:
            if shutil.which(tool_name):
                res = self._sb.run_command(cmd)
                errors = [l for l in (res.stdout + res.stderr).splitlines() if "error" in l.lower()]
                return ToolCheckResult(tool_name, res.success, res.stdout + res.stderr, errors)
        return ToolCheckResult("none", True, "未安装 JS lint 工具", [])


def _lang(path: Path) -> str:
    return {"py": "python", "js": "javascript", "ts": "typescript"}.get(path.suffix.lstrip("."), "unknown")
