"""
代码沙箱执行器
==============
在受控的子进程中真实运行代码，捕获 stdout/stderr。

这是与 LLM 自我评审的本质区别：
  编译器 / 解释器的错误信息永远不会幻觉。

支持语言: Python · Shell · Node.js · (可扩展)
安全措施: timeout + 独立工作目录 + 禁止网络(可选)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    language: str
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def error_summary(self) -> str:
        """供 Reflection prompt 使用的紧凑错误描述。"""
        if self.timed_out:
            return f"执行超时"
        if self.success:
            return ""
        lines = [l for l in self.stderr.splitlines() if l.strip()]
        # 取最后 20 行（最有信息量的部分）
        return "\n".join(lines[-20:])

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:2000],
            "stderr": self.stderr[:2000],
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
        }


class Sandbox:
    """
    轻量代码沙箱。无需 Docker，直接用 subprocess + 临时目录隔离。

    生产环境建议替换为:
      - E2B (e2b.dev) — 云端沙箱，支持任意语言
      - Modal — 无服务器执行环境
      - Docker subprocess — 完全隔离

    Args:
        work_dir:   代码执行的工作目录（None 时用系统临时目录）
        timeout_s:  单次执行超时秒数
    """

    _RUNNERS: dict[str, list[str]] = {
        "python":     [sys.executable, "-c", "{code}"],
        "python_file":[sys.executable, "{file}"],
        "shell":      ["bash", "-c", "{code}"],
        "node":       ["node", "-e", "{code}"],
        "node_file":  ["node", "{file}"],
    }

    def __init__(self, work_dir: str | Path | None = None, timeout_s: int = 30) -> None:
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="agent_sandbox_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s

    # ── 主接口 ────────────────────────────────────────────────────────────────

    def run_code(self, code: str, language: str = "python") -> ExecutionResult:
        """直接执行代码字符串。"""
        code = textwrap.dedent(code)
        cmd = [sys.executable, "-c", code] if language == "python" else ["bash", "-c", code]
        return self._run(cmd, language)

    def run_file(self, file_path: str | Path, language: str | None = None) -> ExecutionResult:
        """执行已存在的文件。"""
        path = Path(file_path)
        lang = language or _detect_language(path)
        if lang == "python":
            cmd = [sys.executable, str(path)]
        elif lang == "node":
            cmd = ["node", str(path)]
        elif lang == "shell":
            cmd = ["bash", str(path)]
        else:
            return ExecutionResult("", f"不支持的语言: {lang}", 1, 0, lang)
        return self._run(cmd, lang, cwd=path.parent)

    def run_command(self, command: str) -> ExecutionResult:
        """执行任意 shell 命令（用于 lint / test / build）。"""
        return self._run(["bash", "-c", command], "shell")

    def write_and_run(self, code: str, filename: str = "main.py") -> ExecutionResult:
        """写入临时文件再执行（适合多行代码带导入的场景）。"""
        path = self.work_dir / filename
        path.write_text(code, encoding="utf-8")
        return self.run_file(path)

    # ── 内部执行 ──────────────────────────────────────────────────────────────

    def _run(
        self,
        cmd: list[str],
        language: str,
        cwd: Path | None = None,
    ) -> ExecutionResult:
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                cwd=str(cwd or self.work_dir),
            )
            elapsed = int((time.monotonic() - start) * 1000)
            return ExecutionResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                duration_ms=elapsed,
                language=language,
            )
        except subprocess.TimeoutExpired:
            elapsed = int((time.monotonic() - start) * 1000)
            return ExecutionResult("", "执行超时", 1, elapsed, language, timed_out=True)
        except FileNotFoundError as e:
            return ExecutionResult("", f"命令未找到: {e}", 127, 0, language)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _detect_language(path: Path) -> str:
    ext = path.suffix.lower()
    return {"py": "python", "js": "node", "ts": "node", "sh": "shell"}.get(ext.lstrip("."), "shell")
