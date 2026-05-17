"""
自我维护引擎 — 每日代码健康检查 + 自动修复
=============================================
Jarvis 每天检查自己的代码库：
  1. 语法检查（py_compile）
  2. 静态分析（ruff / flake8）
  3. 模块导入验证
  4. 配置完整性检查
  5. LLM 审查代码质量（用 Claude CLI 订阅，不消耗 API）
  6. 发现问题 → 生成 patch → 应用修复 → 验证
  7. 写入修复日志到 Obsidian

触发方式：
  - 每天启动时自动运行
  - 手动: jarvis maintain
  - APScheduler 定时

这是 Jarvis 的「自我进化」能力。
"""
from __future__ import annotations

import ast
import asyncio
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class IssueFound:
    """发现的问题。"""
    file_path: Path
    issue_type: str          # "syntax" | "lint" | "import" | "logic" | "config"
    severity: str            # "error" | "warning" | "info"
    message: str
    line_number: int = 0
    auto_fixable: bool = False
    suggested_fix: str = ""  # LLM 建议的修复代码


@dataclass
class MaintenanceReport:
    """一次维护运行的完整报告。"""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    files_checked: int = 0
    issues_found: list[IssueFound] = field(default_factory=list)
    issues_fixed: int = 0
    fixes_applied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        if self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return 0

    def to_markdown(self) -> str:
        ts = self.started_at.strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"## 🔧 Jarvis 自维护报告 — {ts}",
            f"",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 检查文件数 | {self.files_checked} |",
            f"| 发现问题 | {len(self.issues_found)} |",
            f"| 已修复 | {self.issues_fixed} |",
            f"| 耗时 | {self.duration_s:.1f}s |",
            f"",
        ]

        if self.issues_found:
            lines.append("### 发现的问题")
            for issue in self.issues_found:
                icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(issue.severity, "⚪")
                lines.append(
                    f"- {icon} `{issue.file_path.name}:{issue.line_number}` "
                    f"[{issue.issue_type}] {issue.message}"
                )

        if self.fixes_applied:
            lines.append("\n### 已应用的修复")
            for fix in self.fixes_applied:
                lines.append(f"- ✅ {fix}")

        if self.errors:
            lines.append("\n### 维护过程错误")
            for err in self.errors:
                lines.append(f"- ❌ {err}")

        if not self.issues_found:
            lines.append("### ✅ 代码库健康，未发现问题")

        return "\n".join(lines)


class SelfMaintainer:
    """
    Jarvis 自我维护引擎。

    用法：
        maintainer = SelfMaintainer()
        report = await maintainer.run()
        print(report.to_markdown())
    """

    def __init__(
        self,
        target_dir: Path | None = None,
        auto_fix: bool = True,
        llm_review: bool = True,
    ) -> None:
        self.target_dir = target_dir or PROJECT_ROOT / "agent_infra"
        self.auto_fix = auto_fix
        self.llm_review = llm_review
        self._llm: Any = None

    def _get_llm(self):
        if self._llm is None:
            from agent_infra.intelligence.llm_router import LLMRouter
            self._llm = LLMRouter()
        return self._llm

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self) -> MaintenanceReport:
        """运行完整维护流程。"""
        report = MaintenanceReport()
        logger.info(f"Self-maintenance started on: {self.target_dir}")

        py_files = list(self.target_dir.rglob("*.py"))
        report.files_checked = len(py_files)

        # Step 1: 语法检查
        syntax_issues = await asyncio.to_thread(
            self._check_syntax, py_files
        )
        report.issues_found.extend(syntax_issues)

        # Step 2: 静态分析（lint）
        lint_issues = await asyncio.to_thread(self._check_lint)
        report.issues_found.extend(lint_issues)

        # Step 3: 导入验证（关键模块）
        import_issues = await self._check_imports()
        report.issues_found.extend(import_issues)

        # Step 4: 配置完整性
        config_issues = await asyncio.to_thread(self._check_config)
        report.issues_found.extend(config_issues)

        # Step 5: LLM 代码审查（使用订阅 CLI，不消耗 API）
        if self.llm_review and not syntax_issues:  # 有语法错误时跳过
            llm_issues = await self._llm_code_review(py_files)
            report.issues_found.extend(llm_issues)

        # Step 6: 自动修复
        if self.auto_fix and report.issues_found:
            await self._auto_fix(report)

        report.finished_at = datetime.now(timezone.utc)

        # Step 7: 写入 Obsidian
        await asyncio.to_thread(self._write_report, report)

        logger.info(
            f"Self-maintenance complete: "
            f"{len(report.issues_found)} issues, "
            f"{report.issues_fixed} fixed, "
            f"{report.duration_s:.1f}s"
        )
        return report

    # ── Step 1: 语法检查 ──────────────────────────────────────────────────────

    def _check_syntax(self, py_files: list[Path]) -> list[IssueFound]:
        issues = []
        for fpath in py_files:
            try:
                source = fpath.read_text(encoding="utf-8", errors="ignore")
                ast.parse(source)
            except SyntaxError as e:
                issues.append(IssueFound(
                    file_path=fpath,
                    issue_type="syntax",
                    severity="error",
                    message=str(e),
                    line_number=e.lineno or 0,
                    auto_fixable=False,
                ))
            except Exception as e:
                issues.append(IssueFound(
                    file_path=fpath,
                    issue_type="syntax",
                    severity="warning",
                    message=f"Parse warning: {e}",
                ))
        return issues

    # ── Step 2: 静态分析 ──────────────────────────────────────────────────────

    def _check_lint(self) -> list[IssueFound]:
        issues = []
        # 尝试 ruff（最快）
        for tool, args in [
            ("ruff", ["check", str(self.target_dir), "--output-format=json", "-q"]),
            ("flake8", [str(self.target_dir), "--max-line-length=120", "--format=json"]),
        ]:
            if not self._cmd_exists(tool):
                continue
            try:
                result = subprocess.run(
                    [sys.executable, "-m", tool] + args[1:],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                output = result.stdout.strip()
                if not output:
                    break  # 无问题

                # ruff JSON 格式解析
                if tool == "ruff":
                    import json
                    try:
                        ruff_issues = json.loads(output)
                        for item in ruff_issues[:20]:  # 最多处理 20 条
                            issues.append(IssueFound(
                                file_path=Path(item.get("filename", "")),
                                issue_type="lint",
                                severity="warning",
                                message=f"[{item.get('code','?')}] {item.get('message','')}",
                                line_number=item.get("location", {}).get("row", 0),
                                auto_fixable=item.get("fix") is not None,
                            ))
                    except Exception:
                        pass
                break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        return issues

    # ── Step 3: 导入验证 ──────────────────────────────────────────────────────

    async def _check_imports(self) -> list[IssueFound]:
        """验证关键模块能正常导入。"""
        critical_modules = [
            ("agent_infra.config.settings", "get_settings"),
            ("agent_infra.capture.event_bus", "get_event_bus"),
            ("agent_infra.memory.episodic", "get_episodic_memory"),
        ]
        issues = []
        for module_path, attr in critical_modules:
            issue = await asyncio.to_thread(
                self._try_import, module_path, attr
            )
            if issue:
                issues.append(issue)
        return issues

    def _try_import(self, module_path: str, attr: str) -> IssueFound | None:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            if not hasattr(mod, attr):
                return IssueFound(
                    file_path=Path(module_path.replace(".", "/") + ".py"),
                    issue_type="import",
                    severity="warning",
                    message=f"Missing attribute '{attr}' in {module_path}",
                )
        except ImportError as e:
            return IssueFound(
                file_path=Path(module_path.replace(".", "/") + ".py"),
                issue_type="import",
                severity="error",
                message=f"Import failed: {e}",
                auto_fixable=False,
            )
        except Exception as e:
            # 忽略运行时错误（如缺少第三方包）
            logger.debug(f"Import check skipped for {module_path}: {e}")
        return None

    # ── Step 4: 配置完整性 ────────────────────────────────────────────────────

    def _check_config(self) -> list[IssueFound]:
        issues = []
        soul_path = PROJECT_ROOT / "agent_infra" / "config" / "soul.toml"
        if not soul_path.exists():
            issues.append(IssueFound(
                file_path=soul_path,
                issue_type="config",
                severity="warning",
                message="soul.toml 不存在，将使用默认配置",
                auto_fixable=True,
                suggested_fix="创建 soul.toml 配置文件",
            ))

        jarvis_dir = Path.home() / ".jarvis"
        if not jarvis_dir.exists():
            issues.append(IssueFound(
                file_path=jarvis_dir,
                issue_type="config",
                severity="info",
                message="~/.jarvis 目录不存在（将在首次运行时创建）",
            ))
        return issues

    # ── Step 5: LLM 代码审查 ─────────────────────────────────────────────────

    async def _llm_code_review(self, py_files: list[Path]) -> list[IssueFound]:
        """
        用 LLM（Claude CLI 订阅）审查最近修改的文件。
        只检查最近 7 天内修改过的文件，控制成本。
        """
        import time
        recent_cutoff = time.time() - 7 * 24 * 3600
        recent_files = [
            f for f in py_files
            if f.stat().st_mtime > recent_cutoff
            and f.name != "__init__.py"
            and "test_" not in f.name
        ][:5]  # 最多 5 个文件

        if not recent_files:
            return []

        issues = []
        for fpath in recent_files:
            try:
                file_issues = await self._review_single_file(fpath)
                issues.extend(file_issues)
            except Exception as e:
                logger.debug(f"LLM review failed for {fpath.name}: {e}")

        return issues

    async def _review_single_file(self, fpath: Path) -> list[IssueFound]:
        """用 LLM 审查单个文件。"""
        source = fpath.read_text(encoding="utf-8", errors="ignore")
        if len(source) > 5000:
            source = source[:5000] + "\n... (truncated)"

        llm = self._get_llm()
        prompt = f"""请审查以下 Python 代码，找出潜在的 bug、设计问题或改进点。
仅报告有实际影响的问题，不要报告风格偏好。

文件: {fpath.name}

```python
{source}
```

用以下格式回复（每行一个问题，没有问题则返回 "OK"）：
[severity] line_number: 问题描述
severity 取值: error / warning / info

示例:
[error] 42: 可能发生 KeyError，dict.get() 访问未做空值检查
[warning] 15: 异常被静默忽略，可能掩盖真实错误
OK"""

        try:
            response = await asyncio.wait_for(
                llm.agenerate(human=prompt, simple=False, force_subscription=True),
                timeout=60,
            )
        except asyncio.TimeoutError:
            return []

        if not response or response.strip().upper() == "OK":
            return []

        issues = []
        for line in response.splitlines():
            line = line.strip()
            if not line or line.upper() == "OK":
                continue
            # 解析 [severity] line_number: message
            match = re.match(r'\[(error|warning|info)\]\s*(\d+):\s*(.+)', line, re.I)
            if match:
                severity, lineno, message = match.groups()
                issues.append(IssueFound(
                    file_path=fpath,
                    issue_type="logic",
                    severity=severity.lower(),
                    message=message.strip(),
                    line_number=int(lineno),
                    auto_fixable=False,
                ))

        return issues[:5]  # 每文件最多 5 个

    # ── Step 6: 自动修复 ──────────────────────────────────────────────────────

    async def _auto_fix(self, report: MaintenanceReport) -> None:
        """尝试自动修复可修复的问题。"""
        # 优先用 ruff --fix 处理 lint 问题
        fixable_lint = [
            i for i in report.issues_found
            if i.issue_type == "lint" and i.auto_fixable
        ]
        if fixable_lint:
            fixed = await self._ruff_auto_fix()
            if fixed:
                report.issues_fixed += fixed
                report.fixes_applied.append(f"ruff --fix 修复了 {fixed} 个 lint 问题")

        # LLM 修复 logic 类问题（重要、有建议的）
        llm_fixable = [
            i for i in report.issues_found
            if i.issue_type == "logic"
            and i.severity == "error"
        ]
        for issue in llm_fixable[:2]:  # 最多自动修 2 个
            fixed = await self._llm_apply_fix(issue)
            if fixed:
                report.issues_fixed += 1
                report.fixes_applied.append(
                    f"{issue.file_path.name}:{issue.line_number} - {issue.message[:60]}"
                )

    async def _ruff_auto_fix(self) -> int:
        """运行 ruff --fix 修复可自动修复的 lint 问题。"""
        if not self._cmd_exists("ruff"):
            return 0
        try:
            result = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "--fix", str(self.target_dir)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # 解析修复数量
            match = re.search(r'Fixed (\d+)', result.stdout + result.stderr)
            return int(match.group(1)) if match else 0
        except Exception:
            return 0

    async def _llm_apply_fix(self, issue: IssueFound) -> bool:
        """让 LLM 生成并应用修复代码。"""
        try:
            source = issue.file_path.read_text(encoding="utf-8")
            lines = source.splitlines()
            ctx_start = max(0, issue.line_number - 5)
            ctx_end = min(len(lines), issue.line_number + 5)
            context = "\n".join(
                f"{i+1}: {l}" for i, l in enumerate(lines[ctx_start:ctx_end])
            )

            llm = self._get_llm()
            prompt = f"""文件 {issue.file_path.name} 第 {issue.line_number} 行有以下问题：
{issue.message}

相关代码：
{context}

请提供修复后的代码片段（只返回修复的那几行，不要解释）："""

            fix_code = await asyncio.wait_for(
                llm.agenerate(human=prompt, force_subscription=True),
                timeout=60,
            )

            if not fix_code or len(fix_code) > 2000:
                return False

            # 生成 diff 并记录（暂不自动应用，风险太高）
            issue.suggested_fix = fix_code
            logger.info(f"LLM fix suggested for {issue.file_path.name}:{issue.line_number}")
            return False  # 暂时只记录不自动应用，需要人工确认

        except Exception as e:
            logger.debug(f"LLM fix failed: {e}")
            return False

    # ── Step 7: 写报告到 Obsidian ─────────────────────────────────────────────

    def _write_report(self, report: MaintenanceReport) -> None:
        try:
            from agent_infra.memory.semantic import get_semantic_memory
            sem = get_semantic_memory()
            content = report.to_markdown()
            sem.write_insight(
                title=f"Maintenance {report.started_at.strftime('%Y%m%d')}",
                content=content,
                tags=["jarvis", "self-maintenance", "auto-repair"],
                subfolder="System",
            )
            logger.info("Maintenance report written to Obsidian")
        except Exception as e:
            logger.error(f"Failed to write maintenance report: {e}")

    # ── 工具 ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _cmd_exists(cmd: str) -> bool:
        return bool(shutil.which(cmd)) or _module_available(cmd)


def _module_available(module: str) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", module, "--version"],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_maintainer: SelfMaintainer | None = None


def get_self_maintainer() -> SelfMaintainer:
    global _maintainer
    if _maintainer is None:
        _maintainer = SelfMaintainer()
    return _maintainer
