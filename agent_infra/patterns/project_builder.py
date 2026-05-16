"""
Project Builder — Matrix Agent for Project Creation
====================================================
专门处理"创建项目"类请求的多智能体流水线，对标 Claude Code 级别的代码生成质量。

流水线（5 个 Specialist 顺序执行，共享 ProjectSpec 作为唯一上下文）：

  用户请求
    → RequirementsAgent  — 提取结构化需求、技术选型
    → ArchitectAgent     — 设计文件树 + 接口契约
    → CoderAgent         — 按依赖顺序逐文件生成代码（含 lint 自修复）
    → ReviewerAgent      — 跨文件一致性审查
    → ScaffoldAgent      — 生成 pyproject.toml / Dockerfile / CI / README 等配置文件
    → ProjectBuilderNode 将所有 FileDiff 写入 state["file_changes"]
    → （原有图）code_check → reflect → hitl → FileApplyNode → output_guard

通信机制：
  所有 agent 通过 ProjectSpec（Pydantic model）共享结构化上下文，
  存放在 chain_context["project_spec"]，不通过 messages 传递中间状态。

落盘机制：
  FileApplyNode 在 HITL 批准后调用 FileSystem.apply()，唯一的破坏性操作。
"""
from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agent_infra.core.llm import get_llm, get_structured_llm
from agent_infra.core.state import AgentState
from agent_infra.execution.code_tools import CodeToolRunner
from agent_infra.execution.filesystem import FileDiff, FileSystem
from agent_infra.execution.sandbox import Sandbox

logger = logging.getLogger(__name__)

# 文件层依赖顺序：先生成底层，后生成上层
_LAYER_ORDER: dict[str, int] = {
    "config":  0,
    "model":   1,
    "schema":  2,
    "service": 3,
    "router":  4,
    "test":    5,
    "infra":   6,
}


# ── 共享上下文数据结构 ────────────────────────────────────────────────────────

class FileSpec(BaseModel):
    path: str = Field(description="相对于项目根目录的文件路径")
    description: str = Field(description="该文件的职责，一句话")
    layer: str = Field(default="service", description="所属层：config/model/schema/service/router/test/infra")
    content: str = Field(default="", description="CoderAgent 生成的文件内容")


class InterfaceSpec(BaseModel):
    name: str = Field(description="接口名称")
    kind: str = Field(description="类型：endpoint / class / function / schema / constant")
    signature: str = Field(description="完整签名，含参数和返回类型")
    description: str = Field(description="用途说明")


class ReviewIssue(BaseModel):
    file: str
    severity: str  # error / warning
    description: str
    fix: str


class ProjectSpec(BaseModel):
    name: str = ""
    description: str = ""
    output_dir: str = ""
    tech_stack: dict[str, str] = Field(default_factory=dict)
    requirements: list[str] = Field(default_factory=list)
    file_tree: list[FileSpec] = Field(default_factory=list)
    interfaces: list[InterfaceSpec] = Field(default_factory=list)
    review_issues: list[ReviewIssue] = Field(default_factory=list)


# ── 内部结构化输出 schema（仅供 with_structured_output 使用）───────────────────

class _RequirementsOutput(BaseModel):
    name: str = Field(description="项目名称，snake_case")
    description: str = Field(description="项目功能一段话描述")
    output_dir: str = Field(description="建议的输出目录名，snake_case")
    tech_stack: dict[str, str] = Field(
        description="技术选型，如 {framework: fastapi, db: postgresql, auth: jwt, test: pytest}"
    )
    requirements: list[str] = Field(description="功能需求列表，每条 ≤ 40 字，具体可验证")


class _FileSpecOutput(BaseModel):
    path: str
    description: str
    layer: str = "service"


class _InterfaceOutput(BaseModel):
    name: str
    kind: str
    signature: str
    description: str


class _ArchitectOutput(BaseModel):
    file_tree: list[_FileSpecOutput] = Field(description="完整文件列表，含路径/描述/层次")
    interfaces: list[_InterfaceOutput] = Field(description="所有模块共享的接口契约")
    decisions: list[str] = Field(description="关键架构决策，每条说明选择原因")


class _ReviewOutput(BaseModel):
    issues: list[dict[str, str]] = Field(description="问题列表，每项含 file/severity/description/fix")
    overall_quality: int = Field(ge=0, le=10)
    approved: bool
    summary: str


class _ScaffoldOutput(BaseModel):
    files: dict[str, str] = Field(description="配置文件内容，key 为相对路径，value 为完整文件内容")


# ═══════════════════════════════════════════════════════════════════════════════
# Specialist Agents
# ═══════════════════════════════════════════════════════════════════════════════

class RequirementsAgent:
    """
    从用户自然语言描述中提取结构化需求。
    输出：name / description / output_dir / tech_stack / requirements
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.llm = get_structured_llm(_RequirementsOutput, model=model)

    def run(self, user_message: str) -> ProjectSpec:
        system = (
            "你是经验丰富的需求分析师和技术架构师。\n"
            "从用户的项目描述中提取完整、结构化的需求和技术选型。\n\n"
            "规则：\n"
            "- name 使用 snake_case，简短清晰\n"
            "- tech_stack 必须覆盖：框架、数据库、认证方式、测试工具、代码质量工具\n"
            "- 若用户未指定技术栈，根据项目性质推荐成熟稳定的方案\n"
            "- requirements 每条必须具体可测试，避免'用户友好'这类模糊表达\n"
            "- Python 项目默认用 FastAPI + SQLAlchemy + Alembic + pytest + ruff"
        )
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system),
            HumanMessage(content=user_message),
        ])
        result: _RequirementsOutput = (prompt | self.llm).invoke({})
        logger.info(f"[RequirementsAgent] project={result.name}, stack={result.tech_stack}")

        return ProjectSpec(
            name=result.name,
            description=result.description,
            output_dir=result.output_dir or result.name,
            tech_stack=result.tech_stack,
            requirements=result.requirements,
        )


class ArchitectAgent:
    """
    根据 ProjectSpec 设计完整文件树和接口契约。
    输出：file_tree（含层次） + interfaces（跨模块契约） + decisions（架构决策）
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.llm = get_structured_llm(_ArchitectOutput, model=model)

    def run(self, spec: ProjectSpec) -> ProjectSpec:
        requirements_str = "\n".join(f"  - {r}" for r in spec.requirements)
        stack_str = json.dumps(spec.tech_stack, ensure_ascii=False)

        system = (
            "你是首席软件架构师，负责为项目设计清晰、可维护的文件结构。\n\n"
            "layer 分类标准：\n"
            "  config  — 配置加载、环境变量、依赖注入容器\n"
            "  model   — ORM 模型、数据库 schema\n"
            "  schema  — Pydantic 请求/响应 schema（API 契约）\n"
            "  service — 业务逻辑层，不依赖框架\n"
            "  router  — HTTP 路由层，只做参数验证和 service 调用\n"
            "  test    — 单元测试和集成测试\n"
            "  infra   — 数据库连接、缓存、消息队列等基础设施\n\n"
            "要求：\n"
            "  1. 每个文件职责单一，≤ 300 行\n"
            "  2. interfaces 必须覆盖所有跨文件调用的函数/类签名\n"
            "  3. 测试文件覆盖所有 service 和 router\n"
            "  4. 不要生成 __init__.py（会自动创建）"
        )
        human = (
            f"项目：{spec.name}\n"
            f"描述：{spec.description}\n"
            f"技术栈：{stack_str}\n"
            f"功能需求：\n{requirements_str}"
        )
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        result: _ArchitectOutput = (prompt | self.llm).invoke({})
        logger.info(f"[ArchitectAgent] files={len(result.file_tree)}, interfaces={len(result.interfaces)}")

        spec.file_tree = [
            FileSpec(path=f.path, description=f.description, layer=f.layer)
            for f in result.file_tree
        ]
        spec.interfaces = [
            InterfaceSpec(
                name=i.name, kind=i.kind,
                signature=i.signature, description=i.description,
            )
            for i in result.interfaces
        ]
        return spec


class CoderAgent:
    """
    按依赖顺序逐文件生成代码。

    生成策略：
      1. 按 layer 排序（config → model → schema → service → router → test → infra）
      2. 每个文件获取：项目全局上下文 + 接口契约 + 同层/下层已生成文件内容
      3. lint 失败时，携带错误信息自动重试一次
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.llm = get_llm(model=model)
        self._parser = StrOutputParser()
        self._runner = CodeToolRunner(sandbox=Sandbox())

    def run(self, spec: ProjectSpec) -> ProjectSpec:
        # 按层次排序
        sorted_files = sorted(
            spec.file_tree,
            key=lambda f: _LAYER_ORDER.get(f.layer.lower(), 3),
        )

        generated: dict[str, str] = {}
        interfaces_md = _interfaces_to_markdown(spec.interfaces)

        for file_spec in sorted_files:
            logger.info(f"[CoderAgent] generating {file_spec.path}")
            content = self._generate_file(file_spec, spec, generated, interfaces_md)

            # lint 自修复（最多 1 次）
            if file_spec.path.endswith(".py"):
                content = self._lint_fix(file_spec.path, content, spec, generated, interfaces_md)

            generated[file_spec.path] = content
            file_spec.content = content

        return spec

    def _generate_file(
        self,
        file_spec: FileSpec,
        spec: ProjectSpec,
        generated: dict[str, str],
        interfaces_md: str,
        error_feedback: str = "",
    ) -> str:
        context_files = _select_context_files(file_spec, generated)
        context_str = ""
        if context_files:
            context_str = "\n\n已生成的相关文件（供参考）:\n" + "\n\n".join(
                f"### {path}\n```python\n{content[:3000]}\n```"
                for path, content in context_files.items()
            )

        system = (
            f"你是资深工程师，正在构建 **{spec.name}** 项目。\n"
            f"技术栈：{json.dumps(spec.tech_stack, ensure_ascii=False)}\n"
            f"项目描述：{spec.description}\n\n"
            f"接口契约（所有文件必须严格遵守）：\n{interfaces_md}\n\n"
            "代码规范：\n"
            "  1. 生成完整、可直接运行的代码，绝不留 TODO / pass 占位符\n"
            "  2. 所有函数必须有类型注解\n"
            "  3. 边界处理：数据库查询不存在时抛 404，权限不足时抛 403\n"
            "  4. 只返回纯代码，不要包含 ``` 代码围栏或任何解释文字"
        )

        human_parts = [
            f"请生成文件 `{file_spec.path}`",
            f"职责：{file_spec.description}",
        ]
        if error_feedback:
            human_parts.insert(0, f"【上次生成有以下 lint 错误，本次必须修复】\n{error_feedback}\n")
        if context_str:
            human_parts.append(context_str)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system),
            HumanMessage(content="\n\n".join(human_parts)),
        ])
        raw = (prompt | self.llm | self._parser).invoke({})
        return _strip_fences(raw)

    def _lint_fix(
        self,
        path: str,
        content: str,
        spec: ProjectSpec,
        generated: dict[str, str],
        interfaces_md: str,
    ) -> str:
        """写入临时文件 → lint → 有错则带错误重生成一次。"""
        errors = _lint_content(content, self._runner)
        if not errors:
            return content
        logger.warning(f"[CoderAgent] lint errors in {path}, retrying...")
        error_feedback = "\n".join(errors[:20])
        # 构造带错误反馈的 FileSpec 重生成
        file_spec = next((f for f in spec.file_tree if f.path == path), None)
        if file_spec is None:
            return content
        return self._generate_file(file_spec, spec, generated, interfaces_md, error_feedback)


class ReviewerAgent:
    """
    跨文件一致性审查。
    检查：接口签名匹配 / import 路径合法 / 错误处理覆盖 / 安全漏洞
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.llm = get_structured_llm(_ReviewOutput, model=model)

    def run(self, spec: ProjectSpec) -> ProjectSpec:
        if not spec.file_tree:
            return spec

        # 只取核心层文件（测试和配置不纳入跨文件审查）
        core_files = [
            f for f in spec.file_tree
            if f.layer in ("model", "schema", "service", "router") and f.content
        ]
        if not core_files:
            return spec

        files_str = "\n\n".join(
            f"### {f.path} ({f.layer})\n```python\n{f.content[:2000]}\n```"
            for f in core_files
        )
        interfaces_md = _interfaces_to_markdown(spec.interfaces)

        system = (
            "你是高级代码审查员。对以下多文件项目进行跨文件一致性审查。\n\n"
            "重点检查：\n"
            "  1. 接口一致性 — 调用方的参数是否与被调用方的签名匹配\n"
            "  2. import 路径 — 所有 import 是否能在项目内找到对应文件\n"
            "  3. 错误处理 — router 层是否正确处理 service 层可能抛出的异常\n"
            "  4. 安全问题 — SQL 注入、路径穿越、未授权访问\n"
            "  5. 完整性 — 所有接口契约是否全部实现\n\n"
            "severity 只用 'error'（会阻止生成）或 'warning'（仅提示）"
        )
        human = (
            f"接口契约：\n{interfaces_md}\n\n"
            f"项目文件：\n{files_str}"
        )
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        result: _ReviewOutput = (prompt | self.llm).invoke({})
        logger.info(
            f"[ReviewerAgent] quality={result.overall_quality}/10, "
            f"approved={result.approved}, issues={len(result.issues)}"
        )

        spec.review_issues = [
            ReviewIssue(
                file=i.get("file", ""),
                severity=i.get("severity", "warning"),
                description=i.get("description", ""),
                fix=i.get("fix", ""),
            )
            for i in result.issues
        ]
        return spec


class ScaffoldAgent:
    """
    生成项目配置文件：pyproject.toml / Dockerfile / docker-compose.yml /
    .gitignore / Makefile / README.md / GitHub Actions CI
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.llm = get_structured_llm(_ScaffoldOutput, model=model)

    def run(self, spec: ProjectSpec) -> ProjectSpec:
        file_list = "\n".join(f"  - {f.path}" for f in spec.file_tree)
        stack_str = json.dumps(spec.tech_stack, ensure_ascii=False)
        requirements_str = "\n".join(f"  - {r}" for r in spec.requirements)
        has_db = any(
            k in ("db", "database") or v.lower() in ("postgresql", "mysql", "sqlite", "mongodb")
            for k, v in spec.tech_stack.items()
        )

        system = (
            "你是 DevOps 工程师，负责生成项目配置文件。\n\n"
            "必须生成的文件：\n"
            "  - pyproject.toml（含 [project] + [tool.ruff] + [tool.pytest.ini_options]）\n"
            "  - Dockerfile（多阶段构建，生产就绪）\n"
            + ("  - docker-compose.yml（含应用 + 数据库服务）\n" if has_db else "") +
            "  - .gitignore（Python 项目标准）\n"
            "  - Makefile（含 install / lint / test / run / docker-build 目标）\n"
            "  - README.md（含快速开始、环境变量说明、API 文档链接）\n"
            "  - .github/workflows/ci.yml（lint + test + build）\n\n"
            "规则：每个文件必须完整可用，不留 TODO 或注释占位符。"
        )
        human = (
            f"项目名称：{spec.name}\n"
            f"描述：{spec.description}\n"
            f"技术栈：{stack_str}\n"
            f"功能需求：\n{requirements_str}\n\n"
            f"已有源码文件：\n{file_list}"
        )
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        result: _ScaffoldOutput = (prompt | self.llm).invoke({})
        logger.info(f"[ScaffoldAgent] scaffold files={list(result.files.keys())}")

        # 将配置文件追加到 file_tree
        for path, content in result.files.items():
            spec.file_tree.append(
                FileSpec(path=path, description="配置/基础设施文件", layer="infra", content=content)
            )
        return spec


# ═══════════════════════════════════════════════════════════════════════════════
# LangGraph 节点
# ═══════════════════════════════════════════════════════════════════════════════

class ProjectBuilderNode:
    """
    协调 5 个 Specialist Agent，完成从需求到代码的全流程生成。

    输出到 state：
      chain_context["project_spec"] — 完整 ProjectSpec（供 HITL 展示和 FileApplyNode 使用）
      file_changes                  — 所有文件的 FileDiff（含内容，供 HITL 展示 diff）
      tool_errors                   — 所有文件的 lint 错误（供 ReflectionNode 驱动修复）
      messages                      — 生成摘要（供用户查看）
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self._requirements = RequirementsAgent(model=model)
        self._architect = ArchitectAgent(model=model)
        self._coder = CoderAgent(model=model)
        self._reviewer = ReviewerAgent(model=model)
        self._scaffold = ScaffoldAgent(model=model)

    def __call__(self, state: AgentState) -> dict[str, Any]:
        user_message = _latest_human_message(state)
        logger.info("[ProjectBuilderNode] starting pipeline")

        # ── 1. 需求分析 ───────────────────────────────────────────────────────
        spec = self._requirements.run(user_message)

        # ── 2. 架构设计 ───────────────────────────────────────────────────────
        spec = self._architect.run(spec)

        # ── 3. 代码生成（含 lint 自修复）──────────────────────────────────────
        spec = self._coder.run(spec)

        # ── 4. 跨文件审查 ─────────────────────────────────────────────────────
        spec = self._reviewer.run(spec)

        # ── 5. 脚手架配置文件 ─────────────────────────────────────────────────
        spec = self._scaffold.run(spec)

        # ── 6. 准备 FileDiff（不落盘，供 HITL 展示和 FileApplyNode 使用）──────
        fs = FileSystem(root=".")
        file_diffs: list[FileDiff] = []
        tool_errors: list[str] = []

        for file_spec in spec.file_tree:
            if not file_spec.content:
                continue
            target_path = str(Path(spec.output_dir) / file_spec.path)
            diff = fs.prepare_write(target_path, file_spec.content)
            file_diffs.append(diff)

            # 收集 lint 错误（供 ReflectionNode 使用）
            if file_spec.path.endswith(".py") and file_spec.layer not in ("infra",):
                errors = _lint_content(file_spec.content, CodeToolRunner(sandbox=Sandbox()))
                tool_errors.extend(f"[{target_path}] {e}" for e in errors[:5])

        # ── 7. 构建摘要消息 ───────────────────────────────────────────────────
        summary = _build_summary(spec, file_diffs)
        logger.info(f"[ProjectBuilderNode] done: {len(file_diffs)} files, {len(tool_errors)} lint errors")

        return {
            "chain_context": {
                **state.get("chain_context", {}),
                "project_spec": spec.model_dump(),
            },
            "file_changes": [d.to_dict() for d in file_diffs],
            "tool_errors": tool_errors,
            "messages": [AIMessage(content=summary)],
        }


class FileApplyNode:
    """
    HITL 批准后，将 file_changes 中所有文件写入磁盘。

    在非项目创建场景（file_changes 为空）时无操作，不影响现有流程。
    写入路径从 chain_context["project_spec"]["output_dir"] 读取，
    相对于当前工作目录。
    """

    def __call__(self, state: AgentState) -> dict[str, Any]:
        file_changes: list[dict] = state.get("file_changes") or []
        if not file_changes:
            return {}

        applied: list[str] = []
        failed: list[str] = []

        for change in file_changes:
            path = change.get("path", "")
            content = change.get("content", "")
            if not path or not content:
                continue
            try:
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
                applied.append(path)
                logger.info(f"[FileApplyNode] wrote {path}")
            except Exception as e:
                failed.append(f"{path}: {e}")
                logger.error(f"[FileApplyNode] failed to write {path}: {e}")

        status_msg = f"已写入 {len(applied)} 个文件到磁盘。"
        if failed:
            status_msg += f"\n写入失败 {len(failed)} 个：\n" + "\n".join(failed)

        return {
            "messages": [AIMessage(content=status_msg)],
            "file_changes": [],  # 清空，避免下次重复写入
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _latest_human_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _interfaces_to_markdown(interfaces: list[InterfaceSpec]) -> str:
    if not interfaces:
        return "（架构师尚未定义接口契约）"
    lines = []
    for i in interfaces:
        lines.append(f"- **{i.kind}** `{i.name}`: `{i.signature}`  \n  _{i.description}_")
    return "\n".join(lines)


def _select_context_files(
    target: FileSpec,
    generated: dict[str, str],
    max_files: int = 4,
) -> dict[str, str]:
    """
    选择与目标文件最相关的已生成文件作为上下文。
    规则：同目录 > 下层文件（model/schema 对所有人可见）> 最近生成的
    """
    target_layer_rank = _LAYER_ORDER.get(target.layer.lower(), 3)
    target_dir = str(Path(target.path).parent)

    scored: list[tuple[int, str]] = []
    for path in generated:
        score = 0
        # 同目录加分
        if str(Path(path).parent) == target_dir:
            score += 3
        # 下层文件（model/schema）对所有层可见
        file_layer = _guess_layer(path)
        if file_layer in ("model", "schema"):
            score += 2
        # 层次相邻
        file_rank = _LAYER_ORDER.get(file_layer, 3)
        if abs(file_rank - target_layer_rank) <= 1:
            score += 1
        scored.append((score, path))

    scored.sort(key=lambda x: -x[0])
    selected = [path for _, path in scored[:max_files]]
    return {p: generated[p] for p in selected}


def _guess_layer(path: str) -> str:
    """从路径猜测文件层次。"""
    lower = path.lower()
    for layer in ("model", "schema", "service", "router", "test", "config", "infra"):
        if layer in lower:
            return layer
    return "service"


def _lint_content(content: str, runner: CodeToolRunner) -> list[str]:
    """将内容写入临时文件后运行 lint，返回错误行列表。"""
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name
        result = runner.lint(tmp_path)
        return result.errors[:15] if not result.passed else []
    except Exception:
        return []


def _strip_fences(content: str) -> str:
    """移除 LLM 输出中可能包含的 Markdown 代码围栏。"""
    content = content.strip()
    # 去掉开头的 ```python 或 ```
    content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
    # 去掉结尾的 ```
    content = re.sub(r"\n?```$", "", content)
    return content.strip()


def _build_summary(spec: ProjectSpec, diffs: list[FileDiff]) -> str:
    """构建给用户看的生成摘要。"""
    total_lines = sum(d.lines_added for d in diffs)
    errors = [i for i in spec.review_issues if i.severity == "error"]
    warnings = [i for i in spec.review_issues if i.severity == "warning"]

    lines = [
        f"## 项目 `{spec.name}` 生成完毕",
        f"",
        f"**技术栈**：{', '.join(f'{k}: {v}' for k, v in spec.tech_stack.items())}",
        f"**文件数量**：{len(diffs)} 个文件，共 {total_lines} 行代码",
        f"",
        "**文件清单**：",
    ]
    for diff in diffs:
        lines.append(f"  - `{diff.path}` (+{diff.lines_added}行)")

    if errors:
        lines += ["", f"**审查发现 {len(errors)} 个错误需注意**："]
        for issue in errors[:5]:
            lines.append(f"  - [{issue.file}] {issue.description}")
    if warnings:
        lines.append(f"\n**{len(warnings)} 个警告**（不影响运行）")

    lines += [
        "",
        "请审查上方 diff，确认后点击批准将文件写入磁盘。",
    ]
    return "\n".join(lines)
