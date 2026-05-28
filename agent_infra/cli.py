"""
Jarvis CLI — 主入口
====================
typer + rich 命令行界面。

用法:
    python -m agent_infra          # 启动守护进程
    python -m agent_infra status   # 查看系统状态
    python -m agent_infra monitor       # 实时监控仪表盘
    python -m agent_infra feed          # 今日知识推送 + 评分
    python -m agent_infra fetch-knowledge  # 立即抓取新内容
    python -m agent_infra preferences   # 话题偏好权重
    python -m agent_infra digest        # 立即生成日报
    python -m agent_infra search   # 搜索记忆
    python -m agent_infra ask      # 直接提问
    python -m agent_infra pi       # Pi.dev CLI 入口
    python -m agent_infra install  # 安装 Claude Code hooks
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="jarvis",
    help="Jarvis — 你的个人 AI 助手",
    rich_markup_mode="rich",
)
console = Console()


# ── 启动守护进程 ──────────────────────────────────────────────────────────────

@app.command()
def start(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
    no_augment: bool = typer.Option(False, "--no-augment", help="关闭反哺引擎"),
) -> None:
    """启动 Jarvis 守护进程（捕获 + 反哺 + 日报）。"""
    import logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    console.print(Panel.fit(
        "[bold green]Jarvis 启动中...[/bold green]\n"
        "按 Ctrl+C 停止",
        title="🤖 Jarvis",
        border_style="green",
    ))

    asyncio.run(_run_daemon(no_augment=no_augment))


async def _run_daemon(no_augment: bool = False) -> None:
    from agent_infra.capture.event_bus import get_event_bus
    from agent_infra.capture.adapters.clipboard import ClipboardAdapter
    from agent_infra.capture.adapters.claude_code import ClaudeCodeAdapter
    from agent_infra.capture.adapters.files import FileAdapter
    from agent_infra.memory.episodic import get_episodic_memory
    from agent_infra.monitor.task_registry import get_registry

    registry = get_registry()
    registry.register_daemon()

    bus = get_event_bus()

    # 订阅：将所有事件写入情节记忆
    episodic = get_episodic_memory()

    async def _record_to_episodic(event):
        episodic.record(event)

    bus.subscribe(handler=_record_to_episodic)

    # 反哺引擎
    if not no_augment:
        from agent_infra.intelligence.augmentor import get_augmentation_engine
        engine = get_augmentation_engine()
        engine.attach(bus)
        console.print("[green]✓[/green] 反哺引擎已启动")

    # 健康监控
    from agent_infra.healing.monitor import get_health_monitor
    monitor = get_health_monitor()

    # 定时任务（日报 + 自维护 + 知识抓取）
    _schedule_jobs()

    # 补跑关机期间漏掉的任务（后台异步，不阻塞启动）
    asyncio.create_task(_catchup_missed_tasks(), name="catchup")

    # 启动适配器
    adapters = [
        ClipboardAdapter(bus=bus),
        ClaudeCodeAdapter(bus=bus),
        FileAdapter(bus=bus),
    ]

    # 将每个适配器注册进健康监控（自动重启 = stop → start）
    for adapter in adapters:
        if adapter.enabled:
            async def _restart(a=adapter) -> None:
                await a.stop()
                await a.start()
            monitor.register(
                name=adapter.name,
                check_fn=adapter.is_alive,
                restart_fn=_restart,
            )

    console.print("[green]✓[/green] 适配器已加载:")
    for adapter in adapters:
        status = "启用" if adapter.enabled else "禁用"
        console.print(f"  • {adapter.name}: {status}")

    # 并发运行所有任务
    tasks = [
        asyncio.create_task(bus.start(), name="event_bus"),
        asyncio.create_task(monitor.run(), name="health_monitor"),
        *[asyncio.create_task(a.start(), name=a.name) for a in adapters if a.enabled],
    ]

    console.print("\n[bold green]Jarvis 已就绪 🚀[/bold green]\n")

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        console.print("\n[yellow]正在停止...[/yellow]")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        console.print("[green]Jarvis 已停止[/green]")


def _schedule_jobs() -> None:
    """尝试用 APScheduler 调度定时任务（可选依赖）。"""
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from agent_infra.config.settings import get_settings
        from agent_infra.monitor.task_registry import get_registry

        cfg = get_settings()
        registry = get_registry()
        scheduler = AsyncIOScheduler()
        added = []

        # 日报
        if cfg.digest.enabled:
            h, m = map(int, cfg.digest.schedule.split(":"))
            registry.register_task("daily_digest", "日报生成", cfg.digest.schedule)
            scheduler.add_job(_run_digest, "cron", hour=h, minute=m, id="daily_digest")
            added.append(f"日报 {cfg.digest.schedule}")

        # 自我维护
        if cfg.maintenance.enabled:
            h, m = map(int, cfg.maintenance.schedule.split(":"))
            registry.register_task("self_maintain", "自我维护", cfg.maintenance.schedule)
            scheduler.add_job(_run_maintenance, "cron", hour=h, minute=m, id="self_maintain")
            added.append(f"自维护 {cfg.maintenance.schedule}")

        # 知识抓取
        if cfg.knowledge.enabled:
            h, m = map(int, cfg.knowledge.fetch_schedule.split(":"))
            registry.register_task("knowledge_fetch", "知识抓取", cfg.knowledge.fetch_schedule)
            scheduler.add_job(_run_knowledge_fetch, "cron", hour=h, minute=m, id="knowledge_fetch")
            added.append(f"知识抓取 {cfg.knowledge.fetch_schedule}")

        scheduler.start()
        for label in added:
            console.print(f"[green]✓[/green] 调度任务: {label}")
    except ImportError:
        console.print("[yellow]⚠[/yellow] APScheduler 未安装，定时任务未启用")


# 兼容旧调用
def _schedule_digest() -> None:
    _schedule_jobs()


async def _run_digest() -> None:
    from agent_infra.intelligence.digest import get_digest_generator
    from agent_infra.monitor.task_registry import get_registry

    registry = get_registry()
    t0 = registry.mark_running("daily_digest")
    try:
        gen = get_digest_generator()
        await gen.generate_daily()
        registry.mark_done("daily_digest", t0, success=True)
    except Exception as e:
        registry.mark_done("daily_digest", t0, success=False, error=str(e))
        raise


async def _run_maintenance() -> None:
    from agent_infra.automation.self_maintainer import get_self_maintainer
    from agent_infra.monitor.task_registry import get_registry

    registry = get_registry()
    t0 = registry.mark_running("self_maintain")
    try:
        maintainer = get_self_maintainer()
        await maintainer.run()
        registry.mark_done("self_maintain", t0, success=True)
    except Exception as e:
        registry.mark_done("self_maintain", t0, success=False, error=str(e))
        raise


async def _run_knowledge_fetch() -> None:
    from agent_infra.knowledge.feed import run_fetch_only
    from agent_infra.monitor.task_registry import get_registry
    from agent_infra.config.settings import get_settings

    cfg = get_settings()
    registry = get_registry()
    t0 = registry.mark_running("knowledge_fetch")
    try:
        new_count = await run_fetch_only(
            hn_min_score=cfg.knowledge.hn_min_score,
            arxiv_categories=cfg.knowledge.arxiv_categories,
        )
        registry.mark_done("knowledge_fetch", t0, success=True)
        return new_count
    except Exception as e:
        registry.mark_done("knowledge_fetch", t0, success=False, error=str(e))
        raise


async def _catchup_missed_tasks() -> None:
    """启动时检测并补跑关机期间漏掉的 Routine 任务。"""
    from agent_infra.monitor.task_registry import get_registry
    import asyncio as _aio

    registry = get_registry()
    missed = registry.get_missed_tasks()
    if not missed:
        return

    console.print(
        f"\n[yellow]⚡ 检测到 {len(missed)} 个任务在关机期间未执行，正在后台补跑...[/yellow]"
    )

    _runners = {
        "knowledge_fetch": _run_knowledge_fetch,
        "daily_digest":    _run_digest,
        "self_maintain":   _run_maintenance,
    }
    # 按优先级顺序：先知识抓取，再日报，最后维护
    ordered = [tid for tid in _runners if tid in missed]

    for task_id in ordered:
        label = {"knowledge_fetch": "知识抓取", "daily_digest": "日报生成", "self_maintain": "自我维护"}.get(task_id, task_id)
        console.print(f"  [dim]补跑: {label}[/dim]")
        try:
            await _runners[task_id]()
            console.print(f"  [green]✓[/green] {label} 补跑完成")
        except Exception as e:
            console.print(f"  [red]✗[/red] {label} 补跑失败: {e}")


# ── 状态查看 ──────────────────────────────────────────────────────────────────

@app.command()
def status(
    json_output: bool = typer.Option(False, "--json", help="输出统一状态 JSON"),
) -> None:
    """查看 Jarvis 系统状态。"""
    if json_output:
        from agent_infra.harness.status import build_status_payload
        import json

        console.print(json.dumps(build_status_payload(Path.cwd()), ensure_ascii=False, indent=2))
        return

    console.print(Panel.fit("[bold]Jarvis 系统状态[/bold]", border_style="blue"))

    # 配置
    try:
        from agent_infra.config.settings import get_settings
        cfg = get_settings()
        _print_section("配置", {
            "主 LLM": cfg.llm.primary_model,
            "本地 LLM": f"{cfg.llm.local_model} @ {cfg.llm.local_url}",
            "Obsidian Vault": str(cfg.memory.obsidian_vault_path),
        })
    except Exception as e:
        console.print(f"[red]配置加载失败: {e}[/red]")

    # LLM 状态
    try:
        from agent_infra.intelligence.llm_router import LLMRouter
        router = LLMRouter()
        s = router.status()
        _print_section("LLM 路由", {
            "Ollama 可用": "✓" if s["ollama"] else "✗",
            "简单任务走本地": "是" if s["local_for_simple"] else "否",
            "失败回退本地": "是" if s["fallback_to_local"] else "否",
        })
    except Exception as e:
        console.print(f"[yellow]LLM 状态检查失败: {e}[/yellow]")

    # 情节记忆统计
    try:
        from agent_infra.memory.episodic import get_episodic_memory
        mem = get_episodic_memory()
        stats = mem.stats()
        _print_section("情节记忆", {
            "总事件数": stats["total"],
            "今日事件数": stats["today"],
            "来源分布": str(stats["by_source"]),
        })
    except Exception as e:
        console.print(f"[yellow]情节记忆统计失败: {e}[/yellow]")

    # 语义记忆统计
    try:
        from agent_infra.memory.semantic import get_semantic_memory
        sem = get_semantic_memory()
        stats = sem.stats()
        _print_section("语义记忆 (Obsidian)", {
            "Vault 文件数": stats["vault_md_files"],
            "已索引块数": stats["indexed_chunks"],
        })
    except Exception as e:
        console.print(f"[yellow]语义记忆统计失败: {e}[/yellow]")

    # 程序记忆
    try:
        from agent_infra.memory.procedural import get_procedural_memory
        proc = get_procedural_memory()
        stats = proc.stats()
        _print_section("程序记忆 (技能库)", {
            "技能总数": stats["total_skills"],
            "常用技能": ", ".join(s["name"] for s in stats["top_skills"][:3]),
        })
    except Exception as e:
        console.print(f"[yellow]程序记忆统计失败: {e}[/yellow]")

    # 断路器状态
    try:
        from agent_infra.healing.circuit_breaker import all_breaker_status
        breakers = all_breaker_status()
        if breakers:
            _print_section("断路器", {
                b["name"]: b["state"] for b in breakers
            })
    except Exception:
        pass


def _print_section(title: str, data: dict) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="white")
    for k, v in data.items():
        table.add_row(k, str(v))
    console.print(Panel(table, title=f"[bold]{title}[/bold]", border_style="dim"))


# ── 立即生成日报 ──────────────────────────────────────────────────────────────

@app.command()
def digest(
    date: str = typer.Option("", "--date", "-d", help="日期 YYYY-MM-DD，默认今天"),
    no_write: bool = typer.Option(False, "--no-write", help="不写入 Obsidian，只打印"),
) -> None:
    """立即生成日报。"""
    from datetime import datetime, timezone
    import asyncio

    target = None
    if date:
        try:
            target = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            console.print(f"[red]日期格式错误: {date}，应为 YYYY-MM-DD[/red]")
            raise typer.Exit(1)

    async def _run():
        from agent_infra.intelligence.digest import get_digest_generator
        gen = get_digest_generator()
        with console.status("[bold green]生成日报中...[/bold green]"):
            result = await gen.generate_daily(
                date=target,
                write_obsidian=not no_write,
            )
        console.print(Panel(result, title="📋 日报", border_style="green"))

    asyncio.run(_run())


# ── 搜索记忆 ─────────────────────────────────────────────────────────────────

@app.command()
def search(
    query: str = typer.Argument(..., help="搜索关键词"),
    source: str = typer.Option("all", "--source", "-s", help="来源过滤 (all/episodic/semantic)"),
    limit: int = typer.Option(10, "--limit", "-n", help="结果数量"),
) -> None:
    """搜索情节记忆或语义记忆。"""
    if source in ("all", "episodic"):
        console.print(f"\n[bold]情节记忆搜索：{query}[/bold]")
        try:
            from agent_infra.memory.episodic import get_episodic_memory
            mem = get_episodic_memory()
            results = mem.search(query, limit=limit)
            if results:
                for r in results:
                    ts = r.get("timestamp", "")[:16]
                    src = r.get("source", "?")
                    content = r.get("content", "")[:150].replace("\n", " ")
                    console.print(f"  [{ts}] [cyan]{src}[/cyan] {content}")
            else:
                console.print("  [dim]未找到相关记录[/dim]")
        except Exception as e:
            console.print(f"  [red]搜索失败: {e}[/red]")

    if source in ("all", "semantic"):
        console.print(f"\n[bold]语义记忆搜索（Obsidian）：{query}[/bold]")
        try:
            from agent_infra.memory.semantic import get_semantic_memory
            sem = get_semantic_memory()
            results = sem.search(query, n_results=limit)
            if results:
                for r in results:
                    src = r.get("source", "?")
                    score = r.get("score", 0)
                    text = r.get("text", "")[:150].replace("\n", " ")
                    console.print(
                        f"  [score={score:.2f}] [cyan]{src}[/cyan] {text}"
                    )
            else:
                console.print("  [dim]未找到相关记录（vault 可能未索引）[/dim]")
        except Exception as e:
            console.print(f"  [red]语义搜索失败: {e}[/red]")


# ── 直接提问 ─────────────────────────────────────────────────────────────────

@app.command()
def ask(
    question: str = typer.Argument(..., help="你的问题"),
    local: bool = typer.Option(False, "--local", "-l", help="强制使用本地 Ollama"),
) -> None:
    """向 Jarvis 提问（召回记忆 + LLM 回答）。"""
    async def _run():
        from agent_infra.intelligence.llm_router import LLMRouter
        from agent_infra.memory.semantic import get_semantic_memory
        from agent_infra.config.settings import get_settings

        cfg = get_settings()
        router = LLMRouter()
        sem = get_semantic_memory()

        # 召回上下文
        with console.status("[dim]召回记忆...[/dim]"):
            context = sem.format_context(question, n_results=3)

        system = (
            f"你是 {cfg.identity.name}，{cfg.identity.owner} 的个人助手。"
            f"风格：{cfg.identity.persona.style}，{cfg.identity.persona.tone}。"
            f"用中文回答。"
        )
        human = question
        if context:
            human = f"{question}\n\n{context}"

        with console.status("[bold green]思考中...[/bold green]"):
            response = await router.agenerate(
                human=human,
                system=system,
                force_local=local,
            )

        console.print(Panel(response, title="💬 Jarvis", border_style="green"))

    asyncio.run(_run())


# ── 交互式对话（完整 Agent 图）────────────────────────────────────────────────

@app.command()
def chat(
    thread: str = typer.Option("default", "--thread", "-t", help="对话线程 ID（区分不同会话）"),
    code_mode: bool = typer.Option(False, "--code", "-c", help="代码模式（启用 lint + 代码反思）"),
    model: str = typer.Option("claude-sonnet-4-6", "--model", "-m", help="使用的 Claude 模型"),
    hitl: bool = typer.Option(False, "--hitl", help="开启 Human-in-the-Loop 审核"),
) -> None:
    """与 Jarvis 进行多轮对话（完整 Agent 图：路由→规划→反思→记忆）。"""
    from agent_infra.agent import build_agent, stream
    from agent_infra.patterns.hitl import TriggerMode

    hitl_mode: TriggerMode = "conditional" if hitl else "sampling"
    agent = build_agent(model=model, code_mode=code_mode, hitl_mode=hitl_mode)

    console.print(Panel.fit(
        f"[bold green]Jarvis 已就绪[/bold green]  线程: [cyan]{thread}[/cyan]\n"
        "输入 [bold]exit[/bold] 或按 Ctrl+C 退出",
        border_style="green",
    ))

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]再见！[/yellow]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye", "再见"):
            console.print("[yellow]再见！[/yellow]")
            break

        console.print("[bold green]Jarvis:[/bold green] ", end="")
        try:
            for token in stream(agent, user_input, thread_id=thread):
                console.print(token, end="", highlight=False)
            console.print()  # 换行
        except Exception as e:
            console.print(f"\n[red]错误: {e}[/red]")


# ── Pi.dev CLI 入口 ───────────────────────────────────────────────────────────

@app.command()
def pi(
    prompt_parts: list[str] = typer.Argument(
        None,
        metavar="[PROMPT]...",
        help="输入内容；也可以通过 stdin 管道传入",
    ),
    thread: str = typer.Option("pi", "--thread", "-t", help="对话线程 ID"),
    mode: str = typer.Option("agent", "--mode", "-M", help="agent=完整 Agent 图，ask=轻量问答"),
    json_output: bool = typer.Option(False, "--json", help="输出 JSON，适合 Pi 自动读取"),
    local: bool = typer.Option(False, "--local", "-l", help="ask 模式下强制使用本地 Ollama"),
    code_mode: bool = typer.Option(False, "--code", "-c", help="agent 模式下启用代码检查"),
    model: str = typer.Option("claude-sonnet-4-6", "--model", "-m", help="agent 模式使用的模型"),
    hitl: bool = typer.Option(False, "--hitl", help="agent 模式下开启 Human-in-the-Loop 审核"),
) -> None:
    """Pi.dev 可调用的 Jarvis 输入入口。"""
    from agent_infra.pi_bridge import (
        PiInputError,
        build_json_payload,
        resolve_prompt,
        run_agent_prompt,
        run_ask_prompt,
    )

    stdin_text = "" if sys.stdin.isatty() else sys.stdin.read()
    try:
        prompt = resolve_prompt(prompt_parts or (), stdin_text)
    except PiInputError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"agent", "ask"}:
        console.print("[red]mode 必须是 agent 或 ask[/red]")
        raise typer.Exit(1)

    try:
        if normalized_mode == "ask":
            response = asyncio.run(run_ask_prompt(prompt, local=local))
        else:
            response = run_agent_prompt(
                prompt,
                thread=thread,
                model=model,
                code_mode=code_mode,
                hitl=hitl,
            )
    except Exception as e:
        console.print(f"[red]Pi 入口执行失败: {e}[/red]")
        raise typer.Exit(1)

    output = (
        build_json_payload(
            prompt=prompt,
            response=response,
            thread=thread,
            mode=normalized_mode,
        )
        if json_output
        else response
    )
    sys.stdout.write(output)
    if not output.endswith("\n"):
        sys.stdout.write("\n")


# ── 监控仪表盘 ────────────────────────────────────────────────────────────────

@app.command()
def monitor(
    interval: float = typer.Option(3.0, "--interval", "-i", help="刷新间隔（秒）"),
    once: bool = typer.Option(False, "--once", help="只渲染一次，不持续刷新"),
) -> None:
    """实时监控仪表盘：查看每日 Routine 任务状态与当前执行情况。"""
    from agent_infra.monitor.dashboard import run_dashboard
    run_dashboard(refresh_interval=interval, once=once)


# ── 知识推送 ─────────────────────────────────────────────────────────────────

@app.command()
def feed(
    show_only: bool = typer.Option(False, "--show", "-s", help="只展示，不进入评分"),
    fetch: bool = typer.Option(False, "--fetch", "-f", help="立即抓取新内容再展示"),
    size: int = typer.Option(0, "--size", "-n", help="推送条数（0=使用配置默认值）"),
) -> None:
    """今日知识推送：阅读 + 逐条评分，偏好自动学习。"""
    from agent_infra.config.settings import get_settings
    cfg = get_settings()
    feed_size = size if size > 0 else cfg.knowledge.feed_size

    async def _run():
        from agent_infra.knowledge.feed import run_feed
        await run_feed(
            feed_size=feed_size,
            show_only=show_only,
            force_fetch=fetch,
            exploration_ratio=cfg.knowledge.exploration_ratio,
            arxiv_categories=cfg.knowledge.arxiv_categories,
        )

    asyncio.run(_run())


@app.command()
def fetch_knowledge(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """立即抓取最新知识内容（HN + ArXiv）并生成摘要。"""
    from agent_infra.config.settings import get_settings
    cfg = get_settings()

    async def _run():
        from agent_infra.knowledge.feed import run_fetch_only
        with console.status("[bold green]抓取中...[/bold green]"):
            new_count = await run_fetch_only(
                hn_min_score=cfg.knowledge.hn_min_score,
                arxiv_categories=cfg.knowledge.arxiv_categories,
            )
        console.print(f"[green]✓[/green] 新增 {new_count} 条内容（含摘要生成）")
        if verbose:
            from agent_infra.knowledge.store import get_knowledge_store
            stats = get_knowledge_store().stats()
            _print_section("知识库状态", {
                "总条目": stats["total_items"],
                "已推送": stats["shown"],
                "待摘要": stats["pending_summary"],
                "反馈分布": str(stats["feedback"]),
            })

    asyncio.run(_run())


@app.command()
def preferences() -> None:
    """查看并管理话题偏好权重（反馈历史 + 当前权重分布）。"""
    from agent_infra.knowledge.store import get_knowledge_store
    from agent_infra.knowledge.feed import _show_preferences
    store = get_knowledge_store()
    stats = store.stats()
    _print_section("知识库统计", {
        "总条目数": stats["total_items"],
        "已推送": stats["shown"],
        "待摘要": stats["pending_summary"],
        "反馈分布": "  ".join(f"{k}:{v}" for k, v in stats["feedback"].items()) or "暂无",
    })
    console.print()
    _show_preferences(store)


# ── 安装 Claude Code hooks ─────────────────────────────────────────────────────

@app.command()
def install() -> None:
    """安装 Claude Code hooks（一次性配置）。"""
    from agent_infra.capture.adapters.claude_code import ClaudeCodeAdapter
    adapter = ClaudeCodeAdapter()
    adapter.install_hooks()
    console.print("[green]✓[/green] Claude Code hooks 已安装")
    console.print("  配置文件: ~/.claude/settings.json")
    console.print("  重启 Claude Code 后生效")


# ── 手动触发自我维护 ──────────────────────────────────────────────────────────

@app.command()
def maintain(
    no_llm: bool = typer.Option(False, "--no-llm", help="跳过 LLM 代码审查（更快）"),
    no_fix: bool = typer.Option(False, "--no-fix", help="只检查，不自动修复"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """立即运行自我维护：代码检查 + 自动修复。"""
    async def _run():
        from agent_infra.automation.self_maintainer import SelfMaintainer
        maintainer = SelfMaintainer(
            auto_fix=not no_fix,
            llm_review=not no_llm,
        )
        with console.status("[bold]自我维护中...[/bold]"):
            report = await maintainer.run()

        # 打印报告
        console.print(Panel(
            report.to_markdown(),
            title="🔧 自维护报告",
            border_style="blue",
        ))

        # 详细输出
        if verbose and report.issues_found:
            console.print("\n[bold]详细问题列表:[/bold]")
            for issue in report.issues_found:
                icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(issue.severity, "⚪")
                console.print(
                    f"  {icon} [{issue.issue_type}] "
                    f"{issue.file_path.name}:{issue.line_number} — {issue.message}"
                )
            if report.fixes_applied:
                console.print("\n[bold green]已修复:[/bold green]")
                for fix in report.fixes_applied:
                    console.print(f"  ✅ {fix}")

    asyncio.run(_run())


# ── 验证闭环 ────────────────────────────────────────────────────────────────

@app.command()
def verify(
    json_output: bool = typer.Option(False, "--json", help="输出 JSON，适合自动化读取"),
    root: str = typer.Option(".", "--root", help="要验证的项目根目录"),
) -> None:
    """运行构建、lint、测试等验证闭环。"""
    from agent_infra.harness.verification import report_to_json, run_verification, summarize_report

    project_type, report = run_verification(Path(root).resolve())
    output = report_to_json(project_type, report) if json_output else summarize_report(report)
    console.print(output)
    if not all(item.passed for item in report):
        raise typer.Exit(1)


@app.command()
def audit(
    json_output: bool = typer.Option(False, "--json", help="输出 JSON，适合自动化读取"),
    root: str = typer.Option(".", "--root", help="要审计的项目根目录"),
) -> None:
    """审计当前 agent harness 能力覆盖。"""
    from agent_infra.harness.audit import run_harness_audit
    import json

    result = run_harness_audit(Path(root).resolve())
    if json_output:
        console.print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    console.print(f"Harness Audit: {result['overall_score']}/{result['max_score']}")
    for item in result["categories"]:
        status = "PASS" if item["passed"] else "FAIL"
        console.print(f"- {item['category']}: {status} ({item['path']})")
    if result["top_actions"]:
        console.print("\nTop Actions:")
        for action in result["top_actions"]:
            console.print(f"- [{action['category']}] {action['action']} ({action['path']})")


@app.command()
def learn(
    limit: int = typer.Option(50, "--limit", help="扫描最近多少条事件"),
    min_repetitions: int = typer.Option(2, "--min-repetitions", help="重复多少次才提炼为技能"),
) -> None:
    """从近期事件中提炼可复用工作流。"""
    from agent_infra.intelligence.learning import learn_recent_workflows

    skills = learn_recent_workflows(limit=limit, min_repetitions=min_repetitions)
    if not skills:
        console.print("未发现足够稳定的重复工作流。")
        return
    console.print(f"新增 {len(skills)} 个技能:")
    for skill in skills:
        console.print(f"- {skill.name}: {skill.description}")


@app.command()
def adapters() -> None:
    """列出当前已注册的 harness 适配层。"""
    from agent_infra.harness.adapters import get_adapter_registry

    for adapter in get_adapter_registry().list_adapters():
        console.print(
            f"- {adapter.name}: input={','.join(adapter.input_modes)} "
            f"output={','.join(adapter.output_modes)} "
            f"hooks={'yes' if adapter.supports_hooks else 'no'} "
            f"sessions={'yes' if adapter.supports_sessions else 'no'}"
        )


# ── LLM 状态查看 ──────────────────────────────────────────────────────────────

@app.command()
def llm_status() -> None:
    """查看 LLM 路由状态和成本分析。"""
    from agent_infra.intelligence.llm_router import LLMRouter
    router = LLMRouter()

    console.print(Panel.fit(
        router.cost_summary(),
        title="💰 LLM 路由成本分析",
        border_style="green",
    ))

    s = router.status()
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("驱动")
    table.add_column("状态")
    table.add_column("成本")
    table.add_column("说明")

    rows = [
        ("Claude Code CLI", s["claude_cli"], "订阅（零 API 费用）", "需要: claude 命令"),
        ("Codex CLI",       s["codex_cli"], "订阅（零 API 费用）", "需要: codex 命令"),
        ("Ollama 本地",      s["ollama"],    "完全免费",             f"模型: {s['ollama_model']}"),
        ("Claude API",      s["claude_api"], "按量付费",             f"模型: {s['claude_api_model']}"),
    ]

    for name, available, cost, note in rows:
        status_str = "[green]✓ 可用[/green]" if available else "[red]✗ 不可用[/red]"
        table.add_row(name, status_str, cost, note)

    console.print(table)

    # 安装建议
    if not s["claude_cli"]:
        console.print(
            "\n[yellow]💡 安装 Claude Code 可免费使用 Claude 订阅:[/yellow]\n"
            "   npm install -g @anthropic-ai/claude-code\n"
            "   claude login"
        )
    if not s["ollama"]:
        console.print(
            "\n[yellow]💡 安装 Ollama 可完全免费运行本地模型:[/yellow]\n"
            "   brew install ollama\n"
            "   ollama pull hermes3\n"
            "   ollama serve"
        )


# ── 索引 Obsidian vault ───────────────────────────────────────────────────────

@app.command()
def index(
    force: bool = typer.Option(False, "--force", "-f", help="强制重建索引"),
) -> None:
    """索引 Obsidian vault（建立向量检索）。"""
    try:
        from agent_infra.memory.semantic import get_semantic_memory
        sem = get_semantic_memory()
        with console.status("[bold]正在索引 Obsidian vault...[/bold]"):
            stats = sem.index_vault(force=force)
        _print_section("索引完成", {
            "新索引块": stats.get("indexed", 0),
            "跳过（已索引）": stats.get("skipped", 0),
            "错误": stats.get("errors", 0),
        })
    except ImportError as e:
        console.print(f"[red]缺少依赖: {e}[/red]")
        console.print("安装: pip install chromadb sentence-transformers")
    except Exception as e:
        console.print(f"[red]索引失败: {e}[/red]")


# ── 回滚快照 ─────────────────────────────────────────────────────────────────

@app.command()
def rollback(
    snapshot_id: str = typer.Argument(
        default="",
        help="快照 ID（如 20260516_103045）。留空则列出所有可用快照。",
    ),
) -> None:
    """回滚自维护修改到指定快照版本。"""
    from agent_infra.automation import list_snapshots, rollback_snapshot

    snapshots = list_snapshots()

    if not snapshot_id:
        # 列出所有快照
        if not snapshots:
            console.print("[yellow]暂无可用快照。自维护运行后会自动创建。[/yellow]")
            return
        table = Table(show_header=True, header_style="bold cyan", title="可用快照")
        table.add_column("快照 ID", style="bold")
        table.add_column("创建时间")
        table.add_column("修改文件数")
        table.add_column("已应用修复")
        for s in snapshots:
            files = s.get("files", [])
            fixes = [f for f in s.get("fixes_applied", []) if not f.startswith("[快照")]
            table.add_row(
                s["snapshot_id"],
                s.get("created_at", "")[:19].replace("T", " "),
                str(len(files)),
                "\n".join(fixes[:3]) or "—",
            )
        console.print(table)
        console.print("\n[dim]用法: jarvis rollback <快照ID>[/dim]")
        return

    # 执行回滚
    console.print(f"[yellow]正在回滚到快照 {snapshot_id}...[/yellow]")
    restored, errors = rollback_snapshot(snapshot_id)
    if errors:
        for err in errors:
            console.print(f"[red]✗ {err}[/red]")
    if restored:
        console.print(f"[green]✓ 已恢复 {restored} 个文件[/green]")
    else:
        console.print("[red]回滚失败，未恢复任何文件[/red]")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    app()


if __name__ == "__main__":
    main()
