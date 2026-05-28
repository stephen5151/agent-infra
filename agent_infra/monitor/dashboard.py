"""
监控仪表盘 — Rich Live 渲染
=============================
jarvis monitor 命令展示：
  • Daemon 运行状态
  • 每日 Routine 任务（配置 + 下次/上次执行时间）
  • 当前执行中的任务（高亮）
  • 最近活动事件流（情节记忆最新条目）
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich import box

from .task_registry import get_registry

console = Console()


# ── 格式化工具 ────────────────────────────────────────────────────────────────

def _rel_time(iso: str) -> str:
    """将 ISO 时间转为相对描述，例如 '3 小时前' 或 '2 小时后'。"""
    if not iso:
        return "[dim]—[/dim]"
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = dt - now
        secs = int(delta.total_seconds())
        if abs(secs) < 60:
            return "刚刚" if secs >= 0 else "[green]刚完成[/green]"
        mins = abs(secs) // 60
        hours = mins // 60
        days = hours // 24
        if days > 0:
            label = f"{days}天{hours % 24}小时"
        elif hours > 0:
            label = f"{hours}小时{mins % 60}分"
        else:
            label = f"{mins}分钟"
        return f"[cyan]{label}后[/cyan]" if secs > 0 else f"[green]{label}前[/green]"
    except Exception:
        return "[dim]?[/dim]"


def _status_badge(status: str) -> Text:
    badges = {
        "success": Text("✓ 成功", style="bold green"),
        "failed":  Text("✗ 失败", style="bold red"),
        "running": Text("⟳ 执行中", style="bold yellow"),
        "never":   Text("— 未运行", style="dim"),
    }
    return badges.get(status, Text(status, style="dim"))


# ── 面板渲染 ──────────────────────────────────────────────────────────────────

def _build_daemon_panel(snapshot: dict, alive: bool) -> Panel:
    if alive:
        pid = snapshot.get("daemon_pid", "?")
        started_iso = snapshot.get("daemon_started", "")
        started_rel = _rel_time(started_iso) if started_iso else "[dim]未知[/dim]"
        content = Text.assemble(
            ("● ", "bold green"),
            ("守护进程运行中  ", "green"),
            (f"PID {pid}  ", "dim"),
            ("启动于 ", "dim"),
        )
        content.append_text(Text(started_rel))
    else:
        content = Text.assemble(
            ("○ ", "bold red"),
            ("守护进程未运行  ", "red"),
            ("运行 ", "dim"),
            ("jarvis start", "bold cyan"),
            (" 启动", "dim"),
        )
    return Panel(content, title="[bold]Daemon[/bold]", border_style="green" if alive else "red", padding=(0, 1))


def _build_routine_table(tasks: dict) -> Table:
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        expand=True,
        padding=(0, 1),
    )
    table.add_column("任务", min_width=12)
    table.add_column("计划时间", justify="center", min_width=8)
    table.add_column("状态", justify="center", min_width=10)
    table.add_column("上次执行", justify="right", min_width=12)
    table.add_column("下次执行", justify="right", min_width=12)
    table.add_column("耗时", justify="right", min_width=8)

    if not tasks:
        table.add_row("[dim]无已注册任务[/dim]", "", "", "", "", "")
        return table

    for tid, info in tasks.items():
        status = info.get("last_status", "never")
        is_running = status == "running"

        label = info.get("label", tid)
        if is_running:
            label = f"[bold yellow]{label}[/bold yellow]"

        sched = info.get("schedule", "?")
        duration = info.get("last_duration_s", 0)
        dur_str = f"{duration:.1f}s" if duration else "—"

        err = info.get("error", "")
        status_cell = _status_badge(status)
        if err and status == "failed":
            status_cell = Text.assemble(
                _status_badge(status),
                ("\n", ""),
                (err[:40], "dim red"),
            )

        table.add_row(
            label,
            f"[cyan]{sched}[/cyan]",
            status_cell,
            _rel_time(info.get("last_run", "")),
            _rel_time(info.get("next_run", "")),
            dur_str,
        )

    return table


def _build_knowledge_bar() -> Text:
    """知识库一行摘要：未读数 + 已收藏数。"""
    try:
        from agent_infra.knowledge.store import get_knowledge_store
        store = get_knowledge_store()
        stats = store.stats()
        total = stats.get("total_items", 0)
        shown = stats.get("shown", 0)
        unread = total - shown
        bookmarks = stats.get("feedback", {}).get("bookmark", 0)
        likes = stats.get("feedback", {}).get("like", 0)
        pending = stats.get("pending_summary", 0)

        line = Text(no_wrap=True)
        if unread > 0:
            line.append(f"📚 {unread} 条待读", style="cyan")
        else:
            line.append("📚 无待读内容", style="dim")
        if bookmarks:
            line.append(f"  ⭐ {bookmarks} 已收藏", style="yellow")
        if likes:
            line.append(f"  👍 {likes} 喜欢", style="green")
        if pending:
            line.append(f"  ⏳ {pending} 待摘要", style="dim")
        return line
    except Exception:
        return Text("知识库未初始化", style="dim")


def _build_activity_panel(limit: int = 5) -> Panel:
    """从情节记忆拉最新事件 + 知识库摘要行。"""
    rows: list[str] = []

    # 知识库状态行
    kb_line = _build_knowledge_bar()

    try:
        from agent_infra.memory.episodic import get_episodic_memory
        mem = get_episodic_memory()
        events = mem.recent(limit=limit)
        for e in events:
            ts = (e.get("timestamp") or "")[:16].replace("T", " ")
            src = e.get("source", "?")
            content = (e.get("content") or "").replace("\n", " ")[:60]
            rows.append(f"[dim]{ts}[/dim] [cyan]{src}[/cyan] {content}")
    except Exception as ex:
        rows.append(f"[dim]无法读取事件流: {ex}[/dim]")

    from rich.console import Group
    from rich.text import Text as RText

    if rows:
        activity_text = RText("\n".join(rows))
    else:
        activity_text = RText("暂无活动", style="dim")
    body = Group(
        Panel(kb_line, title="[bold]知识推送[/bold]", border_style="cyan", padding=(0, 1)),
        activity_text,
    )
    return Panel(body, title="[bold]最近活动[/bold]", border_style="dim", padding=(0, 1))


# ── 主渲染 ────────────────────────────────────────────────────────────────────

def _render(snapshot: dict, alive: bool) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="daemon", size=3),
        Layout(name="routine"),
        Layout(name="activity", size=10),
    )

    layout["daemon"].update(_build_daemon_panel(snapshot, alive))
    layout["routine"].update(
        Panel(
            _build_routine_table(snapshot.get("tasks", {})),
            title="[bold]每日 Routine[/bold]",
            border_style="blue",
            padding=(0, 0),
        )
    )
    layout["activity"].update(_build_activity_panel())
    return layout


def run_dashboard(refresh_interval: float = 3.0, once: bool = False) -> None:
    """启动 Live 仪表盘。once=True 时只渲染一次（用于 CI / 截图）。"""
    registry = get_registry()

    if once:
        snapshot = registry.snapshot()
        alive = registry.is_daemon_alive()
        console.print(_render(snapshot, alive))
        return

    with Live(
        _render(registry.snapshot(), registry.is_daemon_alive()),
        console=console,
        refresh_per_second=1,
        screen=True,
    ) as live:
        try:
            import time
            while True:
                snapshot = registry.snapshot()
                alive = registry.is_daemon_alive()
                live.update(_render(snapshot, alive))
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            pass
