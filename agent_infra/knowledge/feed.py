"""
每日知识推送 — 展示 + 交互评分
=================================
jarvis feed          → 显示今日推送，进入逐条评分模式
jarvis feed --show   → 只显示，不评分
jarvis feed --fetch  → 立即抓取新内容后再展示

评分操作:
  l / ↑   like      喜欢（+话题权重）
  d / ↓   dislike   不喜欢（-话题权重）
  b       bookmark  收藏（++权重 + 写 Obsidian）
  s / →   skip      跳过（权重不变）
  q       quit      退出评分
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

from .store import get_knowledge_store
from .curator import score_feed, ingest_items
from .fetcher import fetch_all

logger = logging.getLogger(__name__)
console = Console()

RATING_LABELS = {
    "l": ("like",     "[bold green][L] 喜欢[/bold green]"),
    "d": ("dislike",  "[bold red][D] 不感兴趣[/bold red]"),
    "b": ("bookmark", "[bold yellow][B] 收藏[/bold yellow]"),
    "s": ("skip",     "[dim][S] 跳过[/dim]"),
    "q": (None,       "[dim][Q] 退出[/dim]"),
}

SOURCE_ICON = {
    "hackernews": "🔶 HN",
    "arxiv":      "📄 ArXiv",
    "github":     "🐙 GitHub",
}


# ── 单条展示 ──────────────────────────────────────────────────────────────────

def _render_item(item: dict, index: int, total: int) -> None:
    source = SOURCE_ICON.get(item.get("source", ""), "🔗")
    topics = item.get("topics") or []
    topic_str = "  ".join(f"[cyan]{t}[/cyan]" for t in topics) if topics else "[dim]未分类[/dim]"
    quality = item.get("quality", 0)
    raw_score = item.get("raw_score", 0)
    score_str = f"⭐ {raw_score} pts" if raw_score > 0 else ""

    header = Text.assemble(
        (f"[{index}/{total}]  ", "dim"),
        (source, "bold"),
        ("  ", ""),
    )
    header.append_text(Text(topic_str))
    if score_str:
        header.append(f"  {score_str}", style="dim")

    summary = item.get("summary") or item.get("title", "")
    url = item.get("url", "")

    console.print()
    console.rule(style="dim")
    console.print(header)
    console.print(f"[bold]{item.get('title', '')}[/bold]")
    if summary and summary != item.get("title"):
        console.print(f"\n{summary}", style="white")
    console.print(f"\n[dim]{url}[/dim]")


def _prompt_rating() -> str:
    """显示评分选项并等待键盘输入，返回 rating key (l/d/b/s/q)。"""
    opts = "  ".join(label for _, label in RATING_LABELS.values())
    console.print(f"\n{opts}\n")
    while True:
        choice = Prompt.ask("评分", default="s", show_default=False).strip().lower()
        if choice in RATING_LABELS:
            return choice
        console.print("[dim]请输入 l / d / b / s / q[/dim]")


# ── 话题权重概览 ──────────────────────────────────────────────────────────────

def _show_preferences(store) -> None:
    stats = store.get_topic_stats()
    if not stats:
        console.print("[dim]尚无话题偏好数据[/dim]")
        return

    table = Table(
        title="话题偏好权重",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("话题", min_width=12)
    table.add_column("权重", justify="right", min_width=6)
    table.add_column("喜欢", justify="right", min_width=5)
    table.add_column("不感兴趣", justify="right", min_width=8)
    table.add_column("趋势")

    for s in stats:
        w = s["weight"]
        bar_len = round(w * 15)
        bar = "█" * bar_len + "░" * (15 - bar_len)
        if w >= 0.7:
            color = "green"
        elif w <= 0.3:
            color = "red"
        else:
            color = "yellow"
        table.add_row(
            s["topic"],
            f"[{color}]{w:.2f}[/{color}]",
            str(s.get("like_count", 0)),
            str(s.get("dislike_count", 0)),
            f"[{color}]{bar}[/{color}]",
        )
    console.print(table)


# ── 收藏写入 Obsidian ─────────────────────────────────────────────────────────

def _bookmark_to_obsidian(item: dict) -> None:
    try:
        from agent_infra.config.settings import get_settings
        cfg = get_settings()
        folder = cfg.memory.vault_path / "Jarvis Inbox" / "Knowledge Bookmarks"
        folder.mkdir(parents=True, exist_ok=True)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in item.get("title", "untitled"))[:60]
        md_path = folder / f"{today} {safe_title}.md"

        topics = ", ".join(item.get("topics") or [])
        content = (
            f"---\n"
            f"source: {item.get('source', '')}\n"
            f"url: {item.get('url', '')}\n"
            f"topics: [{topics}]\n"
            f"bookmarked: {today}\n"
            f"---\n\n"
            f"# {item.get('title', '')}\n\n"
            f"{item.get('summary', '')}\n\n"
            f"[原文链接]({item.get('url', '')})\n"
        )
        md_path.write_text(content, encoding="utf-8")
        console.print(f"[green]✓[/green] 已收藏到 Obsidian: {md_path.name}")
    except Exception as e:
        logger.warning(f"Obsidian bookmark failed: {e}")
        console.print(f"[yellow]⚠[/yellow] Obsidian 写入失败: {e}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

async def run_feed(
    feed_size: int = 7,
    show_only: bool = False,
    force_fetch: bool = False,
    exploration_ratio: float = 0.2,
    arxiv_categories: list[str] | None = None,
) -> None:
    store = get_knowledge_store()

    # 可选立即抓取
    if force_fetch:
        with console.status("[bold green]抓取最新内容中...[/bold green]"):
            raw_items = await fetch_all(arxiv_categories=arxiv_categories)
            new_count = await ingest_items(raw_items, store=store)
        console.print(f"[green]✓[/green] 新增 {new_count} 条，正在生成摘要...")

    # 还有待富化的条目则提示
    pending = store.pending_summary_count()
    if pending > 0 and not force_fetch:
        console.print(f"[dim]提示: 有 {pending} 条待摘要，可运行 jarvis fetch 抓取并生成[/dim]")

    # 拿候选
    candidates = store.get_feed_candidates(limit=80)
    if not candidates:
        console.print(
            Panel(
                "[yellow]知识库为空。[/yellow]\n\n"
                "运行 [bold cyan]jarvis fetch[/bold cyan] 先抓取内容。",
                title="📚 今日推送",
                border_style="yellow",
            )
        )
        return

    # 权重打分 + 选条目
    weights = store.get_topic_weights()
    feed = score_feed(candidates, weights, feed_size=feed_size, exploration_ratio=exploration_ratio)

    if not feed:
        console.print("[dim]暂无新内容可推送[/dim]")
        return

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    console.print(Panel.fit(
        f"[bold]📚 今日知识推送[/bold]  {today}  [dim]共 {len(feed)} 条[/dim]",
        border_style="blue",
    ))

    rated: dict[str, str] = {}

    for idx, item in enumerate(feed, 1):
        _render_item(item, idx, len(feed))

        if show_only:
            continue

        choice = _prompt_rating()
        rating_key, _ = RATING_LABELS[choice]

        if choice == "q":
            console.print("[dim]退出评分[/dim]")
            break

        if rating_key:
            store.save_feedback(item["id"], rating_key)
            rated[item["id"]] = rating_key
            if rating_key == "bookmark":
                _bookmark_to_obsidian(item)

    # 标记已展示
    store.mark_shown([item["id"] for item in feed])

    # 评分摘要
    if rated and not show_only:
        console.print()
        console.rule("[dim]评分完成[/dim]", style="dim")
        like_n = sum(1 for r in rated.values() if r in ("like", "bookmark"))
        dislike_n = sum(1 for r in rated.values() if r == "dislike")
        console.print(
            f"本次评分: [green]喜欢 {like_n}[/green]  "
            f"[red]不感兴趣 {dislike_n}[/red]  "
            f"共 {len(rated)} 条"
        )
        console.print("[dim]话题偏好已自动更新 → jarvis preferences 查看[/dim]")


async def run_fetch_only(
    hn_min_score: int = 100,
    arxiv_categories: list[str] | None = None,
) -> int:
    """只抓取 + 富化，不展示，供调度任务调用。"""
    store = get_knowledge_store()
    raw_items = await fetch_all(
        hn_min_score=hn_min_score,
        arxiv_categories=arxiv_categories,
    )
    new_count = await ingest_items(raw_items, store=store)
    logger.info(f"Knowledge fetch complete: {new_count} new items")
    return new_count
