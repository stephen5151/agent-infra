"""
日报生成器 — 每日智能摘要
============================
每天 08:00 自动生成昨天的工作摘要，写入 Obsidian。
内容：
  - 今天做了什么（情节记忆）
  - 学到了什么（语义检索）
  - 识别到的模式（程序记忆）
  - 明天建议（LLM 推断）
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from agent_infra.config.settings import get_settings

logger = logging.getLogger(__name__)


class DigestGenerator:
    """
    每日摘要生成器。

    用法：
        gen = DigestGenerator()
        await gen.generate_daily()   # 生成今天的日报
    """

    def __init__(self) -> None:
        self.cfg = get_settings().digest
        self.identity = get_settings().identity
        self._llm = None
        self._episodic = None
        self._semantic = None

    def _get_llm(self):
        if self._llm is None:
            from agent_infra.intelligence.llm_router import LLMRouter
            self._llm = LLMRouter()
        return self._llm

    def _get_episodic(self):
        if self._episodic is None:
            from agent_infra.memory.episodic import get_episodic_memory
            self._episodic = get_episodic_memory()
        return self._episodic

    def _get_semantic(self):
        if self._semantic is None:
            from agent_infra.memory.semantic import get_semantic_memory
            self._semantic = get_semantic_memory()
        return self._semantic

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def generate_daily(
        self,
        date: datetime | None = None,
        write_obsidian: bool | None = None,
    ) -> str:
        """
        生成日报。

        Args:
            date: 报告日期，None = 今天
            write_obsidian: 是否写入 Obsidian，None = 读 cfg

        Returns:
            日报 Markdown 字符串
        """
        target_date = date or datetime.now(timezone.utc)
        write = write_obsidian if write_obsidian is not None else self.cfg.write_to_obsidian

        logger.info(f"Generating daily digest for {target_date.date()}")

        # 收集原始数据
        raw_data = self._collect_data(target_date)

        # LLM 生成日报
        digest_md = await self._generate_digest_markdown(target_date, raw_data)

        # 写入 Obsidian
        if write:
            self._write_to_obsidian(digest_md, target_date)

        # 发送通知
        if self.cfg.notify_voice:
            self._notify(target_date)

        return digest_md

    # ── 数据收集 ─────────────────────────────────────────────────────────────

    def _collect_data(self, date: datetime) -> dict[str, Any]:
        """从情节记忆收集过去 24 小时的原始数据。"""
        episodic = self._get_episodic()

        # 昨天 00:00 到今天 00:00 的所有事件
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        hours_back = (datetime.now(timezone.utc) - start).total_seconds() / 3600

        episodes = episodic.recent(
            hours=min(hours_back + 1, 25),
            min_importance=0.3,
            limit=500,
        )

        # 按来源分组
        by_source: dict[str, list] = {}
        for ep in episodes:
            src = ep["source"]
            by_source.setdefault(src, []).append(ep)

        # 统计
        stats = episodic.stats()

        return {
            "date": date.strftime("%Y-%m-%d"),
            "episodes": episodes,
            "by_source": by_source,
            "total_events": len(episodes),
            "stats": stats,
        }

    # ── LLM 生成 ─────────────────────────────────────────────────────────────

    async def _generate_digest_markdown(
        self, date: datetime, data: dict[str, Any]
    ) -> str:
        """调用 LLM 生成结构化日报。"""
        llm = self._get_llm()

        date_str = date.strftime("%Y年%m月%d日")
        total = data["total_events"]

        # 准备原始内容摘要（控制 token）
        episode_summary = self._format_episodes_for_prompt(data["episodes"])

        system = f"""你是 {self.identity.name}，{self.identity.owner} 的个人 AI 助手。
今天是 {date_str}。
你的任务是基于用户今天的活动日志，生成一份有价值的日报。

日报格式（Markdown）：
## 📊 今日概览
（统计数字：事件数、活跃时间段等）

## ✅ 完成了什么
（具体事项，从 claude_code / clipboard / cli 事件中提炼）

## 💡 学到了什么
（从内容中提炼知识点，不是重复原文）

## 🔍 发现的模式
（有没有重复性工作？有没有可优化的地方？）

## 🚀 明天建议
（基于今天的工作，推断明天可能需要做什么）

风格：简洁直接，用中文，要有实际内容而非空话。
如果今天活动很少，如实说，不要编造。"""

        human = (
            f"今天共 {total} 条活动记录。\n\n"
            f"活动摘要：\n{episode_summary}"
        )

        try:
            digest = await llm.agenerate(system=system, human=human, simple=False)
        except Exception as e:
            logger.error(f"LLM digest generation failed: {e}")
            digest = self._fallback_digest(date_str, data)

        # 添加 frontmatter 头
        header = (
            f"# {date_str} 日报\n\n"
            f"*由 Jarvis 自动生成 · {datetime.now(timezone.utc).strftime('%H:%M UTC')}*\n\n"
        )
        return header + digest

    def _format_episodes_for_prompt(self, episodes: list[dict], max_chars: int = 3000) -> str:
        """将情节列表格式化为 prompt 可用的紧凑格式。"""
        lines = []
        total_len = 0
        for ep in episodes:
            source = ep.get("source", "?")
            content = ep.get("content", "")[:200].replace("\n", " ")
            importance = ep.get("importance", 0.5)
            line = f"[{source}|{importance:.1f}] {content}"
            if total_len + len(line) > max_chars:
                lines.append("...(更多内容已省略)")
                break
            lines.append(line)
            total_len += len(line)
        return "\n".join(lines)

    def _fallback_digest(self, date_str: str, data: dict) -> str:
        """LLM 失败时的降级日报（纯统计）。"""
        by_source = data["by_source"]
        lines = [f"## 今日统计"]
        for src, eps in by_source.items():
            lines.append(f"- {src}: {len(eps)} 条记录")
        return "\n".join(lines)

    # ── 输出 ─────────────────────────────────────────────────────────────────

    def _write_to_obsidian(self, content: str, date: datetime) -> None:
        try:
            from agent_infra.memory.semantic import get_semantic_memory
            semantic = get_semantic_memory()
            path = semantic.write_daily_digest(content, date=date)
            logger.info(f"Daily digest written → {path}")
        except Exception as e:
            logger.error(f"Failed to write digest to Obsidian: {e}")

    def _notify(self, date: datetime) -> None:
        try:
            import subprocess
            date_str = date.strftime("%m月%d日")
            script = (
                f'display notification "你的 {date_str} 日报已生成，查看 Obsidian Inbox" '
                f'with title "Jarvis 日报" sound name "Glass"'
            )
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
        except Exception:
            pass


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_digest: DigestGenerator | None = None


def get_digest_generator() -> DigestGenerator:
    global _digest
    if _digest is None:
        _digest = DigestGenerator()
    return _digest
