"""
反哺引擎 — 智能洞察生成与推送
================================
你输入 → Agent 用更广的知识理解 → 主动推送洞察给你。
这就是「反哺」：不是被动问答，而是主动增强。

工作流：
  1. 订阅 EventBus 高重要性事件
  2. 召回语义记忆（Obsidian RAG）+ 程序记忆（技能库）
  3. Claude API 生成洞察
  4. 写入 Obsidian Inbox
  5. macOS 通知推送

支持 mixed 模式（即时 + 日报）。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable

from agent_infra.capture.event_bus import CaptureEvent, EventBus, EventSource, EventType, get_event_bus
from agent_infra.config.settings import get_settings

logger = logging.getLogger(__name__)


class AugmentationEngine:
    """
    反哺引擎。

    用法：
        engine = AugmentationEngine()
        engine.attach(bus)   # 订阅事件总线
        # 之后 bus.start() 自动驱动引擎
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self.cfg = cfg.augmentation
        self.identity = cfg.identity
        self._llm = None      # 懒加载
        self._semantic = None
        self._procedural = None

    # ── 初始化（懒加载重型依赖）──────────────────────────────────────────────

    def _get_llm(self):
        if self._llm is None:
            from agent_infra.intelligence.llm_router import LLMRouter
            self._llm = LLMRouter()
        return self._llm

    def _get_semantic(self):
        if self._semantic is None:
            from agent_infra.memory.semantic import get_semantic_memory
            self._semantic = get_semantic_memory()
        return self._semantic

    def _get_procedural(self):
        if self._procedural is None:
            from agent_infra.memory.procedural import get_procedural_memory
            self._procedural = get_procedural_memory()
        return self._procedural

    # ── 事件总线挂载 ──────────────────────────────────────────────────────────

    def attach(self, bus: EventBus | None = None) -> None:
        """订阅事件总线，开始监听高重要性事件。"""
        bus = bus or get_event_bus()
        bus.subscribe(
            handler=self.handle_event,
            min_importance=self.cfg.importance_threshold,
        )
        logger.info(
            f"AugmentationEngine attached to EventBus "
            f"(threshold={self.cfg.importance_threshold})"
        )

    async def handle_event(self, event: CaptureEvent) -> None:
        """核心处理：为重要事件生成洞察并推送。"""
        try:
            # 混合模式：即时推送（延迟 N 秒，等待用户完成输入）
            if self.cfg.mode in ("instant", "mixed"):
                await asyncio.sleep(self.cfg.instant_delay_s)
                await self._process_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"AugmentationEngine.handle_event failed: {e}")

    async def _process_event(self, event: CaptureEvent) -> None:
        """生成洞察的完整流程。"""
        logger.info(f"Processing event for augmentation: {event.id} [{event.source.value}]")

        # 1. 召回上下文
        context = await asyncio.to_thread(self._build_context, event.content)

        # 2. 生成洞察
        insights = await self._generate_insights(event, context)
        if not insights:
            return

        # 3. 写入 Obsidian
        for insight in insights:
            await asyncio.to_thread(self._write_insight, event, insight)

        # 4. macOS 通知
        if self.cfg.notify_desktop:
            await asyncio.to_thread(
                self._notify,
                f"Jarvis 洞察 [{event.source.value}]",
                insights[0][:100] + "..." if len(insights[0]) > 100 else insights[0],
            )

        # 5. 标记已处理
        try:
            from agent_infra.memory.episodic import get_episodic_memory
            get_episodic_memory().mark_processed(
                event.id,
                summary=insights[0][:200] if insights else "",
            )
        except Exception:
            pass

    def _build_context(self, content: str) -> str:
        """召回语义记忆 + 程序记忆作为背景。"""
        parts = []
        try:
            semantic = self._get_semantic()
            vault_ctx = semantic.format_context(content, n_results=3)
            if vault_ctx:
                parts.append(vault_ctx)
        except Exception as e:
            logger.debug(f"Semantic context failed: {e}")

        try:
            procedural = self._get_procedural()
            skill_ctx = procedural.format_for_prompt(content)
            if skill_ctx:
                parts.append(skill_ctx)
        except Exception as e:
            logger.debug(f"Procedural context failed: {e}")

        return "\n\n".join(parts)

    async def _generate_insights(
        self, event: CaptureEvent, context: str
    ) -> list[str]:
        """调用 LLM 生成洞察。"""
        llm = self._get_llm()

        source_desc = {
            EventSource.CLIPBOARD: "你刚刚复制了",
            EventSource.CLAUDE_CODE: "你刚完成了一次 Claude Code 对话",
            EventSource.CLI: "你刚执行了命令",
            EventSource.MANUAL: "你刚手动输入了",
        }.get(event.source, "你刚产生了")

        system = f"""你是 {self.identity.name}，{self.identity.owner} 的个人 AI 助手。
你的风格：{self.identity.persona.style}、{self.identity.persona.tone}。

你的任务：基于用户刚产生的内容，结合他的知识库背景，生成 1-3 条有价值的洞察。
洞察要求：
- 补充用户可能没有注意到的关联信息
- 指出潜在风险或改进点
- 推荐下一步行动
- 简洁直接，不废话
- 用中文回复

如果这个内容不值得特别洞察（如简单的复制粘贴、无意义内容），直接返回空字符串。"""

        human = f"{source_desc}以下内容：\n\n{event.content[:2000]}"
        if context:
            human += f"\n\n{context}"

        try:
            response = await llm.agenerate(system=system, human=human, simple=False)
            if not response or response.strip() == "":
                return []
            # 解析为条目列表（每行一条，或整体一条）
            lines = [
                line.strip().lstrip("•-*123456789. ")
                for line in response.splitlines()
                if len(line.strip()) > 20
            ]
            return lines[:self.cfg.max_insights_per_event] if lines else [response.strip()]
        except Exception as e:
            logger.error(f"LLM insight generation failed: {e}")
            return []

    def _write_insight(self, event: CaptureEvent, insight: str) -> None:
        """将洞察写入 Obsidian Inbox。"""
        try:
            semantic = self._get_semantic()
            ts = datetime.now(timezone.utc).strftime("%H:%M")
            content = (
                f"## 来源\n"
                f"- **类型**: {event.source.value}\n"
                f"- **时间**: {ts} UTC\n"
                f"- **重要性**: {event.importance:.1f}\n\n"
                f"## 原始内容摘要\n"
                f"{event.content[:300]}{'...' if len(event.content) > 300 else ''}\n\n"
                f"## Jarvis 洞察\n"
                f"{insight}\n"
            )
            semantic.write_insight(
                title=f"Insight {ts}",
                content=content,
                tags=["jarvis-insight", event.source.value],
            )
        except Exception as e:
            logger.error(f"Failed to write insight to Obsidian: {e}")

    def _notify(self, title: str, message: str) -> None:
        """macOS 桌面通知。"""
        try:
            import subprocess
            script = (
                f'display notification "{message}" '
                f'with title "{title}" '
                f'sound name "Glass"'
            )
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5,
            )
        except Exception as e:
            logger.debug(f"Notification failed: {e}")


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_engine: AugmentationEngine | None = None


def get_augmentation_engine() -> AugmentationEngine:
    global _engine
    if _engine is None:
        _engine = AugmentationEngine()
    return _engine
