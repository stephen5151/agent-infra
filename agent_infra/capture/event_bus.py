"""
事件总线 — 所有捕获数据的统一入口
=====================================
所有感知适配器把数据包装成 CaptureEvent 扔进来，
所有智能模块订阅自己关心的事件类型。

设计原则：
  - Pydantic 类型化，字段清晰
  - 异步非阻塞，不拖慢主流程
  - 持久化队列，进程重启不丢事件
  - 订阅者失败不影响其他订阅者
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── 事件类型定义 ──────────────────────────────────────────────────────────────

class EventSource(str, Enum):
    CLAUDE_CODE  = "claude_code"    # Claude Code 对话
    CLI          = "cli"            # 终端命令
    CLIPBOARD    = "clipboard"      # 剪贴板复制
    FILE         = "file"           # 文件创建/修改
    BROWSER      = "browser"        # 浏览器（未来）
    VOICE        = "voice"          # 语音输入（未来）
    OMI          = "omi"            # Omi 设备（未来）
    MANUAL       = "manual"         # 手动输入
    AGENT        = "agent"          # Agent 自身产生的事件


class EventType(str, Enum):
    INPUT        = "input"          # 你输入了什么
    OUTPUT       = "output"         # 你得到了什么
    ACTION       = "action"         # 你执行了什么操作
    INSIGHT      = "insight"        # Agent 产生的洞察
    DIGEST       = "digest"         # 日报/周报
    SKILL_LEARN  = "skill_learn"    # 学到了新的可复用技能
    HEARTBEAT    = "heartbeat"      # 系统健康检查


class CaptureEvent(BaseModel):
    """标准事件格式，所有适配器必须产生此格式。"""
    id: str                 = Field(default_factory=lambda: str(uuid4()))
    source: EventSource
    type: EventType
    content: str                                    # 主要内容
    summary: str            = ""                    # 简短摘要（可为空，由 agent 生成）
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime     = Field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float       = 0.5                   # 0-1，由适配器初估，agent 可更新
    processed: bool         = False                 # 是否已被智能层处理

    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()

    def to_log_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source.value,
            "type": self.type.value,
            "content": self.content[:500],
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
        }


# ── 订阅者类型 ────────────────────────────────────────────────────────────────

Handler = Callable[[CaptureEvent], Awaitable[None]]


@dataclass
class Subscription:
    handler: Handler
    sources: set[EventSource] | None = None    # None = 订阅所有来源
    types: set[EventType] | None = None        # None = 订阅所有类型
    min_importance: float = 0.0


# ── 事件总线 ──────────────────────────────────────────────────────────────────

class EventBus:
    """
    异步事件总线。

    用法：
        bus = EventBus()
        bus.subscribe(my_handler, sources={EventSource.CLI})
        await bus.publish(event)
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._subscriptions: list[Subscription] = []
        self._queue: asyncio.Queue[CaptureEvent] = asyncio.Queue()
        self._running = False
        self._persist_path = Path(persist_path) if persist_path else None
        self._pending_writes: list[dict] = []

    # ── 订阅 ─────────────────────────────────────────────────────────────────

    def subscribe(
        self,
        handler: Handler,
        sources: set[EventSource] | None = None,
        types: set[EventType] | None = None,
        min_importance: float = 0.0,
    ) -> None:
        """注册事件处理器。"""
        self._subscriptions.append(Subscription(
            handler=handler,
            sources=sources,
            types=types,
            min_importance=min_importance,
        ))

    # ── 发布 ─────────────────────────────────────────────────────────────────

    async def publish(self, event: CaptureEvent) -> None:
        """发布事件到总线（非阻塞）。"""
        await self._queue.put(event)
        if self._persist_path:
            self._append_to_log(event)

    def publish_sync(self, event: CaptureEvent) -> None:
        """同步发布（从非异步代码调用）。"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.publish(event))
            else:
                loop.run_until_complete(self.publish(event))
        except RuntimeError:
            asyncio.run(self.publish(event))

    # ── 运行循环 ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """启动事件分发循环。"""
        self._running = True
        logger.info("EventBus started")
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._dispatch(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"EventBus dispatch error: {e}")

    async def stop(self) -> None:
        self._running = False
        logger.info("EventBus stopped")

    async def _dispatch(self, event: CaptureEvent) -> None:
        """将事件分发给所有匹配的订阅者（并行）。"""
        tasks = []
        for sub in self._subscriptions:
            if not self._matches(event, sub):
                continue
            tasks.append(self._safe_call(sub.handler, event))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_call(self, handler: Handler, event: CaptureEvent) -> None:
        """单个 handler 出错不影响其他 handler。"""
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Handler {handler.__name__} failed: {e}")

    def _matches(self, event: CaptureEvent, sub: Subscription) -> bool:
        if event.importance < sub.min_importance:
            return False
        if sub.sources and event.source not in sub.sources:
            return False
        if sub.types and event.type not in sub.types:
            return False
        return True

    # ── 持久化 ────────────────────────────────────────────────────────────────

    def _append_to_log(self, event: CaptureEvent) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_log_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to persist event: {e}")


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        from agent_infra.config.settings import get_settings
        cfg = get_settings()
        log_path = cfg.memory.episodic_db_path.parent / "events.jsonl"
        _bus = EventBus(persist_path=log_path)
    return _bus
