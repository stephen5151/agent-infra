"""
适配器基类 — pluggy 插件规范
================================
每个数据源是一个适配器插件。
实现 BaseAdapter 即可接入事件总线，无需改动核心代码。
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import pluggy

from agent_infra.capture.event_bus import CaptureEvent, EventBus, get_event_bus

logger = logging.getLogger(__name__)

# pluggy hook 规范名称
PROJECT_NAME = "jarvis"
hookspec = pluggy.HookspecMarker(PROJECT_NAME)
hookimpl = pluggy.HookimplMarker(PROJECT_NAME)


class AdapterSpec:
    """插件 hook 规范定义。"""

    @hookspec
    def on_capture(self, event: CaptureEvent) -> None:
        """有新事件被捕获时调用。"""

    @hookspec
    def on_start(self) -> None:
        """适配器启动时调用。"""

    @hookspec
    def on_stop(self) -> None:
        """适配器停止时调用。"""


class BaseAdapter(ABC):
    """
    所有捕获适配器的基类。

    子类只需实现:
      - name: 适配器标识符
      - run(): 主循环（持续产生事件）
    """

    name: str = "base"
    enabled: bool = True

    def __init__(self, bus: EventBus | None = None) -> None:
        self.bus = bus or get_event_bus()
        self._running = False
        self.logger = logging.getLogger(f"adapter.{self.name}")

    async def start(self) -> None:
        """启动适配器，在后台运行。"""
        if not self.enabled:
            return
        self._running = True
        self.logger.info(f"Adapter [{self.name}] started")
        try:
            await self.run()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Adapter [{self.name}] crashed: {e}")
        finally:
            self._running = False

    async def stop(self) -> None:
        self._running = False

    @abstractmethod
    async def run(self) -> None:
        """子类实现：持续监听并 emit 事件。"""

    async def emit(self, event: CaptureEvent) -> None:
        """发布事件到总线。"""
        await self.bus.publish(event)
