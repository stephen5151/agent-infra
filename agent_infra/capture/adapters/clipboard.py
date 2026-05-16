"""
剪贴板适配器
=============
你复制的内容 = 你认为值得记录的内容。
后台静默监听，自动过滤噪音（太短、重复、密码）。
"""
from __future__ import annotations

import asyncio
import subprocess

from agent_infra.capture.adapters.base import BaseAdapter
from agent_infra.capture.event_bus import CaptureEvent, EventSource, EventType
from agent_infra.config.settings import get_settings


class ClipboardAdapter(BaseAdapter):
    """macOS 剪贴板监听（通过 pbpaste 命令）。"""

    name = "clipboard"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        cfg = get_settings()
        self.enabled = cfg.capture.clipboard_enabled
        self.min_length = cfg.capture.clipboard_min_length
        self.privacy = cfg.privacy
        self._last_content = ""
        self.poll_interval = 1.5

    async def run(self) -> None:
        while self._running:
            content = await self._read_clipboard()
            if self._should_capture(content):
                self._last_content = content
                event = CaptureEvent(
                    source=EventSource.CLIPBOARD,
                    type=EventType.INPUT,
                    content=content,
                    importance=self._estimate_importance(content),
                    metadata={"length": len(content)},
                )
                await self.emit(event)
            await asyncio.sleep(self.poll_interval)

    async def _read_clipboard(self) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "pbpaste",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    def _should_capture(self, content: str) -> bool:
        if not content or content == self._last_content:
            return False
        if len(content) < self.min_length:
            return False
        if self.privacy.is_sensitive(content):
            self.logger.debug("Clipboard: sensitive content skipped")
            return False
        return True

    def _estimate_importance(self, content: str) -> float:
        """简单启发式：代码、URL、长文本重要性更高。"""
        if content.startswith(("http://", "https://")):
            return 0.6
        if any(kw in content for kw in ["def ", "class ", "function ", "const ", "import "]):
            return 0.8   # 代码片段
        if len(content) > 500:
            return 0.7   # 长文本
        return 0.5
