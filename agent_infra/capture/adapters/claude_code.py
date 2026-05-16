"""
Claude Code 适配器
====================
通过 Claude Code 的 hooks 机制捕获所有对话。

工作原理：
  Claude Code 在 ~/.claude/settings.json 中配置 hooks，
  每次工具调用前后、对话结束时调用我们的脚本，
  脚本把数据写入事件队列文件，适配器定期读取。

需要在 ~/.claude/settings.json 添加:
  {
    "hooks": {
      "Stop": [{"matcher": "", "hooks": [
        {"type": "command", "command": "python3 -m agent_infra.capture.adapters.claude_code_hook"}
      ]}]
    }
  }
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from agent_infra.capture.adapters.base import BaseAdapter
from agent_infra.capture.event_bus import (
    CaptureEvent, EventSource, EventType,
)
from agent_infra.config.settings import get_settings


class ClaudeCodeAdapter(BaseAdapter):
    """
    监听 Claude Code hook 写入的事件队列文件。
    Claude Code 每次结束对话时，hook 脚本写入一条 JSON，
    本适配器轮询读取并发布到事件总线。
    """

    name = "claude_code"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        cfg = get_settings()
        self.enabled = cfg.capture.claude_code_enabled
        # hook 脚本写入的队列文件
        self.queue_file = Path.home() / ".jarvis" / "claude_code_queue.jsonl"
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        self.poll_interval = 2.0

    async def run(self) -> None:
        while self._running:
            await self._drain_queue()
            await asyncio.sleep(self.poll_interval)

    async def _drain_queue(self) -> None:
        if not self.queue_file.exists():
            return
        lines = self.queue_file.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return
        # 清空队列文件
        self.queue_file.write_text("", encoding="utf-8")
        for line in lines:
            try:
                data = json.loads(line)
                event = CaptureEvent(
                    source=EventSource.CLAUDE_CODE,
                    type=EventType.OUTPUT,
                    content=data.get("transcript", ""),
                    metadata={
                        "session_id": data.get("session_id", ""),
                        "tool_calls": data.get("tool_calls", []),
                        "stop_reason": data.get("stop_reason", ""),
                    },
                    importance=0.7,
                )
                await self.emit(event)
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Failed to parse Claude Code event: {e}")

    def install_hooks(self) -> None:
        """自动配置 Claude Code hooks（一次性安装）。"""
        settings_path = Path.home() / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)

        existing: dict = {}
        if settings_path.exists():
            existing = json.loads(settings_path.read_text())

        hook_cmd = f"python3 -m agent_infra.capture.adapters.claude_code_hook"
        hook_entry = {"type": "command", "command": hook_cmd}

        hooks = existing.setdefault("hooks", {})
        stop_hooks = hooks.setdefault("Stop", [])

        # 检查是否已安装
        for entry in stop_hooks:
            for h in entry.get("hooks", []):
                if "claude_code_hook" in h.get("command", ""):
                    self.logger.info("Claude Code hooks already installed")
                    return

        stop_hooks.append({"matcher": "", "hooks": [hook_entry]})
        settings_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
        self.logger.info(f"Claude Code hooks installed → {settings_path}")


# ── Hook 脚本入口（被 Claude Code 直接调用）────────────────────────────────────
# python3 -m agent_infra.capture.adapters.claude_code_hook

def _hook_main() -> None:
    """
    Claude Code 调用此脚本时，从 stdin 读取会话数据，
    追加到队列文件，由 ClaudeCodeAdapter 异步读取。
    """
    queue_file = Path.home() / ".jarvis" / "claude_code_queue.jsonl"
    queue_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        data = {"raw": raw[:2000] if raw else ""}

    entry = {
        "session_id": os.environ.get("CLAUDE_SESSION_ID", ""),
        "transcript": data.get("transcript", raw[:3000] if raw else ""),
        "tool_calls": data.get("tool_use", []),
        "stop_reason": data.get("stop_reason", ""),
    }

    with open(queue_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    _hook_main()
