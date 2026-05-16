"""
CLI 命令捕获适配器
===================
监听终端命令历史（~/.zsh_history 或 ~/.bash_history）。
你敲的每条命令 = 你正在做什么的信号。

工作原理：
  - 监听 history 文件的 mtime 变化
  - 读取新增行，过滤噪音命令
  - 发布到事件总线
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from agent_infra.capture.adapters.base import BaseAdapter
from agent_infra.capture.event_bus import CaptureEvent, EventSource, EventType
from agent_infra.config.settings import get_settings


class CLIAdapter(BaseAdapter):
    """监听 shell history 文件，捕获终端命令。"""

    name = "cli"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        cfg = get_settings()
        self.enabled = cfg.capture.cli_enabled
        self.ignore_commands = set(cfg.capture.cli_ignore_commands)
        self.poll_interval = 2.0

        # 自动检测 history 文件
        self.history_file = self._detect_history_file()
        self._last_mtime: float = 0.0
        self._last_size: int = 0

    def _detect_history_file(self) -> Path | None:
        candidates = [
            Path.home() / ".zsh_history",
            Path.home() / ".bash_history",
            Path.home() / ".fish" / "fish_history",
        ]
        for p in candidates:
            if p.exists():
                self.logger.info(f"CLI adapter using history: {p}")
                return p
        self.logger.warning("No shell history file found")
        return None

    async def run(self) -> None:
        if not self.history_file:
            return

        # 初始化 offset
        try:
            stat = self.history_file.stat()
            self._last_mtime = stat.st_mtime
            self._last_size = stat.st_size
        except OSError:
            pass

        while self._running:
            await self._check_history()
            await asyncio.sleep(self.poll_interval)

    async def _check_history(self) -> None:
        try:
            stat = self.history_file.stat()
        except OSError:
            return

        if stat.st_mtime <= self._last_mtime and stat.st_size == self._last_size:
            return

        # 读取新增内容
        try:
            with open(self.history_file, "rb") as f:
                f.seek(max(0, self._last_size - 100))  # 稍微回退防止截断
                new_bytes = f.read()

            self._last_mtime = stat.st_mtime
            self._last_size = stat.st_size
        except OSError:
            return

        # 解析命令
        try:
            new_text = new_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return

        commands = self._parse_commands(new_text)
        for cmd in commands:
            if self._should_capture(cmd):
                event = CaptureEvent(
                    source=EventSource.CLI,
                    type=EventType.ACTION,
                    content=cmd,
                    importance=self._estimate_importance(cmd),
                    metadata={"shell": self._detect_shell()},
                )
                await self.emit(event)

    def _parse_commands(self, text: str) -> list[str]:
        """解析 zsh/bash history 格式，提取命令。"""
        commands = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # zsh 格式: ": 1234567890:0;actual command"
            if line.startswith(":") and ";" in line:
                cmd = line.split(";", 1)[1].strip()
            # fish 格式: "- cmd: actual command"
            elif line.startswith("- cmd:"):
                cmd = line[6:].strip()
            else:
                cmd = line
            if cmd:
                commands.append(cmd)
        return commands

    def _should_capture(self, cmd: str) -> bool:
        if not cmd:
            return False
        base_cmd = cmd.split()[0] if cmd.split() else ""
        if base_cmd in self.ignore_commands:
            return False
        cfg = get_settings()
        if cfg.privacy.is_sensitive(cmd):
            self.logger.debug(f"CLI: sensitive command skipped")
            return False
        return True

    def _estimate_importance(self, cmd: str) -> float:
        """命令重要性估算。"""
        high_importance_prefixes = [
            "git", "docker", "kubectl", "python", "pip",
            "npm", "yarn", "cargo", "make", "ssh", "curl",
        ]
        base = cmd.split()[0] if cmd.split() else ""
        if base in high_importance_prefixes:
            return 0.7
        if any(kw in cmd for kw in ["--prod", "production", "deploy", "release"]):
            return 0.9
        return 0.5

    def _detect_shell(self) -> str:
        shell = os.environ.get("SHELL", "")
        if "zsh" in shell:
            return "zsh"
        if "fish" in shell:
            return "fish"
        return "bash"
