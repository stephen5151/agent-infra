"""
工作记忆 — 当前对话 / 任务上下文窗口
=======================================
类似人类前额叶皮质：短暂持有正在处理的信息。
生命周期：单次对话或单次工作会话。

设计：
  - 固定窗口大小（token 估算），超出自动摘要
  - 支持「关注点」(focus) 标记：agent 主动关注的实体/任务
  - 与情节记忆对接：会话结束时 flush 到 SQLite
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# 粗略 token 估算（1 token ≈ 4 chars）
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class WorkingMemoryEntry:
    role: str        # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.tokens == 0:
            self.tokens = _estimate_tokens(self.content)


class WorkingMemory:
    """
    环形缓冲区工作记忆。

    用法：
        wm = WorkingMemory(max_tokens=8000)
        wm.add("user", "帮我写一个 FastAPI 登录接口")
        wm.add("assistant", "好的，以下是代码...")
        messages = wm.to_messages()   # 转为 LangChain/OpenAI 格式
        ctx = wm.context_string()     # 转为纯文本摘要
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        session_id: str | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.session_id = session_id or self._gen_session_id()
        self._entries: deque[WorkingMemoryEntry] = deque()
        self._total_tokens: int = 0
        self._focus: list[str] = []      # 当前关注的实体/任务
        self._created_at = datetime.now(timezone.utc)
        logger.debug(f"WorkingMemory created: session={self.session_id}")

    @staticmethod
    def _gen_session_id() -> str:
        from uuid import uuid4
        return str(uuid4())[:8]

    # ── 写入 ─────────────────────────────────────────────────────────────────

    def add(
        self,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """添加一条消息到工作记忆。超出 max_tokens 时自动淘汰最旧条目。"""
        entry = WorkingMemoryEntry(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._total_tokens += entry.tokens

        # 超出限制时，淘汰最旧的非 system 消息
        while self._total_tokens > self.max_tokens and len(self._entries) > 1:
            oldest = self._entries[0]
            if oldest.role == "system":
                # 保留 system 消息，淘汰第二旧的
                if len(self._entries) > 2:
                    second = self._entries[1]
                    self._entries.remove(second)
                    self._total_tokens -= second.tokens
                else:
                    break
            else:
                self._entries.popleft()
                self._total_tokens -= oldest.tokens

    def set_focus(self, topics: list[str]) -> None:
        """设置 agent 当前关注的主题/实体（影响检索策略）。"""
        self._focus = topics
        logger.debug(f"Working memory focus: {topics}")

    def clear(self) -> None:
        self._entries.clear()
        self._total_tokens = 0
        self._focus = []

    # ── 读取 ─────────────────────────────────────────────────────────────────

    def to_messages(self) -> list[dict[str, str]]:
        """转为 OpenAI / LangChain 消息格式。"""
        return [{"role": e.role, "content": e.content} for e in self._entries]

    def context_string(self, max_chars: int = 4000) -> str:
        """返回纯文本上下文摘要，用于注入 prompt。"""
        parts = []
        total = 0
        for entry in reversed(self._entries):
            line = f"[{entry.role}] {entry.content}"
            if total + len(line) > max_chars:
                break
            parts.append(line)
            total += len(line)
        parts.reverse()
        return "\n".join(parts)

    def last_assistant_message(self) -> str | None:
        """返回最近一条 assistant 消息内容。"""
        for entry in reversed(self._entries):
            if entry.role == "assistant":
                return entry.content
        return None

    def last_user_message(self) -> str | None:
        """返回最近一条 user 消息内容。"""
        for entry in reversed(self._entries):
            if entry.role == "user":
                return entry.content
        return None

    @property
    def focus(self) -> list[str]:
        return self._focus.copy()

    @property
    def token_count(self) -> int:
        return self._total_tokens

    @property
    def message_count(self) -> int:
        return len(self._entries)

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "messages": self.message_count,
            "tokens": self.token_count,
            "focus": self._focus,
            "age_seconds": (
                datetime.now(timezone.utc) - self._created_at
            ).total_seconds(),
        }

    # ── 持久化到情节记忆 ──────────────────────────────────────────────────────

    def flush_to_episodic(self) -> None:
        """
        将工作记忆内容 flush 到情节记忆（SQLite）。
        通常在会话结束时调用。
        """
        try:
            from agent_infra.capture.event_bus import CaptureEvent, EventSource, EventType
            from agent_infra.memory.episodic import get_episodic_memory

            if not self._entries:
                return

            content = self.context_string(max_chars=10000)
            event = CaptureEvent(
                source=EventSource.AGENT,
                type=EventType.OUTPUT,
                content=content,
                summary=f"Working memory session {self.session_id}",
                metadata={
                    "session_id": self.session_id,
                    "message_count": self.message_count,
                    "focus": self._focus,
                },
                importance=0.6,
            )
            get_episodic_memory().record(event)
            logger.info(f"Working memory flushed to episodic: session={self.session_id}")
        except Exception as e:
            logger.error(f"Failed to flush working memory: {e}")


# ── 全局单例（当前会话）──────────────────────────────────────────────────────

_working: WorkingMemory | None = None


def get_working_memory(reset: bool = False) -> WorkingMemory:
    """获取当前会话的工作记忆单例。reset=True 开始新会话。"""
    global _working
    if _working is None or reset:
        if _working is not None:
            _working.flush_to_episodic()
        _working = WorkingMemory()
    return _working
