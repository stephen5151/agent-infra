"""
断路器 — 自愈核心机制
=======================
防止级联失败：服务连续失败 N 次后，断路器「跳闸」，
后续请求直接快速失败，给服务恢复时间。

状态机：CLOSED → OPEN → HALF_OPEN → CLOSED
  CLOSED: 正常运行
  OPEN:   已跳闸，快速失败
  HALF_OPEN: 超时后允许一次试探，成功则复位
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar

from agent_infra.config.settings import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED    = "closed"      # 正常
    OPEN      = "open"        # 跳闸
    HALF_OPEN = "half_open"   # 试探


class CircuitBreakerError(Exception):
    """断路器跳闸时抛出。"""
    pass


@dataclass
class CircuitStats:
    failures: int = 0
    successes: int = 0
    total_calls: int = 0
    last_failure_time: float = 0.0
    last_state_change: float = field(default_factory=time.time)


class CircuitBreaker:
    """
    断路器。

    用法（装饰器）：
        cb = CircuitBreaker(name="claude_api")

        @cb.protect
        async def call_claude(...):
            ...

    用法（上下文）：
        async with cb:
            await risky_operation()
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int | None = None,
        timeout_s: int | None = None,
    ) -> None:
        cfg = get_settings().healing
        self.name = name
        self.failure_threshold = failure_threshold or cfg.circuit_breaker_threshold
        self.timeout_s = timeout_s or cfg.circuit_breaker_timeout_s
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._lock = asyncio.Lock()

    # ── 状态属性 ─────────────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        # OPEN → HALF_OPEN 的自动转换（超时后）
        if (
            self._state == CircuitState.OPEN
            and time.time() - self._stats.last_failure_time >= self.timeout_s
        ):
            self._transition(CircuitState.HALF_OPEN)
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    # ── 核心调用入口 ──────────────────────────────────────────────────────────

    async def call(self, coro):
        """执行协程，自动记录成功/失败并更新断路器状态。"""
        async with self._lock:
            state = self.state
            if state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker [{self.name}] is OPEN "
                    f"(retry after {self._remaining_timeout():.0f}s)"
                )
            self._stats.total_calls += 1

        try:
            result = await coro
            await self._on_success()
            return result
        except CircuitBreakerError:
            raise
        except Exception as e:
            await self._on_failure(e)
            raise

    def _remaining_timeout(self) -> float:
        elapsed = time.time() - self._stats.last_failure_time
        return max(0, self.timeout_s - elapsed)

    # ── 状态更新 ─────────────────────────────────────────────────────────────

    async def _on_success(self) -> None:
        async with self._lock:
            self._stats.successes += 1
            if self._state == CircuitState.HALF_OPEN:
                logger.info(f"Circuit breaker [{self.name}] HALF_OPEN → CLOSED (recovered)")
                self._transition(CircuitState.CLOSED)
                self._stats.failures = 0

    async def _on_failure(self, exc: Exception) -> None:
        async with self._lock:
            self._stats.failures += 1
            self._stats.last_failure_time = time.time()
            logger.warning(
                f"Circuit breaker [{self.name}] failure "
                f"{self._stats.failures}/{self.failure_threshold}: {exc}"
            )
            if self._state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit breaker [{self.name}] HALF_OPEN → OPEN (still failing)")
                self._transition(CircuitState.OPEN)
            elif (
                self._state == CircuitState.CLOSED
                and self._stats.failures >= self.failure_threshold
            ):
                logger.error(
                    f"Circuit breaker [{self.name}] CLOSED → OPEN "
                    f"({self.failure_threshold} failures, timeout={self.timeout_s}s)"
                )
                self._transition(CircuitState.OPEN)

    def _transition(self, new_state: CircuitState) -> None:
        self._state = new_state
        self._stats.last_state_change = time.time()

    # ── 装饰器 ────────────────────────────────────────────────────────────────

    def protect(self, func: Callable) -> Callable:
        """装饰异步函数，自动加上断路器保护。"""
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any):
            return await self.call(func(*args, **kwargs))
        return wrapper

    # ── 上下文管理器 ──────────────────────────────────────────────────────────

    async def __aenter__(self):
        state = self.state
        if state == CircuitState.OPEN:
            raise CircuitBreakerError(
                f"Circuit breaker [{self.name}] is OPEN"
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type and exc_type is not CircuitBreakerError:
            await self._on_failure(exc_val)
            return False
        if not exc_type:
            await self._on_success()
        return False

    # ── 状态查询 ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """手动复位断路器。"""
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        logger.info(f"Circuit breaker [{self.name}] manually reset")

    def status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failures": self._stats.failures,
            "successes": self._stats.successes,
            "total_calls": self._stats.total_calls,
            "threshold": self.failure_threshold,
            "timeout_s": self.timeout_s,
        }


# ── 全局断路器注册表 ──────────────────────────────────────────────────────────

_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str, **kwargs) -> CircuitBreaker:
    """获取或创建命名断路器。"""
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name, **kwargs)
    return _breakers[name]


def all_breaker_status() -> list[dict]:
    return [b.status() for b in _breakers.values()]
