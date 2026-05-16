"""
健康监控 — 系统自检
=====================
每 30 秒检查所有适配器和服务的健康状态。
发现问题时：
  1. 记录到日志
  2. 尝试自动重启（auto_restart=True）
  3. 桌面通知（严重问题）
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from agent_infra.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    name: str
    check_fn: Callable[[], Awaitable[bool]]
    restart_fn: Callable[[], Awaitable[None]] | None = None
    last_status: bool = True
    last_check_time: float = field(default_factory=time.time)
    consecutive_failures: int = 0


class HealthMonitor:
    """
    系统健康监控守护进程。

    用法：
        monitor = HealthMonitor()
        monitor.register("clipboard", check_fn=clipboard.is_alive)
        await monitor.run()  # 后台循环
    """

    def __init__(self) -> None:
        cfg = get_settings().healing
        self.interval_s = cfg.health_check_interval_s
        self.auto_restart = cfg.auto_restart
        self._checks: list[HealthCheck] = []
        self._running = False
        self._start_time = time.time()

    def register(
        self,
        name: str,
        check_fn: Callable[[], Awaitable[bool]],
        restart_fn: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """注册一个健康检查项。"""
        self._checks.append(HealthCheck(
            name=name,
            check_fn=check_fn,
            restart_fn=restart_fn,
        ))
        logger.debug(f"Health check registered: {name}")

    async def run(self) -> None:
        """主监控循环（后台运行）。"""
        self._running = True
        logger.info(f"HealthMonitor started (interval={self.interval_s}s)")
        while self._running:
            try:
                await self._run_all_checks()
            except Exception as e:
                logger.error(f"HealthMonitor error: {e}")
            await asyncio.sleep(self.interval_s)

    async def stop(self) -> None:
        self._running = False

    async def _run_all_checks(self) -> None:
        for check in self._checks:
            try:
                healthy = await asyncio.wait_for(check.check_fn(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning(f"Health check timeout: {check.name}")
                healthy = False
            except Exception as e:
                logger.warning(f"Health check error [{check.name}]: {e}")
                healthy = False

            check.last_check_time = time.time()

            if healthy:
                if not check.last_status:
                    logger.info(f"[RECOVERED] {check.name} is healthy again")
                check.consecutive_failures = 0
                check.last_status = True
            else:
                check.consecutive_failures += 1
                check.last_status = False
                logger.warning(
                    f"[UNHEALTHY] {check.name} "
                    f"(consecutive failures: {check.consecutive_failures})"
                )

                # 自动重启
                if self.auto_restart and check.restart_fn:
                    try:
                        logger.info(f"Auto-restarting: {check.name}")
                        await asyncio.wait_for(check.restart_fn(), timeout=30.0)
                    except Exception as e:
                        logger.error(f"Auto-restart failed [{check.name}]: {e}")

                # 严重告警（3次以上失败）
                if check.consecutive_failures >= 3:
                    self._alert(check.name, check.consecutive_failures)

    def _alert(self, service_name: str, failures: int) -> None:
        try:
            import subprocess
            script = (
                f'display notification "服务 {service_name} 已连续失败 {failures} 次" '
                f'with title "Jarvis 系统告警" sound name "Sosumi"'
            )
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
        except Exception:
            pass

    def status(self) -> dict:
        uptime = time.time() - self._start_time
        checks_status = []
        for c in self._checks:
            checks_status.append({
                "name": c.name,
                "healthy": c.last_status,
                "consecutive_failures": c.consecutive_failures,
                "last_check": datetime.fromtimestamp(
                    c.last_check_time, tz=timezone.utc
                ).isoformat(),
            })
        return {
            "uptime_seconds": round(uptime),
            "checks": checks_status,
            "all_healthy": all(c.last_status for c in self._checks),
        }


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_monitor: HealthMonitor | None = None


def get_health_monitor() -> HealthMonitor:
    global _monitor
    if _monitor is None:
        _monitor = HealthMonitor()
    return _monitor
