"""
任务注册表 — 持久化每日 Routine 的运行状态
=============================================
守护进程写入 ~/.jarvis/task_state.json，monitor 命令读取并展示。

状态文件格式:
{
  "daemon_pid": 12345,
  "daemon_started": "2026-05-16T10:00:00Z",
  "tasks": {
    "daily_digest": {
      "label": "日报生成",
      "schedule": "08:00",
      "last_run": "2026-05-16T08:00:00Z",
      "last_status": "success",   # success | failed | running | never
      "last_duration_s": 12.3,
      "next_run": "2026-05-17T08:00:00Z",
      "error": ""
    }
  }
}
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

STATE_PATH = Path("~/.jarvis/task_state.json").expanduser()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _next_run_after(schedule_hhmm: str, after: datetime | None = None) -> str:
    """计算下次执行时间（每日）。"""
    h, m = map(int, schedule_hhmm.split(":"))
    base = after or datetime.now(timezone.utc)
    candidate = base.replace(hour=h, minute=m, second=0, microsecond=0)
    if candidate <= base:
        candidate += timedelta(days=1)
    return candidate.isoformat()


class TaskRegistry:
    """读写任务状态文件的单例工具。"""

    def __init__(self, path: Path = STATE_PATH) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── 内部 IO ────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except Exception:
            return {}

    def _save(self, data: dict) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        tmp.replace(self._path)

    # ── 守护进程启动时调用 ─────────────────────────────────────────────────

    def register_daemon(self, pid: int | None = None) -> None:
        data = self._load()
        data["daemon_pid"] = pid or os.getpid()
        data["daemon_started"] = _now()
        data.setdefault("tasks", {})
        self._save(data)

    def register_task(self, task_id: str, label: str, schedule: str) -> None:
        """注册一个日常任务到状态文件（仅补全缺失字段）。"""
        data = self._load()
        tasks = data.setdefault("tasks", {})
        existing = tasks.get(task_id, {})
        tasks[task_id] = {
            "label": label,
            "schedule": schedule,
            "last_run": existing.get("last_run", ""),
            "last_status": existing.get("last_status", "never"),
            "last_duration_s": existing.get("last_duration_s", 0.0),
            "next_run": existing.get("next_run") or _next_run_after(schedule),
            "error": existing.get("error", ""),
        }
        self._save(data)

    # ── 任务执行前后调用 ──────────────────────────────────────────────────

    def mark_running(self, task_id: str) -> float:
        """标记任务开始运行，返回开始时间戳。"""
        data = self._load()
        tasks = data.setdefault("tasks", {})
        if task_id in tasks:
            tasks[task_id]["last_status"] = "running"
            tasks[task_id]["last_run"] = _now()
        self._save(data)
        return time.monotonic()

    def mark_done(
        self,
        task_id: str,
        start_mono: float,
        success: bool,
        error: str = "",
    ) -> None:
        """标记任务完成。"""
        elapsed = round(time.monotonic() - start_mono, 2)
        data = self._load()
        tasks = data.setdefault("tasks", {})
        if task_id in tasks:
            schedule = tasks[task_id].get("schedule", "00:00")
            tasks[task_id]["last_status"] = "success" if success else "failed"
            tasks[task_id]["last_duration_s"] = elapsed
            tasks[task_id]["error"] = error
            tasks[task_id]["next_run"] = _next_run_after(schedule)
        self._save(data)

    # ── Monitor 读取 ──────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """返回完整状态快照（供 dashboard 展示）。"""
        return self._load()

    def is_daemon_alive(self) -> bool:
        data = self._load()
        pid = data.get("daemon_pid")
        if not pid:
            return False
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def get_missed_tasks(self) -> list[str]:
        """
        返回关机期间漏掉的任务 ID 列表。

        判断逻辑：
          - 找到该任务「今天最近一次应执行时刻」（若还未到则取昨天）
          - 若 last_run 早于该时刻（或从未运行），则视为漏跑
        """
        from datetime import timedelta
        data = self._load()
        now = datetime.now(timezone.utc)
        missed: list[str] = []

        for task_id, info in data.get("tasks", {}).items():
            schedule = info.get("schedule", "")
            last_run_iso = info.get("last_run", "")
            last_status = info.get("last_status", "never")

            if not schedule:
                continue
            if last_status == "running":
                continue   # 正在跑，不补跑

            try:
                h, m = map(int, schedule.split(":"))
                # 最近一次应执行时刻（今天，若还没到则取昨天）
                scheduled_today = now.replace(hour=h, minute=m, second=0, microsecond=0)
                if scheduled_today > now:
                    scheduled_today -= timedelta(days=1)

                if not last_run_iso:
                    missed.append(task_id)
                    continue

                last_run_dt = datetime.fromisoformat(last_run_iso)
                if last_run_dt.tzinfo is None:
                    last_run_dt = last_run_dt.replace(tzinfo=timezone.utc)

                if last_run_dt < scheduled_today:
                    missed.append(task_id)
            except Exception:
                pass

        return missed


_registry: TaskRegistry | None = None


def get_registry() -> TaskRegistry:
    global _registry
    if _registry is None:
        _registry = TaskRegistry()
    return _registry
