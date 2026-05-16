"""
程序记忆 — 可复用技能 / 工作流库
====================================
记录你反复做的事情 = 可自动化的技能。
Agent 发现重复模式后，将其提炼为「技能」存入库中，
下次遇到同类任务时，直接调用已知技能而非重新推理。

存储：SQLite（与情节记忆同库）
技能格式：
  - name: 唯一标识
  - description: 人类可读描述
  - trigger_patterns: 触发关键词列表
  - steps: 结构化步骤（JSON）
  - source_events: 来源事件 ID
  - use_count: 被调用次数
  - success_rate: 成功率（0-1）
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator
from uuid import uuid4

from agent_infra.config.settings import get_settings

logger = logging.getLogger(__name__)


CREATE_SKILLS_SQL = """
CREATE TABLE IF NOT EXISTS skills (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    description     TEXT NOT NULL,
    trigger_patterns TEXT DEFAULT '[]',   -- JSON 数组
    steps           TEXT DEFAULT '[]',    -- JSON 数组 [{action, params, description}]
    source_events   TEXT DEFAULT '[]',    -- 来源情节事件 ID
    category        TEXT DEFAULT 'general',
    use_count       INTEGER DEFAULT 0,
    success_count   INTEGER DEFAULT 0,
    enabled         INTEGER DEFAULT 1,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_skills_category ON skills(category);
CREATE INDEX IF NOT EXISTS idx_skills_enabled  ON skills(enabled);
CREATE INDEX IF NOT EXISTS idx_skills_name     ON skills(name);
"""


@dataclass
class Skill:
    """技能定义。"""
    name: str
    description: str
    steps: list[dict[str, Any]]
    trigger_patterns: list[str] = field(default_factory=list)
    source_events: list[str] = field(default_factory=list)
    category: str = "general"
    id: str = field(default_factory=lambda: str(uuid4()))

    @property
    def trigger_text(self) -> str:
        return " | ".join(self.trigger_patterns)

    def matches(self, query: str) -> bool:
        """简单关键词匹配，用于技能检索。"""
        q = query.lower()
        if any(p.lower() in q for p in self.trigger_patterns):
            return True
        if self.name.lower() in q or self.description.lower() in q:
            return True
        return False

    def to_prompt(self) -> str:
        """格式化为可注入 prompt 的步骤说明。"""
        lines = [f"**技能：{self.name}**", f"说明：{self.description}", "步骤："]
        for i, step in enumerate(self.steps, 1):
            action = step.get("action", "")
            desc = step.get("description", action)
            lines.append(f"  {i}. {desc}")
        return "\n".join(lines)


class ProceduralMemory:
    """
    技能库 CRUD + 匹配检索。

    用法：
        pm = ProceduralMemory()
        pm.save_skill(skill)
        matches = pm.find_matching(user_input)
        pm.record_usage(skill_id, success=True)
    """

    def __init__(self, db_path: Path | None = None) -> None:
        cfg = get_settings()
        self.db_path = db_path or cfg.memory.episodic_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"ProceduralMemory initialized → {self.db_path}")

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            for stmt in CREATE_SKILLS_SQL.split(";"):
                stmt = stmt.strip()
                if stmt:
                    try:
                        conn.execute(stmt)
                    except sqlite3.OperationalError as e:
                        if "already exists" not in str(e):
                            raise

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def save_skill(self, skill: Skill) -> None:
        """保存或更新技能。"""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO skills
                    (id, name, description, trigger_patterns, steps, source_events,
                     category, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description=excluded.description,
                    trigger_patterns=excluded.trigger_patterns,
                    steps=excluded.steps,
                    source_events=excluded.source_events,
                    category=excluded.category,
                    updated_at=excluded.updated_at
                """,
                (
                    skill.id, skill.name, skill.description,
                    json.dumps(skill.trigger_patterns, ensure_ascii=False),
                    json.dumps(skill.steps, ensure_ascii=False),
                    json.dumps(skill.source_events, ensure_ascii=False),
                    skill.category, now, now,
                ),
            )
        logger.info(f"Skill saved: {skill.name}")

    def get_skill(self, name: str) -> Skill | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM skills WHERE name=? AND enabled=1", (name,)
            ).fetchone()
        return self._row_to_skill(row) if row else None

    def list_skills(self, category: str | None = None) -> list[Skill]:
        query = "SELECT * FROM skills WHERE enabled=1"
        params: list = []
        if category:
            query += " AND category=?"
            params.append(category)
        query += " ORDER BY use_count DESC"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_skill(r) for r in rows]

    def delete_skill(self, name: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE skills SET enabled=0 WHERE name=?", (name,))

    # ── 检索 ─────────────────────────────────────────────────────────────────

    def find_matching(self, query: str, top_k: int = 3) -> list[Skill]:
        """根据查询找到最匹配的技能。"""
        skills = self.list_skills()
        matched = [s for s in skills if s.matches(query)]
        # 按使用频次降序
        matched.sort(key=lambda s: -self._get_use_count(s.name))
        return matched[:top_k]

    def _get_use_count(self, name: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT use_count FROM skills WHERE name=?", (name,)
            ).fetchone()
        return row["use_count"] if row else 0

    def format_for_prompt(self, query: str) -> str:
        """返回匹配技能的 prompt 友好格式。"""
        skills = self.find_matching(query)
        if not skills:
            return ""
        parts = ["[你已掌握的相关技能]"]
        for skill in skills:
            parts.append(skill.to_prompt())
        return "\n\n".join(parts)

    # ── 使用统计 ─────────────────────────────────────────────────────────────

    def record_usage(self, skill_name: str, success: bool = True) -> None:
        with self._conn() as conn:
            if success:
                conn.execute(
                    "UPDATE skills SET use_count=use_count+1, success_count=success_count+1 WHERE name=?",
                    (skill_name,),
                )
            else:
                conn.execute(
                    "UPDATE skills SET use_count=use_count+1 WHERE name=?",
                    (skill_name,),
                )

    # ── 工具函数 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_skill(row: sqlite3.Row) -> Skill:
        return Skill(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            trigger_patterns=json.loads(row["trigger_patterns"] or "[]"),
            steps=json.loads(row["steps"] or "[]"),
            source_events=json.loads(row["source_events"] or "[]"),
            category=row["category"],
        )

    def stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM skills WHERE enabled=1"
            ).fetchone()[0]
            top = conn.execute(
                "SELECT name, use_count FROM skills WHERE enabled=1 ORDER BY use_count DESC LIMIT 5"
            ).fetchall()
        return {
            "total_skills": total,
            "top_skills": [{"name": r["name"], "use_count": r["use_count"]} for r in top],
        }

    # ── 预置技能 ──────────────────────────────────────────────────────────────

    def seed_default_skills(self) -> None:
        """预置一批常用技能模板，帮助 agent 快速启动。"""
        defaults = [
            Skill(
                name="write_fastapi_endpoint",
                description="用 FastAPI 写一个 REST API 接口",
                trigger_patterns=["fastapi", "接口", "endpoint", "api路由", "rest"],
                category="coding",
                steps=[
                    {"action": "define_router", "description": "定义路由和 HTTP 方法"},
                    {"action": "define_schema", "description": "用 Pydantic 定义请求/响应模型"},
                    {"action": "implement_handler", "description": "实现处理函数，包含错误处理"},
                    {"action": "add_tests", "description": "用 pytest + httpx 添加测试"},
                ],
            ),
            Skill(
                name="debug_python_error",
                description="Python 报错调试流程",
                trigger_patterns=["traceback", "error", "exception", "报错", "调试", "debug"],
                category="debugging",
                steps=[
                    {"action": "read_traceback", "description": "读完整 traceback，定位最内层错误"},
                    {"action": "check_types", "description": "检查变量类型和值"},
                    {"action": "add_logging", "description": "添加 logging 输出中间状态"},
                    {"action": "write_minimal_repro", "description": "写最小复现用例"},
                ],
            ),
            Skill(
                name="daily_review",
                description="每日工作回顾和规划",
                trigger_patterns=["日报", "每日总结", "today review", "daily", "回顾"],
                category="productivity",
                steps=[
                    {"action": "list_completed", "description": "列出今天完成的事项"},
                    {"action": "identify_blockers", "description": "识别遇到的障碍"},
                    {"action": "plan_tomorrow", "description": "规划明天的优先事项"},
                    {"action": "write_obsidian", "description": "写入 Obsidian 日记"},
                ],
            ),
        ]
        for skill in defaults:
            existing = self.get_skill(skill.name)
            if not existing:
                self.save_skill(skill)
        logger.info(f"Seeded {len(defaults)} default skills")


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_procedural: ProceduralMemory | None = None


def get_procedural_memory() -> ProceduralMemory:
    global _procedural
    if _procedural is None:
        _procedural = ProceduralMemory()
    return _procedural
