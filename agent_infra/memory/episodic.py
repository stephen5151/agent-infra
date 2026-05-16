"""
情节记忆 — SQLite 时序活动日志
================================
记录你每一次的输入/输出/行动，就像大脑海马体的日记。
支持：时间范围查询、来源过滤、全文搜索、重要性筛选。

DuckDB 可选扩展，用于跨时段聚合分析（例如识别工作习惯）。
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator, Optional

from agent_infra.capture.event_bus import CaptureEvent, EventSource, EventType
from agent_infra.config.settings import get_settings

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    type        TEXT NOT NULL,
    content     TEXT NOT NULL,
    summary     TEXT DEFAULT '',
    metadata    TEXT DEFAULT '{}',
    importance  REAL DEFAULT 0.5,
    processed   INTEGER DEFAULT 0,
    timestamp   TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodes_timestamp  ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_source     ON episodes(source);
CREATE INDEX IF NOT EXISTS idx_episodes_type       ON episodes(type);
CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes(importance);
CREATE INDEX IF NOT EXISTS idx_episodes_processed  ON episodes(processed);

-- 全文搜索（FTS5）
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    id UNINDEXED,
    content,
    summary,
    content='episodes',
    content_rowid='rowid'
);

-- 触发器保持 FTS 同步
CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, id, content, summary)
    VALUES (new.rowid, new.id, new.content, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, id, content, summary)
    VALUES ('delete', old.rowid, old.id, old.content, old.summary);
    INSERT INTO episodes_fts(rowid, id, content, summary)
    VALUES (new.rowid, new.id, new.content, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, id, content, summary)
    VALUES ('delete', old.rowid, old.id, old.content, old.summary);
END;
"""


# ── 情节记忆主类 ──────────────────────────────────────────────────────────────

class EpisodicMemory:
    """
    SQLite 时序活动日志。

    用法：
        mem = EpisodicMemory()
        await mem.record(event)
        episodes = mem.recent(hours=24)
        results  = mem.search("FastAPI JWT")
    """

    def __init__(self, db_path: Path | None = None) -> None:
        cfg = get_settings()
        self.db_path = db_path or cfg.memory.episodic_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"EpisodicMemory initialized → {self.db_path}")

    def _init_db(self) -> None:
        with self._conn() as conn:
            # PRAGMAs must run outside executescript (it auto-commits)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
        # executescript handles triggers that contain internal semicolons
        raw = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            raw.executescript(CREATE_TABLE_SQL)
        finally:
            raw.close()

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

    # ── 写入 ─────────────────────────────────────────────────────────────────

    def record(self, event: CaptureEvent) -> None:
        """同步记录一个 CaptureEvent 到情节日志。"""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO episodes
                    (id, source, type, content, summary, metadata, importance, processed, timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.source.value,
                    event.type.value,
                    event.content,
                    event.summary,
                    json.dumps(event.metadata, ensure_ascii=False),
                    event.importance,
                    int(event.processed),
                    event.timestamp.isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        logger.debug(f"Episode recorded: {event.id} [{event.source.value}]")

    def mark_processed(self, event_id: str, summary: str = "") -> None:
        """标记事件已被智能层处理，可选更新摘要。"""
        with self._conn() as conn:
            conn.execute(
                "UPDATE episodes SET processed=1, summary=? WHERE id=?",
                (summary, event_id),
            )

    def update_importance(self, event_id: str, importance: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE episodes SET importance=? WHERE id=?",
                (max(0.0, min(1.0, importance)), event_id),
            )

    # ── 查询 ─────────────────────────────────────────────────────────────────

    def recent(
        self,
        hours: float = 24,
        sources: list[str] | None = None,
        types: list[str] | None = None,
        min_importance: float = 0.0,
        limit: int = 200,
    ) -> list[dict]:
        """返回最近 N 小时的事件，按时间降序。"""
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        query = "SELECT * FROM episodes WHERE timestamp >= ?"
        params: list[Any] = [since]

        if sources:
            placeholders = ",".join("?" * len(sources))
            query += f" AND source IN ({placeholders})"
            params.extend(sources)
        if types:
            placeholders = ",".join("?" * len(types))
            query += f" AND type IN ({placeholders})"
            params.extend(types)
        if min_importance > 0:
            query += " AND importance >= ?"
            params.append(min_importance)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """全文搜索情节记忆。"""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT e.* FROM episodes e
                JOIN episodes_fts f ON e.rowid = f.rowid
                WHERE episodes_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def unprocessed(self, limit: int = 50) -> list[dict]:
        """返回尚未被智能层处理的高重要性事件。"""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM episodes
                WHERE processed=0 AND importance >= 0.5
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """返回记忆统计（用于 CLI 健康检查）。"""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            today_start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).isoformat()
            today = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE timestamp >= ?", (today_start,)
            ).fetchone()[0]
            by_source = conn.execute(
                "SELECT source, COUNT(*) as cnt FROM episodes GROUP BY source ORDER BY cnt DESC"
            ).fetchall()
        return {
            "total": total,
            "today": today,
            "by_source": {r["source"]: r["cnt"] for r in by_source},
        }

    # ── 清理 ─────────────────────────────────────────────────────────────────

    def prune(self, keep_days: int = 90) -> int:
        """删除超过 N 天的低重要性事件，返回删除数量。"""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM episodes WHERE timestamp < ? AND importance < 0.7",
                (cutoff,),
            )
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info(f"Pruned {deleted} old low-importance episodes")
        return deleted


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_episodic: EpisodicMemory | None = None


def get_episodic_memory() -> EpisodicMemory:
    global _episodic
    if _episodic is None:
        _episodic = EpisodicMemory()
    return _episodic
