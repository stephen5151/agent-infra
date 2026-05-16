"""
知识库存储 — SQLite
====================
三张表:
  knowledge_items   - 抓取的文章/论文条目
  item_feedback     - 用户评分 (like/dislike/bookmark/skip)
  topic_weights     - 话题偏好权重 (驱动推送比例)

权重调整规则:
  like     → +0.05  (最高 1.0)
  bookmark → +0.10  (最高 1.0)
  dislike  → -0.08  (最低 0.05)
  skip     → 不变
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

DB_PATH = Path("~/.jarvis/knowledge.db").expanduser()

_DDL = """
CREATE TABLE IF NOT EXISTS knowledge_items (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    url          TEXT NOT NULL,
    source       TEXT NOT NULL,          -- hackernews | arxiv | github
    summary      TEXT DEFAULT '',        -- LLM 生成摘要
    topics       TEXT DEFAULT '[]',      -- JSON 数组 ["AI/ML", "Python"]
    quality      REAL DEFAULT 0.5,       -- 0-1，来源评分归一化
    raw_score    INTEGER DEFAULT 0,      -- 原始热度（HN points / arxiv citations）
    fetched_at   TEXT NOT NULL,
    shown_at     TEXT DEFAULT '',        -- 空 = 未推送
    language     TEXT DEFAULT 'en'
);

CREATE TABLE IF NOT EXISTS item_feedback (
    id         TEXT PRIMARY KEY,
    item_id    TEXT NOT NULL,
    rating     TEXT NOT NULL,            -- like | dislike | bookmark | skip
    rated_at   TEXT NOT NULL,
    FOREIGN KEY (item_id) REFERENCES knowledge_items(id)
);

CREATE TABLE IF NOT EXISTS topic_weights (
    topic        TEXT PRIMARY KEY,
    weight       REAL NOT NULL DEFAULT 0.5,
    like_count   INTEGER DEFAULT 0,
    dislike_count INTEGER DEFAULT 0,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_source    ON knowledge_items(source);
CREATE INDEX IF NOT EXISTS idx_items_shown     ON knowledge_items(shown_at);
CREATE INDEX IF NOT EXISTS idx_items_fetched   ON knowledge_items(fetched_at);
CREATE INDEX IF NOT EXISTS idx_feedback_item   ON item_feedback(item_id);
"""

# 话题分类表（用于 LLM prompt 和权重 key 统一）
TOPIC_TAXONOMY = [
    "AI/ML", "编程", "开源", "系统架构", "安全",
    "产品设计", "商业", "创业", "科学", "哲学",
    "效率工具", "健康", "金融", "数学", "其他",
]


class KnowledgeStore:

    def __init__(self, path: Path = DB_PATH) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── 基础 IO ────────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            for stmt in _DDL.split(";"):
                s = stmt.strip()
                if s:
                    try:
                        conn.execute(s)
                    except sqlite3.OperationalError as e:
                        if "already exists" not in str(e):
                            raise

    # ── 写入条目 ──────────────────────────────────────────────────────────

    def upsert_item(
        self,
        title: str,
        url: str,
        source: str,
        topics: list[str],
        quality: float = 0.5,
        raw_score: int = 0,
        summary: str = "",
        language: str = "en",
    ) -> str:
        """插入新条目（URL 重复则跳过），返回 item id。"""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            # 检查是否已存在
            row = conn.execute(
                "SELECT id FROM knowledge_items WHERE url=?", (url,)
            ).fetchone()
            if row:
                return row["id"]
            item_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO knowledge_items
                   (id, title, url, source, summary, topics, quality, raw_score, fetched_at, language)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (item_id, title, url, source,
                 summary, json.dumps(topics, ensure_ascii=False),
                 round(min(1.0, max(0.0, quality)), 4),
                 raw_score, now, language),
            )
        return item_id

    def update_summary(self, item_id: str, summary: str, topics: list[str]) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE knowledge_items SET summary=?, topics=? WHERE id=?",
                (summary, json.dumps(topics, ensure_ascii=False), item_id),
            )

    def mark_shown(self, item_ids: list[str]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.executemany(
                "UPDATE knowledge_items SET shown_at=? WHERE id=?",
                [(now, iid) for iid in item_ids],
            )

    # ── 查询条目 ──────────────────────────────────────────────────────────

    def get_feed_candidates(
        self,
        limit: int = 50,
        unseen_only: bool = True,
    ) -> list[dict]:
        """返回推送候选（有摘要、按质量降序）。"""
        cond = "summary != '' AND summary IS NOT NULL"
        if unseen_only:
            cond += " AND (shown_at IS NULL OR shown_at = '')"
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM knowledge_items WHERE {cond} ORDER BY quality DESC LIMIT ?",
                (limit,),
            ).fetchall()
        items = [dict(r) for r in rows]
        for it in items:
            it["topics"] = json.loads(it.get("topics") or "[]")
        return items

    def get_item(self, item_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge_items WHERE id=?", (item_id,)
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["topics"] = json.loads(item.get("topics") or "[]")
        return item

    def pending_summary_count(self) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM knowledge_items WHERE summary='' OR summary IS NULL"
            ).fetchone()[0]

    # ── 反馈 ─────────────────────────────────────────────────────────────

    def save_feedback(self, item_id: str, rating: str) -> None:
        """保存用户评分并更新话题权重。"""
        now = datetime.now(timezone.utc).isoformat()
        fid = str(uuid.uuid4())
        with self._conn() as conn:
            # 检查是否已评过
            existing = conn.execute(
                "SELECT id FROM item_feedback WHERE item_id=?", (item_id,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE item_feedback SET rating=?, rated_at=? WHERE item_id=?",
                    (rating, now, item_id),
                )
            else:
                conn.execute(
                    "INSERT INTO item_feedback (id, item_id, rating, rated_at) VALUES (?,?,?,?)",
                    (fid, item_id, rating, now),
                )

        # 更新话题权重
        item = self.get_item(item_id)
        if item:
            self._adjust_weights(item["topics"], rating)

    def _adjust_weights(self, topics: list[str], rating: str) -> None:
        delta_map = {"like": 0.05, "bookmark": 0.10, "dislike": -0.08, "skip": 0.0}
        delta = delta_map.get(rating, 0.0)
        if delta == 0.0:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            for topic in topics:
                row = conn.execute(
                    "SELECT weight, like_count, dislike_count FROM topic_weights WHERE topic=?",
                    (topic,),
                ).fetchone()
                if row:
                    new_w = round(min(1.0, max(0.05, row["weight"] + delta)), 4)
                    lc = row["like_count"] + (1 if rating in ("like", "bookmark") else 0)
                    dc = row["dislike_count"] + (1 if rating == "dislike" else 0)
                    conn.execute(
                        "UPDATE topic_weights SET weight=?, like_count=?, dislike_count=?, updated_at=? WHERE topic=?",
                        (new_w, lc, dc, now, topic),
                    )
                else:
                    init_w = round(min(1.0, max(0.05, 0.5 + delta)), 4)
                    conn.execute(
                        "INSERT INTO topic_weights (topic, weight, like_count, dislike_count, updated_at) VALUES (?,?,?,?,?)",
                        (topic, init_w,
                         1 if rating in ("like", "bookmark") else 0,
                         1 if rating == "dislike" else 0,
                         now),
                    )

    # ── 权重查询 ──────────────────────────────────────────────────────────

    def get_topic_weights(self) -> dict[str, float]:
        with self._conn() as conn:
            rows = conn.execute("SELECT topic, weight FROM topic_weights").fetchall()
        return {r["topic"]: r["weight"] for r in rows}

    def get_topic_stats(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM topic_weights ORDER BY weight DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── 统计 ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM knowledge_items").fetchone()[0]
            shown = conn.execute(
                "SELECT COUNT(*) FROM knowledge_items WHERE shown_at != '' AND shown_at IS NOT NULL"
            ).fetchone()[0]
            pending_summary = conn.execute(
                "SELECT COUNT(*) FROM knowledge_items WHERE summary='' OR summary IS NULL"
            ).fetchone()[0]
            feedbacks = conn.execute(
                "SELECT rating, COUNT(*) as cnt FROM item_feedback GROUP BY rating"
            ).fetchall()
        return {
            "total_items": total,
            "shown": shown,
            "pending_summary": pending_summary,
            "feedback": {r["rating"]: r["cnt"] for r in feedbacks},
        }


_store: KnowledgeStore | None = None


def get_knowledge_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
    return _store
