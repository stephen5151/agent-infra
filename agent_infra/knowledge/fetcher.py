"""
知识抓取器 — HackerNews + ArXiv
=================================
零额外依赖：只用 httpx（已在 pyproject.toml）+ 标准库 xml。

HackerNews:
  - Top Stories API（免费，无需 key）
  - 过滤：score > 100，非 job/ask 类

ArXiv:
  - Atom XML API（免费，无需 key）
  - 默认分类：cs.AI, cs.LG, cs.SE, cs.CR（可配置）
"""
from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

HN_BASE = "https://hacker-news.firebaseio.com/v0"
ARXIV_BASE = "https://export.arxiv.org/api/query"
GITHUB_TRENDING = "https://github.com/trending"

TIMEOUT = httpx.Timeout(15.0, connect=5.0)


@dataclass
class RawItem:
    title: str
    url: str
    source: str                  # hackernews | arxiv | github
    raw_score: int = 0
    quality: float = 0.5
    language: str = "en"
    extra: dict = None           # type: ignore

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


# ── HackerNews ────────────────────────────────────────────────────────────────

async def fetch_hackernews(
    client: httpx.AsyncClient,
    limit: int = 30,
    min_score: int = 100,
) -> list[RawItem]:
    """抓取 HN Top Stories，过滤低分 & 招聘贴。"""
    try:
        resp = await client.get(f"{HN_BASE}/topstories.json", timeout=TIMEOUT)
        resp.raise_for_status()
        story_ids: list[int] = resp.json()[:100]
    except Exception as e:
        logger.warning(f"HN topstories fetch failed: {e}")
        return []

    async def _fetch_story(sid: int) -> RawItem | None:
        try:
            r = await client.get(f"{HN_BASE}/item/{sid}.json", timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if not data:
                return None
            # 过滤：无 URL、低分、招聘/问答类
            if not data.get("url"):
                return None
            if data.get("score", 0) < min_score:
                return None
            if data.get("type") not in ("story",):
                return None
            title = data.get("title", "").strip()
            if not title:
                return None
            score = data.get("score", 0)
            quality = min(1.0, score / 800)    # 800 分封顶
            return RawItem(
                title=title,
                url=data["url"],
                source="hackernews",
                raw_score=score,
                quality=round(quality, 3),
                extra={"hn_id": sid, "comments": data.get("descendants", 0)},
            )
        except Exception as e:
            logger.debug(f"HN item {sid} fetch error: {e}")
            return None

    tasks = [_fetch_story(sid) for sid in story_ids[:60]]
    results = await asyncio.gather(*tasks)
    items = [r for r in results if r is not None]
    items.sort(key=lambda x: x.raw_score, reverse=True)
    return items[:limit]


# ── ArXiv ─────────────────────────────────────────────────────────────────────

async def fetch_arxiv(
    client: httpx.AsyncClient,
    categories: list[str] | None = None,
    max_results: int = 10,
) -> list[RawItem]:
    """抓取 ArXiv 最新论文。"""
    cats = categories or ["cs.AI", "cs.LG", "cs.SE"]
    query = " OR ".join(f"cat:{c}" for c in cats)
    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }
    try:
        resp = await client.get(ARXIV_BASE, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"ArXiv fetch failed: {e}")
        return []

    items: list[RawItem] = []
    try:
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            id_el = entry.find("atom:id", ns)
            abstract_el = entry.find("atom:summary", ns)
            if title_el is None or id_el is None:
                continue
            title = (title_el.text or "").strip().replace("\n", " ")
            arxiv_id = (id_el.text or "").strip()
            abstract = (abstract_el.text or "").strip().replace("\n", " ") if abstract_el is not None else ""
            if not title or not arxiv_id:
                continue
            items.append(RawItem(
                title=title,
                url=arxiv_id,
                source="arxiv",
                raw_score=0,
                quality=0.75,
                extra={"abstract": abstract[:500]},
            ))
    except ET.ParseError as e:
        logger.warning(f"ArXiv XML parse error: {e}")

    return items


# ── 主入口 ────────────────────────────────────────────────────────────────────

async def fetch_all(
    hn_limit: int = 30,
    hn_min_score: int = 100,
    arxiv_categories: list[str] | None = None,
    arxiv_limit: int = 10,
) -> list[RawItem]:
    """并发抓取所有来源，返回合并列表。"""
    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": "Jarvis-KnowledgeBot/1.0 (personal assistant)"},
    ) as client:
        hn_task = fetch_hackernews(client, limit=hn_limit, min_score=hn_min_score)
        arxiv_task = fetch_arxiv(client, categories=arxiv_categories, max_results=arxiv_limit)
        hn_items, arxiv_items = await asyncio.gather(hn_task, arxiv_task)

    all_items = hn_items + arxiv_items
    logger.info(f"Fetched {len(hn_items)} HN + {len(arxiv_items)} ArXiv = {len(all_items)} total")
    return all_items
