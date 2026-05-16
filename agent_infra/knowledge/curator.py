"""
知识策展引擎
=============
职责:
  1. 将 RawItem 写入 KnowledgeStore（去重）
  2. 用 LLM 生成中文摘要 + 话题标签
  3. 根据用户话题权重对候选池评分，选出最终推送列表

LLM 调用策略（节省费用）:
  - 优先用本地 Ollama（免费）
  - Ollama 不可用时走 Claude CLI 订阅（零 API 费用）
  - 批量处理，每条限 200 token
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

from .fetcher import RawItem
from .store import KnowledgeStore, TOPIC_TAXONOMY, get_knowledge_store

logger = logging.getLogger(__name__)

# 话题归类 prompt（极简，节省 token）
_TOPIC_PROMPT = """你是一个文章分类助手。根据标题和摘要，从以下列表选出 1-3 个最相关的话题标签，用 JSON 数组返回，不加解释。

可选话题: {taxonomy}

标题: {title}
摘要: {abstract}

只返回 JSON 数组，例如: ["AI/ML", "编程"]"""

_SUMMARY_PROMPT = """用简洁的中文（50-80字）总结以下文章的核心观点，不要加任何前缀。

标题: {title}
摘要: {abstract}"""


async def _call_llm(prompt: str, max_tokens: int = 200) -> str:
    """调用 LLM，优先本地 Ollama。"""
    try:
        from agent_infra.intelligence.llm_router import LLMRouter
        router = LLMRouter()
        return await router.agenerate(
            human=prompt,
            system="你是一个精准的文章分类和摘要助手。",
            max_tokens=max_tokens,
            force_local=True,      # 优先本地，节省费用
        )
    except Exception as e:
        logger.debug(f"LLM call failed: {e}")
        return ""


async def _enrich_item(
    item_id: str,
    title: str,
    abstract: str,
    store: KnowledgeStore,
) -> None:
    """为单个条目生成摘要 + 话题标签并写入 store。"""
    tax_str = ", ".join(TOPIC_TAXONOMY)
    abstract_short = abstract[:300] if abstract else title

    # 并发：话题分类 + 摘要生成
    topic_prompt = _TOPIC_PROMPT.format(
        taxonomy=tax_str, title=title, abstract=abstract_short
    )
    summary_prompt = _SUMMARY_PROMPT.format(title=title, abstract=abstract_short)

    topic_raw, summary = await asyncio.gather(
        _call_llm(topic_prompt, max_tokens=60),
        _call_llm(summary_prompt, max_tokens=150),
    )

    # 解析话题 JSON
    topics: list[str] = []
    try:
        parsed = json.loads(topic_raw.strip())
        if isinstance(parsed, list):
            topics = [t for t in parsed if isinstance(t, str) and t in TOPIC_TAXONOMY]
    except Exception:
        pass

    # 解析失败时 fallback：从分类表关键词匹配
    if not topics:
        text_lower = (title + abstract_short).lower()
        keyword_map = {
            "AI/ML": ["gpt", "llm", "machine learning", "neural", "deep learning", "ai", "model"],
            "编程": ["python", "javascript", "rust", "golang", "code", "programming", "compiler"],
            "安全": ["security", "vulnerability", "exploit", "cve", "hack"],
            "开源": ["open source", "github", "apache", "mit license"],
            "科学": ["research", "paper", "study", "experiment", "discovery"],
        }
        for topic, keywords in keyword_map.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)
                if len(topics) >= 2:
                    break

    if not topics:
        topics = ["其他"]

    # fallback 摘要
    if not summary:
        summary = abstract_short[:120] if abstract_short else title

    store.update_summary(item_id, summary.strip(), topics)
    logger.debug(f"Enriched [{item_id[:8]}] topics={topics}")


# ── 公共接口 ──────────────────────────────────────────────────────────────────

async def ingest_items(
    raw_items: list[RawItem],
    store: KnowledgeStore | None = None,
    enrich: bool = True,
    batch_size: int = 5,
) -> int:
    """
    将 RawItem 列表写入 store，可选 LLM 富化（摘要+话题）。
    返回新写入的条目数。
    """
    db = store or get_knowledge_store()
    new_ids: list[tuple[str, str, str]] = []   # (id, title, abstract)

    for item in raw_items:
        abstract = (item.extra or {}).get("abstract", "")
        item_id = db.upsert_item(
            title=item.title,
            url=item.url,
            source=item.source,
            topics=[],
            quality=item.quality,
            raw_score=item.raw_score,
            summary="",
            language=item.language,
        )
        # upsert 返回已有 id 时 summary 可能已存在，只对新条目富化
        existing = db.get_item(item_id)
        if existing and not existing.get("summary"):
            new_ids.append((item_id, item.title, abstract))

    if enrich and new_ids:
        # 分批并发（避免本地 Ollama 超载）
        for i in range(0, len(new_ids), batch_size):
            batch = new_ids[i : i + batch_size]
            await asyncio.gather(*[
                _enrich_item(iid, title, abstract, db)
                for iid, title, abstract in batch
            ])
        logger.info(f"Enriched {len(new_ids)} new items")

    return len(new_ids)


def score_feed(
    candidates: list[dict],
    weights: dict[str, float],
    feed_size: int = 7,
    exploration_ratio: float = 0.2,
) -> list[dict]:
    """
    根据话题权重对候选排序，混入探索条目。

    - (1 - exploration_ratio) 的位置给高权重话题
    - exploration_ratio 的位置给低权重/未知话题（拓展视野）
    """
    if not candidates:
        return []

    def _item_score(item: dict) -> float:
        topics = item.get("topics") or []
        if not topics:
            return item.get("quality", 0.5)
        avg_weight = sum(weights.get(t, 0.5) for t in topics) / len(topics)
        return avg_weight * 0.7 + item.get("quality", 0.5) * 0.3

    scored = [(c, _item_score(c)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    n_exploit = max(1, round(feed_size * (1 - exploration_ratio)))
    n_explore = feed_size - n_exploit

    top = [c for c, _ in scored[:n_exploit * 2]]
    bottom = [c for c, _ in scored[n_exploit * 2:]]

    exploit_items = top[:n_exploit]

    # 探索：优先选用户从未见过话题的条目
    known_topics = set(weights.keys())
    explore_pool = [
        c for c in bottom
        if any(t not in known_topics for t in (c.get("topics") or []))
    ]
    if len(explore_pool) < n_explore:
        explore_pool = bottom  # fallback：低分条目

    explore_items = random.sample(explore_pool, min(n_explore, len(explore_pool)))

    result = exploit_items + explore_items
    random.shuffle(result)       # 打乱顺序，避免"探索"条目总在最后
    return result[:feed_size]
