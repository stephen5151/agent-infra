"""
Chapter 14 — Knowledge Retrieval (RAG)
========================================
在生成前先从文档库里检索相关片段，注入到上下文窗口，
让 LLM 回答基于真实知识而非纯粹记忆。

Pattern:
  文档入库 → 向量化存储
  查询时: Query → Embed → ANN 检索 → Rerank → 注入 context → LLM 生成

设计原则 (书中 Ch.14):
  1. 检索必须在生成"之前"完成，而非事后记录
  2. 对检索结果做相关性过滤，避免噪音干扰模型
  3. 引用来源，让模型输出可追溯

默认实现:
  - 使用关键词 BM25 风格检索（零外部依赖）
  - 提供 embed_fn 接口：传入任意 langchain Embeddings 即可升级为向量检索
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from agent_infra.core.state import AgentState


# ── 文档存储 ──────────────────────────────────────────────────────────────────

@dataclass
class RAGStore:
    """
    可插拔的文档存储。

    默认使用 BM25 关键词检索（不依赖 embedding API）。
    传入 embed_fn 后自动升级为余弦相似度向量检索。

    Args:
        embed_fn: 可选的向量化函数 (texts: list[str]) -> list[list[float]]
                  例: embed_fn=OpenAIEmbeddings().embed_documents
        top_k:    每次检索返回的最大文档数
        score_threshold: BM25 得分阈值，低于此值的文档被过滤
    """
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None
    top_k: int = 5
    score_threshold: float = 0.1

    _docs: list[Document] = field(default_factory=list, init=False)
    _embeddings: list[list[float]] = field(default_factory=list, init=False)
    _bm25_index: dict[str, Any] = field(default_factory=dict, init=False)

    def add_documents(self, docs: list[Document]) -> None:
        """将文档加入索引。多次调用可增量添加。"""
        self._docs.extend(docs)
        if self.embed_fn:
            vecs = self.embed_fn([d.page_content for d in docs])
            self._embeddings.extend(vecs)
        else:
            self._build_bm25()

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        metas = metadatas or [{} for _ in texts]
        self.add_documents([Document(page_content=t, metadata=m) for t, m in zip(texts, metas)])

    def search(self, query: str) -> list[Document]:
        if not self._docs:
            return []
        if self.embed_fn and self._embeddings:
            return self._vector_search(query)
        return self._bm25_search(query)

    # ── BM25 ─────────────────────────────────────────────────────────────────

    def _build_bm25(self) -> None:
        k1, b = 1.5, 0.75
        corpus = [_tokenize(d.page_content) for d in self._docs]
        df: dict[str, int] = defaultdict(int)
        for tokens in corpus:
            for t in set(tokens):
                df[t] += 1
        N = len(corpus)
        avgdl = sum(len(t) for t in corpus) / max(N, 1)
        self._bm25_index = {"corpus": corpus, "df": df, "N": N, "avgdl": avgdl,
                             "k1": k1, "b": b}

    def _bm25_search(self, query: str) -> list[Document]:
        idx = self._bm25_index
        if not idx:
            return []
        q_tokens = _tokenize(query)
        k1, b = idx["k1"], idx["b"]
        N, avgdl = idx["N"], idx["avgdl"]
        scores: list[float] = []
        for tokens in idx["corpus"]:
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for t in q_tokens:
                if t not in tf:
                    continue
                idf = math.log((N - idx["df"].get(t, 0) + 0.5) /
                               (idx["df"].get(t, 0) + 0.5) + 1)
                score += idf * (tf[t] * (k1 + 1)) / (
                    tf[t] + k1 * (1 - b + b * dl / avgdl))
            scores.append(score)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [self._docs[i] for i, s in ranked[:self.top_k] if s >= self.score_threshold]

    # ── 向量检索 ─────────────────────────────────────────────────────────────

    def _vector_search(self, query: str) -> list[Document]:
        q_vec = self.embed_fn([query])[0]  # type: ignore[index]
        scores = [_cosine(q_vec, d_vec) for d_vec in self._embeddings]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [self._docs[i] for i, s in ranked[:self.top_k] if s >= self.score_threshold]


# ── LangGraph 节点 ────────────────────────────────────────────────────────────

class RAGNode:
    """
    生成前注入检索上下文。

    图中位置: memory_recall → rag_retrieve → router → ...

    用法:
        store = RAGStore()
        store.add_texts(["文档1内容", "文档2内容"])
        rag_node = RAGNode(store)
        graph.add_node("rag_retrieve", rag_node)
    """

    def __init__(self, store: RAGStore, max_context_chars: int = 4000) -> None:
        self.store = store
        self.max_chars = max_context_chars

    def __call__(self, state: AgentState) -> dict[str, Any]:
        query = _latest_human_message(state)
        docs = self.store.search(query)

        # 截断防止超出上下文窗口
        chunks: list[str] = []
        total = 0
        for doc in docs:
            text = doc.page_content
            source = doc.metadata.get("source", "unknown")
            snippet = f"[来源: {source}]\n{text}"
            if total + len(snippet) > self.max_chars:
                break
            chunks.append(snippet)
            total += len(snippet)

        return {"rag_context": chunks}


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z一-鿿]+", text.lower())


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def _latest_human_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""
