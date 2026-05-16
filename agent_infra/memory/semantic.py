"""
语义记忆 — Obsidian 知识库 + RAG 检索
=======================================
你的 Obsidian vault 就是 Agent 的长期记忆。
每次捕获到重要事件，Agent 会将洞察写入 vault；
需要背景知识时，Agent 从 vault 向量检索相关片段。

存储后端：ChromaDB（本地，零配置）
嵌入模型：sentence-transformers（本地，免费）或 Claude API（可配置）

Obsidian vault 路径: ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_infra.config.settings import get_settings

logger = logging.getLogger(__name__)

# ── 可选依赖（运行时懒加载）────────────────────────────────────────────────────

def _try_import_chromadb():
    try:
        import chromadb
        return chromadb
    except ImportError:
        return None

def _try_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        return None


# ── 文本分块器 ────────────────────────────────────────────────────────────────

class MarkdownChunker:
    """将 Markdown 文件切分为语义块，保留标题上下文。"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source_path: str = "") -> list[dict]:
        """返回 [{text, header, source, chunk_idx}] 列表。"""
        chunks = []
        current_header = ""
        buffer: list[str] = []
        buffer_len = 0

        for line in text.splitlines(keepends=True):
            # 检测 Markdown 标题
            header_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if header_match:
                # 先把当前 buffer flush
                if buffer and buffer_len >= 50:
                    chunks.extend(self._flush(buffer, current_header, source_path, len(chunks)))
                    # 保留 overlap
                    overlap_text = "".join(buffer)[-self.overlap:]
                    buffer = [overlap_text] if overlap_text else []
                    buffer_len = len(overlap_text)
                current_header = line.strip()

            buffer.append(line)
            buffer_len += len(line)

            if buffer_len >= self.chunk_size:
                chunks.extend(self._flush(buffer, current_header, source_path, len(chunks)))
                overlap_text = "".join(buffer)[-self.overlap:]
                buffer = [overlap_text] if overlap_text else []
                buffer_len = len(overlap_text)

        # 最后一块
        if buffer and buffer_len >= 50:
            chunks.extend(self._flush(buffer, current_header, source_path, len(chunks)))

        return chunks

    def _flush(self, buffer: list[str], header: str, source: str, idx: int) -> list[dict]:
        text = "".join(buffer).strip()
        if not text:
            return []
        return [{
            "text": text,
            "header": header,
            "source": source,
            "chunk_idx": idx,
        }]


# ── 语义记忆主类 ──────────────────────────────────────────────────────────────

class SemanticMemory:
    """
    Obsidian vault 语义检索 + 洞察写入。

    用法：
        mem = SemanticMemory()
        mem.index_vault()                          # 索引/更新 vault
        results = mem.search("FastAPI JWT 认证")   # 检索相关记忆
        mem.write_insight("## 今天学到", content)  # 写入 Jarvis Inbox
    """

    COLLECTION_NAME = "jarvis_vault"

    def __init__(self) -> None:
        cfg = get_settings()
        self.vault_path = cfg.memory.obsidian_vault_path
        self.inbox_path = cfg.memory.obsidian_inbox_path
        self.chroma_path = cfg.memory.episodic_db_path.parent / "chroma"
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.chunker = MarkdownChunker()

        self._client = None       # ChromaDB client（懒加载）
        self._collection = None   # ChromaDB collection（懒加载）
        self._embedder = None     # SentenceTransformer（懒加载）
        self._indexed_hashes: set[str] = set()  # 已索引文件 hash

        logger.info(f"SemanticMemory initialized → vault: {self.vault_path}")

    # ── 嵌入 & ChromaDB 初始化 ────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            SentenceTransformer = _try_import_sentence_transformers()
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            # 多语言模型，支持中英混合
            self._embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            logger.info("Embedder loaded: paraphrase-multilingual-MiniLM-L12-v2")
        return self._embedder

    def _get_collection(self):
        if self._collection is None:
            chromadb = _try_import_chromadb()
            if chromadb is None:
                raise ImportError(
                    "chromadb not installed. Run: pip install chromadb"
                )
            self._client = chromadb.PersistentClient(path=str(self.chroma_path))
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB collection ready: {self.COLLECTION_NAME}")
        return self._collection

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embedder = self._get_embedder()
        return embedder.encode(texts, show_progress_bar=False).tolist()

    # ── 索引 vault ────────────────────────────────────────────────────────────

    def index_vault(self, force: bool = False) -> dict:
        """
        扫描 Obsidian vault，索引所有 Markdown 文件。
        已索引且未修改的文件跳过（基于内容 hash）。
        返回统计信息。
        """
        if not self.vault_path.exists():
            logger.warning(f"Vault path not found: {self.vault_path}")
            return {"status": "vault_not_found", "indexed": 0, "skipped": 0}

        collection = self._get_collection()
        stats = {"indexed": 0, "skipped": 0, "errors": 0}
        batch_ids: list[str] = []
        batch_docs: list[str] = []
        batch_metas: list[dict] = []
        batch_embeddings: list[list[float]] = []

        md_files = list(self.vault_path.rglob("*.md"))
        logger.info(f"Scanning vault: {len(md_files)} Markdown files")

        for md_file in md_files:
            try:
                text = md_file.read_text(encoding="utf-8", errors="ignore")
                file_hash = hashlib.md5(text.encode()).hexdigest()

                if not force and file_hash in self._indexed_hashes:
                    stats["skipped"] += 1
                    continue

                rel_path = str(md_file.relative_to(self.vault_path))
                chunks = self.chunker.chunk(text, source_path=rel_path)

                for chunk in chunks:
                    chunk_id = f"{file_hash}_{chunk['chunk_idx']}"
                    # 跳过已在 DB 中的
                    existing = collection.get(ids=[chunk_id])
                    if existing["ids"] and not force:
                        continue

                    batch_ids.append(chunk_id)
                    batch_docs.append(chunk["text"])
                    batch_metas.append({
                        "source": rel_path,
                        "header": chunk["header"],
                        "file_hash": file_hash,
                        "indexed_at": datetime.now(timezone.utc).isoformat(),
                    })

                    # 分批嵌入（每批 32 条）
                    if len(batch_ids) >= 32:
                        self._upsert_batch(
                            collection, batch_ids, batch_docs, batch_metas
                        )
                        stats["indexed"] += len(batch_ids)
                        batch_ids, batch_docs, batch_metas = [], [], []

                self._indexed_hashes.add(file_hash)

            except Exception as e:
                logger.warning(f"Failed to index {md_file}: {e}")
                stats["errors"] += 1

        # 最后一批
        if batch_ids:
            self._upsert_batch(collection, batch_ids, batch_docs, batch_metas)
            stats["indexed"] += len(batch_ids)

        logger.info(f"Vault indexed: {stats}")
        return stats

    def _upsert_batch(
        self,
        collection,
        ids: list[str],
        docs: list[str],
        metas: list[dict],
    ) -> None:
        embeddings = self._embed(docs)
        collection.upsert(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metas,
        )

    # ── 检索 ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        向量检索最相关的 vault 片段。
        返回 [{text, source, header, score}] 列表。
        """
        collection = self._get_collection()
        if collection.count() == 0:
            logger.warning("Vault not indexed yet. Call index_vault() first.")
            return []

        query_embedding = self._embed([query])[0]
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text": doc,
                "source": meta.get("source", ""),
                "header": meta.get("header", ""),
                "score": round(1 - dist, 4),  # cosine similarity
            })

        return output

    def format_context(self, query: str, n_results: int = 3) -> str:
        """返回可直接注入 prompt 的上下文字符串。"""
        results = self.search(query, n_results=n_results)
        if not results:
            return ""
        parts = ["[来自你的 Obsidian 知识库]"]
        for i, r in enumerate(results, 1):
            src = r["source"]
            header = f" > {r['header']}" if r["header"] else ""
            parts.append(f"\n--- [{i}] {src}{header} (相关度: {r['score']}) ---\n{r['text']}")
        return "\n".join(parts)

    # ── 写入 Obsidian ─────────────────────────────────────────────────────────

    def write_insight(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
        subfolder: str = "",
    ) -> Path:
        """
        将 Agent 产生的洞察写入 Obsidian Inbox。
        文件名 = timestamp_title.md，不覆盖已有文件。
        返回写入的文件路径。
        """
        # 确保 inbox 目录存在
        inbox = self.inbox_path
        if subfolder:
            inbox = inbox / subfolder
        inbox.mkdir(parents=True, exist_ok=True)

        # 文件名：时间戳 + 清理后的标题
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r'[^\w\s一-鿿-]', '', title)[:50].strip()
        filename = f"{ts}_{safe_title}.md"
        filepath = inbox / filename

        # Obsidian frontmatter
        tag_line = ""
        if tags:
            tag_str = "\n  - ".join(tags)
            tag_line = f"tags:\n  - {tag_str}\n"

        frontmatter = (
            f"---\n"
            f"created: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"source: jarvis-agent\n"
            f"{tag_line}"
            f"---\n\n"
        )

        filepath.write_text(frontmatter + content, encoding="utf-8")
        logger.info(f"Insight written → {filepath}")
        return filepath

    def write_daily_digest(self, content: str, date: datetime | None = None) -> Path:
        """写入每日摘要到 Jarvis Inbox/Daily/ 目录。"""
        d = date or datetime.now(timezone.utc)
        date_str = d.strftime("%Y-%m-%d")
        return self.write_insight(
            title=f"Daily Digest {date_str}",
            content=content,
            tags=["jarvis", "daily-digest"],
            subfolder="Daily",
        )

    # ── 统计 ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        try:
            collection = self._get_collection()
            count = collection.count()
        except Exception:
            count = -1

        vault_files = 0
        if self.vault_path.exists():
            vault_files = len(list(self.vault_path.rglob("*.md")))

        return {
            "vault_path": str(self.vault_path),
            "vault_md_files": vault_files,
            "indexed_chunks": count,
            "inbox_path": str(self.inbox_path),
        }


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_semantic: SemanticMemory | None = None


def get_semantic_memory() -> SemanticMemory:
    global _semantic
    if _semantic is None:
        _semantic = SemanticMemory()
    return _semantic
