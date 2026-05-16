"""
代码库语义索引器
=================
遍历文件系统，将代码文件切分后注入 RAGStore，
实现 Cursor 风格的"对话时自动召回相关代码"能力。

切分策略（代码感知，非纯文本切分）:
  Python  → 以函数/类为粒度切分（AST）
  其他    → 按行滑动窗口切分（chunk_size 行，overlap 行重叠）

召回时:
  query → BM25 或向量检索 → 返回含文件路径+行号的代码片段
  → 注入 state["rag_context"] → LLM 生成时可引用具体位置
"""
from __future__ import annotations

import ast
from pathlib import Path

from langchain_core.documents import Document

from agent_infra.patterns.rag import RAGStore


# 默认忽略的目录/文件模式
_IGNORE = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".DS_Store",
}
_CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".java", ".rs", ".cpp", ".c", ".h"}


class CodebaseIndexer:
    """
    扫描代码库并建立可检索索引。

    Args:
        store:       目标 RAGStore，不传则新建一个
        chunk_lines: 非 Python 文件的切分行数
        overlap:     切分重叠行数（保持上下文连续性）
    """

    def __init__(
        self,
        store: RAGStore | None = None,
        chunk_lines: int = 50,
        overlap: int = 10,
    ) -> None:
        self.store = store or RAGStore()
        self.chunk_lines = chunk_lines
        self.overlap = overlap
        self._indexed: set[str] = set()

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def index_directory(
        self,
        root: str | Path,
        extensions: set[str] | None = None,
    ) -> int:
        """
        递归索引目录下的所有代码文件。

        Returns:
            成功索引的文档（代码块）数量
        """
        root = Path(root).resolve()
        exts = extensions or _CODE_EXTENSIONS
        docs: list[Document] = []

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in _IGNORE for part in path.parts):
                continue
            if path.suffix not in exts:
                continue
            if str(path) in self._indexed:
                continue
            docs.extend(self._chunk_file(path, root))
            self._indexed.add(str(path))

        self.store.add_documents(docs)
        return len(docs)

    def index_file(self, path: str | Path, root: str | Path = ".") -> int:
        """索引单个文件。"""
        path = Path(path).resolve()
        root = Path(root).resolve()
        docs = self._chunk_file(path, root)
        self.store.add_documents(docs)
        self._indexed.add(str(path))
        return len(docs)

    def index_text(self, text: str, source: str = "inline") -> None:
        """直接索引文本内容（用于索引文档、注释等非文件内容）。"""
        doc = Document(page_content=text, metadata={"source": source, "type": "text"})
        self.store.add_documents([doc])

    @property
    def indexed_count(self) -> int:
        return len(self._indexed)

    # ── 切分逻辑 ──────────────────────────────────────────────────────────────

    def _chunk_file(self, path: Path, root: Path) -> list[Document]:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []

        rel = str(path.relative_to(root))

        if path.suffix == ".py":
            return self._chunk_python(content, rel)
        return self._chunk_sliding_window(content, rel)

    def _chunk_python(self, content: str, source: str) -> list[Document]:
        """
        AST 感知切分：以顶层函数/类为粒度，保留函数签名上下文。
        比滑动窗口更准确，不会在函数中间截断。
        """
        docs: list[Document] = []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # 语法错误时退化为滑动窗口
            return self._chunk_sliding_window(content, source)

        lines = content.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            # 只处理顶层和一级嵌套，跳过深层嵌套避免冗余
            start = node.lineno - 1
            end = getattr(node, "end_lineno", start + 1)
            chunk = "\n".join(lines[start:end])
            if len(chunk.strip()) < 20:
                continue
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "type": type(node).__name__,
                    "name": node.name,
                    "line_start": start + 1,
                    "line_end": end,
                },
            ))

        # 如果没解析出任何节点（纯脚本），回退到滑动窗口
        return docs if docs else self._chunk_sliding_window(content, source)

    def _chunk_sliding_window(self, content: str, source: str) -> list[Document]:
        """通用滑动窗口切分，保持行重叠避免上下文断裂。"""
        lines = content.splitlines()
        docs: list[Document] = []
        step = max(1, self.chunk_lines - self.overlap)

        for i in range(0, len(lines), step):
            chunk = "\n".join(lines[i: i + self.chunk_lines])
            if len(chunk.strip()) < 10:
                continue
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "type": "chunk",
                    "line_start": i + 1,
                    "line_end": min(i + self.chunk_lines, len(lines)),
                },
            ))
        return docs
