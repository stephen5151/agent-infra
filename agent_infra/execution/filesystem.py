"""
文件系统操作 + Diff/Patch
===========================
Cursor 风格的核心能力：生成代码 → 以 diff 展示 → 用户 Accept → 写入磁盘。

设计原则:
  - 所有写操作先生成 diff，不直接覆盖
  - diff 存入 state["file_changes"]，供 HITL 展示和回滚
  - apply_diff 才真正落盘，是唯一的"破坏性"操作
"""
from __future__ import annotations

import difflib
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileDiff:
    path: str
    original: str
    updated: str
    unified_diff: str

    @property
    def has_changes(self) -> bool:
        return self.original != self.updated

    @property
    def lines_added(self) -> int:
        return sum(1 for l in self.unified_diff.splitlines() if l.startswith("+") and not l.startswith("+++"))

    @property
    def lines_removed(self) -> int:
        return sum(1 for l in self.unified_diff.splitlines() if l.startswith("-") and not l.startswith("---"))

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "diff": self.unified_diff,
            "content": self.updated,  # FileApplyNode 写盘时使用
        }


class FileSystem:
    """
    受控的文件系统操作层。

    所有写入操作均先生成 FileDiff，通过 apply() 才真正落盘。
    支持单文件操作和批量操作。

    Args:
        root: 可操作的根目录（沙箱边界，拒绝 root 以外的路径）
    """

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()

    # ── 读 ────────────────────────────────────────────────────────────────────

    def read(self, path: str | Path) -> str:
        """读取文件内容。"""
        p = self._resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")
        return p.read_text(encoding="utf-8")

    def exists(self, path: str | Path) -> bool:
        return self._resolve(path).exists()

    def list_files(
        self,
        pattern: str = "**/*",
        exclude: list[str] | None = None,
    ) -> list[str]:
        """列出匹配的文件，返回相对于 root 的路径字符串。"""
        excludes = set(exclude or ["__pycache__", ".git", "node_modules", ".venv", "*.pyc"])
        results = []
        for p in self.root.glob(pattern):
            if not p.is_file():
                continue
            rel = str(p.relative_to(self.root))
            if any(ex in rel for ex in excludes):
                continue
            results.append(rel)
        return sorted(results)

    # ── 写（先生成 diff，不落盘）────────────────────────────────────────────

    def prepare_write(self, path: str | Path, new_content: str) -> FileDiff:
        """
        生成写入 diff，不修改磁盘。
        返回 FileDiff 供展示/审核，调用 apply() 才真正写入。
        """
        p = self._resolve(path)
        original = p.read_text(encoding="utf-8") if p.exists() else ""
        diff = _unified_diff(original, new_content, str(path))
        return FileDiff(path=str(path), original=original, updated=new_content, unified_diff=diff)

    def prepare_patch(self, path: str | Path, patch: str) -> FileDiff:
        """
        将 unified diff 字符串应用到文件，生成 FileDiff（不落盘）。
        适合 LLM 直接输出 diff 格式的场景。
        """
        original = self.read(path) if self.exists(path) else ""
        updated = _apply_unified_diff(original, patch)
        diff = _unified_diff(original, updated, str(path))
        return FileDiff(path=str(path), original=original, updated=updated, unified_diff=diff)

    # ── 落盘 ──────────────────────────────────────────────────────────────────

    def apply(self, file_diff: FileDiff, backup: bool = True) -> None:
        """将 FileDiff 真正写入磁盘（唯一的破坏性操作）。"""
        p = self._resolve(file_diff.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if backup and p.exists():
            shutil.copy2(p, str(p) + ".bak")
        p.write_text(file_diff.updated, encoding="utf-8")

    def apply_many(self, diffs: list[FileDiff], backup: bool = True) -> None:
        """批量落盘，原子性尽力保证（逐个写，失败不回滚）。"""
        for d in diffs:
            self.apply(d, backup=backup)

    def revert(self, path: str | Path) -> bool:
        """从 .bak 文件恢复（撤销最近一次 apply）。"""
        p = self._resolve(path)
        bak = Path(str(p) + ".bak")
        if bak.exists():
            shutil.copy2(bak, p)
            bak.unlink()
            return True
        return False

    # ── 安全边界 ──────────────────────────────────────────────────────────────

    def _resolve(self, path: str | Path) -> Path:
        resolved = (self.root / path).resolve()
        if not str(resolved).startswith(str(self.root)):
            raise PermissionError(f"路径越界: {path} 超出根目录 {self.root}")
        return resolved


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _unified_diff(original: str, updated: str, filename: str) -> str:
    lines_a = original.splitlines(keepends=True)
    lines_b = updated.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(lines_a, lines_b, fromfile=f"a/{filename}", tofile=f"b/{filename}")
    )


def _apply_unified_diff(original: str, patch: str) -> str:
    """
    简单的 unified diff 应用器（纯 stdlib，无需 patch 命令）。
    仅处理标准 +/- 格式，不支持 fuzzy match。
    """
    lines = original.splitlines(keepends=True)
    result = list(lines)
    offset = 0

    for hunk in _parse_hunks(patch):
        start, old_lines, new_lines = hunk
        idx = start - 1 + offset
        result[idx: idx + len(old_lines)] = new_lines
        offset += len(new_lines) - len(old_lines)

    return "".join(result)


def _parse_hunks(patch: str) -> list[tuple[int, list[str], list[str]]]:
    hunks = []
    current_start = 0
    old_lines: list[str] = []
    new_lines: list[str] = []
    in_hunk = False

    for line in patch.splitlines(keepends=True):
        if line.startswith("@@"):
            if in_hunk:
                hunks.append((current_start, old_lines, new_lines))
            parts = line.split()
            # @@ -a,b +c,d @@
            old_info = parts[1].lstrip("-").split(",")
            current_start = int(old_info[0])
            old_lines = []
            new_lines = []
            in_hunk = True
        elif in_hunk:
            if line.startswith("-"):
                old_lines.append(line[1:])
            elif line.startswith("+"):
                new_lines.append(line[1:])
            elif line.startswith(" "):
                old_lines.append(line[1:])
                new_lines.append(line[1:])

    if in_hunk:
        hunks.append((current_start, old_lines, new_lines))
    return hunks
