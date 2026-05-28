"""
文件系统监听适配器
==================
监听指定目录的文件创建/修改事件。
你保存的文件 = 你正在构建的东西。

依赖 watchdog（可选），降级为轮询模式。
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from agent_infra.capture.adapters.base import BaseAdapter
from agent_infra.capture.event_bus import CaptureEvent, EventSource, EventType
from agent_infra.capture.media import MediaKind, build_file_capture_payload
from agent_infra.config.settings import get_settings


class FileAdapter(BaseAdapter):
    """监听文件系统变化。"""

    name = "file"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        cfg = get_settings()
        self.enabled = cfg.capture.file_watch_enabled
        self.watch_paths = [
            Path(p).expanduser()
            for p in cfg.capture.file_watch_paths
        ]
        self.extensions = set(cfg.capture.file_watch_extensions)
        self.poll_interval = 5.0
        self._seen_hashes: dict[Path, str] = {}   # path → content hash

    async def run(self) -> None:
        # 尝试使用 watchdog
        try:
            await self._run_with_watchdog()
        except ImportError:
            self.logger.info("watchdog not installed, using polling mode")
            await self._run_polling()

    async def _run_polling(self) -> None:
        """轮询模式：每 N 秒扫描一次监听目录。"""
        while self._running:
            for watch_path in self.watch_paths:
                if not watch_path.exists():
                    continue
                for ext in self.extensions:
                    for fpath in watch_path.rglob(f"*{ext}"):
                        await self._check_file(fpath)
            await asyncio.sleep(self.poll_interval)

    async def _check_file(self, fpath: Path) -> None:
        """检查文件是否有变化。"""
        try:
            stat = fpath.stat()
            if stat.st_size == 0 or stat.st_size > 1_000_000:
                return  # 跳过空文件和超大文件

            payload = build_file_capture_payload(fpath)
            if payload.media_kind == MediaKind.TEXT:
                content_hash = hashlib.md5(payload.content.encode()).hexdigest()
            else:
                content_hash = hashlib.md5(
                    f"{payload.metadata['path']}:{payload.metadata['size']}".encode()
                ).hexdigest()

            if self._seen_hashes.get(fpath) == content_hash:
                return

            self._seen_hashes[fpath] = content_hash

            # 跳过隐私内容
            cfg = get_settings()
            if payload.media_kind == MediaKind.TEXT and cfg.privacy.is_sensitive(payload.content[:500]):
                return

            event = CaptureEvent(
                source=EventSource.FILE,
                type=EventType.ACTION,
                content=payload.content,
                importance=self._estimate_importance(fpath, payload.content),
                metadata=payload.metadata,
            )
            await self.emit(event)
        except (OSError, PermissionError):
            pass

    async def _run_with_watchdog(self) -> None:
        """watchdog 事件驱动模式（低延迟）。"""
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

        queue: asyncio.Queue = asyncio.Queue()

        class Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory:
                    queue.put_nowait(Path(event.src_path))

            def on_created(self, event):
                if not event.is_directory:
                    queue.put_nowait(Path(event.src_path))

        observer = Observer()
        handler = Handler()
        for watch_path in self.watch_paths:
            if watch_path.exists():
                observer.schedule(handler, str(watch_path), recursive=True)

        observer.start()
        self.logger.info(f"watchdog observer started on: {self.watch_paths}")

        try:
            while self._running:
                try:
                    fpath = await asyncio.wait_for(queue.get(), timeout=1.0)
                    if fpath.suffix in self.extensions:
                        await self._check_file(fpath)
                except asyncio.TimeoutError:
                    continue
        finally:
            observer.stop()
            observer.join()

    def _estimate_importance(self, fpath: Path, content: str) -> float:
        """文件重要性估算。"""
        ext = fpath.suffix
        if ext.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".mp4", ".mov", ".m4v", ".mp3", ".m4a", ".wav"}:
            return 0.6  # 媒体文件先结构化入库，后续再做内容提取
        if ext in (".py", ".ts", ".js", ".rs", ".go"):
            return 0.7  # 代码文件
        if ext == ".md":
            return 0.6  # 文档
        if any(kw in content[:200] for kw in ["TODO", "FIXME", "HACK", "BUG"]):
            return 0.8
        return 0.5
