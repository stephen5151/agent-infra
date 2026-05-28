from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class MediaKind(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    UNKNOWN = "unknown"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".heic",
    ".heif",
    ".tif",
    ".tiff",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"}


@dataclass(frozen=True)
class FileCapturePayload:
    content: str
    metadata: dict
    media_kind: MediaKind


def classify_media_path(path: Path) -> MediaKind:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return MediaKind.IMAGE
    if suffix in VIDEO_EXTENSIONS:
        return MediaKind.VIDEO
    if suffix in AUDIO_EXTENSIONS:
        return MediaKind.AUDIO
    if suffix:
        return MediaKind.TEXT
    return MediaKind.UNKNOWN


def build_file_capture_payload(path: Path, *, text_limit: int = 3000) -> FileCapturePayload:
    stat = path.stat()
    media_kind = classify_media_path(path)
    base_metadata = {
        "path": str(path),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size": stat.st_size,
        "media_kind": media_kind.value,
    }

    if media_kind in {MediaKind.IMAGE, MediaKind.VIDEO, MediaKind.AUDIO}:
        metadata = {
            **base_metadata,
            "content_mode": "structured_media",
            "extraction_status": "pending",
            "extraction_plan": _extraction_plan(media_kind),
        }
        return FileCapturePayload(
            content=_structured_media_text(path, media_kind, stat.st_size, metadata["extraction_plan"]),
            metadata=metadata,
            media_kind=media_kind,
        )

    content = path.read_text(encoding="utf-8", errors="ignore")[:text_limit]
    metadata = {
        **base_metadata,
        "content_mode": "text",
        "media_kind": MediaKind.TEXT.value,
    }
    return FileCapturePayload(content=content, metadata=metadata, media_kind=MediaKind.TEXT)


def _structured_media_text(path: Path, media_kind: MediaKind, size: int, extraction_plan: list[str]) -> str:
    plan = "\n".join(f"- {step}" for step in extraction_plan)
    return (
        "# Structured Media Intake\n\n"
        f"媒体类型: {media_kind.value}\n"
        f"文件名: {path.name}\n"
        f"文件路径: {path}\n"
        f"文件扩展名: {path.suffix.lower() or 'unknown'}\n"
        f"文件大小: {size} bytes\n"
        "提取状态: pending\n\n"
        "## 后续提取计划\n"
        f"{plan}\n"
    )


def _extraction_plan(media_kind: MediaKind) -> list[str]:
    if media_kind == MediaKind.IMAGE:
        return ["OCR 提取可见文字", "多模态模型生成画面描述", "保留文件路径作为原始来源"]
    if media_kind == MediaKind.VIDEO:
        return ["提取字幕或语音转写", "抽取关键帧并生成画面描述", "汇总时间线和主要主题"]
    if media_kind == MediaKind.AUDIO:
        return ["语音转写", "按主题切分段落", "提取可复用摘要"]
    return ["按文本内容读取"]
