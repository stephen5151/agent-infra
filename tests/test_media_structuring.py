import tempfile
import unittest
from pathlib import Path

from agent_infra.capture.media import (
    MediaKind,
    build_file_capture_payload,
    classify_media_path,
)
from agent_infra.config.settings import get_settings


class MediaStructuringTests(unittest.TestCase):
    def test_classifies_image_and_video_files(self) -> None:
        self.assertEqual(classify_media_path(Path("photo.JPG")), MediaKind.IMAGE)
        self.assertEqual(classify_media_path(Path("clip.mp4")), MediaKind.VIDEO)

    def test_media_file_becomes_structured_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "demo.png"
            fpath.write_bytes(b"\x89PNG\r\n")

            payload = build_file_capture_payload(fpath)

        self.assertIn("媒体类型: image", payload.content)
        self.assertIn("文件名: demo.png", payload.content)
        self.assertIn("提取状态: pending", payload.content)
        self.assertEqual(payload.metadata["content_mode"], "structured_media")
        self.assertEqual(payload.metadata["media_kind"], "image")
        self.assertEqual(payload.metadata["extraction_status"], "pending")

    def test_text_file_keeps_text_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "note.md"
            fpath.write_text("hello from text", encoding="utf-8")

            payload = build_file_capture_payload(fpath)

        self.assertEqual(payload.content, "hello from text")
        self.assertEqual(payload.metadata["content_mode"], "text")
        self.assertEqual(payload.metadata["extension"], ".md")

    def test_default_capture_config_includes_media_extensions(self) -> None:
        extensions = set(get_settings().capture.file_watch_extensions)

        self.assertIn(".png", extensions)
        self.assertIn(".mp4", extensions)
        self.assertIn(".m4a", extensions)


if __name__ == "__main__":
    unittest.main()
