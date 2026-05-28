import tempfile
import unittest
from pathlib import Path

from agent_infra.harness.status import build_status_payload


class HarnessStatusTests(unittest.TestCase):
    def test_status_payload_has_stable_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = build_status_payload(Path(tmp))

        self.assertEqual(payload["schema_version"], "jarvis.hud-status.v1")
        self.assertIn("context", payload)
        self.assertIn("sessionControls", payload)
        self.assertIn("checks", payload)
        self.assertIn("risk", payload)
        self.assertIn("status", payload["sessionControls"]["supported"])
