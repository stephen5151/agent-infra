import tempfile
import unittest
from pathlib import Path

from agent_infra.harness.audit import run_harness_audit


class HarnessAuditTests(unittest.TestCase):
    def test_audit_reports_missing_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent_infra").mkdir()
            (root / "agent_infra" / "core").mkdir(parents=True)
            (root / "agent_infra" / "core" / "tools.py").write_text("", encoding="utf-8")

            result = run_harness_audit(root)

        self.assertGreater(result["max_score"], 0)
        self.assertTrue(any(item["category"] == "tool_coverage" for item in result["categories"]))
        self.assertTrue(any(item["score"] < item["max_score"] for item in result["categories"]))
        self.assertTrue(result["top_actions"])
        self.assertEqual(result["rubric_version"], "jarvis.audit.v1")
        tool_category = next(item for item in result["categories"] if item["category"] == "tool_coverage")
        self.assertIn("dimensions", tool_category)
        self.assertIn("evidence", tool_category)
        self.assertIn("depth", tool_category["dimensions"])
        self.assertEqual(tool_category["max_score"], 10)

    def test_audit_rewards_tests_and_operational_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent_infra" / "core").mkdir(parents=True)
            (root / "agent_infra" / "core" / "tools.py").write_text("", encoding="utf-8")
            (root / "agent_infra" / "harness").mkdir(parents=True)
            (root / "agent_infra" / "harness" / "verification.py").write_text("", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_tools.py").write_text("import unittest\n", encoding="utf-8")

            result = run_harness_audit(root)

        quality = next(item for item in result["categories"] if item["category"] == "quality_gates")
        self.assertGreaterEqual(quality["dimensions"]["presence"], 1)
        self.assertGreaterEqual(quality["dimensions"]["evidence"], 1)
        self.assertGreaterEqual(quality["score"], 4)
