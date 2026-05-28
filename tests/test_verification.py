import tempfile
import unittest
from pathlib import Path

from agent_infra.harness.verification import (
    VerificationResult,
    detect_verification_plan,
    python_sources,
    summarize_report,
)


class VerificationTests(unittest.TestCase):
    def test_detect_python_project_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (root / "pkg").mkdir()
            (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_demo.py").write_text("import unittest\n", encoding="utf-8")

            plan = detect_verification_plan(root)

        self.assertEqual(plan.project_type, "python")
        self.assertEqual([check.name for check in plan.checks], ["build", "lint", "tests"])

    def test_detect_generic_project_without_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("hello", encoding="utf-8")

            plan = detect_verification_plan(root)

        self.assertEqual(plan.project_type, "generic")
        self.assertEqual([check.name for check in plan.checks], ["files"])

    def test_summarize_report_includes_failed_checks(self) -> None:
        report = [
            VerificationResult(name="build", passed=True, command="python -m py_compile", output="ok"),
            VerificationResult(name="tests", passed=False, command="python -m unittest", output="1 failed"),
        ]

        summary = summarize_report(report)

        self.assertIn("Build: PASS", summary)
        self.assertIn("Tests: FAIL", summary)
        self.assertIn("Overall: NOT READY", summary)
        self.assertIn("1 failed", summary)

    def test_python_sources_excludes_virtualenv_and_hidden_worktrees(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (root / ".venv").mkdir()
            (root / ".venv" / "ignored.py").write_text("", encoding="utf-8")
            (root / ".claude").mkdir()
            (root / ".claude" / "ignored.py").write_text("", encoding="utf-8")

            sources = python_sources(root)

        self.assertEqual(sources, ["pkg/__init__.py"])


if __name__ == "__main__":
    unittest.main()
