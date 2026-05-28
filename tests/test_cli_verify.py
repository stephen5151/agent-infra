import unittest
from unittest.mock import patch

try:
    from typer.testing import CliRunner

    from agent_infra.cli import app
except ModuleNotFoundError:  # pragma: no cover
    CliRunner = None
    app = None


@unittest.skipIf(CliRunner is None, "typer is not installed")
class VerifyCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_verify_command_prints_summary(self) -> None:
        with patch("agent_infra.harness.verification.run_verification") as run_verification:
            run_verification.return_value = ("python", [])
            with patch("agent_infra.harness.verification.summarize_report", return_value="Overall: READY"):
                result = self.runner.invoke(app, ["verify"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Overall: READY", result.output)

    def test_verify_command_supports_json_output(self) -> None:
        with patch("agent_infra.harness.verification.run_verification") as run_verification:
            run_verification.return_value = ("python", [])
            with patch("agent_infra.harness.verification.report_to_json", return_value='{"overall":"ready"}'):
                result = self.runner.invoke(app, ["verify", "--json"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('{"overall":"ready"}', result.output)


if __name__ == "__main__":
    unittest.main()
