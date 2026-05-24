import json
import unittest
from unittest.mock import patch

try:
    from typer.testing import CliRunner

    from agent_infra.cli import app
except ModuleNotFoundError:  # pragma: no cover - depends on optional dev environment
    CliRunner = None
    app = None


@unittest.skipIf(CliRunner is None, "typer is not installed")
class PiCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_pi_command_accepts_prompt_arguments(self) -> None:
        with patch("agent_infra.pi_bridge.run_agent_prompt", return_value="reply") as run_agent:
            result = self.runner.invoke(app, ["pi", "hello", "pi"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output, "reply\n")
        run_agent.assert_called_once()
        self.assertEqual(run_agent.call_args.args[0], "hello pi")

    def test_pi_command_accepts_stdin_and_json_output(self) -> None:
        with patch("agent_infra.pi_bridge.run_agent_prompt", return_value="reply"):
            result = self.runner.invoke(
                app,
                ["pi", "--json", "--thread", "thread-1"],
                input="hello from stdin",
            )

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["prompt"], "hello from stdin")
        self.assertEqual(payload["response"], "reply")
        self.assertEqual(payload["thread"], "thread-1")
        self.assertEqual(payload["mode"], "agent")

    def test_pi_command_rejects_unknown_mode(self) -> None:
        result = self.runner.invoke(app, ["pi", "--mode", "bad", "hello"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("mode", result.output)


if __name__ == "__main__":
    unittest.main()
