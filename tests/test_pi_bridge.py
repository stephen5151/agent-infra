import json
import unittest

from agent_infra.pi_bridge import PiInputError, build_json_payload, resolve_prompt


class PiBridgeTests(unittest.TestCase):
    def test_resolve_prompt_prefers_arguments(self) -> None:
        prompt = resolve_prompt(("hello", "from", "args"), "ignored stdin")

        self.assertEqual(prompt, "hello from args")

    def test_resolve_prompt_reads_stdin_when_no_arguments(self) -> None:
        prompt = resolve_prompt((), "  hello from stdin\n")

        self.assertEqual(prompt, "hello from stdin")

    def test_resolve_prompt_rejects_empty_input(self) -> None:
        with self.assertRaises(PiInputError):
            resolve_prompt((), "   \n")

    def test_build_json_payload_is_machine_readable(self) -> None:
        payload = build_json_payload(
            prompt="Question?",
            response="Answer.",
            thread="pi",
            mode="agent",
        )

        decoded = json.loads(payload)
        self.assertEqual(decoded["prompt"], "Question?")
        self.assertEqual(decoded["response"], "Answer.")
        self.assertEqual(decoded["thread"], "pi")
        self.assertEqual(decoded["mode"], "agent")


if __name__ == "__main__":
    unittest.main()
