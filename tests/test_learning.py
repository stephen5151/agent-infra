import unittest

from agent_infra.intelligence.learning import derive_skills_from_events


class LearningTests(unittest.TestCase):
    def test_repeated_cli_commands_become_skill_candidates(self) -> None:
        events = [
            {"id": "1", "source": "cli", "type": "action", "content": "pytest tests/test_a.py"},
            {"id": "2", "source": "cli", "type": "action", "content": "pytest tests/test_b.py"},
            {"id": "3", "source": "cli", "type": "action", "content": "git status"},
        ]

        skills = derive_skills_from_events(events, min_repetitions=2)

        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].name, "cli_pytest_workflow")
        self.assertIn("pytest", skills[0].trigger_patterns)
