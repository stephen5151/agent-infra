import unittest

from agent_infra.harness.adapters import get_adapter_registry


class HarnessAdapterTests(unittest.TestCase):
    def test_registry_contains_current_input_surfaces(self) -> None:
        registry = get_adapter_registry()
        adapters = {adapter.name: adapter for adapter in registry.list_adapters()}

        self.assertIn("jarvis-cli", adapters)
        self.assertIn("jarvis-chat", adapters)
        self.assertIn("pi-cli-bridge", adapters)
        self.assertIn("text", adapters["jarvis-cli"].output_modes)
