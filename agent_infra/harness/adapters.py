from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HarnessAdapter:
    name: str
    input_modes: tuple[str, ...]
    output_modes: tuple[str, ...]
    supports_hooks: bool
    supports_sessions: bool


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, HarnessAdapter] = {}

    def register(self, adapter: HarnessAdapter) -> None:
        self._adapters[adapter.name] = adapter

    def list_adapters(self) -> list[HarnessAdapter]:
        return sorted(self._adapters.values(), key=lambda item: item.name)


_registry: AdapterRegistry | None = None


def get_adapter_registry() -> AdapterRegistry:
    global _registry
    if _registry is None:
        registry = AdapterRegistry()
        registry.register(HarnessAdapter("jarvis-cli", ("text",), ("text", "json"), False, True))
        registry.register(HarnessAdapter("jarvis-chat", ("text",), ("text",), False, True))
        registry.register(HarnessAdapter("pi-cli-bridge", ("text", "stdin"), ("text", "json"), False, True))
        _registry = registry
    return _registry
