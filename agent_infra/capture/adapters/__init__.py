from agent_infra.capture.adapters.base import BaseAdapter
from agent_infra.capture.adapters.clipboard import ClipboardAdapter
from agent_infra.capture.adapters.claude_code import ClaudeCodeAdapter
from agent_infra.capture.adapters.cli_adapter import CLIAdapter
from agent_infra.capture.adapters.files import FileAdapter

__all__ = [
    "BaseAdapter",
    "ClipboardAdapter",
    "ClaudeCodeAdapter",
    "CLIAdapter",
    "FileAdapter",
]
