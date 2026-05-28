from __future__ import annotations

import subprocess
from pathlib import Path

from agent_infra.harness.adapters import get_adapter_registry
from agent_infra.memory.working import get_working_memory


def build_status_payload(root: Path) -> dict:
    wm = get_working_memory()
    return {
        "schema_version": "jarvis.hud-status.v1",
        "context": {
            "harness": "jarvis-cli",
            "repo": root.name,
            "branch": _git_output(root, ["git", "branch", "--show-current"]),
            "worktree": str(root),
            "session_id": wm.session_id,
            "context_pressure": round(wm.token_count / max(wm.max_tokens, 1), 4),
        },
        "checks": {
            "working_memory_messages": wm.message_count,
            "working_memory_tokens": wm.token_count,
        },
        "risk": {
            "dirty_worktree": bool(_git_output(root, ["git", "status", "--short"])),
        },
        "sessionControls": {
            "supported": ["create", "resume", "status", "diff"],
            "adapters": [adapter.name for adapter in get_adapter_registry().list_adapters()],
        },
    }


def _git_output(root: Path, command: list[str]) -> str:
    try:
        proc = subprocess.run(
            command,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        return ""
    return ""
