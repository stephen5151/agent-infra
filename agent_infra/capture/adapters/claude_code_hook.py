"""
Claude Code Hook 脚本入口
==========================
Claude Code 在对话结束时通过 Stop hook 调用本脚本：
    python3 -m agent_infra.capture.adapters.claude_code_hook

脚本从 stdin 读取 JSON 格式的会话数据，追加到队列文件
(~/.jarvis/claude_code_queue.jsonl)，供 ClaudeCodeAdapter 异步消费。

安装方式（一次性）:
    jarvis install
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    queue_file = Path.home() / ".jarvis" / "claude_code_queue.jsonl"
    queue_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        data = {"raw": raw[:2000] if "raw" in dir() else ""}

    entry = {
        "session_id": os.environ.get("CLAUDE_SESSION_ID", ""),
        "transcript": data.get("transcript", raw[:3000] if raw else ""),
        "tool_calls": data.get("tool_use", []),
        "stop_reason": data.get("stop_reason", ""),
    }

    with open(queue_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
