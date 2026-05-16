"""
Output formatting utilities for Agent responses.
Provides helpers for rendering agent results to different targets:
  - terminal (rich)
  - plain text
  - markdown
"""
from __future__ import annotations

from typing import Any


def format_response(response: str, format: str = "plain") -> str:
    """
    Format an agent response for output.

    Args:
        response: Raw response string from the agent
        format:   "plain" | "markdown" | "rich"

    Returns:
        Formatted string ready for display
    """
    if format == "plain":
        return response.strip()
    if format == "markdown":
        return response.strip()
    if format == "rich":
        # Escape rich markup characters that aren't intentional
        return response.replace("[", r"\[")
    return response


def truncate(text: str, max_chars: int = 2000, suffix: str = "...") -> str:
    """Truncate long text to max_chars, adding suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """
    Extract all Markdown code blocks from text.

    Returns:
        List of (language, code) tuples
    """
    import re
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(lang.lower(), code.strip()) for lang, code in matches if code.strip()]


def render_plan(plan: list[dict[str, Any]]) -> str:
    """Render a plan (list of PlanStep dicts) as a numbered markdown list."""
    lines = []
    for step in plan:
        status_icon = {"done": "✅", "in_progress": "🔄", "pending": "⏳", "failed": "❌"}.get(
            step.get("status", "pending"), "⏳"
        )
        lines.append(f"{step.get('step', 0) + 1}. {status_icon} {step.get('task', '')}")
    return "\n".join(lines)


__all__ = [
    "format_response",
    "truncate",
    "extract_code_blocks",
    "render_plan",
]
