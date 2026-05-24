from __future__ import annotations

import json
from typing import Iterable


class PiInputError(ValueError):
    """Raised when the Pi CLI bridge receives no usable input."""


def resolve_prompt(args: Iterable[str], stdin_text: str) -> str:
    prompt = " ".join(part for part in args if part).strip()
    if prompt:
        return prompt

    prompt = stdin_text.strip()
    if prompt:
        return prompt

    raise PiInputError("No input received. Pass a prompt argument or pipe text to stdin.")


def build_json_payload(*, prompt: str, response: str, thread: str, mode: str) -> str:
    return json.dumps(
        {
            "prompt": prompt,
            "response": response,
            "thread": thread,
            "mode": mode,
        },
        ensure_ascii=False,
    )


def run_agent_prompt(
    prompt: str,
    *,
    thread: str,
    model: str,
    code_mode: bool,
    hitl: bool,
) -> str:
    from agent_infra.agent import build_agent, run
    from agent_infra.patterns.hitl import TriggerMode

    hitl_mode: TriggerMode = "conditional" if hitl else "sampling"
    agent = build_agent(model=model, code_mode=code_mode, hitl_mode=hitl_mode)
    return run(agent, prompt, thread_id=thread)


async def run_ask_prompt(prompt: str, *, local: bool) -> str:
    from agent_infra.config.settings import get_settings
    from agent_infra.intelligence.llm_router import LLMRouter
    from agent_infra.memory.semantic import get_semantic_memory

    cfg = get_settings()
    router = LLMRouter()
    sem = get_semantic_memory()
    context = sem.format_context(prompt, n_results=3)

    system = (
        f"你是 {cfg.identity.name}，{cfg.identity.owner} 的个人助手。"
        f"风格：{cfg.identity.persona.style}，{cfg.identity.persona.tone}。"
        f"用中文回答。"
    )
    human = f"{prompt}\n\n{context}" if context else prompt
    return await router.agenerate(human=human, system=system, force_local=local)
