"""LLM factory — single place to swap models or add caching."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage


@lru_cache(maxsize=8)
def get_llm(
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    cache: bool = True,
) -> BaseChatModel:
    """
    Return a cached ChatAnthropic instance.

    Args:
        model:       Claude model ID
        temperature: Sampling temperature
        max_tokens:  Max output tokens
        cache:       Enable prompt caching (reduces cost ~90% for repeated system prompts)
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY"),
    }
    return ChatAnthropic(**kwargs)


def cached_system(text: str) -> SystemMessage:
    """
    Return a SystemMessage with cache_control ephemeral set on the text block.

    Anthropic prompt caching requires the cache_control marker inside the
    message content block — the beta header alone is not enough.
    Wrapping every static system prompt with this helper ensures the prefix
    is cached across calls and input tokens are billed at the cached rate.
    """
    return SystemMessage(content=[
        {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}},
    ])


def get_structured_llm(schema: type[Any], **kwargs: Any) -> Any:
    """Return an LLM bound to a Pydantic output schema via with_structured_output."""
    return get_llm(**kwargs).with_structured_output(schema)
