"""LLM factory — single place to swap models or add caching."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel


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
    # Prompt caching: inject cache_control on model_kwargs so the API
    # caches the system prompt block across calls with the same prefix.
    if cache:
        kwargs["model_kwargs"] = {
            "extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"},
        }
    return ChatAnthropic(**kwargs)


def get_structured_llm(schema: type[Any], **kwargs: Any) -> Any:
    """Return an LLM bound to a Pydantic output schema via with_structured_output."""
    return get_llm(**kwargs).with_structured_output(schema)
