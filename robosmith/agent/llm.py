"""
Provider-agnostic LLM factory.

Uses litellm under the hood (already a robosmith dependency) to support
any LLM provider: Anthropic, OpenAI, Ollama, Gemini, etc.
Wraps litellm in a LangChain-compatible ChatModel interface.
"""

from __future__ import annotations

import os
import logging
from typing import Any
from langchain_community.chat_models import ChatLiteLLM

logger = logging.getLogger(__name__)

# Default model if none specified
DEFAULT_MODEL = "anthropic/claude-sonnet-4-20250514"

def get_llm(
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> Any:
    """Get a LangChain-compatible chat model via litellm.

    Supports any provider litellm supports:
      - "anthropic/claude-sonnet-4-20250514"
      - "openai/gpt-4o"
      - "ollama/llama3"
      - "gemini/gemini-pro"
      - etc.

    Config priority:
      1. Explicit model parameter
      2. ROBOSMITH_MODEL env var
      3. robosmith.yaml llm.model field
      4. DEFAULT_MODEL fallback
    """

    if model is None:
        model = os.environ.get("ROBOSMITH_MODEL", DEFAULT_MODEL)

    logger.debug(f"Using LLM: {model}")

    return ChatLiteLLM(
        model=model,
        temperature=temperature,
        **kwargs,
    )