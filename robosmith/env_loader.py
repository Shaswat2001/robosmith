"""
.env.local loader and LLM provider auto-detection.

Loads .env.local (or .env) from the current working directory and sets
the variables as environment variables. Then detects which LLM provider
is available based on which API keys are present.

Provider priority order:
  1. ANTHROPIC_API_KEY  → anthropic  (Claude models)
  2. OPENAI_API_KEY     → openai     (GPT models)
  3. GEMINI_API_KEY     → gemini     (Gemini models)
  4. GROQ_API_KEY       → groq       (Llama via Groq)
  5. OPENROUTER_API_KEY → openrouter (multi-provider gateway)
"""

from __future__ import annotations

import os
from pathlib import Path
from robosmith._logging import logger

# Default main/fast model per provider (litellm format: "provider/model")
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {
        "model":      "anthropic/claude-sonnet-4-6",
        "fast_model": "anthropic/claude-haiku-4-5-20251001",
    },
    "openai": {
        "model":      "openai/gpt-4o",
        "fast_model": "openai/gpt-4o-mini",
    },
    "gemini": {
        "model":      "gemini/gemini-2.0-flash",
        "fast_model": "gemini/gemini-2.0-flash",
    },
    "groq": {
        "model":      "groq/llama-3.3-70b-versatile",
        "fast_model": "groq/llama-3.1-8b-instant",
    },
    "openrouter": {
        "model":      "openrouter/anthropic/claude-sonnet-4-6",
        "fast_model": "openrouter/meta-llama/llama-3.1-8b-instruct:free",
    },
    "ollama": {
        "model":      "ollama/llama3",
        "fast_model": "ollama/llama3",
    },
}

# Maps env var name → provider name
_KEY_TO_PROVIDER: list[tuple[str, str]] = [
    ("ANTHROPIC_API_KEY",  "anthropic"),
    ("OPENAI_API_KEY",     "openai"),
    ("GEMINI_API_KEY",     "gemini"),
    ("GOOGLE_API_KEY",     "gemini"),
    ("GROQ_API_KEY",       "groq"),
    ("OPENROUTER_API_KEY", "openrouter"),
]

def load_env_local(paths: list[str] | None = None) -> dict[str, str]:
    """
    Load .env.local (and .env as fallback) into os.environ.

    Parses KEY=VALUE lines, strips quotes, ignores comments.
    Does NOT override already-set env vars (same semantics as dotenv).

    Args:
        paths: List of file paths to try, in order. Defaults to
               [".env.local", ".env"] in the current directory.

    Returns:
        Dict of keys that were newly loaded (for logging).
    """
    if paths is None:
        paths = [".env.local", ".env"]

    loaded: dict[str, str] = {}
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue

        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue

                key, _, raw_value = line.partition("=")
                key = key.strip()
                value = raw_value.strip()

                # Strip surrounding quotes (single or double)
                if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]

                # Don't override vars already in the environment
                if key and key not in os.environ:
                    os.environ[key] = value
                    loaded[key] = value

            logger.debug(f"Loaded {len(loaded)} vars from {path}")
            break  # stop after first found file
        except Exception as e:
            logger.debug(f"Could not read {path}: {e}")

    return loaded

def detect_provider() -> str | None:
    """
    Return the first provider whose API key is present in the environment.

    Returns None if no known key is found (caller should warn the user).
    """
    for env_var, provider in _KEY_TO_PROVIDER:
        if os.environ.get(env_var):
            return provider
    return None

def resolve_llm(
    llm_arg: str | None = None,
    *,
    config_model: str | None = None,
) -> tuple[str, str]:
    """
    Resolve the main model and fast_model strings to use.

    Priority:
      1. --llm CLI flag (provider name or full "provider/model" string)
      2. robosmith.yaml llm.model (config_model)
      3. ROBOSMITH_MODEL env var
      4. Auto-detect from available API key in environment

    Returns:
        (model, fast_model) — both in litellm "provider/model" format.
    """
    # Explicit CLI arg: accept "openai", "anthropic", or "openai/gpt-4o"
    if llm_arg:
        if "/" in llm_arg:
            # Full model string like "openai/gpt-4o"
            provider = llm_arg.split("/")[0]
            defaults = PROVIDER_DEFAULTS.get(provider, {})
            return llm_arg, defaults.get("fast_model", llm_arg)
        else:
            # Provider name like "openai"
            provider = llm_arg.lower()
            defaults = PROVIDER_DEFAULTS.get(provider)
            if defaults:
                return defaults["model"], defaults["fast_model"]
            logger.warning(f"Unknown provider '{provider}', falling back to auto-detect")

    # Config file / env var override
    if config_model:
        provider = config_model.split("/")[0] if "/" in config_model else None
        defaults = PROVIDER_DEFAULTS.get(provider or "", {})
        return config_model, defaults.get("fast_model", config_model)

    env_model = os.environ.get("ROBOSMITH_MODEL")
    if env_model:
        provider = env_model.split("/")[0] if "/" in env_model else None
        defaults = PROVIDER_DEFAULTS.get(provider or "", {})
        return env_model, defaults.get("fast_model", env_model)

    # Auto-detect from available keys
    provider = detect_provider()
    if provider:
        defaults = PROVIDER_DEFAULTS[provider]
        logger.debug(f"Auto-detected provider: {provider} ({defaults['model']})")
        return defaults["model"], defaults["fast_model"]

    # No key found — return anthropic default (will fail with a clear error at call time)
    logger.warning(
        "No LLM API key found in environment. "
        "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, or GROQ_API_KEY "
        "in .env.local or your shell environment."
    )
    defaults = PROVIDER_DEFAULTS["anthropic"]
    return defaults["model"], defaults["fast_model"]
