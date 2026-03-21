"""
Base LLM agent — the foundation for all LLM-powered stages.

Every stage that talks to an LLM (reward design, task intake, code review)
inherits from this. It handles the boring stuff: retries, logging, error
handling, token tracking.

Usage::

    from robosmith.agents.base import BaseAgent
    from robosmith.config import LLMConfig

    agent = BaseAgent(LLMConfig())
    response = agent.chat("What RL algorithm should I use for locomotion?")
    print(response)  # "PPO is generally the best starting point for..."

    # With a system prompt
    agent = BaseAgent(LLMConfig(), system_prompt="You are a robotics expert.")
    response = agent.chat("Design a reward for picking up a cube.")

    # Structured output (returns parsed dict/list)
    result = agent.chat_json("Return a JSON list of 3 reward component names.")
"""

from __future__ import annotations

import json
import time
import litellm
from typing import Any

from loguru import logger

from robosmith.config import LLMConfig


class BaseAgent:
    """
    Thin wrapper around LiteLLM for consistent LLM access.

    Subclass this for specialized agents (RewardAgent, ReviewAgent, etc.)
    that add domain-specific system prompts and parsing logic.
    """

    def __init__(
        self,
        config: LLMConfig,
        system_prompt: str = "",
        use_fast_model: bool = False,
    ) -> None:
        self.config = config
        self.system_prompt = system_prompt
        self.model = config.fast_model if use_fast_model else config.model

        # Track usage across calls
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def chat(self, user_message: str, temperature: float | None = None) -> str:
        """
        Send a message to the LLM and return the text response.
        """
        
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_message})

        temp = temperature if temperature is not None else self.config.temperature

        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.debug(f"LLM call (attempt {attempt}): model={self.model}, len={len(user_message)}")
                start = time.time()
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                )

                elapsed = time.time() - start
                text = response.choices[0].message.content or ""

                # Track tokens
                usage = response.usage
                if usage:
                    self.total_input_tokens += usage.prompt_tokens
                    self.total_output_tokens += usage.completion_tokens

                self.total_calls += 1
                logger.debug(f"LLM response in {elapsed:.1f}s — {len(text)} chars")

                return text

            except Exception as e:
                last_error = e
                error_str = str(e)

                # User-friendly error messages for common issues
                if "API Key" in error_str or "AuthenticationError" in error_str:
                    friendly = "Missing API key. Set ANTHROPIC_API_KEY (or OPENAI_API_KEY) in your environment."
                elif "RateLimitError" in error_str or "429" in error_str:
                    friendly = "Rate limited by LLM provider. Waiting before retry..."
                elif "timeout" in error_str.lower() or "ConnectionError" in error_str:
                    friendly = "Connection to LLM provider timed out."
                else:
                    friendly = error_str[:120]

                logger.warning(f"LLM call failed (attempt {attempt}/{self.config.max_retries}): {friendly}")
                if attempt < self.config.max_retries:
                    # Longer backoff for rate limits (15s, 30s, 60s)
                    if "RateLimitError" in error_str or "429" in error_str:
                        wait = 15 * attempt
                        logger.info(f"Rate limited — waiting {wait}s before retry...")
                        time.sleep(wait)
                    else:
                        time.sleep(2 ** attempt)  # Exponential backoff

        # Final error — also friendly
        error_str = str(last_error)
        if "API Key" in error_str or "AuthenticationError" in error_str:
            raise RuntimeError(
                "No API key found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY:\n"
                "  export ANTHROPIC_API_KEY='sk-ant-...'"
            )
        raise RuntimeError(f"LLM call failed after {self.config.max_retries} attempts: {last_error}")

    def chat_json(self, user_message: str, temperature: float | None = None) -> Any:
        """
        Send a message and parse the response as JSON.

        Appends an instruction to return valid JSON. Strips markdown
        code fences if the LLM wraps the response in them.
        """
        prompt = user_message + "\n\nRespond with valid JSON only. No explanation, no markdown."
        raw = self.chat(prompt, temperature=temperature)

        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove first line (```json or ```) and last line (```)
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}\nRaw: {raw[:200]}")
            raise ValueError(f"LLM returned invalid JSON: {e}") from e

    def usage_summary(self) -> dict:
        """Return a summary of token usage across all calls."""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }