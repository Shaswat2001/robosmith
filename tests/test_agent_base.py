"""Tests for forge.agents.base — LLM agent foundation."""

from unittest.mock import MagicMock, patch

import pytest

from forge.agents.base import BaseAgent
from forge.config import LLMConfig


def _mock_response(text: str, input_tokens: int = 10, output_tokens: int = 20):
    """Create a mock LiteLLM response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage = MagicMock()
    response.usage.prompt_tokens = input_tokens
    response.usage.completion_tokens = output_tokens
    return response


class TestBaseAgentChat:
    @patch("litellm.completion")
    def test_basic_chat(self, mock_completion):
        mock_completion.return_value = _mock_response("Hello, I am a robot expert.")

        agent = BaseAgent(LLMConfig())
        result = agent.chat("What is RL?")

        assert result == "Hello, I am a robot expert."
        assert agent.total_calls == 1
        mock_completion.assert_called_once()

    @patch("litellm.completion")
    def test_system_prompt_included(self, mock_completion):
        mock_completion.return_value = _mock_response("PPO is best.")

        agent = BaseAgent(LLMConfig(), system_prompt="You are a robotics expert.")
        agent.chat("What algorithm?")

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a robotics expert."
        assert messages[1]["role"] == "user"

    @patch("litellm.completion")
    def test_fast_model_used(self, mock_completion):
        mock_completion.return_value = _mock_response("Quick answer.")

        config = LLMConfig(model="big-model", fast_model="small-model")
        agent = BaseAgent(config, use_fast_model=True)
        agent.chat("Quick question")

        call_args = mock_completion.call_args
        assert call_args.kwargs["model"] == "small-model"

    @patch("litellm.completion")
    def test_token_tracking(self, mock_completion):
        mock_completion.return_value = _mock_response("Answer", input_tokens=50, output_tokens=100)

        agent = BaseAgent(LLMConfig())
        agent.chat("Question 1")
        agent.chat("Question 2")

        usage = agent.usage_summary()
        assert usage["total_calls"] == 2
        assert usage["total_input_tokens"] == 100
        assert usage["total_output_tokens"] == 200

    @patch("litellm.completion")
    def test_temperature_override(self, mock_completion):
        mock_completion.return_value = _mock_response("Creative answer")

        agent = BaseAgent(LLMConfig(temperature=0.7))
        agent.chat("Be creative", temperature=1.5)

        call_args = mock_completion.call_args
        assert call_args.kwargs["temperature"] == 1.5


class TestBaseAgentRetry:
    @patch("litellm.completion")
    def test_retries_on_failure(self, mock_completion):
        mock_completion.side_effect = [
            Exception("Rate limited"),
            _mock_response("Success on retry"),
        ]

        agent = BaseAgent(LLMConfig(max_retries=3))
        result = agent.chat("Test")

        assert result == "Success on retry"
        assert mock_completion.call_count == 2

    @patch("litellm.completion")
    def test_raises_after_max_retries(self, mock_completion):
        mock_completion.side_effect = Exception("Always fails")

        agent = BaseAgent(LLMConfig(max_retries=2))
        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            agent.chat("Test")


class TestBaseAgentJSON:
    @patch("litellm.completion")
    def test_parse_json_dict(self, mock_completion):
        mock_completion.return_value = _mock_response('{"reward": "distance_to_goal"}')

        agent = BaseAgent(LLMConfig())
        result = agent.chat_json("Give me a reward component")

        assert isinstance(result, dict)
        assert result["reward"] == "distance_to_goal"

    @patch("litellm.completion")
    def test_parse_json_list(self, mock_completion):
        mock_completion.return_value = _mock_response('["distance", "grasp", "safety"]')

        agent = BaseAgent(LLMConfig())
        result = agent.chat_json("List 3 reward components")

        assert isinstance(result, list)
        assert len(result) == 3

    @patch("litellm.completion")
    def test_strips_markdown_fences(self, mock_completion):
        mock_completion.return_value = _mock_response('```json\n{"key": "value"}\n```')

        agent = BaseAgent(LLMConfig())
        result = agent.chat_json("Give me JSON")

        assert result == {"key": "value"}

    @patch("litellm.completion")
    def test_invalid_json_raises(self, mock_completion):
        mock_completion.return_value = _mock_response("This is not JSON at all")

        agent = BaseAgent(LLMConfig())
        with pytest.raises(ValueError, match="invalid JSON"):
            agent.chat_json("Give me JSON")