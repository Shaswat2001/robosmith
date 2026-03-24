"""
Decision agent — makes intelligent pipeline decisions using LLM reasoning.

Instead of hardcoded rules (success_rate > 0.5 → refine_reward), this agent
looks at the full context (eval results, training curves, reward components,
task description) and reasons about what to do next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from robosmith.agents.base import BaseAgent
from robosmith.config import Decision, LLMConfig

DECISION_SYSTEM_PROMPT = """\
You are an expert RL researcher making decisions about a robot training pipeline.

Given evaluation results, training curves, and task context, you must decide:
1. ACCEPT — the policy meets the success criteria, ship it
2. REFINE_REWARD — the reward function needs adjustment
3. SWITCH_ALGO — the RL algorithm isn't working, try a different one
4. PIVOT — fundamental approach isn't working, need to rethink

You must respond with valid JSON:
{
    "decision": "accept" | "refine_reward" | "switch_algo" | "pivot",
    "reasoning": "1-2 sentence explanation of why",
    "suggestions": ["specific actionable suggestion 1", "suggestion 2"],
    "confidence": 0.0-1.0
}

Be specific in your suggestions — don't just say "improve the reward".
Say exactly what component to change and how.
"""

@dataclass
class PipelineDecision:
    """A decision about what to do next in the pipeline."""

    action: Decision
    reasoning: str = ""
    suggestions: list[str] = field(default_factory=list)
    confidence: float = 0.5

class DecisionAgent:
    """LLM-based decision agent for intelligent pipeline iteration."""

    def __init__(self, config: LLMConfig) -> None:
        self._agent = BaseAgent(
            config,
            system_prompt=DECISION_SYSTEM_PROMPT,
            use_fast_model=True,  # Decisions don't need the expensive model
        )

    def decide(
        self,
        eval_report: Any = None,
        training_result: Any = None,
        task_spec: Any = None,
        reward_code: str = "",
        iteration: int = 1,
        max_iterations: int = 3,
    ) -> PipelineDecision:
        """
        Make an intelligent decision about what to do next.

        Args:
            eval_report: Evaluation results (EvalReport).
            training_result: Training metrics (TrainingResult).
            task_spec: The task specification.
            reward_code: The current reward function code.
            iteration: Current iteration number.
            max_iterations: Max allowed iterations.

        Returns:
            PipelineDecision with action, reasoning, and suggestions.
        """
        prompt = self._build_prompt(
            eval_report, training_result, task_spec,
            reward_code, iteration, max_iterations,
        )

        try:
            result = self._agent.chat_json(prompt, temperature=0.3)

            action_str = result.get("decision", "refine_reward").lower()
            action_map = {
                "accept": Decision.ACCEPT,
                "refine_reward": Decision.REFINE_REWARD,
                "switch_algo": Decision.SWITCH_ALGO,
                "pivot": Decision.REFINE_REWARD,  # Pivot maps to refine for now
            }
            action = action_map.get(action_str, Decision.REFINE_REWARD)

            return PipelineDecision(
                action=action,
                reasoning=result.get("reasoning", ""),
                suggestions=result.get("suggestions", []),
                confidence=float(result.get("confidence", 0.5)),
            )

        except Exception as e:
            logger.warning(f"Decision agent failed, using rule-based fallback: {e}")
            return self._rule_based_fallback(eval_report, training_result)

    def _build_prompt(
        self,
        eval_report, training_result, task_spec,
        reward_code, iteration, max_iterations,
    ) -> str:
        lines = [f"TASK: {task_spec.task_description if task_spec else 'unknown'}"]
        lines.append(f"ITERATION: {iteration}/{max_iterations}")
        lines.append("")

        if eval_report:
            lines.append("EVALUATION RESULTS:")
            lines.append(f"  Success rate: {eval_report.success_rate:.0%}")
            lines.append(f"  Mean reward: {eval_report.mean_reward:.2f} (±{eval_report.std_reward:.2f})")
            lines.append(f"  Mean episode length: {eval_report.mean_episode_length:.0f}")
            lines.append(f"  Best: {eval_report.best_reward:.2f}, Worst: {eval_report.worst_reward:.2f}")
            if eval_report.criteria_results:
                lines.append("  Criteria:")
                for name, result in eval_report.criteria_results.items():
                    passed = result.get("passed")
                    val = result.get("value")
                    lines.append(f"    {name}: {'PASS' if passed else 'FAIL'} (value={val})")
            lines.append("")

        if training_result and training_result.metrics_history:
            history = training_result.metrics_history
            rewards = [h.get("mean_reward", 0) for h in history]
            lines.append("TRAINING CURVE:")
            lines.append(f"  Algorithm: {training_result.algorithm}")
            lines.append(f"  Timesteps: {training_result.total_timesteps:,}")
            lines.append(f"  Reward trajectory: {' → '.join(f'{r:.1f}' for r in rewards[::max(1, len(rewards)//5)])}")
            lines.append(f"  Start: {rewards[0]:.2f}, Final: {rewards[-1]:.2f}")

            # Detect patterns
            if len(rewards) >= 3:
                improving = rewards[-1] > rewards[0]
                flat = max(rewards[-3:]) - min(rewards[-3:]) < 0.5
                if flat:
                    lines.append("  PATTERN: Reward is FLAT (not learning)")
                elif improving:
                    lines.append("  PATTERN: Reward is IMPROVING")
                else:
                    lines.append("  PATTERN: Reward is DECREASING")
            lines.append("")

        if reward_code:
            lines.append("CURRENT REWARD FUNCTION (first 300 chars):")
            lines.append(reward_code[:300])
            lines.append("")

        lines.append("What should we do next? Respond with JSON.")
        return "\n".join(lines)

    @staticmethod
    def _rule_based_fallback(eval_report, training_result) -> PipelineDecision:
        """Fallback to rule-based decision if LLM fails."""
        if eval_report is None:
            return PipelineDecision(action=Decision.REFINE_REWARD, reasoning="No eval data")

        if eval_report.success_rate >= 0.8:
            return PipelineDecision(action=Decision.ACCEPT, reasoning="High success rate")

        if eval_report.mean_reward <= 0 and eval_report.mean_episode_length < 20:
            return PipelineDecision(
                action=Decision.SWITCH_ALGO,
                reasoning="Very low reward and short episodes",
            )

        return PipelineDecision(
            action=Decision.REFINE_REWARD,
            reasoning=f"Success rate {eval_report.success_rate:.0%} below threshold",
        )