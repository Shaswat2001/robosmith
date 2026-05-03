"""
Reward agent — generates reward function Python code via LLM.

This is the Eureka-style core of RoboSmith. Given a task
description and environment info, the LLM writes executable Python
reward functions that can be plugged into an RL training loop.
"""

from __future__ import annotations

from robosmith._logging import logger

from .prompt import REWARD_SYSTEM_PROMPT
from .types import RewardCandidate
from robosmith.config import LLMConfig
from robosmith.agent.models.base import BaseAgent
from robosmith.stages.reward_design.analysis import analyzeRewardCode

class RewardAgent(BaseAgent):
    """
    LLM agent specialized for generating reward function code.

    Subclasses BaseAgent with a domain-specific system prompt and
    methods for generating, validating, and evolving reward candidates.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config, system_prompt=REWARD_SYSTEM_PROMPT)

    def generate(
        self,
        task_description: str,
        obs_space_info: str,
        action_space_info: str,
        num_candidates: int = 4,
        env_code_context: str = "",
        literature_context: str = "",
    ) -> list[RewardCandidate]:
        """
        Generate multiple reward function candidates for a task.

        Args:
            task_description: What the robot should do.
            obs_space_info: Description of the observation space.
            action_space_info: Description of the action space.
            num_candidates: How many candidates to generate.
            env_code_context: Optional environment source code for richer context.
            literature_context: Summary of relevant papers from literature scout.

        Returns:
            List of RewardCandidate objects (validated, ready to evaluate).
        """
        prompt = self._build_generation_prompt(
            task_description, obs_space_info, action_space_info,
            env_code_context, literature_context,
        )

        candidates = []
        for i in range(num_candidates):
            logger.info(f"Generating reward candidate {i + 1}/{num_candidates}")

            # Use slightly varied temperature for diversity across candidates
            # Cap at 1.0 — above that, code quality drops sharply
            temp = 0.7 + (i * 0.05)
            temp = min(temp, 1.0)

            raw_code = self.chat(prompt, temperature=temp)
            code = self._clean_code(raw_code)

            candidate = RewardCandidate(
                code=code,
                candidate_id=i,
                generation=0,
                analysis=analyzeRewardCode(code, obs_space_info),
            )

            if candidate.is_valid():
                candidates.append(candidate)
                logger.info(f"Candidate {i} — valid")
            else:
                logger.warning(f"Candidate {i} — invalid: {candidate.error}")

        logger.info(f"Generated {len(candidates)}/{num_candidates} valid candidates")
        return candidates

    def evolve(
        self,
        task_description: str,
        obs_space_info: str,
        action_space_info: str,
        previous_best: RewardCandidate,
        training_feedback: str,
        generation: int = 1,
        num_candidates: int = 4,
    ) -> list[RewardCandidate]:
        """
        Generate improved candidates based on feedback from training.

        This is the evolutionary reflection step — the LLM sees what
        happened when we trained with the previous reward and writes
        better versions.
        """
        prompt = self._build_evolution_prompt(
            task_description, obs_space_info, action_space_info,
            previous_best, training_feedback,
        )

        candidates = []
        for i in range(num_candidates):
            logger.info(f"Evolving reward candidate {i + 1}/{num_candidates} (gen {generation})")

            raw_code = self.chat(prompt, temperature=0.8)
            code = self._clean_code(raw_code)

            candidate = RewardCandidate(
                code=code,
                candidate_id=i,
                generation=generation,
                analysis=analyzeRewardCode(code, obs_space_info),
            )

            if candidate.is_valid():
                candidates.append(candidate)
            else:
                logger.warning(f"Evolved candidate {i} — invalid: {candidate.error}")

        logger.info(f"Evolved {len(candidates)}/{num_candidates} valid candidates (gen {generation})")
        return candidates

    # ── Prompt construction ──

    def _build_generation_prompt(
        self,
        task_description: str,
        obs_space_info: str,
        action_space_info: str,
        env_code_context: str,
        literature_context: str = "",
    ) -> str:
        prompt = f"""Write a reward function for this robot learning task.

                    TASK: {task_description}

                    OBSERVATION SPACE: {obs_space_info}

                    ACTION SPACE: {action_space_info}
                    """
        if literature_context:
            prompt += f"\nRELEVANT PRIOR WORK (use these insights to inform your reward design):\n{literature_context}\n"

        if env_code_context:
            prompt += f"\nENVIRONMENT CODE (for context):\n{env_code_context}\n"

        prompt += """
                    Write a single Python function compute_reward(obs, action, next_obs, info) that returns (float, dict).
                    Decompose the reward into named components. Use only numpy.
                    Return ONLY the function code, no explanation."""

        return prompt

    def _build_evolution_prompt(
        self,
        task_description: str,
        obs_space_info: str,
        action_space_info: str,
        previous_best: RewardCandidate,
        training_feedback: str,
    ) -> str:
        return f"""Improve this reward function based on training feedback.

                TASK: {task_description}

                OBSERVATION SPACE: {obs_space_info}
                ACTION SPACE: {action_space_info}

                PREVIOUS REWARD FUNCTION:
                {previous_best.code}

                TRAINING FEEDBACK:
                {training_feedback}

                Write an improved compute_reward function. Fix the issues described in the feedback.
                Keep the same signature: compute_reward(obs, action, next_obs, info) -> tuple[float, dict].
                Return ONLY the function code, no explanation."""

    # ── Code cleaning ──1
    @staticmethod
    def _clean_code(raw: str) -> str:
        """Strip markdown fences and whitespace from LLM output."""
        code = raw.strip()

        # Remove markdown code fences
        if code.startswith("```"):
            lines = code.split("\n")
            # Drop first line (```python or ```)
            lines = lines[1:]
            # Drop last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        return code.strip()
