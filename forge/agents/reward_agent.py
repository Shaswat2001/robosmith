"""
Reward agent — generates reward function Python code via LLM.

This is the Eureka-style core of Embodied Agent Forge. Given a task
description and environment info, the LLM writes executable Python
reward functions that can be plugged into an RL training loop.

The reward function signature is standardized::

    def compute_reward(obs, action, next_obs, info) -> tuple[float, dict]:
        # ... reward logic ...
        return total_reward, {"component1": val1, "component2": val2}
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from forge.agents.base import BaseAgent
from forge.config import LLMConfig

REWARD_SYSTEM_PROMPT = """\
You are an expert reinforcement learning reward engineer. Your job is to write \
Python reward functions for robot learning tasks.

RULES:
1. The function signature is ALWAYS:
   def compute_reward(obs, action, next_obs, info) -> tuple[float, dict]:

2. obs and next_obs are numpy arrays. action is a numpy array.
   info is a dict that may contain extra environment data.

3. Return a tuple of (total_reward, components_dict).
   The components_dict maps component names to their individual values.
   Example: return total_reward, {"distance": -dist, "grasp": grasp_bonus}

4. Import only numpy (as np). No other imports.

5. Write dense reward functions — not sparse. Give continuous feedback.

6. Decompose the reward into clear components:
   - task_reward: progress toward the goal
   - shaping_reward: helpful intermediate signals
   - safety_reward: penalties for dangerous states (optional)

7. Keep rewards well-scaled. Individual components should be roughly in [-1, 1].
   Use normalization or clipping if needed.

8. Return ONLY the Python function. No explanation, no markdown, no examples.
"""

@dataclass
class RewardCandidate:
    """A single generated reward function candidate."""

    code: str
    function_name: str = "compute_reward"
    candidate_id: int = 0
    generation: int = 0  # Which evolutionary iteration produced this

    # Filled in after evaluation
    score: float | None = None
    metrics: dict = field(default_factory=dict)
    error: str | None = None

    def is_valid(self) -> bool:
        """Check if the code compiles and has the right signature."""
        try:
            compiled = compile(self.code, f"<reward_{self.candidate_id}>", "exec")
            namespace: dict = {}
            exec(compiled, namespace)

            func = namespace.get(self.function_name)
            if func is None:
                self.error = f"Function '{self.function_name}' not found in generated code"
                return False

            if not callable(func):
                self.error = f"'{self.function_name}' is not callable"
                return False

            return True

        except SyntaxError as e:
            self.error = f"Syntax error: {e}"
            return False
        except Exception as e:
            self.error = f"Validation error: {e}"
            return False

    def get_function(self):  # noqa: ANN201
        """Compile and return the actual callable reward function."""
        namespace: dict = {"np": __import__("numpy")}
        exec(self.code, namespace)  # noqa: S102
        return namespace[self.function_name]

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
    ) -> list[RewardCandidate]:
        """
        Generate multiple reward function candidates for a task.
        """
        prompt = self._build_generation_prompt(
            task_description, obs_space_info, action_space_info, env_code_context
        )

        candidates = []
        for i in range(num_candidates):
            logger.info(f"Generating reward candidate {i + 1}/{num_candidates}")

            # Use higher temperature for diversity across candidates
            temp = 0.7 + (i * 0.1)  # 0.7, 0.8, 0.9, 1.0, ...
            temp = min(temp, 1.3)

            raw_code = self.chat(prompt, temperature=temp)
            code = self._clean_code(raw_code)

            candidate = RewardCandidate(
                code=code,
                candidate_id=i,
                generation=0,
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
            )

            if candidate.is_valid():
                candidates.append(candidate)
            else:
                logger.warning(f"Evolved candidate {i} — invalid: {candidate.error}")

        logger.info(f"Evolved {len(candidates)}/{num_candidates} valid candidates (gen {generation})")
        return candidates

    # Prompt construction
    def _build_generation_prompt(
        self,
        task_description: str,
        obs_space_info: str,
        action_space_info: str,
        env_code_context: str,
    ) -> str:
        prompt = f"""Write a reward function for this robot learning task.

TASK: {task_description}

OBSERVATION SPACE: {obs_space_info}

ACTION SPACE: {action_space_info}
"""
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

    # ── Code cleaning ──

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