"""
Stage 4: Reward design.

The evolutionary loop:
  1. Create the environment, extract obs/action space info
  2. Ask the RewardAgent to generate K candidate reward functions
  3. Evaluate each candidate by running N episodes with random actions
  4. Score candidates by total reward and episode length
  5. Optionally: evolve — feed the best + feedback to the LLM for improvement

The "evaluate with random actions" step is a quick sanity check, not
real RL training. It answers: "does this reward function produce a
signal at all, or is it always zero / NaN / crashing?"

Real RL training happens in Stage 5. This stage just finds a reward
function worth training with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from robosmith.agents.reward_agent import RewardAgent, RewardCandidate
from robosmith.config import LLMConfig, RewardSearchConfig, TaskSpec
from robosmith.envs.registry import EnvEntry
from robosmith.envs.wrapper import make_env

@dataclass
class EvalResult:
    """Result of evaluating one reward candidate on the environment."""

    candidate_id: int
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    num_episodes: int
    had_errors: bool = False
    error_message: str = ""


@dataclass
class RewardDesignResult:
    """Output of the full reward design stage."""

    best_candidate: RewardCandidate
    all_candidates: list[RewardCandidate]
    eval_results: list[EvalResult]
    generations_run: int

def extract_space_info(env) -> tuple[str, str]:  # noqa: ANN001
    """
    Extract human-readable observation and action space descriptions
    from a gymnasium environment. This is what the LLM sees as context.
    """
    obs_space = env.observation_space
    act_space = env.action_space

    # Observation space
    if hasattr(obs_space, "shape"):
        obs_info = f"{type(obs_space).__name__}(shape={obs_space.shape}, dtype={obs_space.dtype})"
        if hasattr(obs_space, "low") and hasattr(obs_space, "high"):
            low = np.min(obs_space.low)
            high = np.max(obs_space.high)
            obs_info += f" range=[{low:.1f}, {high:.1f}]"
    elif hasattr(obs_space, "spaces"):
        # Dict observation space (e.g. Fetch envs)
        parts = []
        for key, space in obs_space.spaces.items():
            parts.append(f"{key}: {type(space).__name__}({getattr(space, 'shape', '?')})")
        obs_info = "Dict(" + ", ".join(parts) + ")"
    else:
        obs_info = str(obs_space)

    # Action space
    if hasattr(act_space, "shape"):
        act_info = f"{type(act_space).__name__}(shape={act_space.shape}, dtype={act_space.dtype})"
        if hasattr(act_space, "low") and hasattr(act_space, "high"):
            act_info += f" range=[{act_space.low.min():.1f}, {act_space.high.max():.1f}]"
    else:
        act_info = str(act_space)

    return obs_info, act_info

def evaluate_candidate(
    candidate: RewardCandidate,
    env_entry: EnvEntry,
    num_episodes: int = 5,
    max_steps_per_episode: int = 200,
) -> EvalResult:
    """
    Evaluate a reward candidate by running episodes with random actions.

    This is a quick sanity check — does the reward function:
    - Run without crashing?
    - Produce non-zero values?
    - Produce finite (not NaN/Inf) values?
    """
    try:
        reward_fn = candidate.get_function()
    except Exception as e:
        return EvalResult(
            candidate_id=candidate.candidate_id,
            mean_reward=float("-inf"),
            std_reward=0.0,
            mean_episode_length=0.0,
            num_episodes=0,
            had_errors=True,
            error_message=f"Failed to compile: {e}",
        )

    try:
        env = make_env(env_entry)
    except Exception as e:
        return EvalResult(
            candidate_id=candidate.candidate_id,
            mean_reward=float("-inf"),
            std_reward=0.0,
            mean_episode_length=0.0,
            num_episodes=0,
            had_errors=True,
            error_message=f"Failed to create env: {e}",
        )

    episode_rewards = []
    episode_lengths = []
    errors = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        had_error = False

        for step in range(max_steps_per_episode):
            action = env.action_space.sample()
            next_obs, _env_reward, terminated, truncated, info = env.step(action)

            # Flatten dict observations (Fetch envs return dicts)
            obs_array = _flatten_obs(obs)
            next_obs_array = _flatten_obs(next_obs)

            try:
                reward, components = reward_fn(obs_array, action, next_obs_array, info)

                # Check for bad values
                if not np.isfinite(reward):
                    had_error = True
                    errors.append(f"Ep {ep} step {step}: reward is {reward}")
                    break

                total_reward += reward
            except Exception as e:
                had_error = True
                errors.append(f"Ep {ep} step {step}: {e}")
                break

            obs = next_obs
            steps += 1

            if terminated or truncated:
                break

        if not had_error:
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

    env.close()

    if not episode_rewards:
        return EvalResult(
            candidate_id=candidate.candidate_id,
            mean_reward=float("-inf"),
            std_reward=0.0,
            mean_episode_length=0.0,
            num_episodes=0,
            had_errors=True,
            error_message="; ".join(errors[:3]),
        )

    return EvalResult(
        candidate_id=candidate.candidate_id,
        mean_reward=float(np.mean(episode_rewards)),
        std_reward=float(np.std(episode_rewards)),
        mean_episode_length=float(np.mean(episode_lengths)),
        num_episodes=len(episode_rewards),
        had_errors=len(errors) > 0,
        error_message="; ".join(errors[:3]) if errors else "",
    )


def run_reward_design(
    task_spec: TaskSpec,
    env_entry: EnvEntry,
    llm_config: LLMConfig,
    search_config: RewardSearchConfig | None = None,
    num_candidates: int = 4,
    num_eval_episodes: int = 5,
    literature_context: str = "",
) -> RewardDesignResult:
    """
    Run the full evolutionary reward design stage.

    The Eureka-style loop:
      1. Generate K candidates via LLM
      2. Evaluate each in the sim (short rollouts with random actions)
      3. Pick the best, format training feedback
      4. Feed best + feedback to LLM → generate improved candidates
      5. Repeat for N iterations
    """
    config = search_config or RewardSearchConfig()
    num_iterations = config.num_iterations

    # Step 1: Extract space info from the environment
    logger.info(f"Extracting space info from {env_entry.env_id}")
    env = make_env(env_entry)
    obs_info, act_info = extract_space_info(env)
    env.close()

    logger.info(f"Obs space: {obs_info}")
    logger.info(f"Act space: {act_info}")

    agent = RewardAgent(llm_config)
    all_candidates: list[RewardCandidate] = []
    all_eval_results: list[EvalResult] = []
    global_best: RewardCandidate | None = None

    for gen in range(num_iterations):
        logger.info(f"Reward evolution — generation {gen + 1}/{num_iterations}")

        # Step 2: Generate candidates
        if gen == 0:
            # First generation: generate from scratch with literature context
            candidates = agent.generate(
                task_description=task_spec.task_description,
                obs_space_info=obs_info,
                action_space_info=act_info,
                num_candidates=num_candidates,
                literature_context=literature_context
            )
        else:
            # Later generations: evolve from the best so far
            feedback = _format_feedback(global_best, all_eval_results[-num_candidates:])
            candidates = agent.evolve(
                task_description=task_spec.task_description,
                obs_space_info=obs_info,
                action_space_info=act_info,
                previous_best=global_best,
                training_feedback=feedback,
                generation=gen,
                num_candidates=num_candidates,
            )

        if not candidates:
            logger.warning(f"Generation {gen + 1} produced zero valid candidates")
            if global_best is not None:
                break  # Use what we have
            raise RuntimeError("Reward agent generated zero valid candidates")

        # Step 3: Evaluate each candidate
        logger.info(f"Evaluating {len(candidates)} candidates ({num_eval_episodes} episodes each)")
        gen_eval_results = []
        for candidate in candidates:
            result = evaluate_candidate(
                candidate, env_entry,
                num_episodes=num_eval_episodes,
            )
            candidate.score = result.mean_reward
            candidate.metrics = {
                "mean_reward": result.mean_reward,
                "std_reward": result.std_reward,
                "mean_episode_length": result.mean_episode_length,
            }
            gen_eval_results.append(result)

            status = "error" if result.had_errors else f"reward={result.mean_reward:.2f}"
            logger.info(f"  Gen {gen + 1} candidate {candidate.candidate_id}: {status}")

        all_candidates.extend(candidates)
        all_eval_results.extend(gen_eval_results)

        # Step 4: Update global best
        valid = [c for c in candidates if c.score is not None and np.isfinite(c.score)]
        if valid:
            gen_best = max(valid, key=lambda c: c.score)
            if global_best is None or gen_best.score > global_best.score:
                improved = global_best is None or gen_best.score > global_best.score
                old_score = global_best.score if global_best else None
                global_best = gen_best
                _gens_without_improvement = 0
                logger.info(
                    f"Generation {gen + 1} best: {gen_best.score:.2f}"
                    + (f" (improved from {old_score:.2f})" if old_score is not None and improved else "")
                )
            else:
                _gens_without_improvement += 1
                logger.info(
                    f"Generation {gen + 1} best: {gen_best.score:.2f} "
                    f"(no improvement over {global_best.score:.2f})"
                )
                # Early termination: no improvement for 2+ generations
                if _gens_without_improvement >= 2 and gen < num_iterations - 1:
                    logger.info("Reward converged — stopping early (no improvement for 2 generations)")
                    break

    if global_best is None:
        raise RuntimeError("All reward candidates across all generations failed evaluation")

    logger.info(
        f"Reward design complete — best score: {global_best.score:.2f}, "
        f"generation: {global_best.generation}, "
        f"total candidates: {len(all_candidates)}"
    )

    return RewardDesignResult(
        best_candidate=global_best,
        all_candidates=all_candidates,
        eval_results=all_eval_results,
        generations_run=num_iterations,
    )

def _format_feedback(best: RewardCandidate, recent_evals: list[EvalResult]) -> str:
    """
    Format evaluation results as human-readable feedback for the LLM.

    This is the 'reward reflection' step from Eureka — telling the LLM
    what happened so it can improve.
    """
    lines = [
        f"Previous best reward function scored {best.score:.2f}.",
        "",
        "Evaluation results across candidates:",
    ]

    for ev in recent_evals:
        if ev.had_errors:
            lines.append(f"  Candidate {ev.candidate_id}: FAILED — {ev.error_message}")
        else:
            lines.append(
                f"  Candidate {ev.candidate_id}: "
                f"mean_reward={ev.mean_reward:.2f}, "
                f"std={ev.std_reward:.2f}, "
                f"ep_length={ev.mean_episode_length:.0f}"
            )

    if best.metrics:
        lines.append("")
        lines.append("Best candidate reward components:")
        for k, v in best.metrics.items():
            lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append(
        "Improve the reward function. Focus on: "
        "making the task reward signal stronger, "
        "ensuring reward components are well-scaled, "
        "and adding shaping terms if the agent isn't making progress."
    )

    return "\n".join(lines)

def _flatten_obs(obs) -> np.ndarray:  # noqa: ANN001
    """Flatten observation to a 1D numpy array. Handles dict obs (Fetch envs)."""
    if isinstance(obs, dict):
        arrays = [np.asarray(v).flatten() for v in obs.values()]
        return np.concatenate(arrays)
    return np.asarray(obs).flatten()
