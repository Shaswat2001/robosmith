"""
Stage 4: Reward design.

The evolutionary loop:
  1. Create the environment, extract obs/action space info
  2. Ask the RewardAgent to generate K candidate reward functions
  3. Evaluate each candidate by running N episodes with random actions
  4. Score candidates by total reward and episode length
  5. Optionally: evolve — feed the best + feedback to the LLM for improvement
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from robosmith.agents.reward_agent import RewardAgent, RewardCandidate
from robosmith.config import LLMConfig, RewardSearchConfig, TaskSpec
from robosmith.envs.registry import EnvEntry
from robosmith.agents.base import BaseAgent
from robosmith.envs.wrapper import make_env

from .utils import EvalResult, RewardDesignResult

def extract_space_info(env, env_entry=None, llm_config=None) -> tuple[str, str]:  # noqa: ANN001
    """
    Extract human-readable observation and action space descriptions.

    Three-tier strategy:
    1. Runtime introspection — pull docs from the env class/metadata
    2. Sample-based analysis — reset/step the env and describe what we see
    3. LLM lookup fallback — ask the LLM to look up the env's obs layout

    This is what the reward-writing LLM sees as context, so the more
    detail we can extract, the better the reward functions will be.
    """
    obs_space = env.observation_space
    act_space = env.action_space
    env_id = env_entry.env_id if env_entry else ""

    # ── Observation space ──

    # Check for dict/goal-conditioned spaces FIRST (Dict has shape=None)
    if hasattr(obs_space, "spaces") and hasattr(obs_space.spaces, "items"):
        parts = []
        for key, space in obs_space.spaces.items():
            shape = getattr(space, "shape", "?")
            parts.append(f"{key}: {type(space).__name__}({shape})")
        obs_info = "Dict(" + ", ".join(parts) + ")"
        obs_info += "\n\nIMPORTANT: This is a Dict observation space (GoalEnv)."
        obs_info += "\nThe obs passed to compute_reward is a FLAT numpy array, concatenated as:"
        offset = 0
        for key, space in obs_space.spaces.items():
            dim = int(np.prod(space.shape)) if hasattr(space, "shape") else 0
            obs_info += f"\n  obs[{offset}:{offset + dim}] = '{key}' ({dim} dims)"
            offset += dim
        obs_info += f"\nTotal flattened size: {offset}"
        if "achieved_goal" in obs_space.spaces and "desired_goal" in obs_space.spaces:
            obs_info += "\n\nThis is a GOAL-CONDITIONED environment:"
            obs_info += "\n  - 'observation': robot state (joint positions, velocities)"
            obs_info += "\n  - 'achieved_goal': current state of the thing being controlled"
            obs_info += "\n  - 'desired_goal': target state we want to reach"
            obs_info += "\n  KEY: Reward should minimize distance between achieved_goal and desired_goal."
    elif hasattr(obs_space, "shape") and obs_space.shape is not None:
        obs_info = f"{type(obs_space).__name__}(shape={obs_space.shape}, dtype={obs_space.dtype})"
        if hasattr(obs_space, "low") and hasattr(obs_space, "high"):
            low = np.min(obs_space.low)
            high = np.max(obs_space.high)
            obs_info += f" range=[{low:.1f}, {high:.1f}]"
    else:
        obs_info = str(obs_space)

    # Tier 1: Extract docs from the environment class itself
    obs_doc = _introspect_env_obs(env, env_id)

    # Tier 2: Sample-based analysis if no docs found
    if not obs_doc:
        obs_doc = _analyze_obs_by_sampling(env)

    # Tier 3: LLM lookup for unknown environments
    if not obs_doc and llm_config:
        obs_doc = _llm_lookup_obs(env_id, obs_space, llm_config)

    if obs_doc:
        obs_info += "\n" + obs_doc

    # ── Action space ──

    if hasattr(act_space, "shape"):
        act_info = f"{type(act_space).__name__}(shape={act_space.shape}, dtype={act_space.dtype})"
        if hasattr(act_space, "low") and hasattr(act_space, "high"):
            act_info += f" range=[{act_space.low.min():.1f}, {act_space.high.max():.1f}]"
    else:
        act_info = str(act_space)

    return obs_info, act_info

def _introspect_env_obs(env, env_id: str) -> str:
    """
    Tier 1: Extract observation documentation from the environment itself.

    Checks: class docstring, metadata, MuJoCo model names, space field names.
    """
    lines = []

    # Check if the unwrapped env has observation documentation
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

    # Pull from class docstring — many gymnasium envs document obs layout there
    doc = type(unwrapped).__doc__ or ""
    if "observation" in doc.lower() and "obs[" in doc.lower():
        # Extract just the observation section
        obs_section = _extract_section(doc, ["observation", "obs space", "state space"])
        if obs_section:
            lines.append("From env documentation:")
            lines.append(obs_section[:500])  # Truncate

    # Check for MuJoCo-specific model info
    if hasattr(unwrapped, "model") and hasattr(unwrapped.model, "body_names"):
        try:
            body_names = [unwrapped.model.body(i).name for i in range(unwrapped.model.nbody)]
            joint_names = [unwrapped.model.joint(i).name for i in range(unwrapped.model.njnt)]
            if body_names:
                lines.append(f"MuJoCo bodies: {', '.join(body_names[:10])}")
            if joint_names:
                lines.append(f"MuJoCo joints: {', '.join(joint_names[:10])}")
        except Exception:
            pass

    # Check for named observation fields (GoalEnv, dict spaces)
    if hasattr(env.observation_space, "spaces"):
        lines.append("Named observation fields:")
        for key, space in env.observation_space.spaces.items():
            shape = getattr(space, "shape", "?")
            lines.append(f"  {key}: shape={shape}")

    # Check env metadata
    if hasattr(unwrapped, "metadata"):
        meta = unwrapped.metadata
        if "obs_keys" in meta:
            lines.append(f"Observation keys: {meta['obs_keys']}")

    return "\n".join(lines) if lines else ""

def _analyze_obs_by_sampling(env) -> str:
    """
    Tier 2: Sample the environment and describe what we see.

    Resets and steps the env to get actual observation values,
    then describes their ranges and characteristics.
    """
    try:
        obs, info = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        obs_flat = np.asarray(obs).flatten() if not isinstance(obs, dict) else None
        next_flat = np.asarray(next_obs).flatten() if not isinstance(next_obs, dict) else None

        if obs_flat is None:
            return ""

        lines = [f"Sample observation analysis ({len(obs_flat)} dims):"]

        # Analyze ranges of observation dimensions
        if len(obs_flat) <= 30:
            # Small enough to show per-dimension stats
            for i in range(len(obs_flat)):
                val = obs_flat[i]
                delta = next_flat[i] - obs_flat[i] if next_flat is not None else 0
                lines.append(f"  obs[{i}] = {val:.4f} (delta after 1 step: {delta:+.4f})")
        else:
            # Too many dims — show summary blocks
            block_size = max(1, len(obs_flat) // 8)
            for start in range(0, len(obs_flat), block_size):
                end = min(start + block_size, len(obs_flat))
                block = obs_flat[start:end]
                lines.append(
                    f"  obs[{start}:{end}] range=[{block.min():.3f}, {block.max():.3f}] "
                    f"mean={block.mean():.3f}"
                )

        # Check info dict for useful keys
        useful_info_keys = [k for k in step_info.keys() if k not in ("TimeLimit.truncated",)]
        if useful_info_keys:
            lines.append(f"  info keys: {', '.join(useful_info_keys)}")

        return "\n".join(lines)

    except Exception as e:
        logger.debug(f"Obs sampling failed: {e}")
        return ""

def _llm_lookup_obs(env_id: str, obs_space, llm_config) -> str:
    """
    Tier 3: Ask the LLM to look up the observation layout.

    Uses the fast model to describe what each observation dimension means
    based on its knowledge of the environment.
    """
    try:

        agent = BaseAgent(llm_config, use_fast_model=True)

        shape = getattr(obs_space, "shape", "?")
        prompt = f"""Describe the observation space layout for the gymnasium environment '{env_id}'.
The observation space has shape {shape}.

For each important dimension or group of dimensions, explain:
- What it represents (position, velocity, angle, etc.)
- Its typical range
- Which dimensions are most important for reward design

Be concise — this will be used as context for reward function generation.
Focus on the KEY dimensions for the task.
Respond with the observation layout description only, no preamble."""

        response = agent.chat(prompt, temperature=0.3)
        return f"LLM-retrieved observation layout for {env_id}:\n{response[:600]}"

    except Exception as e:
        logger.debug(f"LLM obs lookup failed: {e}")
        return ""

def _extract_section(text: str, keywords: list[str]) -> str:
    """Extract a section from a docstring by keyword."""
    lines = text.split("\n")
    capturing = False
    captured = []

    for line in lines:
        lower = line.lower().strip()
        if any(kw in lower for kw in keywords):
            capturing = True
            continue
        if capturing:
            if line.strip() == "" and captured:
                break  # End of section
            if line.strip().startswith("##") or line.strip().startswith("==="):
                break  # Next section
            captured.append(line)

    return "\n".join(captured).strip()

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
    training_reflection: str = "",
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

    # Adaptive budget: simple envs need fewer iterations
    obs_dim = _get_obs_dim(env_entry)
    if obs_dim <= 10:
        # Simple env (Pendulum, CartPole) — 2 gens, 3 candidates
        num_iterations = min(config.num_iterations, 2)
        num_candidates = min(num_candidates, 3)
        logger.info(f"Simple env (obs_dim={obs_dim}) — reduced budget: {num_iterations} gens, {num_candidates} candidates")
    elif obs_dim >= 50:
        # Complex env (Humanoid, HandReach) — use full budget
        num_iterations = config.num_iterations
        logger.info(f"Complex env (obs_dim={obs_dim}) — full budget: {num_iterations} gens, {num_candidates} candidates")
    else:
        num_iterations = config.num_iterations

    # Step 1: Extract space info from the environment
    logger.info(f"Extracting space info from {env_entry.env_id}")
    env = make_env(env_entry)
    obs_info, act_info = extract_space_info(env, env_entry, llm_config)
    env.close()

    logger.info(f"Obs space: {obs_info}")
    logger.info(f"Act space: {act_info}")

    agent = RewardAgent(llm_config)
    all_candidates: list[RewardCandidate] = []
    all_eval_results: list[EvalResult] = []
    global_best: RewardCandidate | None = None
    _gens_without_improvement = 0

    for gen in range(num_iterations):
        logger.info(f"Reward evolution — generation {gen + 1}/{num_iterations}")

        # Step 2: Generate candidates
        if gen == 0:
            # First generation: generate from scratch
            # Include literature context + training reflection from previous outer iteration
            combined_context = literature_context
            if training_reflection:
                combined_context = (
                    (combined_context + "\n\n" if combined_context else "")
                    + training_reflection
                )

            candidates = agent.generate(
                task_description=task_spec.task_description,
                obs_space_info=obs_info,
                action_space_info=act_info,
                num_candidates=num_candidates,
                literature_context=combined_context,
            )
        else:
            # Later generations: evolve from the best so far
            # If no global best yet (all previous candidates errored), regenerate from scratch
            if global_best is None:
                # Collect error messages from failed candidates to help the LLM
                error_msgs = [r.error_message for r in gen_eval_results if r.had_errors and r.error_message]
                error_context = ""
                if error_msgs:
                    unique_errors = list(set(e[:100] for e in error_msgs))[:3]
                    error_context = (
                        "\n\nPREVIOUS CANDIDATES ALL FAILED WITH THESE ERRORS:\n"
                        + "\n".join(f"  - {e}" for e in unique_errors)
                        + "\n\nFix these issues in your next attempt. "
                        "Make sure array indexing matches the observation layout described above."
                    )
                logger.warning(f"Generation {gen + 1}: no viable best from previous gen — regenerating from scratch")
                candidates = agent.generate(
                    task_description=task_spec.task_description,
                    obs_space_info=obs_info + error_context,
                    action_space_info=act_info,
                    num_candidates=num_candidates,
                    literature_context=combined_context if gen == 1 else "",
                )
            else:
                feedback = _format_feedback(global_best, all_eval_results[-num_candidates:])
                if training_reflection:
                    feedback = feedback + "\n\n" + training_reflection

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
        "REWARD EVALUATION RESULTS (random policy rollouts):",
        f"  Best candidate scored {best.score:.2f}.",
        "",
        "  All candidates this generation:",
    ]

    # Show all candidates, ranked
    sorted_evals = sorted(recent_evals, key=lambda e: e.mean_reward, reverse=True)
    for rank, ev in enumerate(sorted_evals, 1):
        if ev.had_errors:
            lines.append(f"    #{rank} Candidate {ev.candidate_id}: CRASHED — {ev.error_message[:80]}")
        else:
            lines.append(
                f"    #{rank} Candidate {ev.candidate_id}: "
                f"reward={ev.mean_reward:.2f} (±{ev.std_reward:.2f}), "
                f"ep_length={ev.mean_episode_length:.0f}"
            )

    # Score spread analysis
    valid_scores = [ev.mean_reward for ev in recent_evals if not ev.had_errors]
    if len(valid_scores) >= 2:
        spread = max(valid_scores) - min(valid_scores)
        lines.append("")
        lines.append(f"  Score spread: {spread:.2f} (max={max(valid_scores):.2f}, min={min(valid_scores):.2f})")
        if spread < 5:
            lines.append("  NOTE: Very narrow spread — candidates are too similar. Try more diverse approaches.")
        elif spread > 100:
            lines.append("  NOTE: Wide spread — some approaches work much better. Focus on what the best one does differently.")

    # Component breakdown from best candidate
    if best.metrics:
        lines.append("")
        lines.append("  Best candidate metrics:")
        for k, v in best.metrics.items():
            if isinstance(v, float):
                lines.append(f"    {k}: {v:.4f}")
            else:
                lines.append(f"    {k}: {v}")

    lines.append("")
    lines.append(
        "INSTRUCTIONS: Write an improved reward function that addresses the issues above. "
        "If the best score is negative, the reward function needs fundamental changes. "
        "If positive but low, focus on strengthening the task-relevant components. "
        "Keep reward components well-scaled (roughly [-1, 1] each) and use dense signals."
    )

    return "\n".join(lines)

def _flatten_obs(obs) -> np.ndarray:  # noqa: ANN001
    """Flatten observation to a 1D numpy array. Handles dict obs (Fetch envs)."""
    if isinstance(obs, dict):
        arrays = [np.asarray(v).flatten() for v in obs.values()]
        return np.concatenate(arrays)
    return np.asarray(obs).flatten()

def _get_obs_dim(env_entry) -> int:
    """Get observation dimensionality without creating the env."""
    try:
        env = make_env(env_entry)
        obs_space = env.observation_space
        if hasattr(obs_space, "spaces"):
            # Dict space — sum all sub-spaces
            dim = sum(int(np.prod(s.shape)) for s in obs_space.spaces.values() if hasattr(s, "shape"))
        elif hasattr(obs_space, "shape") and obs_space.shape:
            dim = int(np.prod(obs_space.shape))
        else:
            dim = 0
        env.close()
        return dim
    except Exception:
        return 20  # Default assumption

