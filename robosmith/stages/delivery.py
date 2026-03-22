"""
Stage 7: Delivery.

Packages everything from a successful Forge run into a clean artifact bundle:
  - reward_function.py   — the winning reward function (standalone, runnable)
  - task_spec.json       — the full task specification
  - eval_report.json     — evaluation metrics and decision history
  - report.md            — human-readable report card
  - policy_*.zip         — trained model checkpoint (if training ran)
  - run_state.json       — full pipeline state for reproducibility

Optionally pushes to HuggingFace Hub.
"""

from __future__ import annotations

import json
import time
import shutil
import imageio
import numpy as np
from pathlib import Path
import gymnasium as gym
from loguru import logger
from dataclasses import dataclass

from robosmith.config import RunState, TaskSpec
from robosmith.envs.registry import EnvRegistry
from robosmith.stages.evaluation import EvalReport
from robosmith.agents.reward_agent import RewardCandidate

@dataclass
class DeliveryResult:
    """Output of the delivery stage."""

    artifacts_dir: Path
    files_written: list[str]
    pushed_to_hub: bool = False
    hub_url: str | None = None

def run_delivery(
    state: RunState,
    reward_candidate: RewardCandidate | None = None,
    eval_report: EvalReport | None = None,
    training_result: object | None = None,
) -> DeliveryResult:
    """
    Package all artifacts from a Forge run.
    """
    artifacts_dir = Path(state.artifacts_dir) if state.artifacts_dir else Path("./forge_output")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    files_written: list[str] = []

    # 1. Save reward function as standalone Python file
    if reward_candidate:
        reward_path = artifacts_dir / "reward_function.py"
        _write_reward_file(reward_path, reward_candidate, state.task_spec)
        files_written.append("reward_function.py")
        logger.info(f"Saved reward function → {reward_path}")

    # 2. Save task spec
    spec_path = artifacts_dir / "task_spec.json"
    spec_path.write_text(state.task_spec.model_dump_json(indent=2))
    files_written.append("task_spec.json")

    # 3. Save eval report
    if eval_report:
        eval_path = artifacts_dir / "eval_report.json"
        eval_data = {
            "success_rate": eval_report.success_rate,
            "mean_reward": eval_report.mean_reward,
            "std_reward": eval_report.std_reward,
            "mean_episode_length": eval_report.mean_episode_length,
            "worst_reward": eval_report.worst_reward,
            "best_reward": eval_report.best_reward,
            "decision": eval_report.decision.value,
            "decision_reason": eval_report.decision_reason,
            "criteria_results": eval_report.criteria_results,
            "num_episodes": len(eval_report.episodes),
        }
        eval_path.write_text(json.dumps(eval_data, indent=2))
        files_written.append("eval_report.json")

    # 4. Save run state
    state_path = artifacts_dir / "run_state.json"
    state_path.write_text(state.model_dump_json(indent=2))
    files_written.append("run_state.json")

    # 5. Record video of the trained policy
    if training_result and hasattr(training_result, "model_path") and training_result.model_path:
        video_path = _record_policy_video(
            state=state,
            model_path=training_result.model_path,
            artifacts_dir=artifacts_dir,
        )
        if video_path:
            files_written.append(video_path.name)

    # 6. Generate markdown report card
    report_path = artifacts_dir / "report.md"
    _write_report_card(report_path, state, reward_candidate, eval_report, training_result)
    files_written.append("report.md")
    logger.info(f"Saved report card → {report_path}")

    # 7. Push to HuggingFace Hub (if requested)
    pushed = False
    hub_url = None
    if state.task_spec.push_to_hub:
        pushed, hub_url = _push_to_hub(state.task_spec.push_to_hub, artifacts_dir)
        if pushed:
            files_written.append(f"(pushed to {hub_url})")

    logger.info(f"Delivery complete — {len(files_written)} artifacts in {artifacts_dir}")

    return DeliveryResult(
        artifacts_dir=artifacts_dir,
        files_written=files_written,
        pushed_to_hub=pushed,
        hub_url=hub_url,
    )

def _record_policy_video(
    state: RunState,
    model_path: Path,
    artifacts_dir: Path,
    num_episodes: int = 1,
    max_steps: int = 500,
) -> Path | None:
    """
    Record a video of the trained policy acting in the environment.

    Uses gymnasium's RecordVideo wrapper. Falls back to frame-by-frame
    capture with imageio if RecordVideo is unavailable.
    """
    env_id = state.task_spec.environment_id
    if not env_id:
        logger.warning("No environment ID — skipping video recording")
        return None

    try:

        registry = EnvRegistry()
        env_entry = registry.get(env_id)
        if not env_entry:
            logger.warning(f"Environment '{env_id}' not found — skipping video")
            return None

        # Try to load the trained model
        try:
            from stable_baselines3 import PPO, SAC, TD3

            algo_name = model_path.stem.replace("policy_", "").lower()
            algo_map = {"ppo": PPO, "sac": SAC, "td3": TD3}
            AlgoClass = algo_map.get(algo_name, PPO)
            model = AlgoClass.load(str(model_path))
        except Exception as e:
            logger.warning(f"Could not load model for video: {e}")
            return None

        # Create env with render_mode="rgb_array" for video capture

        video_dir = artifacts_dir / "_tmp_video"
        shutil.rmtree(video_dir, ignore_errors=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        # Method 1: gymnasium RecordVideo (needs moviepy)
        try:
            env = gym.make(env_entry.env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_dir),
                episode_trigger=lambda ep: True,
                name_prefix="policy",
            )

            for ep in range(num_episodes):
                obs, info = env.reset()
                for step in range(max_steps):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
            env.close()

            video_files = list(video_dir.glob("*.mp4"))
            if video_files:
                latest = max(video_files, key=lambda p: p.stat().st_mtime)
                final_path = artifacts_dir / "policy_rollout.mp4"
                if final_path.exists():
                    final_path.unlink()
                shutil.move(str(latest), str(final_path))
                shutil.rmtree(video_dir, ignore_errors=True)
                return final_path

        except Exception as e:
            logger.debug(f"RecordVideo failed ({e}), trying imageio")
            try:
                env.close()
            except Exception:
                pass

        # Method 2: manual frame capture with imageio
        try:
            env = gym.make(env_id, render_mode="rgb_array")
            frames = []

            for ep in range(num_episodes):
                obs, info = env.reset()

                # capture initial frame
                frame = env.render()
                if frame is not None:
                    frame = np.asarray(frame)
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    frames.append(frame)

                for step in range(max_steps):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)

                    frame = env.render()
                    if frame is not None:
                        frame = np.asarray(frame)
                        if frame.dtype != np.uint8:
                            frame = frame.astype(np.uint8)
                        frames.append(frame)

                    if terminated or truncated:
                        break

            env.close()

            if frames:
                first_shape = frames[0].shape
                frames = [f for f in frames if f.shape == first_shape]

                video_path = artifacts_dir / "policy_rollout.mp4"
                imageio.mimwrite(str(video_path), frames, fps=30, macro_block_size=None)

                shutil.rmtree(video_dir, ignore_errors=True)
                return video_path
        except ImportError:
            logger.info("For video recording: pip install imageio[ffmpeg] moviepy")
        except Exception as e:
            logger.warning(f"Video recording failed: {e}")

        # Clean up empty video dir
        shutil.rmtree(video_dir, ignore_errors=True)

    except Exception as e:
        logger.warning(f"Video recording failed: {e}")

    return None

def _write_reward_file(
    path: Path,
    candidate: RewardCandidate,
    task_spec: TaskSpec,
) -> None:
    """Write the reward function as a standalone, documented Python file."""
    header = f'''"""
Reward function generated by RoboSmith.

Task: {task_spec.task_description}
Robot: {task_spec.robot_type.value} ({task_spec.robot_model or "auto"})
Environment: {task_spec.environment_id or "auto"}
Generation: {candidate.generation}
Score: {candidate.score}

Usage:
    import numpy as np
    from reward_function import compute_reward

    obs = env.reset()
    action = policy(obs)
    next_obs, _, _, _, info = env.step(action)
    reward, components = compute_reward(obs, action, next_obs, info)
"""

import numpy as np

'''
    path.write_text(header + candidate.code + "\n")

def _write_report_card(
    path: Path,
    state: RunState,
    reward_candidate: RewardCandidate | None,
    eval_report: EvalReport | None,
    training_result: object | None,
) -> None:
    """Generate a human-readable markdown report card."""
    spec = state.task_spec
    lines: list[str] = []

    lines.append(f"# RoboSmith report: {spec.task_description}")
    lines.append("")
    lines.append(f"**Run ID:** `{state.run_id}`")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Iterations:** {state.iteration}")
    lines.append("")

    # Task summary
    lines.append("## Task")
    lines.append("")
    lines.append(f"- **Description:** {spec.task_description}")
    lines.append(f"- **Robot:** {spec.robot_type.value} ({spec.robot_model or 'auto'})")
    lines.append(f"- **Environment:** {spec.environment_id or 'auto'}")
    lines.append(f"- **Algorithm:** {spec.algorithm.value}")
    lines.append("")

    # Evaluation results
    if eval_report:
        lines.append("## Evaluation results")
        lines.append("")
        lines.append(f"- **Success rate:** {eval_report.success_rate:.0%}")
        lines.append(f"- **Mean reward:** {eval_report.mean_reward:.2f} (±{eval_report.std_reward:.2f})")
        lines.append(f"- **Best reward:** {eval_report.best_reward:.2f}")
        lines.append(f"- **Worst reward:** {eval_report.worst_reward:.2f}")
        lines.append(f"- **Mean episode length:** {eval_report.mean_episode_length:.0f}")
        lines.append(f"- **Episodes evaluated:** {len(eval_report.episodes)}")
        lines.append(f"- **Decision:** {eval_report.decision.value} — {eval_report.decision_reason}")
        lines.append("")

        if eval_report.criteria_results:
            lines.append("### Success criteria")
            lines.append("")
            for criterion, result in eval_report.criteria_results.items():
                status = "PASS" if result["passed"] else "FAIL"
                value = result.get("value", "N/A")
                lines.append(f"- `{criterion}` → {value} [{status}]")
            lines.append("")

    # Training info
    if training_result and hasattr(training_result, "algorithm"):
        lines.append("## Training")
        lines.append("")
        lines.append(f"- **Algorithm:** {training_result.algorithm}")
        lines.append(f"- **Total timesteps:** {training_result.total_timesteps:,}")
        lines.append(f"- **Training time:** {training_result.training_time_seconds:.1f}s")
        lines.append(f"- **Final mean reward:** {training_result.final_mean_reward:.2f}")
        if training_result.model_path:
            lines.append(f"- **Model checkpoint:** `{training_result.model_path.name}`")
        # Check if video was recorded
        video_path = Path(state.artifacts_dir) / "policy_rollout.mp4" if state.artifacts_dir else None
        if video_path and video_path.exists():
            lines.append(f"- **Policy video:** `policy_rollout.mp4`")
        lines.append("")

    # Reward function
    if reward_candidate:
        lines.append("## Reward function")
        lines.append("")
        lines.append(f"- **Generation:** {reward_candidate.generation}")
        lines.append(f"- **Eval score:** {reward_candidate.score}")
        lines.append("")
        lines.append("```python")
        lines.append(reward_candidate.code.strip())
        lines.append("```")
        lines.append("")

    # Pipeline stages
    lines.append("## Pipeline stages")
    lines.append("")
    for name, record in state.stages.items():
        duration = f"{record.duration_seconds:.1f}s" if record.duration_seconds else "—"
        lines.append(f"- **{name}:** {record.status.value} ({duration})")
    lines.append("")

    # Decision history
    if state.decision_history:
        lines.append("## Decision history")
        lines.append("")
        for d in state.decision_history:
            lines.append(f"- Iteration {d.get('iteration', '?')}: "
                         f"{d.get('decision', '?')} — {d.get('reason', '')}")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by [RoboSmith](https://github.com/Shaswat2001/robosmith)*")

    path.write_text("\n".join(lines))

def _push_to_hub(repo_id: str, artifacts_dir: Path) -> tuple[bool, str | None]:
    """Push artifacts to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface-hub not installed — skipping push")
        return False, None

    try:
        api = HfApi()
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(
            folder_path=str(artifacts_dir),
            repo_id=repo_id,
            commit_message="RoboSmith run artifacts",
        )
        url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Pushed to HuggingFace Hub → {url}")
        return True, url
    except Exception as e:
        logger.warning(f"Failed to push to Hub: {e}")
        return False, None