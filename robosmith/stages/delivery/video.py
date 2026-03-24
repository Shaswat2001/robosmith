
from __future__ import annotations

import shutil
import imageio
import numpy as np
import gymnasium as gym
from pathlib import Path
from loguru import logger

from robosmith.config import RunState
from robosmith.envs.registry import EnvRegistry
from robosmith.trainers.registry import TrainerRegistry

def load_policy_for_video(model_path: Path):
    """
    Load a trained policy for video recording.

    Tries the trainer registry first, falls back to SB3 direct loading.
    Returns an object with a .predict(obs, deterministic=True) method.
    """
    # Try trainer registry
    try:
        registry = TrainerRegistry()

        name = model_path.stem.lower()
        if "cleanrl" in name:
            backend = "cleanrl"
        elif "il" in name:
            backend = "il_trainer"
        elif "offline" in name:
            backend = "offline_rl_trainer"
        else:
            backend = "sb3"

        trainer = registry.get_trainer(backend=backend)
        return trainer.load_policy(model_path)
    except Exception:
        pass

    # Fallback: direct SB3
    from stable_baselines3 import PPO, SAC, TD3
    algo_name = model_path.stem.replace("policy_", "").lower()
    algo_map = {"ppo": PPO, "sac": SAC, "td3": TD3}
    AlgoClass = algo_map.get(algo_name, PPO)
    return AlgoClass.load(str(model_path))

def record_policy_video(
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
            model = load_policy_for_video(model_path)
        except Exception as e:
            logger.warning(f"Could not load model for video: {e}")
            return None

        # Create env with render_mode="rgb_array" for video capture

        video_dir = artifacts_dir / "_tmp_video"
        shutil.rmtree(video_dir, ignore_errors=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        # Method 1: gymnasium RecordVideo (needs moviepy)
        try:
            env = gym.make(env_entry, render_mode="rgb_array")
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

            video_files = sorted(video_dir.glob("*.mp4"))
            if video_files:
                final_path = artifacts_dir / "policy_rollout.mp4"
                video_files[-1].rename(final_path)
                shutil.rmtree(video_dir, ignore_errors=True)
                logger.info(f"Recorded policy video → {final_path}")
                return final_path

        except Exception as e:
            logger.debug(f"RecordVideo failed ({e}), trying imageio")
            try:
                env.close()
            except Exception:
                pass

        # Method 2: manual frame capture with imageio
        try:
            env = gym.make(env_entry, render_mode="rgb_array")
            frames: list = []

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
                    frame = env.render()
                    if frame is not None:
                        frames.append(np.asarray(frame))
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
                imageio.mimwrite(str(video_path), frames, fps=30)

                shutil.rmtree(video_dir, ignore_errors=True)
                logger.info(f"Recorded policy video → {video_path}")
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
