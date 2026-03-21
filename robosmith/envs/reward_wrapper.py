"""
Custom reward wrapper - injects a Forge reward function into any Gymnasium env. 

The RL algorithm sees our LLM-generated reward instead of the environment's
default reward. The original reward is still available in the info dict 
for comparison. 
"""

from __future__ import annotations
 
from typing import Any, Callable
 
import gymnasium as gym
import numpy as np

class ForgeRewardWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that replaces the environment reward with a custom one.
 
    The custom reward function has the signature:
        def compute_reward(obs, action, next_obs, info) -> tuple[float, dict]
 
    The wrapper:
    - Stores the previous obs so it can pass (obs, action, next_obs, info)
    - Replaces the step reward with our custom reward
    - Adds the original reward and component breakdown to info
    """

    def __init__(
        self,
        env: gym.Env,
        reward_fn: Callable
    ) -> None:
        super().__init__(env)
        self.reward_fn = reward_fn
        self._prev_obs: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = self._to_array(obs)
        return obs, info
    
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        next_obs, original_reward, terminated, truncated, info = self.env.step(action)
 
        # Compute our custom reward
        obs_array = self._prev_obs if self._prev_obs is not None else self._to_array(next_obs)
        next_obs_array = self._to_array(next_obs)
        action_array = np.asarray(action, dtype=np.float32)
 
        try:
            custom_reward, components = self.reward_fn(
                obs_array, action_array, next_obs_array, info
            )
            custom_reward = float(custom_reward)
 
            # Safety check
            if not np.isfinite(custom_reward):
                custom_reward = 0.0
                components["_error"] = "non-finite reward, clamped to 0"
 
        except Exception as e:
            custom_reward = 0.0
            components = {"_error": str(e)}
 
        # Store for next step
        self._prev_obs = next_obs_array
 
        # Add metadata to info
        info["custom_reward"] = custom_reward
        info["reward_components"] = components
        info["original_reward"] = original_reward
 
        return next_obs, custom_reward, terminated, truncated, info

    @staticmethod
    def _to_array(obs: Any) -> np.ndarray:
        """Convert observation to flat numpy array."""
        if isinstance(obs, dict):
            arrays = [np.asarray(v).flatten() for v in obs.values()]
            return np.concatenate(arrays)
        return np.asarray(obs).flatten()