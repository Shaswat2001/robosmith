"""
Trajectory analyzer.

Takes episodes from any reader and computes standardized diagnostics:
success rate, action stats, episode length distribution, failure mode
clustering, and run-to-run comparison.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any

from robosmith.diagnostics.diag_models import (
    ActionStats,
    EpisodeSummary,
    FailureCluster,
    TrajectoryCompareResult,
    TrajectoryDiagResult,
)
from robosmith.diagnostics.trajectory_reader import Episode, get_reader

logger = logging.getLogger(__name__)

def analyze_trajectory(path: str) -> TrajectoryDiagResult:
    """Run full trajectory diagnostics on a file or directory."""
    reader = get_reader(path)
    episodes = list(reader.read_episodes(path))

    if not episodes:
        raise ValueError(f"No episodes found in '{path}'")

    # ── Basic counts ──
    num_episodes = len(episodes)
    lengths = [ep.length for ep in episodes]
    total_frames = sum(lengths)

    # ── Success rate ──
    episodes_with_success = [ep for ep in episodes if ep.success is not None]
    if episodes_with_success:
        successes = sum(1 for ep in episodes_with_success if ep.success)
        failures = len(episodes_with_success) - successes
        success_rate = successes / len(episodes_with_success)
    else:
        successes = None
        failures = None
        success_rate = None

    # ── Episode lengths ──
    lengths_arr = np.array(lengths)

    # ── Action stats ──
    action_dim = None
    action_stats_list: list[ActionStats] = []
    episodes_with_actions = [ep for ep in episodes if ep.actions.size > 0 and ep.actions.ndim > 1]

    if episodes_with_actions:
        all_actions = np.concatenate([ep.actions for ep in episodes_with_actions], axis=0)
        action_dim = all_actions.shape[1]

        for d in range(action_dim):
            col = all_actions[:, d]
            col_min = float(np.min(col))
            col_max = float(np.max(col))

            # Clipping rate: fraction of values at the extremes (within 1% of range)
            range_val = col_max - col_min
            if range_val > 0:
                clip_threshold = range_val * 0.01
                at_low = np.sum(col <= col_min + clip_threshold)
                at_high = np.sum(col >= col_max - clip_threshold)
                clipping_rate = float((at_low + at_high) / len(col))
            else:
                clipping_rate = 1.0

            action_stats_list.append(ActionStats(
                dim=d,
                mean=float(np.mean(col)),
                std=float(np.std(col)),
                min=col_min,
                max=col_max,
                clipping_rate=clipping_rate,
            ))

    # ── Reward stats ──
    reward_mean = reward_std = reward_min = reward_max = None
    episodes_with_rewards = [ep for ep in episodes if ep.rewards is not None and len(ep.rewards) > 0]

    if episodes_with_rewards:
        all_rewards = np.concatenate([ep.rewards for ep in episodes_with_rewards])
        reward_mean = float(np.mean(all_rewards))
        reward_std = float(np.std(all_rewards))
        reward_min = float(np.min(all_rewards))
        reward_max = float(np.max(all_rewards))

    # ── Episode summaries ──
    episode_summaries = []
    for ep in episodes:
        total_reward = None
        if ep.rewards is not None and len(ep.rewards) > 0:
            total_reward = float(np.sum(ep.rewards))

        termination = None
        if ep.success is True:
            termination = "success"
        elif ep.success is False:
            if ep.dones is not None and len(ep.dones) > 0 and ep.dones[-1] > 0.5:
                termination = "terminated"
            else:
                termination = "timeout"

        episode_summaries.append(EpisodeSummary(
            episode_idx=ep.index,
            length=ep.length,
            success=ep.success,
            total_reward=total_reward,
            termination_reason=termination,
        ))

    # ── Failure clustering ──
    failure_clusters = None
    failed_episodes = [ep for ep in episodes if ep.success is False]
    if len(failed_episodes) >= 3:
        failure_clusters = _cluster_failures(failed_episodes)

    return TrajectoryDiagResult(
        source=path,
        format=reader.get_format_name(),
        num_episodes=num_episodes,
        total_frames=total_frames,
        success_rate=success_rate,
        successes=successes,
        failures=failures,
        episode_length_mean=float(np.mean(lengths_arr)),
        episode_length_std=float(np.std(lengths_arr)),
        episode_length_min=int(np.min(lengths_arr)),
        episode_length_max=int(np.max(lengths_arr)),
        action_dim=action_dim,
        action_stats=action_stats_list,
        reward_mean=reward_mean,
        reward_std=reward_std,
        reward_min=reward_min,
        reward_max=reward_max,
        episodes=episode_summaries,
        failure_clusters=failure_clusters,
    )

def compare_trajectories(path_a: str, path_b: str) -> TrajectoryCompareResult:
    """Compare two trajectory sets side by side."""
    result_a = analyze_trajectory(path_a)
    result_b = analyze_trajectory(path_b)

    # Action divergence per dimension
    action_divergence: list[float] = []
    if result_a.action_stats and result_b.action_stats:
        min_dims = min(len(result_a.action_stats), len(result_b.action_stats))
        for d in range(min_dims):
            mean_diff = abs(result_a.action_stats[d].mean - result_b.action_stats[d].mean)
            std_diff = abs(result_a.action_stats[d].std - result_b.action_stats[d].std)
            action_divergence.append(float(mean_diff + std_diff))

    # Success rate delta
    sr_delta = None
    if result_a.success_rate is not None and result_b.success_rate is not None:
        sr_delta = result_b.success_rate - result_a.success_rate

    # Biggest degradation description
    biggest = None
    if sr_delta is not None and sr_delta < 0:
        biggest = f"Success rate dropped from {result_a.success_rate:.1%} to {result_b.success_rate:.1%}"
    elif action_divergence:
        worst_dim = int(np.argmax(action_divergence))
        biggest = f"Action dim {worst_dim} has largest divergence ({action_divergence[worst_dim]:.4f})"

    return TrajectoryCompareResult(
        source_a=path_a,
        source_b=path_b,
        success_rate_a=result_a.success_rate,
        success_rate_b=result_b.success_rate,
        success_rate_delta=sr_delta,
        episode_length_mean_a=result_a.episode_length_mean,
        episode_length_mean_b=result_b.episode_length_mean,
        action_divergence=action_divergence,
        biggest_degradation=biggest,
    )

def _cluster_failures(failed_episodes: list[Episode]) -> list[FailureCluster]:
    """Simple failure clustering based on episode length and action patterns.

    Uses a heuristic approach:
    1. Short episodes (< 25th percentile length) -> "early termination"
    2. Max-length episodes -> "timeout"
    3. Remaining episodes clustered by final action statistics
    """
    clusters: list[FailureCluster] = []
    lengths = np.array([ep.length for ep in failed_episodes])
    p25 = np.percentile(lengths, 25)
    max_len = np.max(lengths)

    # Cluster 1: Early termination
    early = [ep for ep in failed_episodes if ep.length < p25]
    if early:
        clusters.append(FailureCluster(
            cluster_id=0,
            count=len(early),
            description=f"Early termination (length < {int(p25)} steps)",
            example_episodes=[ep.index for ep in early[:5]],
            common_features={"avg_length": float(np.mean([ep.length for ep in early]))},
        ))

    # Cluster 2: Timeout (at or near max length)
    timeout_threshold = max_len * 0.95
    timeouts = [ep for ep in failed_episodes if ep.length >= timeout_threshold and ep.length >= p25]
    if timeouts:
        clusters.append(FailureCluster(
            cluster_id=1,
            count=len(timeouts),
            description=f"Timeout (length >= {int(timeout_threshold)} steps)",
            example_episodes=[ep.index for ep in timeouts[:5]],
            common_features={"avg_length": float(np.mean([ep.length for ep in timeouts]))},
        ))

    # Cluster 3: Mid-length failures (everything else)
    mid = [
        ep for ep in failed_episodes
        if ep.length >= p25 and ep.length < timeout_threshold
    ]
    if mid:
        # Analyze final actions for mid-length failures
        common: dict[str, Any] = {"avg_length": float(np.mean([ep.length for ep in mid]))}
        episodes_with_actions = [ep for ep in mid if ep.actions.size > 0 and ep.actions.ndim > 1]
        if episodes_with_actions:
            final_actions = np.stack([ep.actions[-1] for ep in episodes_with_actions])
            common["final_action_mean"] = final_actions.mean(axis=0).tolist()
            common["final_action_std"] = final_actions.std(axis=0).tolist()

        clusters.append(FailureCluster(
            cluster_id=2,
            count=len(mid),
            description="Mid-episode failure (not early, not timeout)",
            example_episodes=[ep.index for ep in mid[:5]],
            common_features=common,
        ))

    return clusters