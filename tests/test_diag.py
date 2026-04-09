"""
Tests for robosmith.inspect trajectory diagnostics.

Creates synthetic HDF5 files to test the full pipeline without external data.
"""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from robosmith.diagnostics.diag_models import TrajectoryDiagResult, TrajectoryCompareResult
from robosmith.diagnostics.trajectory_reader import HDF5TrajectoryReader, Episode, get_reader
from robosmith.diagnostics.trajectory_analyzer import analyze_trajectory, compare_trajectories


@pytest.fixture
def synthetic_hdf5(tmp_path):
    """Create a synthetic HDF5 file with robomimic-style structure."""
    fpath = tmp_path / "rollouts.hdf5"
    with h5py.File(str(fpath), "w") as f:
        data = f.create_group("data")

        # 5 episodes: 3 success, 2 failure
        for i in range(5):
            length = 50 + i * 10  # 50, 60, 70, 80, 90
            demo = data.create_group(f"demo_{i}")
            demo.attrs["num_samples"] = length

            # Actions: 7-dim, normally distributed
            actions = np.random.randn(length, 7).astype(np.float32)
            actions = np.clip(actions, -1, 1)
            demo.create_dataset("actions", data=actions)

            # Rewards: small positive
            rewards = np.random.uniform(0, 0.1, size=(length,))
            if i < 3:
                rewards[-1] = 1.0  # success signal
            demo.create_dataset("rewards", data=rewards)

            # Dones: 1 at end for successful episodes
            dones = np.zeros(length)
            if i < 3:
                dones[-1] = 1.0
            demo.create_dataset("dones", data=dones)

            # Obs
            obs = demo.create_group("obs")
            obs.create_dataset("robot0_eef_pos", data=np.random.randn(length, 3))

    return fpath


@pytest.fixture
def two_hdf5_dirs(tmp_path):
    """Create two directories with HDF5 files for comparison."""
    for name, success_count in [("baseline", 8), ("perturbed", 3)]:
        d = tmp_path / name
        d.mkdir()
        fpath = d / "rollouts.hdf5"
        with h5py.File(str(fpath), "w") as f:
            data = f.create_group("data")
            for i in range(10):
                length = 100
                demo = data.create_group(f"demo_{i}")
                actions = np.random.randn(length, 7).astype(np.float32)
                demo.create_dataset("actions", data=actions)
                dones = np.zeros(length)
                if i < success_count:
                    dones[-1] = 1.0
                demo.create_dataset("dones", data=dones)
                demo.create_dataset("rewards", data=np.zeros(length))

    return tmp_path / "baseline", tmp_path / "perturbed"


# ── Reader Tests ──────────────────────────────────────────────


class TestHDF5Reader:
    def test_can_read_hdf5_file(self, synthetic_hdf5):
        reader = HDF5TrajectoryReader()
        assert reader.can_read(str(synthetic_hdf5)) is True

    def test_can_read_hdf5_dir(self, synthetic_hdf5):
        reader = HDF5TrajectoryReader()
        assert reader.can_read(str(synthetic_hdf5.parent)) is True

    def test_cannot_read_random_dir(self, tmp_path):
        reader = HDF5TrajectoryReader()
        assert reader.can_read(str(tmp_path)) is False

    def test_read_episodes(self, synthetic_hdf5):
        reader = HDF5TrajectoryReader()
        episodes = list(reader.read_episodes(str(synthetic_hdf5)))
        assert len(episodes) == 5
        assert episodes[0].length == 50
        assert episodes[4].length == 90
        assert episodes[0].action_dim == 7

    def test_success_detection(self, synthetic_hdf5):
        reader = HDF5TrajectoryReader()
        episodes = list(reader.read_episodes(str(synthetic_hdf5)))
        assert episodes[0].success is True
        assert episodes[1].success is True
        assert episodes[2].success is True
        assert episodes[3].success is False
        assert episodes[4].success is False

    def test_auto_detect_reader(self, synthetic_hdf5):
        reader = get_reader(str(synthetic_hdf5))
        assert isinstance(reader, HDF5TrajectoryReader)


# ── Analyzer Tests ────────────────────────────────────────────


class TestTrajectoryAnalyzer:
    def test_basic_analysis(self, synthetic_hdf5):
        result = analyze_trajectory(str(synthetic_hdf5))

        assert isinstance(result, TrajectoryDiagResult)
        assert result.num_episodes == 5
        assert result.format == "hdf5"
        assert result.action_dim == 7
        assert result.success_rate == pytest.approx(0.6)  # 3/5
        assert result.successes == 3
        assert result.failures == 2

    def test_episode_lengths(self, synthetic_hdf5):
        result = analyze_trajectory(str(synthetic_hdf5))
        assert result.episode_length_min == 50
        assert result.episode_length_max == 90
        assert result.episode_length_mean == pytest.approx(70.0)

    def test_action_stats(self, synthetic_hdf5):
        result = analyze_trajectory(str(synthetic_hdf5))
        assert len(result.action_stats) == 7
        for s in result.action_stats:
            assert -2.0 < s.mean < 2.0
            assert s.std > 0
            assert s.min >= -1.0  # clipped
            assert s.max <= 1.0

    def test_reward_stats(self, synthetic_hdf5):
        result = analyze_trajectory(str(synthetic_hdf5))
        assert result.reward_mean is not None
        assert result.reward_min is not None

    def test_episode_summaries(self, synthetic_hdf5):
        result = analyze_trajectory(str(synthetic_hdf5))
        assert len(result.episodes) == 5
        assert result.episodes[0].success is True
        assert result.episodes[0].termination_reason == "success"
        assert result.episodes[4].success is False

    def test_json_serializable(self, synthetic_hdf5):
        result = analyze_trajectory(str(synthetic_hdf5))
        data = json.loads(result.model_dump_json(exclude_none=True))
        assert data["num_episodes"] == 5
        assert data["action_dim"] == 7
        assert len(data["action_stats"]) == 7

    def test_directory_analysis(self, synthetic_hdf5):
        """Analyzing a directory should find the HDF5 file inside."""
        result = analyze_trajectory(str(synthetic_hdf5.parent))
        assert result.num_episodes == 5


# ── Comparison Tests ──────────────────────────────────────────


class TestTrajectoryCompare:
    def test_compare(self, two_hdf5_dirs):
        baseline, perturbed = two_hdf5_dirs
        result = compare_trajectories(str(baseline), str(perturbed))

        assert isinstance(result, TrajectoryCompareResult)
        assert result.success_rate_a == pytest.approx(0.8)
        assert result.success_rate_b == pytest.approx(0.3)
        assert result.success_rate_delta == pytest.approx(-0.5)
        assert result.biggest_degradation is not None
        assert "dropped" in result.biggest_degradation.lower()

    def test_compare_json_serializable(self, two_hdf5_dirs):
        baseline, perturbed = two_hdf5_dirs
        result = compare_trajectories(str(baseline), str(perturbed))
        data = json.loads(result.model_dump_json(exclude_none=True))
        assert data["success_rate_delta"] == pytest.approx(-0.5)


# ── Failure Clustering Tests ──────────────────────────────────


class TestFailureClustering:
    def test_failure_clusters_with_enough_failures(self, tmp_path):
        """Create a dataset with many failures to test clustering."""
        fpath = tmp_path / "failures.hdf5"
        with h5py.File(str(fpath), "w") as f:
            data = f.create_group("data")
            for i in range(20):
                # Vary lengths to create different clusters
                if i < 5:
                    length = 10  # early termination
                elif i < 15:
                    length = 100  # timeout
                else:
                    length = 50  # mid-episode
                demo = data.create_group(f"demo_{i}")
                demo.create_dataset("actions", data=np.random.randn(length, 4))
                demo.create_dataset("dones", data=np.zeros(length))  # all failures
                demo.create_dataset("rewards", data=np.zeros(length))

        result = analyze_trajectory(str(fpath))
        assert result.success_rate == 0.0
        assert result.failure_clusters is not None
        assert len(result.failure_clusters) >= 2  # at least early + timeout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])