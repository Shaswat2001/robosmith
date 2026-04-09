"""
Trajectory readers for different formats.

Each reader takes a file path or directory and yields standardized
episode data that the diagnostic analyzer can consume.
"""

from __future__ import annotations

import h5py
import json
import logging
import numpy as np
from pathlib import Path
from typing import Any, Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from huggingface_hub import hf_hub_download, list_repo_tree

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

@dataclass
class Episode:
    """Standardized episode representation across all formats."""

    index: int
    actions: np.ndarray  # (T, action_dim)
    rewards: np.ndarray | None = None  # (T,)
    dones: np.ndarray | None = None  # (T,)
    success: bool | None = None
    states: np.ndarray | None = None  # (T, state_dim)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.actions.shape[0]

    @property
    def action_dim(self) -> int:
        return self.actions.shape[1] if self.actions.ndim > 1 else 1

class TrajectoryReader(ABC):
    """Base class for trajectory readers."""

    @abstractmethod
    def can_read(self, path: str) -> bool:
        ...

    @abstractmethod
    def read_episodes(self, path: str) -> Iterator[Episode]:
        ...

    @abstractmethod
    def get_format_name(self) -> str:
        ...

class HDF5TrajectoryReader(TrajectoryReader):
    """Read trajectories from robomimic/LIBERO-style HDF5 files.

    Expected structure:
        data/
            demo_0/
                actions: (T, action_dim)
                rewards: (T,)
                dones: (T,)
                obs/
                    <obs_key>: (T, ...)
            demo_1/
                ...
    """

    def can_read(self, path: str) -> bool:
        p = Path(path)
        if p.is_file() and p.suffix in (".hdf5", ".h5"):
            return True
        if p.is_dir():
            return any(p.glob("*.hdf5")) or any(p.glob("*.h5"))
        return False

    def get_format_name(self) -> str:
        return "hdf5"

    def read_episodes(self, path: str) -> Iterator[Episode]:

        p = Path(path)
        files = [p] if p.is_file() else sorted(p.glob("*.hdf5")) + sorted(p.glob("*.h5"))

        episode_idx = 0
        for fpath in files:
            with h5py.File(str(fpath), "r") as f:
                data_group = f.get("data", f)  # some files have data/ prefix, some don't

                demo_keys = sorted(
                    [k for k in data_group.keys() if k.startswith("demo")],
                    key=lambda x: int(x.split("_")[-1]) if "_" in x else 0,
                )

                for demo_key in demo_keys:
                    demo = data_group[demo_key]

                    actions = np.array(demo["actions"]) if "actions" in demo else np.array([])
                    rewards = np.array(demo["rewards"]) if "rewards" in demo else None
                    dones = np.array(demo["dones"]) if "dones" in demo else None

                    # Detect success
                    success = None
                    if dones is not None and len(dones) > 0:
                        success = bool(dones[-1] > 0.5)
                    # Some datasets have task_successes
                    if "task_successes" in demo:
                        task_succ = np.array(demo["task_successes"])
                        if len(task_succ) > 0:
                            success = bool(task_succ[-1] > 0.5)

                    # States (if available)
                    states = None
                    if "obs" in demo:
                        obs_group = demo["obs"]
                        # Try common state keys
                        for state_key in ["robot0_eef_pos", "joint_pos", "state", "proprio"]:
                            if state_key in obs_group:
                                states = np.array(obs_group[state_key])
                                break

                    yield Episode(
                        index=episode_idx,
                        actions=actions,
                        rewards=rewards,
                        dones=dones,
                        success=success,
                        states=states,
                    )
                    episode_idx += 1

class LeRobotTrajectoryReader(TrajectoryReader):
    """Read trajectories from LeRobot format (local parquet files or Hub).

    Reads parquet data files and uses meta/info.json to segment episodes.
    """

    def can_read(self, path: str) -> bool:
        p = Path(path)
        # Local LeRobot dataset: has meta/info.json and data/ directory
        if p.is_dir():
            return (p / "meta" / "info.json").exists() and (p / "data").exists()
        # Hub repo_id
        if "/" in path and not p.exists():
            try:
                hf_hub_download(path, "meta/info.json", repo_type="dataset")
                return True
            except Exception:
                return False
        return False

    def get_format_name(self) -> str:
        return "lerobot"

    def read_episodes(self, path: str) -> Iterator[Episode]:

        p = Path(path)

        if p.is_dir():
            yield from self._read_local(p)
        else:
            yield from self._read_hub(path)

    def _read_local(self, root: Path) -> Iterator[Episode]:
        

        # Read meta
        with open(root / "meta" / "info.json") as f:
            meta = json.load(f)

        total_episodes = meta.get("total_episodes", 0)

        # Read all parquet data files
        data_dir = root / "data"
        parquet_files = sorted(data_dir.rglob("*.parquet"))

        if not parquet_files:
            return

        # Concatenate all parquet files
        tables = [pq.read_table(str(f)) for f in parquet_files]
        table = pa.concat_tables(tables)
        df = table.to_pandas()

        # Segment by episode index
        if "episode_index" in df.columns:
            for ep_idx in range(total_episodes):
                ep_data = df[df["episode_index"] == ep_idx]
                if ep_data.empty:
                    continue

                actions = self._extract_array(ep_data, "action")
                states = self._extract_array(ep_data, "observation.state")

                # Check for success signal
                success = None
                if "next.done" in ep_data.columns:
                    success = bool(ep_data["next.done"].iloc[-1])
                elif "next.success" in ep_data.columns:
                    success = bool(ep_data["next.success"].iloc[-1])

                rewards = None
                if "next.reward" in ep_data.columns:
                    rewards = ep_data["next.reward"].values.astype(np.float64)

                yield Episode(
                    index=ep_idx,
                    actions=actions if actions is not None else np.array([]),
                    rewards=rewards,
                    success=success,
                    states=states,
                )

    def _read_hub(self, repo_id: str) -> Iterator[Episode]:
        """Read from Hub by downloading parquet shards."""

        meta_path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
        with open(meta_path) as f:
            meta = json.load(f)

        # Find and download first data parquet (for quick diagnostics)
        files = [f.rfilename for f in list_repo_tree(repo_id, repo_type="dataset")]
        parquet_files = sorted([f for f in files if f.startswith("data/") and f.endswith(".parquet")])

        if not parquet_files:
            return

        # Download first shard only for quick diag

        local_path = hf_hub_download(repo_id, parquet_files[0], repo_type="dataset")
        table = pq.read_table(local_path)
        df = table.to_pandas()

        if "episode_index" not in df.columns:
            return

        for ep_idx in df["episode_index"].unique():
            ep_data = df[df["episode_index"] == ep_idx]

            actions = self._extract_array(ep_data, "action")

            success = None
            if "next.done" in ep_data.columns:
                success = bool(ep_data["next.done"].iloc[-1])

            rewards = None
            if "next.reward" in ep_data.columns:
                rewards = ep_data["next.reward"].values.astype(np.float64)

            yield Episode(
                index=int(ep_idx),
                actions=actions if actions is not None else np.array([]),
                rewards=rewards,
                success=success,
            )

    def _extract_array(self, df: Any, prefix: str) -> np.ndarray | None:
        """Extract a multi-dim array from columns with a shared prefix.

        LeRobot stores action as a list column 'action' or as
        individual columns 'action.0', 'action.1', etc.
        """
        if prefix in df.columns:
            col = df[prefix]
            first_val = col.iloc[0]
            if isinstance(first_val, (list, np.ndarray)):
                return np.stack(col.values)
            else:
                return col.values.reshape(-1, 1)

        # Try numbered columns
        numbered = sorted([c for c in df.columns if c.startswith(f"{prefix}.")])
        if numbered:
            return df[numbered].values.astype(np.float64)

        return None

# ── Auto-detect reader ────────────────────────────────────────
_READERS = [HDF5TrajectoryReader(), LeRobotTrajectoryReader()]

def get_reader(path: str) -> TrajectoryReader:
    """Auto-detect the right reader for a given path."""
    for reader in _READERS:
        if reader.can_read(path):
            return reader
    raise ValueError(
        f"No trajectory reader found for '{path}'. "
        f"Supported formats: HDF5 (.hdf5/.h5), LeRobot (parquet+meta)"
    )
