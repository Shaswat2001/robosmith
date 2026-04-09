"""
LeRobot dataset inspector (v2 and v3 format).

Reads dataset metadata from the HuggingFace Hub by fetching only the
metadata files (meta/info.json, meta/tasks.jsonl) without downloading
the full dataset.
"""

from __future__ import annotations

import json
import logging
import numpy as np
from typing import Any
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_tree

from robosmith.inspect.models import (
    CameraSpec,
    ColumnStats,
    DataQualityIssue,
    DatasetFormat,
    DatasetInspectResult,
    Severity,
    StorageInfo,
)
from robosmith.inspect.registry import BaseDatasetInspector, dataset_registry

logger = logging.getLogger(__name__)

class LeRobotInspector(BaseDatasetInspector):
    """Inspector for LeRobot format datasets on HuggingFace Hub (v2 and v3)."""

    name = "lerobot"

    def can_handle(self, identifier: str, **kwargs: Any) -> bool:
        """Check if this looks like a LeRobot dataset on the Hub.

        Strategy: try to download meta/info.json. If it exists, it's a LeRobot dataset.
        This is fast (single small file) and reliable.
        """
        if "/" not in identifier:
            return False

        try:

            hf_hub_download(
                identifier,
                "meta/info.json",
                repo_type="dataset",
            )
            return True
        except ImportError:
            logger.warning("huggingface_hub not installed. Install with: pip install huggingface-hub")
            return False
        except Exception as e:
            logger.debug(f"Not a LeRobot dataset ({identifier}): {e}")
            return False

    def inspect(self, identifier: str, **kwargs: Any) -> DatasetInspectResult:
        """Inspect a LeRobot dataset from the Hub."""
        meta = self._fetch_meta(identifier)

        if not meta:
            raise ValueError(f"Could not fetch meta/info.json from {identifier}")

        # ── Determine format version ──
        codebase_version = meta.get("codebase_version", "unknown")
        if codebase_version.startswith("v3"):
            dataset_format = DatasetFormat.LEROBOT  # v3 is an evolution of v2
        elif codebase_version.startswith("v2"):
            dataset_format = DatasetFormat.LEROBOT
        else:
            dataset_format = DatasetFormat.LEROBOT

        # ── Parse features from meta/info.json ──
        features = meta.get("features", {})

        cameras = self._parse_cameras(features)
        action_dim, action_keys = self._parse_action(features)
        state_dim, state_keys = self._parse_state(features)

        # ── Episode count ──
        episodes = meta.get("total_episodes", 0)
        total_frames = meta.get("total_frames", 0)
        fps = meta.get("fps", None)

        # ── Task descriptions ──
        tasks = self._fetch_tasks(identifier)

        # ── Storage info ──
        storage = StorageInfo(
            format="parquet+mp4",
            size_bytes=None,
            size_gb=None,
        )

        return DatasetInspectResult(
            repo_id=identifier,
            dataset_format=dataset_format,
            episodes=episodes,
            total_frames=total_frames,
            fps=fps,
            cameras=cameras,
            action_dim=action_dim,
            action_keys=action_keys,
            state_dim=state_dim,
            state_keys=state_keys,
            task_descriptions=tasks,
            storage=storage,
        )

    def inspect_schema(self, identifier: str, **kwargs: Any) -> dict[str, ColumnStats]:
        """Get detailed column stats by reading a parquet shard."""
        try:
            
            # Find the first parquet data file
            files = [f.rfilename for f in list_repo_tree(identifier, repo_type="dataset")]
            parquet_files = [
                f for f in files
                if f.startswith("data/") and f.endswith(".parquet")
            ]

            if not parquet_files:
                return {}

            local_path = hf_hub_download(
                identifier, parquet_files[0], repo_type="dataset"
            )
            table = pq.read_table(local_path)

            stats: dict[str, ColumnStats] = {}
            for i, field in enumerate(table.schema):
                col = table.column(i)
                col_stats = ColumnStats(dtype=str(field.type))

                try:

                    values = col.to_numpy(zero_copy_only=False)
                    if np.issubdtype(values.dtype, np.number):
                        col_stats.min = float(np.nanmin(values))
                        col_stats.max = float(np.nanmax(values))
                        col_stats.mean = float(np.nanmean(values))
                        col_stats.std = float(np.nanstd(values))
                        col_stats.nan_count = int(np.isnan(values).sum())
                        col_stats.constant = bool(col_stats.std == 0.0)
                except (TypeError, ValueError, ImportError):
                    pass

                stats[field.name] = col_stats

            return stats

        except ImportError as e:
            logger.warning(f"Missing dependency for schema inspection: {e}")
            return {}
        except Exception as e:
            logger.warning(f"Schema inspection failed: {e}")
            return {}

    def inspect_quality(self, identifier: str, **kwargs: Any) -> list[DataQualityIssue]:
        """Check for common data quality issues."""
        issues: list[DataQualityIssue] = []
        schema = self.inspect_schema(identifier)

        for col_name, stats in schema.items():
            if stats.nan_count > 0:
                issues.append(DataQualityIssue(
                    severity=Severity.CRITICAL,
                    issue_type="nan_values",
                    detail=f"Column '{col_name}' has {stats.nan_count} NaN values",
                    affected_columns=[col_name],
                ))

            if stats.constant and stats.min is not None:
                issues.append(DataQualityIssue(
                    severity=Severity.WARNING,
                    issue_type="constant_column",
                    detail=f"Column '{col_name}' is constant (value={stats.min})",
                    affected_columns=[col_name],
                ))

        return issues

    # ── Private helpers ────────────────────────────────────────
    def _fetch_meta(self, repo_id: str) -> dict[str, Any]:
        """Fetch meta/info.json from the Hub."""
        try:
            path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not fetch meta/info.json: {e}")
            return {}

    def _parse_cameras(self, features: dict[str, Any]) -> dict[str, CameraSpec]:
        """Extract camera specs from the features dict.

        In LeRobot v3, features looks like:
        {
            "observation.images.cam_high": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                ...
            }
        }
        """
        cameras: dict[str, CameraSpec] = {}
        for key, spec in features.items():
            if not key.startswith("observation.images."):
                continue

            cam_name = key.replace("observation.images.", "")
            shape = spec.get("shape", [])

            if len(shape) >= 2:
                # shape is [height, width, channels] based on "names" field
                names = spec.get("names", [])
                # names can be a list like ["height", "width", "channel"]
                # or a dict like {"height": ..., "width": ...}
                first_name = None
                if isinstance(names, list) and names:
                    first_name = names[0] if isinstance(names[0], str) else None
                elif isinstance(names, dict):
                    first_name = list(names.keys())[0] if names else None

                if first_name == "height":
                    cameras[cam_name] = CameraSpec(
                        height=shape[0],
                        width=shape[1],
                        channels=shape[2] if len(shape) > 2 else 3,
                        encoding=spec.get("video_info", {}).get("video.codec"),
                    )
                else:
                    cameras[cam_name] = CameraSpec(
                        height=shape[0],
                        width=shape[1],
                        channels=shape[2] if len(shape) > 2 else 3,
                    )
            else:
                cameras[cam_name] = CameraSpec(width=640, height=480, channels=3)

        return cameras

    def _flatten_names(self, names: Any) -> list[str]:
        """Flatten a names field that can be a list, dict, or None.

        LeRobot v3 uses dicts like {"motors": ["left_waist", "right_gripper", ...]}.
        Earlier versions use flat lists like ["joint_0", "joint_1", ...].
        """
        if names is None:
            return []
        if isinstance(names, list):
            # Could be list of strings or list of dicts
            flat: list[str] = []
            for item in names:
                if isinstance(item, str):
                    flat.append(item)
                elif isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, list):
                            flat.extend(str(x) for x in v)
                        else:
                            flat.append(str(v))
            return flat
        if isinstance(names, dict):
            # {"motors": ["left_waist", ...]} -> flatten all values
            flat = []
            for v in names.values():
                if isinstance(v, list):
                    flat.extend(str(x) for x in v)
                else:
                    flat.append(str(v))
            return flat
        return []

    def _parse_action(self, features: dict[str, Any]) -> tuple[int | None, list[str]]:
        """Extract action dimension and keys from features.

        Handles both flat action and nested action.* keys.
        """
        # Flat action key
        if "action" in features:
            spec = features["action"]
            shape = spec.get("shape", [])
            if shape:
                names = self._flatten_names(spec.get("names"))
                return shape[0], names

        # Nested action.* keys (like DROID format)
        action_keys = sorted([k for k in features if k.startswith("action.") and k != "action"])
        if action_keys:
            total_dim = 0
            names: list[str] = []
            for key in action_keys:
                spec = features[key]
                shape = spec.get("shape", [])
                dim = shape[0] if shape else 1
                total_dim += dim
                names.append(key.replace("action.", ""))
            return total_dim, names

        return None, []

    def _parse_state(self, features: dict[str, Any]) -> tuple[int | None, list[str]]:
        """Extract state dimension and keys from features."""
        # Flat state key
        if "observation.state" in features:
            spec = features["observation.state"]
            shape = spec.get("shape", [])
            if shape:
                names = self._flatten_names(spec.get("names"))
                return shape[0], names

        # Nested observation.state.* keys
        state_keys = sorted([
            k for k in features
            if k.startswith("observation.state.") and k != "observation.state"
        ])
        if state_keys:
            total_dim = 0
            names: list[str] = []
            for key in state_keys:
                spec = features[key]
                shape = spec.get("shape", [])
                dim = shape[0] if shape else 1
                total_dim += dim
                names.append(key.replace("observation.state.", ""))
            return total_dim, names

        return None, []

    def _fetch_tasks(self, repo_id: str) -> list[str]:
        """Fetch task descriptions from meta/tasks.jsonl or meta/tasks.parquet."""

        # Try JSONL first (v2 format)
        try:
            path = hf_hub_download(repo_id, "meta/tasks.jsonl", repo_type="dataset")
            tasks = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        if "task" in data:
                            tasks.append(data["task"])
            if tasks:
                return tasks
        except Exception:
            pass

        # Try Parquet (v3 format)
        try:
            path = hf_hub_download(repo_id, "meta/tasks.parquet", repo_type="dataset")
            table = pq.read_table(path)
            if "task" in table.column_names:
                return table.column("task").to_pylist()
        except Exception:
            pass

        return []

# ── Register ──────────────────────────────────────────────────
dataset_registry.register("lerobot", LeRobotInspector)