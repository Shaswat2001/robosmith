"""
LeRobot policy inspector.

Reads config.json from a HuggingFace Hub model repo to extract
architecture, action dim, expected cameras, state keys, normalization, etc.
Works for all LeRobot policy types: SmolVLA, Pi0, Pi0.5, ACT, Diffusion, etc.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from huggingface_hub import hf_hub_download

from robosmith.inspect.models import (
    ActionHeadType,
    PolicyInspectResult,
)
from robosmith.inspect.registry import BasePolicyInspector, policy_registry

logger = logging.getLogger(__name__)

# Map LeRobot policy type strings to our ActionHeadType enum
_ACTION_HEAD_MAP: dict[str, ActionHeadType] = {
    "smolvla": ActionHeadType.FLOW_MATCHING,
    "pi0": ActionHeadType.FLOW_MATCHING,
    "pi0fast": ActionHeadType.AUTOREGRESSIVE,
    "pi05": ActionHeadType.FLOW_MATCHING,
    "act": ActionHeadType.DETERMINISTIC,
    "diffusion": ActionHeadType.DIFFUSION,
    "vqbet": ActionHeadType.DISCRETE_TOKENS,
    "tdmpc": ActionHeadType.GAUSSIAN,
    "xvla": ActionHeadType.FLOW_MATCHING,
}

# Map policy type to known VLM backbone (if applicable)
_VLM_BACKBONE_MAP: dict[str, str] = {
    "smolvla": "SmolVLM2-500M-Video-Instruct",
    "pi0": "PaliGemma-3B",
    "pi0fast": "PaliGemma-3B",
    "pi05": "PaliGemma-3B",
    "xvla": "InternVL2",
}


class LeRobotPolicyInspector(BasePolicyInspector):
    """Inspector for LeRobot policy checkpoints on HuggingFace Hub."""

    name = "lerobot_policy"

    def can_handle(self, identifier: str, **kwargs: Any) -> bool:
        """Check if this looks like a LeRobot policy on the Hub.

        Strategy: try to download config.json from the model repo.
        If it exists and has a 'type' field, it's a LeRobot policy.
        """
        if "/" not in identifier:
            return False

        try:

            path = hf_hub_download(identifier, "config.json", repo_type="model")
            with open(path) as f:
                config = json.load(f)
            # LeRobot policy configs have a "type" field and "input_features"
            return "type" in config and (
                "input_features" in config or "output_features" in config
            )
        except ImportError:
            logger.warning("huggingface_hub not installed.")
            return False
        except Exception as e:
            logger.debug(f"Not a LeRobot policy ({identifier}): {e}")
            return False

    def inspect(self, identifier: str, **kwargs: Any) -> PolicyInspectResult:
        """Inspect a LeRobot policy from the Hub."""
        config = self._fetch_config(identifier)

        if not config:
            raise ValueError(f"Could not fetch config.json from {identifier}")

        policy_type = config.get("type", "unknown")

        # ── Parse input features ──
        input_features = config.get("input_features", {})
        cameras = self._parse_cameras(input_features)
        state_keys = self._parse_state_keys(input_features)
        state_dim = self._get_state_dim(input_features)

        # ── Parse output features ──
        output_features = config.get("output_features", {})
        action_dim = self._get_action_dim(output_features)

        # ── Action head type ──
        action_head = _ACTION_HEAD_MAP.get(policy_type, ActionHeadType.UNKNOWN)

        # ── VLM backbone ──
        vlm_name = config.get("vlm_model_name")
        if not vlm_name:
            vlm_name = _VLM_BACKBONE_MAP.get(policy_type)

        # ── Normalization ──
        norm_mapping = config.get("normalization_mapping", {})
        norm_str = self._format_normalization(norm_mapping)

        # ── Image size ──
        resize = config.get("resize_imgs_with_padding")

        # ── Chunk size ──
        chunk_size = config.get("chunk_size") or config.get("n_action_steps")

        # ── Language ──
        accepts_language = config.get("tokenizer_max_length") is not None

        # ── Requirements (infer from policy type) ──
        requirements = self._infer_requirements(policy_type)

        return PolicyInspectResult(
            model_id=identifier,
            architecture=policy_type.upper() if policy_type else "unknown",
            base_vlm=vlm_name,
            action_head=action_head,
            action_dim=action_dim,
            action_chunk_size=chunk_size,
            expected_cameras=cameras,
            expected_state_keys=state_keys,
            normalization=norm_str,
            input_image_size=resize,
            accepts_language_instruction=accepts_language,
            inference_dtype=config.get("dtype"),
            requirements=requirements,
            training_config=config if kwargs.get("include_config") else None,
        )

    # ── Private helpers ────────────────────────────────────────
    def _fetch_config(self, repo_id: str) -> dict[str, Any]:
        """Fetch config.json from the Hub."""
        try:
            path = hf_hub_download(repo_id, "config.json", repo_type="model")
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not fetch config.json: {e}")
            return {}

    def _parse_cameras(self, input_features: dict[str, Any]) -> list[str]:
        """Extract camera key names from input_features."""
        cameras = []
        for key, spec in input_features.items():
            feat_type = spec.get("type", "")
            if feat_type == "VISUAL":
                cameras.append(key)
        return sorted(cameras)

    def _parse_state_keys(self, input_features: dict[str, Any]) -> list[str]:
        """Extract state key names from input_features."""
        state_keys = []
        for key, spec in input_features.items():
            feat_type = spec.get("type", "")
            if feat_type == "STATE":
                state_keys.append(key)
        return sorted(state_keys)

    def _get_state_dim(self, input_features: dict[str, Any]) -> int | None:
        """Get total state dimension from input_features."""
        total = 0
        for key, spec in input_features.items():
            if spec.get("type") == "STATE":
                shape = spec.get("shape", [])
                if shape:
                    total += shape[0]
        return total if total > 0 else None

    def _get_action_dim(self, output_features: dict[str, Any]) -> int | None:
        """Get action dimension from output_features."""
        if "action" in output_features:
            shape = output_features["action"].get("shape", [])
            if shape:
                return shape[0]
        return None

    def _format_normalization(self, norm_mapping: dict[str, str]) -> str | None:
        """Format normalization mapping as a readable string."""
        if not norm_mapping:
            return None
        parts = [f"{k.lower()}={v.lower()}" for k, v in norm_mapping.items()]
        return ", ".join(parts)

    def _infer_requirements(self, policy_type: str) -> list[str]:
        """Infer package requirements from policy type."""
        base = ["torch>=2.0", "lerobot"]
        if policy_type in ("smolvla", "pi0", "pi0fast", "pi05", "xvla"):
            base.append("transformers>=4.40")
        if policy_type in ("smolvla",):
            base.append("flash-attn (optional)")
        return base

# ── Register ──────────────────────────────────────────────────
policy_registry.register("lerobot_policy", LeRobotPolicyInspector)
