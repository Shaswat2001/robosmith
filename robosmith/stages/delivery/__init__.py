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

from .run import run_delivery
from .video import record_policy_video, load_policy_for_video
from .report import push_to_hub, write_report_card, write_reward_file

__all__ = ["run_delivery", "record_policy_video", "load_policy_for_video", "push_to_hub", "write_report_card", "write_reward_file"]