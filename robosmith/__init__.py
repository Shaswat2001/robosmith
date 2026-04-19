"""
RoboSmith
=========

Natural language → trained robot policy.

An autonomous smith that crafts robot behaviors. Describe a task
in plain English, and RoboSmith handles environment setup, reward
design, RL training, evaluation, and deployment.

    from robosmith import TaskSpec, ForgeConfig
    from robosmith.agent.graphs.run import run_pipeline

    spec = TaskSpec(task_description="A Franka arm that picks up a red cube")
    result = run_pipeline(spec)

"""

__version__ = "0.2.0"

from robosmith.config import ForgeConfig as SmithConfig
from robosmith.config import TaskSpec

# Keep old names as aliases for backward compat
ForgeConfig = SmithConfig

__all__ = [
    "SmithConfig",
    "ForgeConfig",
    "TaskSpec",
]
