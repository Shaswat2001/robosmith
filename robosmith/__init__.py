"""
RoboSmith
=========

Natural language → trained robot policy.

An autonomous smith that crafts robot behaviors. Describe a task
in plain English, and RoboSmith handles environment setup, reward
design, RL training, evaluation, and deployment.

    from robosmith import TaskSpec, SmithController

    spec = TaskSpec(task_description="A Franka arm that picks up a red cube")
    controller = SmithController(spec)
    result = controller.run()

"""

__version__ = "0.1.0"

from robosmith.config import ForgeConfig as SmithConfig
from robosmith.config import TaskSpec
from robosmith.controller import ForgeController as SmithController

# Keep old names as aliases for backward compat
ForgeConfig = SmithConfig
ForgeController = SmithController

__all__ = [
    "SmithConfig",
    "SmithController",
    "TaskSpec",
    # Aliases
    "ForgeConfig",
    "ForgeController",
]
