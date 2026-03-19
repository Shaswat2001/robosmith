"""
Embodied Agent Forge
====================

Natural language → trained robot policy.

An end-to-end autonomous pipeline that transforms a natural language
task description into a trained, evaluated, and deployable robot policy.

    from forge import TaskSpec, ForgeController

    spec = TaskSpec.from_natural_language("A Franka arm that picks up a red cube")
    controller = ForgeController(spec)
    result = controller.run()

"""

__version__ = "0.1.0"

from forge.config import ForgeConfig, TaskSpec
from forge.controller import ForgeController

__all__ = [
    "ForgeConfig",
    "ForgeController",
    "TaskSpec",
]
