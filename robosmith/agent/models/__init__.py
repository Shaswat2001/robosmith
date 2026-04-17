__all__ = ["BaseAgent", "RewardAgent", "RewardCandidate"]

def __getattr__(name: str):
    if name == "BaseAgent":
        from robosmith.agent.models.base import BaseAgent

        return BaseAgent
    if name == "RewardAgent":
        from robosmith.agent.models.reward import RewardAgent

        return RewardAgent
    if name == "RewardCandidate":
        from robosmith.agent.models.reward import RewardCandidate

        return RewardCandidate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
