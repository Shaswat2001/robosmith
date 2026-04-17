from .types import RewardCandidate

__all__ = [
    "RewardAgent",
    "RewardCandidate"
]

def __getattr__(name: str):
    if name == "RewardAgent":
        from .agent import RewardAgent

        return RewardAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
