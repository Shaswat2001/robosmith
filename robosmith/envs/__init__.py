from robosmith.envs.registry import EnvEntry, EnvRegistry

__all__ = ["EnvEntry", "EnvRegistry", "make_env"]

def __getattr__(name: str):
    if name == "make_env":
        from robosmith.envs.wrapper import make_env

        return make_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
