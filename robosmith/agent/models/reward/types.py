"""Pure data types for reward generation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class RewardCandidate:
    """A single generated reward function candidate."""

    code: str
    function_name: str = "compute_reward"
    candidate_id: int = 0
    generation: int = 0

    # Filled in after evaluation
    score: float | None = None
    metrics: dict = field(default_factory=dict)
    error: str | None = None

    def is_valid(self) -> bool:
        """Check if the code compiles and has the right signature."""
        try:
            compiled = compile(self.code, f"<reward_{self.candidate_id}>", "exec")
            namespace: dict[str, Any] = {}
            exec(compiled, namespace)

            func = namespace.get(self.function_name)
            if func is None:
                self.error = f"Function '{self.function_name}' not found in generated code"
                return False

            if not callable(func):
                self.error = f"'{self.function_name}' is not callable"
                return False

            return True

        except SyntaxError as e:
            self.error = f"Syntax error: {e}"
            return False
        except Exception as e:
            self.error = f"Validation error: {e}"
            return False

    def get_function(self) -> Callable[..., Any]:
        """Compile and return the actual callable reward function."""
        namespace: dict[str, Any] = {"np": __import__("numpy")}
        exec(self.code, namespace)
        return namespace[self.function_name]
