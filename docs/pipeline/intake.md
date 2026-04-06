# Stage 1: Intake

The intake stage is the front door of the pipeline. It takes a natural language task description — anything from "Walk forward" to "A Franka arm that picks up a red cube and places it on the shelf" — and parses it into a structured `TaskSpec` that every downstream stage can work with.

## Why this stage exists

Natural language is ambiguous. "Walk forward" could mean a bipedal humanoid, a quadruped, or even a snake robot. "Balance the pole" could be an inverted pendulum (classic control) or a humanoid holding a stick (manipulation). The intake stage resolves these ambiguities using an LLM, producing a structured specification with explicit robot type, environment type, success criteria, and algorithm preferences.

Without this stage, every downstream module would need to interpret raw text independently, leading to inconsistencies. The intake stage centralizes all interpretation in one place.

## How it works

```python
from robosmith.stages.intake import parse_task

task_spec = parse_task("Walk forward", llm_config)
# TaskSpec(
#     task_description="Walk forward",
#     robot_type=RobotType.QUADRUPED,
#     environment_type=EnvironmentType.FLOOR,
#     algorithm=AlgorithmChoice.AUTO,
#     ...
# )
```

The function sends the task description to the **fast LLM model** (e.g., Claude Haiku) with a structured prompt. The prompt includes:

1. The raw task description
2. A list of valid robot types, environment types, and algorithms
3. Examples of tricky classifications (e.g., "balance the pendulum" → classic control, not arm)
4. Instructions to return JSON with all required fields

The LLM returns a JSON object that maps directly to `TaskSpec` fields. The response is parsed with `chat_json()`, which handles markdown code fence stripping and retries on parse failures.

## What gets extracted

| Field | How it's determined | Example |
|-------|-------------------|---------|
| `robot_type` | LLM classifies from task description | "Walk forward" → `QUADRUPED` |
| `environment_type` | Inferred from robot + task | Quadruped → `FLOOR` |
| `success_criteria` | Default `success_rate >= 0.8` unless task implies otherwise | — |
| `algorithm` | Usually `AUTO` unless task strongly suggests one | — |
| `robot_model` | Extracted if mentioned | "Franka arm" → `"franka"` |
| `safety_constraints` | Extracted if mentioned | "Don't flip over" → constraint |

## Classic control detection

One of the trickiest classification problems is distinguishing classic control tasks from robotics tasks. "Balance the pendulum" and "Swing up the pole" are classic control problems that use specific environments (Pendulum-v1, CartPole-v1), not robot arm simulators.

The prompt includes explicit examples:

- "Balance the pendulum upright" → classic control, not arm
- "Swing up and balance the pole" → CartPole, not manipulation
- "Move the cart to keep the pole balanced" → CartPole
- "Mountain car: reach the flag" → classic control

This prevents the common failure mode where the LLM maps every "balance" task to a robot arm.

## Usage guidelines

**Keep task descriptions concise but specific.** "Walk forward" works. "Make a robot walk" also works. Very long descriptions with multiple goals may confuse the LLM — break them into separate runs if needed.

**Mention the robot type if it matters.** "A humanoid that walks forward" is more specific than "Walk forward" and removes ambiguity about whether you want a quadruped or biped.

**Success criteria are rarely needed in the prompt.** The default (`success_rate >= 0.8`) works for most tasks. If you need custom criteria, use the config file instead.

## Error handling

If the LLM fails to return valid JSON after retries, the intake stage falls back to a minimal `TaskSpec` with defaults (ARM robot, TABLETOP environment, AUTO algorithm) and logs a warning. The pipeline continues — downstream stages like env synthesis will refine the match based on the raw task description.

## Source

`robosmith/stages/intake/parsing.py` — main parsing logic

`robosmith/stages/intake/prompt.py` — prompt templates
