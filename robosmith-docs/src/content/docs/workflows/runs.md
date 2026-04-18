---
title: Runs And Resume
description: List, inspect, compare, clean, and resume RoboSmith pipeline runs.
---

## Run Directories

Each `robosmith run` creates a timestamped directory in `robosmith_runs/` unless
you configure another `runs_dir`.

```text
robosmith_runs/
  run_20260415_182058_a64796/
    checkpoint.json
    run_state.json
    task_spec.json
    eval_report.json
    reward_function.py
    report.md
```

The run ID format is:

```text
run_YYYYMMDD_HHMMSS_<short-uuid>
```

Most run commands accept either the full run ID or a unique prefix.

## List Runs

```bash
robosmith runs list
robosmith runs list --limit 50
robosmith runs list --status success
robosmith runs list --runs-dir ./my_runs
```

The list view reads `run_state.json`, `eval_report.json`, and `task_spec.json`
when present. It shows the task, status, decision, success rate, mean reward,
and iteration count.

## Inspect A Run

```bash
robosmith runs inspect run_20260415
robosmith runs inspect run_20260415 --log
robosmith runs inspect run_20260415 --reward
```

`--log` prints the recorded graph step log. `--reward` prints
`reward_function.py` if delivery produced it.

## Compare Runs

```bash
robosmith runs compare run_20260415 run_20260416
```

Comparison reads both run directories and displays:

| Metric | Source |
| --- | --- |
| Status and decision | `run_state.json`, `eval_report.json` |
| Task and robot | `task_spec.json` |
| Environment | `run_state.json` |
| Reward metrics | `eval_report.json` |
| Iterations | `run_state.json` |

Numeric differences are highlighted in the terminal output.

## Resume

```bash
robosmith resume run_20260415
robosmith resume run_20260415 --runs-dir ./my_runs --verbose
```

Resume requires `checkpoint.json`. It loads the checkpoint, reconstructs typed
objects, skips nodes listed in `completed_nodes`, and continues the graph.

This is most useful when a run was interrupted after a completed stage. If a run
finished successfully and no checkpoint is present, inspect the artifacts instead
of resuming.

## Clean Old Runs

```bash
robosmith runs clean --older-than 14 --dry-run
robosmith runs clean --older-than 14 --yes
```

`--dry-run` reports what would be deleted and how much space would be freed.
Use `--yes` to skip the confirmation prompt.

## Python Resume

```python
from pathlib import Path
from robosmith.agent.graphs.run import resume_pipeline

state = resume_pipeline(
    run_id="run_20260415",
    runs_dir=Path("./robosmith_runs"),
)
print(state["status"])
```
