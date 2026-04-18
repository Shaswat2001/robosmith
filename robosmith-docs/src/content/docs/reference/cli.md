---
title: CLI Commands
description: Complete command and flag reference for the RoboSmith CLI.
---

## Top-Level Commands

| Command | Purpose |
| --- | --- |
| `robosmith run` | Run the full natural-language-to-policy pipeline. |
| `robosmith resume` | Resume an interrupted pipeline run from `checkpoint.json`. |
| `robosmith envs` | List and filter registered simulation environments. |
| `robosmith trainers` | Show available trainer backends and algorithms. |
| `robosmith deps` | Show installed and missing optional dependencies. |
| `robosmith config` | Print default `ForgeConfig` as JSON. |
| `robosmith version` | Print package version. |
| `robosmith inspect` | Inspect datasets, environments, policies, robots, and compatibility. |
| `robosmith diag` | Analyze or compare trajectory rollouts. |
| `robosmith gen` | Generate wrapper code. |
| `robosmith auto` | Run compound agentic workflows. |
| `robosmith runs` | List, inspect, compare, and clean run directories. |

## `robosmith run`

```bash
robosmith run --task "A Franka arm picks up a red cube"
```

| Flag | Default | Description |
| --- | --- | --- |
| `--task`, `-t` | required | Natural language task description. |
| `--robot`, `-r` | `arm` | Robot type: `arm`, `quadruped`, `biped`, `dexterous_hand`, `mobile_base`, `custom`. |
| `--model`, `-m` | none | Specific robot model, for example `franka`, `unitree_go2`, `shadow_hand`. |
| `--algo`, `-a` | `auto` | Algorithm: `ppo`, `sac`, `td3`, `auto`. |
| `--time-budget` | `60` | Max training time in minutes. |
| `--num-envs` | `1024` | Number of parallel environments requested. |
| `--push-to-hub` | none | HuggingFace repo ID for pushed artifacts. |
| `--candidates`, `-c` | `4` | Reward candidates per iteration. |
| `--skip`, `-s` | none | Stages to skip: `scout`, `intake`, `delivery`. |
| `--backend`, `-b` | auto | Training backend: `sb3`, `cleanrl`, or another registered backend. |
| `--llm`, `-L` | auto | Provider or exact LiteLLM model string. |
| `--scout` | `semantic_scholar` | Search backend: `semantic_scholar`, `arxiv`, `both`. |
| `--config` | auto-detect | Path to `robosmith.yaml`. |
| `--dry-run` | false | Parse and plan only; do not train. |
| `--verbose`, `-v` | false | Write debug logs to `robosmith_runs/latest.log`. |

## `robosmith resume`

```bash
robosmith resume run_20260415_182058_a64796
robosmith resume run_20260415 --runs-dir ./robosmith_runs --verbose
```

| Argument or flag | Description |
| --- | --- |
| `run_id` | Full run ID or unique prefix. |
| `--runs-dir` | Base directory containing `run_*` directories. |
| `--verbose`, `-v` | Debug logging. |

## `robosmith envs`

```bash
robosmith envs --robot arm --framework gym --tags "pick place"
```

| Flag | Description |
| --- | --- |
| `--robot`, `-r` | Filter by robot type. |
| `--framework`, `-f` | Filter by framework. |
| `--tags`, `-t` | Comma-separated tags to search. |
| `--env-type`, `-e` | Filter by environment type. |

## `robosmith inspect`

```bash
robosmith inspect dataset lerobot/aloha_mobile_cabinet --schema --quality
robosmith inspect env Ant-v5 --obs-docs --sample
robosmith inspect policy lerobot/smolvla_base
robosmith inspect robot path/to/robot.urdf
robosmith inspect compat POLICY TARGET --fix
```

| Subcommand | Extra flags |
| --- | --- |
| `dataset IDENTIFIER` | `--json`, `--schema`, `--quality`, `--sample N` |
| `env IDENTIFIER` | `--json`, `--obs-docs`, `--sample` |
| `policy IDENTIFIER` | `--json`, `--config`, `--requirements` |
| `robot IDENTIFIER` | `--json` |
| `compat A B [C]` | `--json`, `--fix` |

## `robosmith diag`

```bash
robosmith diag trajectory path/to/rollout.hdf5
robosmith diag compare rollout_a.hdf5 rollout_b.hdf5 --json
```

| Subcommand | Description |
| --- | --- |
| `trajectory PATH` | Analyze one trajectory set. |
| `compare PATH_A PATH_B` | Compare two trajectory sets. |

Both support `--json`.

## `robosmith gen`

```bash
robosmith gen wrapper POLICY TARGET
robosmith gen wrapper POLICY TARGET --no-llm -o adapter.py
```

| Flag | Description |
| --- | --- |
| `--output`, `-o` | Write generated code to a file. |
| `--no-llm` | Use template-based generation. |

## `robosmith auto`

```bash
robosmith auto integrate POLICY TARGET --verbose -o adapter.py
```

| Flag | Description |
| --- | --- |
| `--output`, `-o` | Write generated wrapper to a file. |
| `--json`, `-j` | Output machine-readable final state. |
| `--verbose`, `-v` | Show step-by-step execution. |

## `robosmith runs`

```bash
robosmith runs list --limit 20
robosmith runs inspect run_20260415 --log --reward
robosmith runs compare run_20260415 run_20260416
robosmith runs clean --older-than 14 --dry-run
```

| Subcommand | Important flags |
| --- | --- |
| `list` | `--runs-dir`, `--status`, `--limit` |
| `inspect RUN_ID` | `--runs-dir`, `--log`, `--reward` |
| `compare RUN_A RUN_B` | `--runs-dir` |
| `clean` | `--runs-dir`, `--older-than`, `--dry-run`, `--yes` |
