# Custom Agents

RoboSmith uses LLM agents for task parsing, reward design, and pipeline decisions. You can extend or replace any of them.

## Agent Architecture

```
agents/
├── base.py            # BaseAgent — LLM wrapper with retry + JSON parsing
├── reward_agent.py    # RewardAgent — generates and evolves reward functions
└── decision_agent.py  # DecisionAgent — pipeline iteration decisions
```

## BaseAgent

All agents inherit from `BaseAgent`, which handles LLM calls, retries, rate limiting, and JSON parsing:

```python
from robosmith.agents.base import BaseAgent

agent = BaseAgent(
    config=llm_config,
    system_prompt="You are an expert...",
    use_fast_model=True,  # Use the cheaper model
)

# Text response
response = agent.chat("What algorithm should I use?")

# Structured JSON response (with automatic retry on parse failure)
result = agent.chat_json("Respond with JSON: {algorithm: str, reason: str}")
```

## RewardAgent

The reward agent generates and evolves reward functions. It's the core of stage 4:

```python
from robosmith.agents.reward_agent import RewardAgent

agent = RewardAgent(llm_config)

# Generate fresh candidates
candidates = agent.generate(
    task_description="Walk forward",
    obs_space_info="Box(shape=(105,), ...)",
    action_space_info="Box(shape=(8,), ...)",
    num_candidates=4,
    literature_context="Recent papers suggest...",
)

# Evolve from the best candidate
evolved = agent.evolve(
    task_description="Walk forward",
    obs_space_info="...",
    action_space_info="...",
    previous_best=best_candidate,
    training_feedback="Reward was flat at 10.83...",
    generation=2,
    num_candidates=4,
)
```

## DecisionAgent

Makes intelligent decisions about what to do next after evaluation:

```python
from robosmith.agents.decision_agent import DecisionAgent

agent = DecisionAgent(llm_config)
decision = agent.decide(
    eval_report=report,
    training_result=result,
    task_spec=spec,
    reward_code=reward_fn_source,
    iteration=1,
    max_iterations=3,
)

print(decision.action)      # Decision.REFINE_REWARD
print(decision.reasoning)   # "Reward is flat, increase velocity weight"
print(decision.suggestions) # ["Change obs[13] weight from 2.0 to 5.0"]
print(decision.confidence)  # 0.8
```

## Creating Your Own Agent

```python
from robosmith.agents.base import BaseAgent

class MyCustomAgent:
    def __init__(self, config):
        self._agent = BaseAgent(
            config,
            system_prompt="You are an expert at ...",
            use_fast_model=True,
        )

    def analyze(self, data: str) -> dict:
        return self._agent.chat_json(f"Analyze this: {data}")
```
