---
name: orbit-eval
description: Evaluate bot performance by running multiple matches and analyzing results.
user_invocable: true
---

# Orbit Wars Evaluation

Run structured evaluation of bot performance.

## Setup
Ask the user for:
- **Bot A**: path to the bot being evaluated (default: `bots/main.py`)
- **Bot B**: path to the opponent bot (default: `bots/baseline.py`)
- **Number of matches**: how many games to run (default: 10)

## Execution

Run matches and collect results:

```python
from kaggle_environments import make

env = make("orbit_wars")
results = []
n_matches = 10  # adjust per user request

for i in range(n_matches):
    result = env.run(["bots/bot_a.py", "bots/bot_b.py"])
    final = result[-1]
    # Collect scores/outcomes
    results.append(final)
    env.reset()
```

## Analysis

Report:
```
## Evaluation: [Bot A] vs [Bot B]
- Matches: [N]
- Bot A wins: [count] ([%])
- Bot B wins: [count] ([%])
- Draws: [count]
- Average game length: [turns]

## Observations
- [Patterns in wins/losses]
- [Turns where games typically decide]
- [Notable behaviors]

## Signal vs Noise
- With [N] matches, results are [statistically meaningful / too noisy]
- Recommended: run [more/fewer] matches for confidence
```

**Important**: With fewer than 20 matches, results may not be statistically significant. Flag this to the user.
