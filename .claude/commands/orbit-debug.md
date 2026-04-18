---
name: orbit-debug
description: Debug a specific issue in an Orbit Wars bot by reproducing and inspecting game state.
user_invocable: true
---

# Orbit Wars Debug Session

Structured debugging for bot issues. Follow this sequence:

## 1. Reproduce the Failure
Ask the user for:
- Which bot file has the issue
- What behavior is wrong (crash, bad decisions, specific turn)
- Any error messages

Run a match to reproduce:
```bash
python -m kaggle_environments run --environment orbit_wars --agents bots/<bot>.py bots/baseline.py --debug
```

## 2. Inspect Game State
If the issue is a bad decision (not a crash):
- Add temporary logging to the bot's `agent_fn` to print observation state at the problematic turn
- Re-run and capture output
- Identify what the bot "saw" vs what it "did"

## 3. Review Decision Logic
- Trace the code path for the specific game state
- Check: planet ownership, fleet counts, distances, turn number
- Identify where the logic diverges from expected behavior

## 4. Validate Hypothesis
- Propose a specific explanation for the bug
- Describe what a fix would need to change (do NOT implement — that's a new plan/implement cycle)

## Output
```
## Debug Report
- **Issue**: [description]
- **Reproduced**: [yes/no]
- **Root Cause**: [specific code location and logic error]
- **Game State at Failure**: [relevant observation data]
- **Recommended Fix**: [description for planner, not code]
```
