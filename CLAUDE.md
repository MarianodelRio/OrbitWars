# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Kaggle Orbit Wars bot — an RTS-style competition where agents control planets and fleets, returning actions each turn. Development is iterative: design heuristic → implement → simulate → evaluate → refine.

Full game rules: [`game/rules.md`](game/rules.md)
Active model: [`PlanetPolicyModel`](bots/neural/planet_policy_model.py) (entity-centric policy/value network).

## Structure

```
bots/           # Bot implementations (one file per bot variant)
experiments/    # Evaluation scripts and results
tests/          # Simulation and unit tests
docs/           # Workflow documentation
```

- `bots/` contains self-contained bot files. Each bot is a function that receives observation and returns actions.
- Simulation runs locally via `kaggle_environments`.

## Development Commands

```bash
# Install dependencies
pip install kaggle-environments

# Run a local match between two bots
python scripts/run_match.py

# Run tests
python -m pytest tests/

# Quick evaluation (multiple matches)
# Use /orbit-eval skill for structured evaluation
```

## Role System (Strict)

This project uses three isolated roles. **Never mix responsibilities.**

### Planner
- Analyzes tasks and proposes step-by-step plans
- Identifies which files to touch (max 3)
- Detects risks and defines acceptance criteria
- **Does NOT write or modify code**

### Implementer
- Executes exactly what the plan says
- Makes minimal, focused changes
- **Does NOT redesign, extend scope, or add unrequested features**

### Reviewer
- Evaluates changes against the plan
- Detects bugs, edge cases, and regressions
- **Does NOT implement fixes — reports them for a new cycle**

## Context Isolation Rules

- One task = one clean cycle (plan → implement → review)
- Each role starts with `/clear` or its dedicated `/[role]-start` skill
- Pass context between roles via explicit artifacts (plan text, file paths, diff summary) — not conversation history
- Maximum 1–3 files changed per cycle
- Never break local simulation — verify with a test match before completing

## Completion Checklist

Before marking any change done:
1. Simulation runs without errors (`kaggle_environments` match completes)
2. Changes match the plan (no scope creep)
3. No side effects on existing bots
4. Tests pass if applicable

## Conventions

- Bot files: `bots/<name>.py` — descriptive name reflecting strategy (e.g., `aggressive_expand.py`, `defensive_turtle.py`)
- Experiments: `experiments/<date>_<description>/` — contain eval scripts and result logs
- Keep bots self-contained: no cross-bot imports
- Each bot must define an `agent_fn(observation, configuration)` entry point

## Principles

- Context isolation > convenience
- Small iterations > large changes
- Clarity > complexity
- Reproducibility > speed
