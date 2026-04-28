# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Kaggle Orbit Wars bot — an RTS-style competition where agents control planets and fleets, returning actions each turn. Development is iterative: design heuristic → implement → simulate → evaluate → refine.

Full game rules: [`game/rules.md`](game/rules.md)
Active model: [`PlanetPolicyModel`](bots/neural/planet_policy_model.py) (entity-centric policy/value network).

## Structure

```
bots/
  heuristic/      # Rule-based bots (baseline, sniper, oracle_sniper)
  neural/         # PlanetPolicyModel, StateBuilder, ActionCodec, NeuralBot
  scoring/        # Scoring-based heuristic bot
  registry.py     # Short-name → module:fn mapping
  interface.py    # Bot base class + make_agent()
training/
  envs/           # OrbitWarsEnv (non-gym kaggle wrapper)
  rl/             # PPO loss, GAE, RolloutBuffer, OpponentPool
  rewards/        # PotentialReward (shaping + events + terminal)
  trainers/       # ILTrainer, RLTrainer
  evaluation/     # Evaluator (match-based win-rate)
  utils/          # RLConfig, RunConfig, CheckpointManager, MetricsLogger
dataset/          # IL dataset building (HDF5 cache, torch adapter)
game/
  env/            # kaggle env runner + load_agent
  eval/           # match metrics
  logic/          # combat, geometry, threat
  state/          # state models
  data/           # HDF5 writer
tournament/       # ELO ranking
submission/       # Kaggle submission packaging
scripts/          # CLI helpers (matches, tournament, train_il, train_rl)
tests/
  unit/           # Unit tests (rl/, action_codec, state_builder, …)
  integration/    # Integration tests (bots, rl env step)
docs/             # Workflow documentation
train.py          # Unified CLI entry point (auto-detects IL vs RL from config)
```

- `bots/` contains self-contained bot files. Each bot exposes an `agent_fn(obs, config)` at module level.
- `train.py` is the primary entry point for both IL and RL training; mode is auto-detected from the config file.
- Simulation runs locally via `kaggle_environments`.

## Development Commands

```bash
# Install dependencies
pip install kaggle-environments

make match           # single match
make tournament      # round-robin tournament with ELO
make data            # generate IL training data
make cache           # build HDF5 training cache
make train           # IL training
make train-phase1 IL_CKPT=<path>   # RL phase 1 (requires IL checkpoint)
make train-phase2    # RL phase 2 (warmstarts from phase 1 rl_last.pt)
make train-phase3    # RL phase 3 (warmstarts from phase 2 rl_last.pt)
make pipeline        # full pipeline, blocking
make eval CKPT=<path>              # evaluate a checkpoint
make watch RUN=<path>              # stream live training metrics
make submit-neural   # package and submit neural bot
make test            # all tests
make test-unit       # unit tests only
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
