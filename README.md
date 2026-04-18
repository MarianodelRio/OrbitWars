# OrbitWars

Kaggle Orbit Wars bot — an RTS-style competition where agents control planets and
fleets, submitting actions each turn. This repo is for iterative bot development:
design heuristic, implement, simulate locally, evaluate, and refine.

## Structure

```
bots/           Bot implementations (one file per strategy variant)
experiments/    Evaluation scripts and result logs
tests/          Simulation and unit tests
scripts/        Utility scripts (run_match.py, etc.)
docs/           Workflow documentation
game/           Game engine assets and rules
```

Game rules reference: [`game/rules.md`](game/rules.md)

## Setup

```bash
pip install kaggle-environments
```

## Usage

Run a local match between two bots:

```bash
python scripts/run_match.py
```

Run tests:

```bash
python -m pytest tests/
```

## Bot Conventions

- One file per bot: `bots/<strategy_name>.py`
- Each bot must define an `agent_fn(observation, configuration)` entry point
- Bots are self-contained — no cross-bot imports

## Evaluation

Use the `/orbit-eval` skill inside Claude Code for structured multi-match evaluation,
or run evaluation scripts directly from `experiments/`.

## Development Cycle

Each change follows a strict plan → implement → review cycle with context isolation
between roles. See `docs/` for workflow details.
