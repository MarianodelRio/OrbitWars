# Orbit Wars

## What Is This

A competitive bot for the [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) Kaggle competition — an RTS game where agents control planets and fleets on a 100x100 board with a central sun. Development is iterative: implement heuristics, generate training data, train a neural model, evaluate, and refine. The primary bot is `PlanetPolicyModel`, an entity-centric transformer trained by imitation learning and PPO.

## Quick Start

```bash
# Install dependencies
pip install kaggle-environments

# Run a local match (edit scripts/matches/config.json first)
python scripts/matches/run.py

# Train — Imitation Learning
python train.py --config training/il_config.json

# Train — Reinforcement Learning
python train.py --config training/rl_config.json
```

## Documentation Map

| Document | What It Covers |
|---|---|
| [`docs/model.md`](docs/model.md) | PlanetPolicyModel architecture, config, input features, action codec, LSTM state |
| [`docs/training.md`](docs/training.md) | IL and RL training loops, all config fields, losses, GPU setup, monitoring |
| [`docs/bots.md`](docs/bots.md) | Bot catalog, agent_fn contract, registry API, NeuralBot, creating new bots |
| [`docs/data.md`](docs/data.md) | HDF5 schema, DataCatalog, EpisodeReader, NeuralILDataset, PrecomputedILDataset |
| [`docs/game.md`](docs/game.md) | Observation structure, turn order, combat, fleet speed, orbiting planets, comets |
| [`docs/evaluation.md`](docs/evaluation.md) | Single match, tournament, Elo rating, Evaluator class |
| [`docs/submission.md`](docs/submission.md) | Packaging heuristic and neural bots, submission command, troubleshooting |
| [`game/rules.md`](game/rules.md) | Canonical game rules reference |

## System Map

```
Raw data (HDF5 episodes)
        |
   DataCatalog.scan()
        |
   EpisodeReader / StepRecord
        |
   StateBuilder (planet/fleet/global features)
   ActionCodec  (encode/decode labels)
        |
   NeuralILDataset  ──(cache)──>  PrecomputedILDataset
        |                                |
        +--------------------------------+
                    |
               ILTrainer
           (IL cross-entropy + MSE)
                    |
              checkpoint.pt
                    |
         NeuralBot.load(path)
                    |
      RLTrainer (PPO + opponent pool)
                    |
              checkpoint.pt
                    |
   package_neural.py  ──>  submission/main.py  ──>  Kaggle
```

## Running Tests

```bash
python -m pytest tests/ -q
```

Unit tests cover geometry, combat, and model shapes. Integration tests verify that each bot produces valid output for a full match.

## Development Workflow

This project uses a strict plan-implement-review cycle with three isolated roles (Planner, Implementer, Reviewer). Maximum 1–3 files changed per cycle. Never break local simulation. See [`CLAUDE.md`](CLAUDE.md) for the full role definitions, context isolation rules, and completion checklist.
