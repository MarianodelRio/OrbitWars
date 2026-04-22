# Orbit Wars

A competitive bot for the [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) Kaggle competition — an RTS game where agents control planets and fleets on a 100×100 board with a central sun.

## Project Overview

Development follows an iterative loop: design a heuristic strategy, implement it as a bot, simulate matches locally, evaluate win rates, and refine. Bots are plain Python files with a single entry point. Packaging for Kaggle produces a self-contained `submission/main.py` with no internal imports.

## Project Structure

```
OrbitWars/
├── game/                    # Game engine — rules, simulation, shared utilities
│   ├── rules.md             # Full game rules reference
│   ├── state/models.py      # Planet, Fleet, GameState dataclasses + parse_obs()
│   ├── logic/
│   │   ├── geometry.py      # Math: dist, eta, orbit_predict, path_crosses_sun
│   │   ├── combat.py        # simulate_combat() — deterministic combat resolution
│   │   └── threat.py        # incoming_fleets(), is_under_attack()
│   └── env/
│       ├── runner.py        # run_match() — single match executor + HTML renderer
│       └── evaluator.py     # evaluate() — run N matches, return statistics
│
├── bots/                    # Bot implementations — one strategy per file
│   ├── interface.py         # Abstract Bot base class
│   └── heuristic/           # Handcrafted rule-based bots
│       ├── baseline.py      # Do-nothing baseline (return [])
│       ├── random_bot.py    # Sends half ships to a random target
│       └── sniper.py        # Captures nearest planet when garrison allows
│
├── dataset/                 # Dataset pipeline — see docs/dataset.md
│   ├── catalog.py           # DataCatalog + EpisodeMeta (episode discovery)
│   ├── episode.py           # EpisodeReader + StepRecord (HDF5 access)
│   ├── builder.py           # SampleBuilder + TrainingSample
│   ├── config.py            # PipelineConfig (JSON-driven factory)
│   ├── torch_adapter.py     # OrbitDataset + LazyOrbitDataset
│   └── transforms/          # State, action, reward transforms; step filters
│
├── training/                # Training infrastructure (RL / self-play)
│   ├── envs/
│   │   └── gym_wrapper.py   # Gymnasium-compatible wrapper (step() stub)
│   ├── rewards/
│   │   └── shaped.py        # Dense reward: Δplanets + 0.01·Δships + 0.1·Δproduction
│   └── train.py             # Self-play loop entry point
│
├── tournament/              # Tournament system
│   ├── runner.py            # run_tournament() — round-robin using evaluate()
│   ├── elo.py               # update_elo() — standard Elo rating update
│   └── results/             # Stored tournament result logs
│
├── submission/              # Kaggle submission artifact
│   └── main.py              # Self-contained agent (no external imports)
│
├── scripts/                 # Developer CLI tools
│   ├── run_match.py         # Run a single local match
│   ├── evaluate.py          # Head-to-head evaluation over N matches
│   ├── run_tournament.py    # Round-robin tournament across registered bots
│   └── package_submission.py # Copy current best bot → submission/main.py
│
└── tests/
    ├── unit/                # Pure logic tests (geometry, combat, models)
    └── integration/         # Full match tests (bot sanity + regression)
```

## Setup

```bash
pip install kaggle-environments
```

## Running a Local Match

```bash
# Default: sniper vs baseline, 500 steps
python scripts/run_match.py

# With HTML replay
python scripts/run_match.py --render --output game.html

# Custom bots
python scripts/run_match.py \
    --bot1 bots.heuristic.sniper:agent_fn \
    --bot2 bots.heuristic.random_bot:agent_fn \
    --steps 200
```

Open `game.html` in a browser to watch the full replay.

## Evaluating a Bot

```bash
# Win rate over 20 matches
python scripts/evaluate.py \
    --bot1 bots.heuristic.sniper:agent_fn \
    --bot2 bots.heuristic.random_bot:agent_fn \
    --n 20 --steps 500
```

Outputs: win rate, average ships, draws, average game length.

## Running the Tournament

```bash
# 5-match round-robin at 200 steps (default)
python scripts/run_tournament.py

# More matches for statistical confidence
python scripts/run_tournament.py --n 20 --steps 500
```

Prints a standings table sorted by wins. To register a new bot, add it to `REGISTERED_BOTS` in `scripts/run_tournament.py`.

## Adding a New Bot

1. Create `bots/heuristic/my_bot.py` (or `bots/rl/my_bot.py` for trained agents).
2. Define the entry point:
   ```python
   def agent_fn(obs, config=None):
       # obs has: player, planets, fleets, angular_velocity
       # return list of [from_planet_id, angle_radians, num_ships]
       return []
   ```
3. Add it to `REGISTERED_BOTS` in `scripts/run_tournament.py` to include it in tournaments.
4. Run `python scripts/run_match.py --bot1 bots.heuristic.my_bot:agent_fn` to test it.

Each bot must be self-contained (no cross-bot imports). See `game/rules.md` for the full action spec.

## Packaging for Kaggle

```bash
# Package the current best bot (default: sniper)
python scripts/package_submission.py

# Package a specific bot
python scripts/package_submission.py --bot bots.heuristic.sniper:agent_fn

# Submit to Kaggle
kaggle competitions submit orbit-wars -f submission/main.py -m "description"
```

`submission/main.py` is auto-generated — do not edit it manually. It is fully self-contained (no imports from `game/` or `bots/`) as required by Kaggle.

## Dataset Pipeline

`dataset/` converts recorded `.h5` episode files into training samples for imitation
learning or offline RL. See [docs/dataset.md](docs/dataset.md) for the full API
reference and [docs/pipeline_config.md](docs/pipeline_config.md) for JSON config.

## Running Tests

```bash
python -m pytest tests/ -q
```

Unit tests cover geometry, combat, and model parsing. Integration tests verify that each bot produces valid output and that the sniper beats the baseline in the majority of games.

## Development Workflow

This project follows a plan → implement → review cycle. See `CLAUDE.md` for role definitions and context isolation rules. Use the `/orbit-cycle` skill in Claude Code to run a full cycle.

For game mechanics, see [`game/rules.md`](game/rules.md).
