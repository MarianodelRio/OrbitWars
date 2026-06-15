# Orbit Wars — Kaggle RL Bot

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Orbit_Wars-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/competitions/orbit-wars)

An end-to-end deep RL agent for the [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) Kaggle competition — a real-time strategy game where bots control planets and fleets on a 100×100 board. The agent uses a custom entity-centric transformer trained first by imitation learning, then fine-tuned with multi-phase PPO and league-style self-play.

---

## Problem

Orbit Wars is a competitive RTS: 2–4 bots simultaneously manage fleets and planets over 500 turns. Each turn, a player can launch fleets from any owned planet toward any target. Planets orbit the sun, fleets travel at log-scaled speeds, and the player with the most total ships (on planets + in transit) wins.

The action space is combinatorial — any planet can target any other planet at any fraction of its garrison — making it unsuitable for classical RL approaches that assume small discrete action spaces. The full state includes orbiting planets, moving fleets, and threat vectors, which requires spatial and relational reasoning.

---

## Architecture — PlanetPolicyModel

`PlanetPolicyModel` is an **entity-centric transformer**: each planet makes an independent decision (action type, target, amount) while sharing context with all other planets and fleets through attention.

```
planet_features (B,P,24)    fleet_features (B,FL,16)    global_features (B,16)
        │                           │                           │
  Planet encoder              Fleet encoder                     │
  Linear→GELU→Linear→LN      Linear→GELU→Linear→LN            │
        │                           │                           │
        └────────── Cross-attention (fleets→planets) ──────────┘
                    Q=planets · K/V=fleets · 4 heads
                                 │
                     4× PlanetBlock
                     pre-LN self-attention + relational bias
                     (pairwise distance, angle, ownership projected into attn mask)
                                 │
                      Attention pooling + global MLP
                                 │
                           LSTM (episode memory across 500 turns)
                                 │
              ┌──────────────────┼──────────────────┐
         action type         target              amount
         logits (3)       (pointer net)       logits (8 bins)

  + 3 value heads: win/loss · score diff · shaped return
  + 5 auxiliary heads: per-planet ownership · opponent launch · multi-horizon returns
```

**Key design choices:**
- **Relational bias** — pairwise features (normalized distance, angle diff, ownership) projected into additive attention mask biases, giving the self-attention layers geometric priors without positional encodings.
- **Pointer network** for target selection — avoids a fixed target vocabulary; generalizes across maps with variable planet counts (20–40 planets).
- **Discrete amount bins** — 8 log-spaced ship-fraction values `[0.05, 0.1, …, 1.0]` instead of a continuous head.
- **LSTM recurrence** — single-step LSTM preserves garrison history and prior attack context across all 500 turns.
- **Auxiliary supervision** — per-planet ownership and opponent-launch prediction heads provide dense training signal beyond the sparse terminal reward.

Model size: ~8M parameters (E=192, G=384, 4 transformer layers, 8 heads, ffn=768).

---

## Training Pipeline

```
Heuristic bots (baseline · sniper · oracle_sniper · scoring)
    │  play matches → HDF5 episodes                        make data
    ▼
Imitation Learning                                         make train
  cross-entropy on action type, target, amount
  MSE on outcome + 5 auxiliary heads
  AdamW · cosine LR with warmup · bfloat16 AMP · 50 epochs
    │
    ▼  IL checkpoint
RL Phase 1 — Anchored exploration                          make train-phase1
  PPO + BC-KL penalty · fixed heuristic opponents
    │
    ▼
RL Phase 2 — Opponent pool                                 make train-phase2
  PPO · full heuristic pool · KL penalty decay
    │
    ▼
RL Phase 3 — League play                                   make train-phase3
  PPO · 15% self-play · frozen snapshot pool
  KL anchor decayed over 400 iterations
  bfloat16 AMP + torch.compile
    │
    ▼
submission/main.py → Kaggle                                make submit-neural
```

**PPO configuration:**

| Hyperparameter | Value |
|---|---|
| Algorithm | PPO + GAE (λ=0.95, γ=0.997) |
| Rollout steps | 8 192 per iteration |
| PPO epochs / batch | 4 epochs · batch size 512 |
| Value heads | 3 (outcome · score diff · shaped return) |
| Entropy | Per-head coefficients (action / target / amount) |
| KL anchor | BC-KL penalty, decayed over 400 iterations |
| Mixed precision | bfloat16 AMP + `torch.compile` |
| Opponents | Heuristic pool + 15% self-play + frozen snapshots |

---

## Results

Heuristic bot round-robin (10 matches per pair):

| Rank | Bot | Wins | Elo |
|---|---|---|---|
| 1 | `oracle_sniper` | 19 / 20 | 1159 |
| 2 | `random` | 10 / 20 | 1029 |
| 3 | `baseline` | 1 / 20 | 812 |

The neural bot trains against the full heuristic pool throughout all RL phases.

---

## Project Structure

```
bots/
  heuristic/          # Rule-based bots: baseline, sniper, oracle_sniper
  neural/             # PlanetPolicyModel, StateBuilder, ActionCodec, NeuralBot
  scoring/            # Scoring-based heuristic bot
  registry.py         # Short name → module:fn mapping
training/
  envs/               # OrbitWarsEnv (kaggle env wrapper)
  rl/                 # PPO loss, GAE, RolloutBuffer, OpponentPool
  rewards/            # PotentialReward (potential shaping + events + terminal)
  trainers/           # ILTrainer, RLTrainer
  evaluation/         # Evaluator (match-based win-rate)
  utils/              # RLConfig, RunConfig, CheckpointManager, MetricsLogger
dataset/              # IL dataset: HDF5 cache, torch adapter, transforms
game/
  env/                # kaggle env runner + agent loader
  eval/               # match metrics
  logic/              # combat, geometry, threat
  state/              # observation models
docs/                 # Detailed documentation per subsystem
tests/
  unit/               # geometry, combat, model shapes, PPO loss, action codec
  integration/        # bot smoke tests, dataset, RL env step
scripts/              # CLI helpers for matches, tournament, packaging
train.py              # Unified entry point (auto-detects IL vs RL from config)
```

---

## Installation

```bash
git clone https://github.com/MarianodelRio/OrbitWars
cd OrbitWars
bash setup.sh          # creates .venv, installs PyTorch (auto-detects CUDA), requirements
source .venv/bin/activate
```

Requirements: Python 3.10+, PyTorch ≥ 2.0, `kaggle-environments`, `h5py`, `numpy`.

**Hardware note:** RL training was run on a GCP compute instance with an NVIDIA L4 GPU (24 GB VRAM). The RL loop is CPU-bound (rollout collection via `kaggle_environments`), so a machine with multiple vCPUs matters more than GPU tier. `setup.sh` auto-detects the CUDA driver version and installs the correct PyTorch wheel.

---

## Usage

```bash
make match                                   # single local match
make tournament                              # round-robin tournament with Elo

make data                                    # generate IL training episodes (HDF5)
make cache                                   # build precomputed HDF5 training cache

make train                                   # Imitation Learning
make train-phase1 IL_CKPT=runs/.../best.pt   # RL phase 1 (requires IL checkpoint)
make train-phase2                            # RL phase 2 (warmstarts from phase 1)
make train-phase3                            # RL phase 3 (warmstarts from phase 2)
make pipeline                                # full pipeline, blocking

make eval CKPT=runs/.../rl_best_winrate.pt   # evaluate a checkpoint
make watch RUN=runs/<run>/<id>               # stream live training metrics

make submit-neural                           # package and submit to Kaggle
```

---

## Tests

```bash
make test        # all tests (unit + integration)
make test-unit   # unit tests only
```

Tests cover geometry, combat logic, model shapes, PPO loss, GAE, action codec, opponent pool, and full match integration.

---

## Documentation

| Document | Content |
|---|---|
| [docs/model.md](docs/model.md) | PlanetPolicyModel architecture, all feature tables, action codec |
| [docs/training.md](docs/training.md) | IL and RL training loops, full config reference |
| [docs/bots.md](docs/bots.md) | Bot catalog, `agent_fn` contract, registry API |
| [docs/data.md](docs/data.md) | HDF5 schema, DataCatalog, EpisodeReader, dataset pipeline |
| [docs/game.md](docs/game.md) | Observation structure, combat mechanics, fleet speed formula |
| [docs/evaluation.md](docs/evaluation.md) | Tournaments, Elo rating, Evaluator class |
| [docs/submission.md](docs/submission.md) | Packaging and submitting to Kaggle |
| [game/rules.md](game/rules.md) | Full game rules reference |
| [docs/training_guide.md](docs/training_guide.md) | GCP setup and end-to-end training walkthrough |
