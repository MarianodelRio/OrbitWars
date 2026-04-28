# RL Training for Orbit Wars

## Purpose

This document describes the PPO-based reinforcement learning training pipeline for the Orbit Wars bot. The RL trainer uses `PlanetPolicyModel` (the same model as IL training) and trains it via self-play against a pool of opponents (heuristics and past snapshots of itself).

---

## How to Run

```bash
python scripts/train_rl.py --config training/rl_config.json
```

A minimal `rl_config.json`:

```json
{
  "run_name": "rl_v1",
  "run_id": "exp01",
  "total_iterations": 1000,
  "n_rollout_steps": 2048,
  "steps_per_episode": 500,
  "snapshot_every": 50,
  "eval_every": 100
}
```

---

## Key Config Fields

| Field | Default | Description |
|---|---|---|
| `total_iterations` | 1000 | Number of PPO update iterations |
| `n_rollout_steps` | 2048 | Steps collected per iteration before PPO update |
| `steps_per_episode` | 500 | Max steps per episode (matches kaggle env default) |
| `ppo_epochs` | 4 | PPO epochs per collected rollout |
| `ppo_batch_size` | 256 | Mini-batch size for PPO update |
| `clip_eps` | 0.2 | PPO clipping epsilon |
| `lr` | 3e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda for advantage estimation |
| `snapshot_every` | 50 | Save opponent snapshot every N iterations |
| `eval_every` | 100 | Run evaluation every N iterations |
| `max_snapshots` | 5 | Maximum past snapshots in opponent pool (oldest evicted) |
| `heuristic_opponents` | [baseline, proximity] | Heuristic agent paths for opponent pool |

---

## Reward Shaping

The reward uses potential-based shaping to provide dense feedback:

```
phi(s) = w_planets * (my_planets / total_planets)
       + w_production * (my_production / total_production)
       + w_ships * (my_ships / total_ships)

shaped_reward = lam * clip(gamma * phi(s') - phi(s), -clip_abs, clip_abs)
```

Terminal rewards are +1 (win), -1 (loss), 0 (draw) from the kaggle environment.

Total step reward = `terminal_reward + shaped_reward`.

Default weights: `w_planets=1.0`, `w_production=0.5`, `w_ships=0.1`, `lam=0.05`, `clip_abs=0.2`.

---

## Snapshots and Opponent Pool

Every `snapshot_every` iterations, the current model is saved as a snapshot checkpoint under:

```
runs/<run_name>/<run_id>/checkpoints/snapshots/snap_NNNNNN.pt
```

These snapshots are added to the opponent pool for future rollout collection (self-play). When the pool exceeds `max_snapshots`, the oldest snapshot is evicted. Heuristic opponents remain permanently in the pool.

At each iteration, one opponent is sampled uniformly at random from the pool.

---

## Monitoring Training

Training metrics are logged to CSV files under `runs/<run_name>/<run_id>/metrics/`:

- `rl_train.csv`: per-iteration PPO metrics (policy loss, value loss, entropy, KL, clip fraction, explained variance, episode count and mean reward)
- `rl_eval.csv`: evaluation results (win/draw/loss rates per opponent) at `eval_every` intervals

Example monitoring with pandas:

```python
import pandas as pd
df = pd.read_csv("runs/rl_v1/exp01/metrics/rl_train.csv")
df.plot(x="iteration", y=["policy_loss", "value_loss", "entropy"])
```
