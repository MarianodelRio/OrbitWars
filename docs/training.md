# Training Guide

## Quick Start

```bash
# Imitation Learning (IL)
python train.py --config training/il_config.json

# Reinforcement Learning (PPO)
python train.py --config training/rl_config.json

# Dry run — print config and exit without training
python train.py --config training/rl_config.json --dry-run

# Override device
python train.py --config training/il_config.json --device cuda
```

Alternatively, use Makefile targets:

```bash
make train                                              # IL training
make train-phase1 IL_CKPT=runs/.../best.pt             # RL phase 1 (requires IL checkpoint)
make train-phase2                                       # RL phase 2 (warmstarts from phase 1 rl_last.pt)
make train-phase3                                       # RL phase 3 (warmstarts from phase 2 rl_last.pt)
make pipeline                                           # full pipeline, blocking (runs all steps sequentially)
make cache                                              # build HDF5 training cache
make eval CKPT=runs/.../rl_best_winrate.pt             # evaluate a checkpoint
make watch RUN=runs/<run>/<id>                          # stream live training metrics
```

Mode is auto-detected: configs containing `total_iterations` + `n_rollout_steps` are treated as RL; configs containing `epochs` are treated as IL.

---

## Imitation Learning (IL)

### How It Works

1. `DataCatalog.scan()` discovers all `.h5` episodes under `data/matches/` and `data/tournaments/`.
2. Optional filters (bot name, winner-only, min/max steps) are applied.
3. Episodes are shuffled and split into train/val by episode (not by step).
4. `NeuralILDataset` (lazy) or `PrecomputedILDataset` (cached) provides `(state, labels)` batches.
5. The model is trained with a combination of cross-entropy and MSE losses for multiple epochs.
6. Checkpoints are saved each epoch; best checkpoint by validation loss is tagged `best`.

### il_config.json Reference

All fields correspond to `RunConfig` in `training/utils/run_config.py`.

| Field | Type | Default | Description |
|---|---|---|---|
| `run_name` | str | required | Directory name under `runs/` |
| `run_id` | str | required | Sub-run ID (auto-incremented if empty) |
| `model_config` | dict | required | PlanetPolicyConfig fields as a dict |
| `lr` | float | required | Learning rate |
| `batch_size` | int | required | Samples per gradient step |
| `epochs` | int | required | Total training epochs |
| `val_split` | float | required | Fraction of episodes for validation |
| `eval_every` | int | required | Run evaluator every N epochs |
| `eval_opponents` | list | required | Registry names to evaluate against |
| `n_eval_matches` | int | required | Matches per evaluation opponent |
| `data_pipeline` | dict | required | Catalog and builder configuration (see below) |
| `device` | str | required | `"cpu"`, `"cuda"`, or `"auto"` |
| `seed` | int | required | Random seed for episode shuffle |
| `weight_decay` | float | 1e-4 | AdamW / Adam weight decay |
| `action_type_loss_weight` | float | 1.0 | Weight on action type CE loss |
| `value_loss_weight` | float | 0.5 | Weight on outcome MSE loss |
| `use_class_weights` | bool | True | Inverse-frequency class weighting for CE losses |
| `resume_from` | str / None | None | Path to checkpoint to resume from |
| `angular_diff_threshold` | float | π/4 | Max angle error for target inference in codec |
| `lr_schedule` | str | `"constant"` | One of `"constant"`, `"cosine"`, `"step"`, `"cosine_with_warmup"` |
| `early_stopping_patience` | int | 0 | Stop if no improvement for N epochs (0=disabled) |
| `optimizer` | str | `"adam"` | `"adam"` or `"adamw"` |
| `lr_min` | float | 1e-5 | Minimum LR for cosine schedule |
| `warmup_epochs` | int | 0 | Warm-up epochs for `cosine_with_warmup` |
| `num_workers` | int | 0 | DataLoader worker count |
| `pin_memory` | bool | False | Pin tensors to GPU memory for faster transfer |
| `persistent_workers` | bool | False | Keep DataLoader workers alive between epochs |
| `use_amp` | bool | False | Automatic Mixed Precision (CUDA only) |
| `amp_dtype` | str | `"bfloat16"` | AMP dtype: `"bfloat16"` or `"float16"` |
| `augment_reflection` | bool | False | Random x/y-axis reflection augmentation |
| `score_diff_loss_weight` | float | 0.3 | Weight on score-diff MSE auxiliary loss |
| `aux_ownership_weight` | float | 0.1 | Weight on per-planet ownership BCE loss |
| `aux_opponent_launch_weight` | float | 0.1 | Weight on per-planet opponent-launch BCE loss |

The `data_pipeline` dict contains two sub-dicts:
- `catalog`: `roots` (list of paths), `filter` (bot, opponent, winner_only, done_reason, min_steps, max_steps, max_episodes)
- `builder`: `perspective` ("winner"/"loser"/"both"), `step_filter`, `cache_path`

### Losses

All losses use `ignore_index = -1` — labels set to `-1` (padding planets and non-owned planets) are excluded from every loss.

| Loss | Term | Notes |
|---|---|---|
| Action type CE | `action_type_loss_weight * CE(at_logits, at_labels)` | Optional inverse-frequency class weights |
| Target CE | `CE(target_logits, target_labels)` | Flat CE over pointer labels |
| Amount CE | `CE(amount_logits, amount_labels)` | Optional inverse-frequency class weights |
| Outcome MSE | `value_loss_weight * MSE(v_outcome, value_target)` | Win=1, Loss=-1, Draw=0 |
| Score diff MSE | `score_diff_loss_weight * MSE(v_score_diff, score_diff)` | |
| Ownership BCE | `aux_ownership_weight * BCE(aux_ownership_10, ownership_10)` | Per-planet, masked to real planets |
| Opponent launch BCE | `aux_opponent_launch_weight * BCE(aux_opponent_launch, opponent_launch)` | Per-planet, masked |

Class weights are clipped to `[0.1, 10.0]` relative to their mean to prevent extreme values.

### Checkpoint Layout

```
runs/<run_name>/<run_id>/
  checkpoints/
    epoch_001.pt
    epoch_002.pt
    ...
    best.pt           # epoch with lowest val_loss
    last.pt           # most recent epoch
  metrics/
    train_metrics.csv
    val_metrics.csv
  eval/
    epoch_NNN_vs_<opponent>.json
  config.json
```

---

## Three-Phase RL Curriculum

RL training is split into three sequential phases, each building on the previous checkpoint.

**Phase 1** (`training/rl_phase1.json`) — 300 iterations
- Opponents: heuristic-only (no self-play).
- KL-BC regularization and IL distillation anchors are active to keep the policy close to the IL warmstart.
- Requires `IL_CKPT` pointing to a trained `best.pt` from the IL run.
- Launch with: `make train-phase1 IL_CKPT=runs/<run>/<id>/checkpoints/best.pt`

**Phase 2** (`training/rl_phase2.json`) — 500 iterations
- Opponents: 40% self-play, remainder from heuristic pool.
- KL-BC and IL distillation anchors are removed.
- Warmstarts automatically from phase 1 `rl_last.pt`.
- Launch with: `make train-phase2`

**Phase 3** (`training/rl_phase3.json`) — 700 iterations
- Opponents: 70% self-play.
- Lower learning rate (`lr=1e-4`) for fine-tuning.
- Warmstarts automatically from phase 2 `rl_last.pt`.
- Launch with: `make train-phase3`

**Key checkpoint:** `rl_best_winrate.pt` — saved whenever the model achieves its highest mean win-rate across eval opponents during RL training. This is the checkpoint to submit.

To run all three phases sequentially in one blocking call: `make pipeline`

---

## Reinforcement Learning (RL) with PPO

### Training Loop

`RLTrainer.train()` runs a fixed number of iterations:

1. **Collect rollout**: Play `n_rollout_steps` steps against a sampled opponent. LSTM hidden state persists within each episode and resets on episode end. After collection, GAE advantages are computed with `gamma` and `gae_lambda`.
2. **PPO update**: Run `ppo_epochs` passes over the buffer in mini-batches of size `ppo_batch_size`. Computes clipped policy loss, value loss, and entropy bonus. Optionally mixes in IL distillation batches or a KL-BC regularization term.
3. **Logging**: Metrics are logged to CSV; the buffer is cleared.
4. **Snapshot**: Every `snapshot_every` iterations, the current model is saved as a snapshot and added to the opponent pool.
5. **Checkpoint**: Every `save_every` iterations, a full checkpoint including optimizer state is saved to `rl_last.pt`.
6. **Evaluation**: Every `eval_every` iterations, the current model is evaluated against `eval_opponents`.

Auto-resume: if `runs/<run_name>/<run_id>/checkpoints/rl_last.pt` exists, training resumes from that iteration.

### Reward

The reward at each step is the sum of three components:

**Shaped reward** (potential-based):
```
r_shaped = lam * (gamma * Phi(s') - Phi(s))
Phi(s) = w_production * (my_production / total_production)
       + w_planets    * (my_planets / total_planets)
       + w_ships      * log(1 + my_ships) / log(1001)

where my_ships = sum of ships on owned planets + sum of ships in owned fleets
```

Fleet ships are included so the agent is not penalized for launching fleets.

**Event rewards:**

| Event | Value |
|---|---|
| Capture enemy planet | `r_event_capture_enemy` (default 0.5) |
| Lose a planet | `r_event_lose_planet` (default -0.3) |
| Eliminate opponent | `r_event_eliminate_opponent` (default 1.0) |
| Capture comet | `r_event_capture_comet` (default 0.2) |
| Explore bonus (first combat on a planet) | `r_explore` (default 0.01, active for first `explore_iterations`) |

**Terminal rewards:**

| Outcome | Value |
|---|---|
| Win | `r_terminal_win + r_terminal_margin_coef * margin` (defaults: 10.0 + 5.0*margin) |
| Loss | `r_terminal_loss + r_terminal_margin_coef * margin` (defaults: -10.0 + 5.0*margin) |

`margin = (my_ships - max_opp_ships) / (my_ships + max_opp_ships)`.

### PPO Loss

```
total_loss = policy_loss - entropy_term + vf_coef * value_loss + kl_bc_coef * kl_bc
```

- **Policy loss**: clipped surrogate with `clip_eps`.
- **Entropy term**: asymmetric per-head coefficients: `entropy_coef_action_type`, `entropy_coef_target`, `entropy_coef_amount`.
- **Value loss**: MSE between predicted `v_shaped` and GAE returns, scaled by `vf_coef`.
- **KL-BC term**: KL divergence from the current policy to a frozen BC reference policy (optional).

### Opponent Pool

The pool maintains up to `max_snapshots` historical snapshots of the trained model plus any configured `heuristic_opponents`. At the start of each rollout, an opponent is sampled:
- With probability `self_play_prob`: use the current model.
- Otherwise: sample uniformly from the pool (heuristic + snapshots).

A `frozen_checkpoint` can be specified to seed the pool with a fixed pre-trained opponent.

### IL Distillation

If `il_distill_ratio > 0` and `il_data_cache_path` points to a valid HDF5 cache, then during each PPO mini-batch loop, with probability `il_distill_ratio` a mini-batch is drawn from the IL dataset instead of the RL buffer, and a cross-entropy loss (action type + target + amount, `ignore_index=-1`) is applied. This forces the policy to stay close to imitation data.

### KL-to-BC Regularization

If `bc_policy_path` is set, a frozen copy of that checkpoint is loaded as the BC reference model. During PPO updates, a KL divergence penalty between the current policy and the BC policy is added to the loss. The coefficient decays linearly from `kl_bc_coef_start` to `kl_bc_coef_end` over `kl_bc_coef_decay_iters` iterations.

### rl_config.json Reference

All fields correspond to `RLConfig` in `training/utils/rl_config.py`.

| Field | Type | Default | Description |
|---|---|---|---|
| `n_rollout_steps` | int | 2048 | Steps collected per iteration |
| `n_envs` | int | 1 | Number of parallel environments (currently single-env) |
| `steps_per_episode` | int | 500 | Max steps per episode |
| `ppo_epochs` | int | 4 | Gradient passes over the rollout buffer |
| `ppo_batch_size` | int | 256 | Mini-batch size for PPO updates |
| `clip_eps` | float | 0.2 | PPO clipping epsilon |
| `vf_coef` | float | 0.5 | Value loss coefficient |
| `ent_coef` | float | 0.01 | Global entropy coefficient (overridden per-head by asymmetric coefs) |
| `max_grad_norm` | float | 0.5 | Gradient clip norm |
| `lr` | float | 3e-4 | Learning rate |
| `normalize_advantages` | bool | True | Standardize advantages per mini-batch |
| `gamma` | float | 0.99 | Discount factor |
| `gae_lambda` | float | 0.95 | GAE lambda |
| `w_planets` | float | 1.0 | Potential weight for planet share |
| `w_production` | float | 0.5 | Potential weight for production share |
| `w_ships` | float | 0.1 | Potential weight for ship log-share |
| `reward_lambda` | float | 0.1 | Shaping scale `lam` |
| `reward_clip_abs` | float | 0.2 | Deprecated; ignored |
| `r_terminal_win` | float | 10.0 | Win terminal reward |
| `r_terminal_loss` | float | -10.0 | Loss terminal reward |
| `r_terminal_margin_coef` | float | 5.0 | Margin multiplier on terminal reward |
| `r_event_capture_enemy` | float | 0.5 | Reward for capturing an enemy planet |
| `r_event_capture_comet` | float | 0.2 | Reward for capturing a comet |
| `r_event_eliminate_opponent` | float | 1.0 | Reward for eliminating an opponent |
| `r_event_lose_planet` | float | -0.3 | Penalty for losing a planet |
| `r_event_ships_wasted_coef` | float | 0.0 | Penalty coef for ships lost in failed attacks |
| `r_explore` | float | 0.01 | Exploration bonus for first planet contact |
| `explore_iterations` | int | 200 | Iterations during which exploration bonus is active |
| `max_snapshots` | int | 5 | Max historical model snapshots in opponent pool |
| `snapshot_every` | int | 50 | Save snapshot every N iterations |
| `heuristic_opponents` | list | `["bots.heuristic.baseline:agent_fn"]` | Heuristic opponents for the pool |
| `frozen_checkpoint` | str / None | None | Checkpoint to seed the pool with |
| `self_play_prob` | float | 0.3 | Probability of using current model as opponent |
| `eval_every` | int | 100 | Evaluate every N iterations |
| `n_eval_matches` | int | 10 | Matches per evaluation opponent |
| `eval_opponents` | list | `["heuristic.baseline"]` | Registry names for evaluation |
| `save_every` | int | 100 | Save checkpoint every N iterations |
| `lr_schedule` | str | `"cosine"` | `"cosine"` or `"constant"` |
| `run_name` | str | `"rl_run"` | Directory name under `runs/` |
| `run_id` | str | `""` | Sub-run ID |
| `device` | str | `"cpu"` | `"cpu"`, `"cuda"`, or `"auto"` |
| `seed` | int | 42 | Random seed |
| `total_iterations` | int | 1000 | Total training iterations |
| `model_config` | dict | `{}` | PlanetPolicyConfig overrides |
| `bc_policy_path` | str | `""` | Path to frozen BC reference checkpoint |
| `kl_bc_coef_start` | float | 1.0 | Initial KL-BC coefficient |
| `kl_bc_coef_end` | float | 0.1 | Final KL-BC coefficient after decay |
| `kl_bc_coef_decay_iters` | int | 500 | Iterations over which KL-BC coef decays |
| `il_distill_ratio` | float | 0.1 | Fraction of mini-batches replaced by IL data |
| `il_data_cache_path` | str | `""` | Path to pre-computed IL cache HDF5 |
| `entropy_coef_action_type` | float | 0.02 | Per-head entropy coef for action type |
| `entropy_coef_target` | float | 0.005 | Per-head entropy coef for target |
| `entropy_coef_amount` | float | 0.005 | Per-head entropy coef for amount |

---

## Evaluation CLI

```bash
# Evaluate a checkpoint against all registered bots (20 matches each)
python train.py eval --checkpoint runs/<run_name>/<run_id>/checkpoints/best.pt

# Specific opponents and match count
python train.py eval \
    --checkpoint runs/.../checkpoints/rl_last.pt \
    --opponents heuristic.baseline heuristic.sniper \
    --n-matches 30
```

Output is printed as JSON with per-opponent win rate, draw rate, loss rate, and average scores.

---

## GPU Training

### Prerequisites

- GCP VM with GPU (T4 or better)
- NVIDIA driver installed on the VM
- Python 3.10+ on the VM
- `gcloud` CLI configured locally with `--project` and `--zone`

### Setup in 3 commands

```bash
git clone <repo-url> && cd OrbitWars
bash setup.sh
source .venv/bin/activate
```

### Transfer data from Windows

Option A (primary):
```bash
gcloud compute scp --recurse ./data/ <VM_NAME>:~/OrbitWars/data/ --project <PROJECT> --zone <ZONE>
```

Option B (requires SSH key or WSL/Git Bash):
```bash
rsync -avz ./data/ <user>@<VM_IP>:~/OrbitWars/data/
```

### Training commands

```bash
make train                                              # IL training
make train-phase1 IL_CKPT=runs/.../best.pt             # RL phase 1 (heuristic opponents + BC anchors)
make pipeline                                           # full pipeline, blocking
```

### Troubleshooting

- `nvidia-smi: command not found` — NVIDIA driver is not installed; `setup.sh` falls back to the CPU wheel automatically.
- `torch.cuda.is_available()` returns `False` on a GPU VM — check your driver version; re-run `setup.sh` (it is idempotent).

---

## Monitoring

Use `make watch RUN=runs/<run_name>/<run_id>` to stream live training metrics to the terminal.

Training metrics are written as CSV files to `runs/<run_name>/<run_id>/metrics/`:

```
runs/<run_name>/<run_id>/metrics/
  train_metrics.csv      # per-epoch: epoch, loss (IL) or per-iter: iteration, policy_loss, value_loss, entropy (RL)
  val_metrics.csv        # per-epoch: epoch, loss (IL only)
```

Example: load and plot IL training curve:

```python
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("runs/my_run/run_001/metrics/train_metrics.csv")
val   = pd.read_csv("runs/my_run/run_001/metrics/val_metrics.csv")

plt.plot(train["epoch"], train["loss"], label="train")
plt.plot(val["epoch"],   val["loss"],   label="val")
plt.legend()
plt.show()
```
