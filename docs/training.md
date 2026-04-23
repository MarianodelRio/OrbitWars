# Training Guide

## Quick start

Verify that game data exists before launching training:

```bash
python scripts/probe_pipeline.py
```

Then launch imitation learning training:

```bash
python scripts/train_il.py --config training/il_config.json
# or equivalently
make train
```

## What a run generates

With the default `run_name: "neural_il"` and `run_id: ""` (auto-increment to `run_001`), the run directory is `runs/neural_il/run_001/`:

```
runs/neural_il/run_001/
├── config.json                          # Full resolved config snapshot written at startup
├── checkpoints/
│   ├── best.pt                          # Checkpoint with the lowest validation loss so far
│   ├── last.pt                          # Checkpoint from the most recent completed epoch
│   └── epoch_NNN.pt                     # Per-epoch checkpoint (e.g. epoch_001.pt, epoch_005.pt)
├── metrics/
│   ├── train.csv                        # Per-epoch training loss: fields epoch, loss
│   └── val.csv                          # Per-epoch validation loss: fields epoch, loss
└── eval/
    └── epoch_NNN_vs_<opponent>.json     # Evaluation result JSON written every eval_every epochs
```

## `il_config.json` field reference

### Top-level fields

| Field | Type | Default | Description |
|---|---|---|---|
| `run_name` | string | `"neural_il"` | Parent directory under `runs/` that groups related runs |
| `run_id` | string | `""` | Empty string triggers auto-increment to the next available `run_NNN`; a non-empty value forces an exact name and will overwrite an existing run directory |
| `lr` | float | `0.001` | Adam learning rate |
| `batch_size` | int | `64` | Mini-batch size for training and validation |
| `epochs` | int | `20` | Total number of training epochs |
| `val_split` | float | `0.2` | Fraction of episodes held out for validation; the split is by episode index (not by step) to prevent turn-level data leakage |
| `eval_every` | int | `5` | Run match evaluation every this many epochs |
| `eval_opponents` | list[string] | `["heuristic.baseline"]` | Opponents to evaluate against; must be keys registered in `OPPONENT_REGISTRY` in `training/evaluation/evaluator.py`; the two currently registered values are `"heuristic.baseline"` and `"heuristic.proximity_conqueror"` |
| `n_eval_matches` | int | `10` | Number of matches per opponent per evaluation |
| `data_pipeline` | object | — | Data loading and feature-building configuration; see `docs/pipeline_config.md` for the full field reference |
| `device` | string | `"cpu"` | PyTorch device string (`"cpu"`, `"cuda"`, `"mps"`) |
| `seed` | int | `42` | Random seed for reproducibility |
| `resume_from` | string\|null | `null` | Path to a `.pt` checkpoint whose weights are loaded into the model **before** training starts. Relative paths are resolved from the repository root. The run always gets a new `run_id` — metrics history is not carried over. |

### `model_config` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `input_dim` | int | `1050` | Flattened input vector size; must equal `max_planets * 7 + max_fleets * 7`; if `max_planets` changes, recalculate this value or the model will silently accept wrong-shaped tensors |
| `hidden_dims` | list[int] | `[256, 128]` | Width of each hidden layer in the MLP encoder |
| `max_planets` | int | `50` | Maximum number of planets in the state vector; must match the value used in `StateBuilder` — a mismatch causes checkpoint load failures at inference time |
| `n_amount_bins` | int | `5` | Number of discrete fleet-size bins used by `ActionCodec` |
| `dropout` | float | `0.1` | Dropout probability applied in the MLP encoder |

## Loading a checkpoint

### Use `best.pt` as a playable bot

```python
from bots.neural.bot import NeuralBot

bot = NeuralBot.load("runs/neural_il/run_001/checkpoints/best.pt")
# bot is a Bot subclass; wrap it with make_agent to get a callable agent_fn
from bots.interface import make_agent
agent_fn = make_agent(bot)
```

### Play matches or tournaments with a trained checkpoint

Any bot spec in `scripts/matches/config.json` or `scripts/tournament/config.json` that points to the neural bot can include a `?checkpoint=` suffix:

```json
{
  "bot1": "bots.neural.bot:agent_fn?checkpoint=runs/neural_il/run_001/checkpoints/best.pt",
  "bot2": "bots.heuristic.sniper:agent_fn"
}
```

The checkpoint is loaded lazily on the first call (once per match session). Relative paths are resolved from the repository root. Without `?checkpoint=` the neural bot uses random weights.

The same syntax works inside the tournament `bots` registry:

```json
"bots": {
  "neural_v1": "bots.neural.bot:agent_fn?checkpoint=runs/neural_il/run_001/checkpoints/best.pt",
  "sniper":    "bots.heuristic.sniper:agent_fn"
}
```

### Continue training from a checkpoint (`resume_from`)

Set `resume_from` in `training/il_config.json` to load weights before training starts:

```json
{
  "resume_from": "runs/neural_il/run_001/checkpoints/best.pt"
}
```

The script prints a confirmation line and then trains normally. A new `run_id` is assigned automatically (`run_002`, `run_003`, …) so the original run is never overwritten. Set `resume_from` back to `null` to start from random weights again.

### Run standalone evaluation against registered opponents

```python
from pathlib import Path
from training.evaluation.evaluator import Evaluator

evaluator = Evaluator.from_checkpoint(
    checkpoint_path=Path("runs/neural_il/run_001/checkpoints/best.pt"),
    opponents=["heuristic.baseline", "heuristic.proximity_conqueror"],
    n_matches=20,
)
results = evaluator.run()
```

`Evaluator.from_checkpoint` accepts `checkpoint_path` (Path), `opponents` (list of registry key strings), and `n_matches` (int, default 10). It returns an `Evaluator` instance; call `.run()` to execute the matches.

## Adding an opponent to the evaluator

`OPPONENT_REGISTRY` in `training/evaluation/evaluator.py` lines 14-17 is the sole place to register new opponents. Add an entry with the format:

```python
OPPONENT_REGISTRY = {
    "heuristic.baseline": "bots.heuristic.baseline:agent_fn",
    "heuristic.proximity_conqueror": "bots.heuristic.proximity_conqueror:agent_fn",
    "my_bot.name": "bots.my_bot:agent_fn",   # new entry
}
```

The key is the string used in `eval_opponents` inside `il_config.json`. The value is a `module.path:attribute` string that `load_agent` resolves via `importlib`.

## RL note

`RLTrainer` exists as a placeholder class in `training/trainers/`. The checkpoint format, `CheckpointManager`, and `Evaluator` are already designed to be reusable for an RL training loop without changes. What remains to implement RL fine-tuning is: (1) `OrbitWarsEnv.step()` — the Gym-style step function that applies an action and returns the next observation and reward; and (2) a policy-gradient loop (e.g. PPO or REINFORCE) that calls `step()`, accumulates returns, and updates the model. See `training/trainers/il_trainer.py` as a reference for how the existing trainer hooks fit together.
