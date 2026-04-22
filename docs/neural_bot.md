# Neural Bot — Architecture & Usage

## What it is

`bots/neural/` is a policy-value neural bot for Orbit Wars. It uses a multi-head MLP
trained via imitation learning on recorded heuristic matches. At inference it runs a
single forward pass and returns one launch action (or no-op) per turn.

---

## Files at a glance

```
bots/neural/
├── types.py          # Pure-numpy dataclasses. No PyTorch.
├── state_builder.py  # obs/StepRecord → ModelInput (numpy array + context)
├── action_codec.py   # Raw actions ↔ ModelLabels (encode for training, decode for inference)
├── model.py          # PolicyValueModel — PyTorch nn.Module
├── bot.py            # NeuralBot — wraps model for live play. agent_fn entry point.
└── training.py       # NeuralILDataset + build_il_dataset — IL data pipeline
```

---

## Data flow

### Inference (live game)

```
obs dict
  └─ StateBuilder.from_obs(obs, player)
       └─ ModelInput(array: np.ndarray (1050,), context: ActionContext)
            └─ NeuralBot.act() — tensor conversion + forward pass (no_grad)
                 └─ PolicyOutput (logits per head + value scalar)
                      └─ ActionCodec.decode(output, context, planets)
                           └─ [] or [[planet_id, angle, n_ships]]
```

### Imitation learning (offline)

```
HDF5 file
  └─ EpisodeReader.step(t) → StepRecord
       ├─ StateBuilder.from_step(step, player) → ModelInput
       └─ ActionCodec.encode(raw_actions, context, planets, value) → ModelLabels
            └─ ILSample(state_array, labels)
                 └─ NeuralILDataset → torch DataLoader
                      └─ PolicyValueModel (train)
```

---

## Key types

### `ActionContext`  (`types.py`)

Captures the per-turn planet index mapping needed by encode/decode.

| Field | Shape | Description |
|---|---|---|
| `planet_ids` | `(n,) int32` | Real game IDs of planets this turn |
| `planet_positions` | `(n, 2) float32` | Raw x, y (not normalised) |
| `my_planet_mask` | `(n,) bool` | True where owner == current player |
| `n_planets` | `int` | Number of active planets |

### `ModelInput`  (`types.py`)

Output of `StateBuilder`. Framework-agnostic.

| Field | Type | Description |
|---|---|---|
| `array` | `np.ndarray (D,) float32` | Flat state vector, ready for `torch.from_numpy` |
| `context` | `ActionContext` | Index mappings for encode/decode |

`D = max_planets * 7 + max_fleets * 7 = 1050` with defaults (50 planets, 100 fleets).

### `ModelLabels`  (`types.py`)

Training labels for one turn. All indices are `-1` for NO_OP turns.

| Field | Values |
|---|---|
| `action_type` | `0` = NO_OP, `1` = LAUNCH |
| `source_idx` | index into `planet_ids` |
| `target_idx` | index into `planet_ids` |
| `amount_bin` | `0–4` → fractions `[0.1, 0.25, 0.5, 0.75, 1.0]` of source ships |
| `value_target` | `+1.0` win, `-1.0` loss, `0.0` draw |

### `PolicyOutput`  (`model.py`)

Returned by `PolicyValueModel.forward`. All tensors have batch dim `B`.

| Field | Shape |
|---|---|
| `action_type_logits` | `(B, 2)` |
| `source_logits` | `(B, 50)` |
| `target_logits` | `(B, 50)` |
| `amount_logits` | `(B, 5)` |
| `value` | `(B, 1)` — tanh output |

---

## `StateBuilder`  (`state_builder.py`)

Stateless; safe for `DataLoader(num_workers > 0)`.

```python
from bots.neural.state_builder import StateBuilder

sb = StateBuilder(max_planets=50, max_fleets=100)  # defaults
print(sb.input_dim)  # 1050

# from a live obs dict
model_input = sb.from_obs(obs, player=0)

# from an offline StepRecord
model_input = sb.from_step(step, player=0)
```

**Planet features** (7 per slot, zero-padded): `owner_self`, `owner_enemy`, `owner_neutral`,
`x/100`, `y/100`, `ships/200` (clipped), `production/5`.

**Fleet features** (7 per slot, zero-padded): `owner_self`, `owner_enemy`,
`x/100`, `y/100`, `sin(angle)`, `cos(angle)`, `ships/200` (clipped).

---

## `ActionCodec`  (`action_codec.py`)

```python
from bots.neural.action_codec import ActionCodec

codec = ActionCodec(n_amount_bins=5)  # BINS = [0.1, 0.25, 0.5, 0.75, 1.0]

# Training: raw dataset action → ModelLabels
labels = codec.encode(raw_actions, context, planets, value_target=1.0)

# Inference: PolicyOutput → game action list
actions = codec.decode(policy_output, context, planets)
# returns [] or [[planet_id, angle, n_ships]]
```

`encode` selects the dominant action (largest ship count), maps `from_planet_id` to an
index, infers target via minimum angular difference, and bins the ship fraction.

`decode` runs argmax per head with masking: source is restricted to `my_planet_mask`,
target excludes source, then computes the launch angle from raw planet positions.

---

## `PolicyValueModel`  (`model.py`)

```python
from bots.neural.model import PolicyValueModel, PolicyValueConfig

config = PolicyValueConfig(
    input_dim=1050,
    hidden_dims=[256, 128],   # any depth works
    max_planets=50,
    n_amount_bins=5,
    dropout=0.1,
)
model = PolicyValueModel(config)

# Forward pass
import torch
state = torch.zeros(1, 1050)
output = model(state)   # PolicyOutput
print(output.value)     # (1, 1) — tanh
```

The MLP encoder (`self.encoder`, a public `nn.Sequential`) can be swapped for an
attention encoder without touching the heads or `NeuralBot`.

---

## `NeuralBot`  (`bot.py`)

```python
from bots.neural.bot import NeuralBot
from bots.neural.model import PolicyValueModel, PolicyValueConfig
from bots.neural.state_builder import StateBuilder
from bots.neural.action_codec import ActionCodec

config = PolicyValueConfig(input_dim=1050, hidden_dims=[256, 128])
bot = NeuralBot(
    model=PolicyValueModel(config),
    state_builder=StateBuilder(),
    codec=ActionCodec(),
    device="cpu",
)

actions = bot.act(obs)   # works with dict obs or object obs
```

### Loading a trained checkpoint

```python
bot = NeuralBot.load("path/to/checkpoint.pt", device="cpu")
```

Checkpoint format (what `torch.save` should receive):

```python
{
    "config": PolicyValueConfig(...),
    "state_dict": model.state_dict(),
    "max_planets": 50,
    "max_fleets": 100,
    "n_amount_bins": 5,
}
```

### `agent_fn` for kaggle_environments

`bot.py` exposes a module-level `agent_fn(obs, config)` that satisfies the
`kaggle_environments` submission interface. It lazy-initialises an untrained model.
Replace it with `NeuralBot.load()` once you have trained weights.

---

## `NeuralILDataset` + `build_il_dataset`  (`training.py`)

```python
from dataset.catalog import DataCatalog
from bots.neural.state_builder import StateBuilder
from bots.neural.action_codec import ActionCodec
from bots.neural.training import build_il_dataset

catalog = DataCatalog.scan()                         # all recorded matches
catalog = catalog.filter(bot="scoring", winner_only=True)

dataset = build_il_dataset(
    catalog,
    state_builder=StateBuilder(),
    codec=ActionCodec(),
    perspective="winner",   # "winner" | "loser" | "both"
)

from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

for batch in loader:
    # batch keys: state, action_type, source_idx, target_idx, amount_bin, value_target
    print(batch["state"].shape)   # (64, 1050)
```

`perspective="winner"` skips draws and only includes the winning player's turns —
useful for pure IL from winning behaviour.

Each `__getitem__` does the numpy→tensor conversion on the fly, so samples stay as
numpy until the DataLoader worker picks them up.

---

## Running the tests

```bash
# All tests (takes ~90s due to integration tests)
python -m pytest tests/ -q

# Just the neural foundation unit tests (~0.3s)
python -m pytest tests/unit/test_neural_foundation.py -v

# Only neural-related tests
python -m pytest tests/ -k neural -v
```

The 16 unit tests in `test_neural_foundation.py` cover:

| Group | What's tested |
|---|---|
| `ActionContext` | empty construction, field shapes |
| `ModelLabels` | NO_OP and LAUNCH construction |
| `StateBuilder` | `input_dim`, planet features, fleet features, empty state, `from_obs`/`from_step`/`__call__` equivalence |
| `ActionCodec.encode` | empty actions → NO_OP, valid LAUNCH, unknown planet ID |
| `ActionCodec.decode` | empty context → returns `[]` |

---

## What's not yet implemented

| Feature | Where to add |
|---|---|
| Training loop (loss, optimiser, epochs) | `training/train.py` or new `bots/neural/trainer.py` |
| Checkpoint saving | alongside the training loop |
| Action masking in cross-entropy loss (ignore `-1` labels) | training loop |
| Multiple actions per turn | requires sequential decoder — deferred |
| Attention / entity-centric encoder | swap `model.encoder` only |
| RL fine-tuning | `OrbitWarsEnv.step()` not yet implemented |
