# Neural Bot — Architecture & Usage

## What it is

`bots/neural/` is a policy-value neural bot for Orbit Wars. It supports two model architectures:

- **Flat MLP** (`PolicyValueModel`) — encodes the full state as a flattened 1050-dim vector and applies a multi-head MLP.
- **Pointer Network** (`PointerNetworkModel`) — encodes each planet individually with a shared MLP, uses dot-product attention to produce source/target logits that directly address planets. This enables better generalisation across different planet counts.

Both models are trained via imitation learning on recorded heuristic matches. At inference each runs a single forward pass and returns one launch action (or no-op) per turn.

---

## Files at a glance

```
bots/neural/
├── types.py           # Pure-numpy dataclasses. No PyTorch.
├── state_builder.py   # obs/StepRecord → ModelInput (flat) or StructuredModelInput (pointer)
├── action_codec.py    # Raw actions ↔ ModelLabels (encode for training, decode for inference)
├── model.py           # PolicyValueModel — flat MLP. PyTorch nn.Module.
├── pointer_model.py   # PointerNetworkModel — attention-based. PyTorch nn.Module.
├── bot.py             # NeuralBot — wraps either model for live play. agent_fn entry point.
└── training.py        # NeuralILDataset + build_il_dataset — IL data pipeline
```

---

## Data flow

### Inference — Flat MLP

```
obs dict
  └─ StateBuilder.from_obs(obs, player)
       └─ ModelInput(array: np.ndarray (1050,), context: ActionContext)
            └─ NeuralBot.act() — tensor conversion + forward pass (no_grad)
                 └─ PolicyOutput (logits per head + value scalar)
                      └─ ActionCodec.decode(output, context, planets)
                           └─ [] or [[planet_id, angle, n_ships]]
```

### Inference — Pointer Network

```
obs dict
  └─ StateBuilder.from_obs_structured(obs, player)
       └─ StructuredModelInput(planet_features (50,7), fleet_features (700,),
                               planet_mask (50,), context: ActionContext)
            └─ NeuralBot.act() — 3 tensors + forward pass (no_grad)
                 └─ PointerPolicyOutput (logits per head + value scalar)
                      └─ ActionCodec.decode(output, context, planets)
                           └─ [] or [[planet_id, angle, n_ships]]
```

`NeuralBot.act()` detects the model type automatically via `isinstance` — no caller change needed.

### Imitation learning — Flat MLP

```
HDF5 file
  └─ EpisodeReader.step(t) → StepRecord
       ├─ StateBuilder.from_step(step, player) → ModelInput
       └─ ActionCodec.encode(raw_actions, context, planets, value) → ModelLabels
            └─ ILSample(state_array, labels)
                 └─ NeuralILDataset(use_pointer=False) → torch DataLoader
                      └─ PolicyValueModel (train)
```

### Imitation learning — Pointer Network

```
HDF5 file
  └─ EpisodeReader.step(t) → StepRecord
       ├─ StateBuilder.from_step_structured(step, player) → StructuredModelInput
       └─ ActionCodec.encode(raw_actions, context, planets, value) → ModelLabels
            └─ ILSample(planet_features, fleet_features, planet_mask, labels)
                 └─ NeuralILDataset(use_pointer=True) → torch DataLoader
                      └─ PointerNetworkModel (train)
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

# --- Flat mode (PolicyValueModel) ---
model_input = sb.from_obs(obs, player=0)    # from live obs dict
model_input = sb.from_step(step, player=0)  # from offline StepRecord
# returns ModelInput(array: np.ndarray (1050,), context: ActionContext)

# --- Structured mode (PointerNetworkModel) ---
si = sb.from_obs_structured(obs, player=0)
si = sb.from_step_structured(step, player=0)
# returns StructuredModelInput TypedDict:
#   planet_features: np.ndarray (max_planets, 7) float32
#   fleet_features:  np.ndarray (max_fleets*7,)  float32
#   planet_mask:     np.ndarray (max_planets,)   bool  — True = real planet
#   context:         ActionContext
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
`decode` accepts either `PolicyOutput` or `PointerPolicyOutput` — same fields, same behaviour.

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

## `PointerNetworkModel`  (`pointer_model.py`)

```python
from bots.neural.pointer_model import PointerNetworkModel, PointerNetworkConfig

config = PointerNetworkConfig(
    planet_input_dim=7,
    fleet_input_dim=700,    # max_fleets * 7
    planet_embed_dim=64,
    global_dim=128,
    max_planets=50,
    max_fleets=100,
    n_amount_bins=5,
    dropout=0.1,
)
model = PointerNetworkModel(config)

# Forward pass
import torch
B = 4
planet_features = torch.randn(B, 50, 7)   # (batch, max_planets, 7)
fleet_features  = torch.randn(B, 700)     # (batch, max_fleets*7)
planet_mask     = torch.ones(B, 50, dtype=torch.bool)
planet_mask[:, 30:] = False               # mark slots 30-49 as padding

output = model(planet_features, fleet_features, planet_mask)  # PointerPolicyOutput
print(output.source_logits.shape)   # (B, 50)
print(output.value.shape)           # (B, 1)  — tanh
```

**Architecture:**
1. `planet_encoder` (shared MLP 7 → 64 → 64 with LayerNorm + dropout) embeds every planet slot independently.
2. `fleet_proj` (Linear 700 → 64) and `global_proj` (Linear 128 → 128, where 128 = global_dim) project fleet and global context.
3. **Source attention**: query from global context, key from planet embeddings; dot product scaled by `√planet_embed_dim`. Padding slots (where `planet_mask = False`) receive `-inf` logits before softmax.
4. **Target attention**: conditioned on a soft source embedding (weighted sum via source probabilities); same masking, plus the attended source slot is additionally masked out.
5. `action_type_head`, `amount_head`, `value_head` are shared linear heads on the global context.

`PointerPolicyOutput` has the same fields as `PolicyOutput` so `ActionCodec.decode()` works with both.

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

`NeuralBot.act()` automatically detects whether the wrapped model is a `PointerNetworkModel`
and calls `from_obs_structured` instead of `from_obs`. No caller changes required.

### Loading a trained checkpoint

```python
bot = NeuralBot.load("path/to/checkpoint.pt", device="cpu")
```

The checkpoint stores a `model_type` key (`"flat"` or `"pointer"`) so `NeuralBot.load()`
instantiates the correct class automatically. Checkpoint format (what `torch.save` should receive):

```python
{
    "config": PolicyValueConfig(...),   # or PointerNetworkConfig(...)
    "model_type": "flat",               # or "pointer"
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

The following were listed as pending and are now implemented: training loop (`training/trainers/il_trainer.py`), checkpoint saving (`training/utils/checkpointing.py`), action masking via `ignore_index=-1`.

| Feature | Where to add |
|---|---|
| Multiple actions per turn | requires sequential decoder — deferred |
| Attention / entity-centric encoder | swap `model.encoder` only |
| RL fine-tuning | `OrbitWarsEnv.step()` not yet implemented |
