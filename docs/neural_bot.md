# Neural Bot — Architecture & Usage

## What it is

`bots/neural/` is a policy-value neural bot for Orbit Wars. It supports three model architectures:

- **Flat MLP** (`PolicyValueModel`) — encodes the full state as a flattened 1050-dim vector and applies a multi-head MLP. Single global action per turn.
- **Pointer Network** (`PointerNetworkModel`) — encodes each planet individually with a shared MLP, uses dot-product attention to select source/target planets. Better generalisation across planet counts.
- **Planet Policy** (`PlanetPolicyModel`) — entity-centric v2 architecture. Each owned planet independently decides whether and where to launch. Self-attention across planets captures inter-dependencies. This is the primary active-development model.

All models are trained via imitation learning on recorded heuristic matches. `NeuralBot` detects the model type automatically and dispatches to the correct state builder and codec.

---

## Files at a glance

```
bots/neural/
├── types.py                 # Pure-numpy dataclasses (ActionContext, ModelLabels, PerPlanetLabels…)
├── state_builder.py         # obs/StepRecord → flat ModelInput or pointer StructuredModelInput
├── state_builder_v2.py      # obs/StepRecord → StructuredStateV2 (entity tensors, v2)
├── action_codec.py          # Single dominant-action encode/decode (flat + pointer)
├── action_codec_v2.py       # Per-planet encode/decode for PlanetPolicyModel
├── model.py                 # PolicyValueModel — flat MLP
├── pointer_model.py         # PointerNetworkModel — attention-based
├── planet_policy_model.py   # PlanetPolicyModel — per-planet entity-centric (v2)
├── bot.py                   # NeuralBot — wraps any model for live play
└── training.py              # ILSample, NeuralILDataset, build_il_dataset
```

---

## Data flow

### Inference — Planet Policy (v2)

```
obs dict
  └─ StateBuilderV2.from_obs(obs, player)
       └─ StructuredStateV2:
            planet_features (50, 10) float32
            fleet_features  (200, 8) float32
            fleet_mask      (200,)   bool
            planet_mask     (50,)    bool
            global_features (4,)     float32
            context: ActionContext
            └─ NeuralBot.act() — 5 tensors + forward pass (no_grad)
                 └─ PlanetPolicyOutput:
                      action_type_logits (B, 50, 2)
                      target_logits      (B, 50, 50)
                      amount_logits      (B, 50, 5)
                      value              (B, 1)
                      └─ ActionCodecV2.decode_per_planet(output, context, planets)
                           └─ [[planet_id, angle, n_ships], ...]  (one per launching planet)
```

### Inference — Flat MLP

```
obs dict
  └─ StateBuilder.from_obs(obs, player)
       └─ ModelInput(array: np.ndarray (1050,), context: ActionContext)
            └─ NeuralBot.act() — tensor + forward pass (no_grad)
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

### Imitation learning — Planet Policy (v2)

```
HDF5 file
  └─ EpisodeReader.step(t) → StepRecord
       ├─ StateBuilderV2.from_step(step, player) → StructuredStateV2
       └─ ActionCodecV2.encode_per_planet(raw_actions, context, planets, value, max_planets)
            └─ PerPlanetLabels((max_planets,) × 3 label arrays + my_planet_mask)
                 └─ ILSample(planet_features_v2, fleet_features_v2, fleet_mask,
                              global_features, labels_v2)
                      └─ NeuralILDataset(use_planet_policy=True) → torch DataLoader
                           └─ PlanetPolicyModel (train) — per-planet CE loss
```

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

Output of `StateBuilder` (flat mode). Framework-agnostic.

| Field | Type | Description |
|---|---|---|
| `array` | `np.ndarray (D,) float32` | Flat state vector, ready for `torch.from_numpy` |
| `context` | `ActionContext` | Index mappings for encode/decode |

`D = max_planets * 7 + max_fleets * 7 = 1050` with defaults (50 planets, 100 fleets).

### `StructuredStateV2`  (`state_builder_v2.py`)

Output of `StateBuilderV2`. Contains separate entity tensors.

| Key | Shape | Description |
|---|---|---|
| `planet_features` | `(max_planets, 10) float32` | Per-planet features, zero-padded |
| `fleet_features` | `(max_fleets, 8) float32` | Per-fleet features, zero-padded |
| `fleet_mask` | `(max_fleets,) bool` | True for real fleets |
| `planet_mask` | `(max_planets,) bool` | True for real planets |
| `global_features` | `(4,) float32` | Game-level context |
| `context` | `ActionContext` | Index mappings for encode/decode |

**Planet features** (10 per slot): `is_mine`, `is_enemy`, `is_neutral`, `x/100`, `y/100`,
`ships/200` (clipped), `production/5`, `radius/3`, `is_comet`, `dist_from_center/50`.

**Fleet features** (8 per slot): `is_mine`, `is_enemy`, `x/100`, `y/100`,
`sin(angle)`, `cos(angle)`, `ships/200` (clipped), `dist_from_center/50`.

**Global features** (4): `turn/500`, `my_ship_fraction`, `my_planet_fraction`, `fleet_density`.

### `ModelLabels`  (`types.py`)

Training labels for one turn (flat/pointer mode). All indices are `-1` for NO_OP turns.

| Field | Values |
|---|---|
| `action_type` | `0` = NO_OP, `1` = LAUNCH |
| `source_idx` | index into `planet_ids` |
| `target_idx` | index into `planet_ids` |
| `amount_bin` | `0–4` → fractions `[0.1, 0.25, 0.5, 0.75, 1.0]` of source ships |
| `value_target` | `+1.0` win, `-1.0` loss, `0.0` draw |

### `PerPlanetLabels`  (`types.py`)

Training labels for one turn (planet-policy v2 mode). One label per planet slot.
Non-mine and padding slots use `-1` as `ignore_index` in the loss.

| Field | Shape | Description |
|---|---|---|
| `planet_action_types` | `(max_planets,) int32` | `0`=NO_OP, `1`=LAUNCH, `-1`=PADDING |
| `planet_target_idxs` | `(max_planets,) int32` | Target planet index or `-1` |
| `planet_amount_bins` | `(max_planets,) int32` | Amount bin `0–4` or `-1` |
| `my_planet_mask` | `(max_planets,) bool` | True for own planets |
| `value_target` | `float` | `+1.0` win, `-1.0` loss, `0.0` draw |

### `PlanetPolicyOutput`  (`planet_policy_model.py`)

Returned by `PlanetPolicyModel.forward`. Per-planet decisions for the whole fleet.

| Field | Shape | Description |
|---|---|---|
| `action_type_logits` | `(B, max_planets, 2)` | NO_OP vs LAUNCH per planet |
| `target_logits` | `(B, max_planets, max_planets)` | Pointer attention matrix |
| `amount_logits` | `(B, max_planets, 5)` | Ship fraction bin per planet |
| `value` | `(B, 1)` | Game outcome estimate (tanh) |

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

## `PlanetPolicyModel`  (`planet_policy_model.py`)

The primary v2 model. Each owned planet independently decides whether and where to launch,
sharing information via self-attention.

```python
from bots.neural.planet_policy_model import PlanetPolicyModel, PlanetPolicyConfig

config = PlanetPolicyConfig(
    Dp=10, Df=8, Dg=4,          # input dims
    E=64, F=32, G=128,           # embedding dims
    max_planets=50, max_fleets=200,
    n_amount_bins=5, dropout=0.1, n_attn_heads=2,
)
model = PlanetPolicyModel(config)

import torch
B = 4
planet_features = torch.randn(B, 50, 10)
fleet_features  = torch.randn(B, 200, 8)
fleet_mask      = torch.ones(B, 200, dtype=torch.bool)
global_features = torch.randn(B, 4)
planet_mask     = torch.ones(B, 50, dtype=torch.bool)
planet_mask[:, 30:] = False   # padding

output = model(planet_features, fleet_features, fleet_mask, global_features, planet_mask)
print(output.action_type_logits.shape)  # (4, 50, 2)
print(output.target_logits.shape)       # (4, 50, 50)
print(output.value.shape)               # (4, 1)
```

**5-stage architecture:**

1. **Planet Encoder** (shared MLP): `Linear(Dp, E) → ReLU → Linear(E, E) → ReLU`
2. **Fleet Encoder + masked mean-pool**: `Linear(Df, F) → ReLU` → fleet context `(B, F)`
3. **Self-Attention** (1 layer, `n_attn_heads` heads): `MHA(planet_emb, planet_emb, planet_emb, key_padding_mask=~planet_mask)` + LayerNorm residual
4. **Global MLP**: `concat(planet_pool, fleet_ctx, global_features) → Linear(E+F+Dg, G) → ReLU → Linear(G, G)` → global repr `(B, G)`
5. **Per-planet heads** (`h_i = concat(planet_ctx_i, global_repr)`, shape `(B, E+G)`):
   - `action_type_head: Linear(E+G, 2)`
   - `amount_head: Linear(E+G, 5)`
   - **Pointer**: `queries = W_query(h)`, `keys = W_key(planet_ctx)`, `target_logits = (queries @ keys.T) / √E`
   - `value_head: Linear(G, 1)` with tanh

The target pointer is a full `(B, P, P)` matrix — row `i` gives the distribution over which planet planet `i` should target.

---

## `StateBuilderV2`  (`state_builder_v2.py`)

Stateless, same interface as `StateBuilder` but outputs structured entity tensors.

```python
from bots.neural.state_builder_v2 import StateBuilderV2

sb = StateBuilderV2(max_planets=50, max_fleets=200)

# Live inference
state = sb.from_obs(obs, player=0)          # → StructuredStateV2
state = sb.from_obs_structured(obs, player) # alias

# Offline dataset building
state = sb.from_step(step, player=0)        # → StructuredStateV2

# Access structured arrays
print(state["planet_features"].shape)  # (50, 10)
print(state["fleet_features"].shape)   # (200, 8)
print(state["global_features"].shape)  # (4,)
print(state["planet_mask"].shape)      # (50,)  bool
print(state["fleet_mask"].shape)       # (200,) bool
```

---

## `ActionCodecV2`  (`action_codec_v2.py`)

Per-planet encode/decode for `PlanetPolicyModel`. Preserves all actions per turn
instead of discarding all but the dominant one.

```python
from bots.neural.action_codec_v2 import ActionCodecV2

codec = ActionCodecV2(
    n_amount_bins=5,
    angular_diff_threshold=math.pi / 4,  # reject noisy target inferences
)

# Training: all raw actions → PerPlanetLabels
labels = codec.encode_per_planet(
    raw_actions,   # (n_actions, 3) — [from_planet_id, angle, n_ships]
    context,       # ActionContext
    planets,       # (n_planets, 7) — col 5 = ships
    value_target,  # float
    max_planets,   # padding size
)
# labels.planet_action_types.shape == (max_planets,)
# labels.my_planet_mask shows which slots have real labels

# Inference: PlanetPolicyOutput → game action list
actions = codec.decode_per_planet(output, context, planet_features, max_planets)
# returns [[planet_id, angle, n_ships], ...]  — one entry per planet that launches
```

**Key difference from `ActionCodec.encode`:** instead of picking one dominant action,
`encode_per_planet` maps every action to its source planet independently. A turn where
a bot launches from 8 of 12 planets produces 8 LAUNCH labels + 4 NO_OP labels,
capturing the full teacher signal instead of discarding 7/8 of it.

Target inference still uses angular matching (`best_diff ≤ angular_diff_threshold`);
if no planet is within threshold, `target_idx` is set to `-1` (ignored in training loss).

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

import torch
B = 4
planet_features = torch.randn(B, 50, 7)
fleet_features  = torch.randn(B, 700)
planet_mask     = torch.ones(B, 50, dtype=torch.bool)
planet_mask[:, 30:] = False

output = model(planet_features, fleet_features, planet_mask)  # PointerPolicyOutput
print(output.source_logits.shape)   # (B, 50)
```

---

## `NeuralBot`  (`bot.py`)

```python
from bots.neural.bot import NeuralBot

# Wrap any model (auto-detects type)
bot = NeuralBot(model=model, state_builder=state_builder, codec=codec, device="cpu")
actions = bot.act(obs)   # works with dict obs or object obs

# Load trained checkpoint
bot = NeuralBot.load("path/to/checkpoint.pt", device="cpu")
# model_type in checkpoint ("flat" | "pointer" | "planet_policy") selects correct class
```

**Checkpoint format:**

```python
# planet_policy
{
    "model_type": "planet_policy",
    "config": dataclasses.asdict(PlanetPolicyConfig(...)),
    "state_dict": model.state_dict(),
    "max_planets": 50, "max_fleets": 200, "n_amount_bins": 5,
}
# flat / pointer
{
    "model_type": "flat",   # or "pointer"
    "config": PolicyValueConfig(...),  # or PointerNetworkConfig
    "state_dict": model.state_dict(),
    "max_planets": 50, "max_fleets": 100, "n_amount_bins": 5,
}
```

### `agent_fn` for kaggle_environments

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
