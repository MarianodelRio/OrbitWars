# Dataset Pipeline

## Overview

`dataset/` turns HDF5 episode files recorded during matches into training samples
for imitation learning (IL) or offline reinforcement learning (RL).

What it **does**:
- Discovers and filters `.h5` episode files by metadata (bot names, outcome, length).
- Reads turn-by-turn state/action data with optional in-memory caching.
- Builds `TrainingSample` objects with pluggable transforms for state, action, reward,
  and step filtering.
- Wraps samples in a PyTorch `Dataset` for use with `DataLoader`.

What it **does not do**:
- Record episodes — that is handled by `game/data/hdf5_writer.py`.
- Define model architectures or training loops — those live in `training/`.
- Normalize or featurize data — use transforms for that.

---

## Architecture

```
DataCatalog.scan()          # discover .h5 files, read only root attrs
    │
    └─ .filter(...)         # return a sub-catalog by episode metadata
           │
           └─ EpisodeMeta   # lightweight episode descriptor (no I/O)
                  │
                  └─ EpisodeReader (context manager)
                         │  .step(t)      → StepRecord
                         │  .steps(s, e)  → Iterator[StepRecord]
                         │  .all_steps()  → list[StepRecord]
                         │
                         └─ SampleBuilder.build_episode(reader)
                                │  (applies state/action/reward transforms + filter)
                                └─ list[TrainingSample]

                  DataCatalog + SampleBuilder
                         └─ SampleBuilder.build_from_catalog(catalog)
                                └─ Iterator[TrainingSample]

                  list[TrainingSample]
                         └─ OrbitDataset  →  torch DataLoader
```

---

## Quickstart

End-to-end IL pipeline in ~15 lines:

```python
from dataset.catalog import DataCatalog
from dataset.builder import SampleBuilder
from dataset.transforms.state import RawStateTransform
from dataset.transforms.action import RawActionTransform
from dataset.transforms.filters import HasActionFilter
from dataset.torch_adapter import OrbitDataset
from torch.utils.data import DataLoader
import torch

catalog = DataCatalog.scan()                          # scan data/matches/ + data/tournaments/
catalog = catalog.filter(winner_only=False)           # (no-op here; shown for clarity)

builder = SampleBuilder(
    state_transform=RawStateTransform(),
    action_transform=RawActionTransform(),
    perspective="winner",
    mode="il_step",
    step_filter=HasActionFilter(),
)

samples = list(builder.build_from_catalog(catalog))
print(f"{len(samples)} samples from {len(catalog.episodes)} episodes")

def state_to_tensor(s):
    return torch.from_numpy(s["planets"])             # (n_planets, 7)

def action_to_tensor(a):
    return torch.from_numpy(a)                        # (n_actions, 3)

dataset = OrbitDataset(samples, state_to_tensor, action_to_tensor)
loader  = DataLoader(dataset, batch_size=64, shuffle=True)
```

---

## DataCatalog

### `DataCatalog.scan(roots=None)`

Scans directories for `.h5` files by reading only HDF5 root attributes (fast —
no turn data is loaded).

```python
catalog = DataCatalog.scan()                          # uses repo defaults
catalog = DataCatalog.scan(roots=[Path("data/matches")])
```

- `roots=None` → uses `data/matches/` and `data/tournaments/` relative to repo root.
- Returns a `DataCatalog` holding a list of `EpisodeMeta`.
- Skips and prints a warning for any file that cannot be read.

### `catalog.filter(...)`

Returns a **new** `DataCatalog` with the matching subset. All parameters are optional.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bot` | `str \| None` | `None` | Keep episodes where this string appears in `bot0` or `bot1` |
| `opponent` | `str \| None` | `None` | Keep episodes where this string appears in the *other* player's name |
| `winner_only` | `bool` | `False` | Keep only episodes where `bot` won. Requires `bot` to be set |
| `done_reason` | `str \| None` | `None` | Keep only episodes with this end reason (`"step_limit"` or `"elimination"`) |
| `min_steps` | `int \| None` | `None` | Discard episodes with fewer than N turns |
| `max_steps` | `int \| None` | `None` | Discard episodes with more than N turns |

```python
catalog = DataCatalog.scan().filter(
    bot="scoring",
    winner_only=True,
    done_reason="elimination",
    min_steps=50,
)
```

### `catalog.episodes`

`list[EpisodeMeta]` — the current filtered set.

---

## EpisodeMeta

Lightweight descriptor read from HDF5 root attributes. No turn data is stored.

| Field | Type | Description |
|---|---|---|
| `path` | `Path` | Absolute path to the `.h5` file |
| `bot0` | `str` | Name of player 0's bot |
| `bot1` | `str` | Name of player 1's bot |
| `winner` | `int` | `0` or `1` for the winning player; `-1` for draw |
| `done_reason` | `str` | `"step_limit"` — max turns reached; `"elimination"` — one player lost all planets |
| `total_steps` | `int` | Number of real turns in the episode (≤ 500) |
| `final_ships_p0` | `float` | Total ships owned by player 0 at game end |
| `final_ships_p1` | `float` | Total ships owned by player 1 at game end |

---

## EpisodeReader

Turn-by-turn reader for a single HDF5 episode. Must be used as a context manager.

```python
from dataset.episode import EpisodeReader

with EpisodeReader(meta) as reader:
    print(reader.total_steps)        # int

    step = reader.step(0)            # StepRecord for turn 0

    for step in reader.steps():      # all turns
        ...

    for step in reader.steps(10, 50):  # turns 10..49
        ...

    all_steps = reader.all_steps()   # list[StepRecord] — entire episode in RAM
```

### `cache=True`

Load all HDF5 arrays into RAM at `__enter__` time. Subsequent `step()` calls do no
I/O. Use when you iterate over an episode multiple times or build RL transitions
(which look ahead one step):

```python
with EpisodeReader(meta, cache=True) as reader:
    samples = builder.build_episode(reader)
```

### Methods

| Method | Returns | Description |
|---|---|---|
| `step(t)` | `StepRecord` | Single turn `t`. `IndexError` if out of range |
| `steps(start=0, end=None)` | `Iterator[StepRecord]` | Yields turns `[start, end)`. No more than one turn in RAM at a time |
| `all_steps()` | `list[StepRecord]` | Entire episode. Only use if the episode fits comfortably in RAM |

---

## StepRecord — Field Reference

All arrays are **unpadded** — slice lengths match the real counts for that turn.

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `turn` | scalar | `int` | Zero-based turn index |
| `planets` | `(n_planets, 7)` | `float32` | Planet vectors: `[id, owner, x, y, radius, ships, production]` |
| `fleets` | `(n_fleets, 7)` | `float32` | Fleet vectors: `[id, owner, x, y, angle, from_planet_id, ships]` |
| `actions_p0` | `(n_actions_p0, 3)` | `float32` | Player 0 actions: `[from_planet_id, angle, ships]`. Shape `(0, 3)` if no actions |
| `actions_p1` | `(n_actions_p1, 3)` | `float32` | Player 1 actions. Shape `(0, 3)` if no actions |
| `comet_planet_ids` | `(k,)` | `int32` | Planet IDs currently acting as comets. Empty array if none |
| `is_terminal` | scalar | `bool` | `True` only at the final turn of the episode |

### Planet column details (`planets[:, col]`)

| Col | Name | Notes |
|---|---|---|
| 0 | `id` | Stable planet ID across all turns |
| 1 | `owner` | `0` = player 0, `1` = player 1, `-1` = neutral (unowned) |
| 2 | `x` | Position in [0, 100] |
| 3 | `y` | Position in [0, 100] |
| 4 | `radius` | Planet size |
| 5 | `ships` | Current garrison ship count |
| 6 | `production` | Ships produced per turn when owned |

### Fleet column details (`fleets[:, col]`)

| Col | Name | Notes |
|---|---|---|
| 0 | `id` | Unique fleet ID for this turn |
| 1 | `owner` | `0` or `1` — always owned, never `-1` |
| 2 | `x` | Current position |
| 3 | `y` | Current position |
| 4 | `angle` | Direction of travel (radians) |
| 5 | `from_planet_id` | Origin planet ID |
| 6 | `ships` | Ship count in this fleet |

---

## SampleBuilder

Converts an `EpisodeReader` (or an entire `DataCatalog`) into `TrainingSample` objects.

```python
from dataset.builder import SampleBuilder

builder = SampleBuilder(
    state_transform=RawStateTransform(),
    action_transform=RawActionTransform(),
    reward_transform=None,          # required for rl_transition
    step_filter=HasActionFilter(),
    perspective="winner",
    mode="il_step",
)
```

### Constructor parameters

| Parameter | Type | Valid values | Description |
|---|---|---|---|
| `state_transform` | callable | any `StateTransform` | Maps `(StepRecord, player) → state` |
| `action_transform` | callable | any `ActionTransform` | Maps `(StepRecord, player) → action` |
| `reward_transform` | callable or `None` | any `RewardTransform` or `None` | Required when `mode="rl_transition"` |
| `step_filter` | callable or `None` | any `StepFilter` or `None` | `None` = include all turns |
| `perspective` | `str \| int` | `"winner"`, `"both"`, `0`, `1` | Which player's samples to extract |
| `mode` | `str` | `"il_step"`, `"rl_transition"` | Learning paradigm |

#### `perspective` values

| Value | Behaviour |
|---|---|
| `"winner"` | Only the winning player's steps. Draw episodes (`winner == -1`) are skipped entirely |
| `"both"` | Both players' steps from every episode |
| `0` | Always player 0's steps |
| `1` | Always player 1's steps |

#### `mode` values

| Value | `reward` | `next_state` | Use case |
|---|---|---|---|
| `"il_step"` | `None` | `None` | Imitation learning — clone the winning policy |
| `"rl_transition"` | from `reward_transform` | state at `t+1` | Offline RL — Q-learning, actor-critic |

### `build_episode(reader)`

```python
with EpisodeReader(meta, cache=True) as reader:
    samples = builder.build_episode(reader)   # list[TrainingSample]
```

Use when you need all samples from a single episode at once.

### `build_from_catalog(catalog)`

```python
for sample in builder.build_from_catalog(catalog):
    ...                                       # Iterator[TrainingSample]
```

Streams samples from every episode in the catalog. Memory-efficient for large datasets.

---

## TrainingSample Fields

| Field | Type | Description |
|---|---|---|
| `state` | any | Output of `state_transform` |
| `action` | any | Output of `action_transform` |
| `reward` | `float \| None` | `None` in `"il_step"` mode; float in `"rl_transition"` mode |
| `next_state` | any or `None` | State at `t+1`; `None` in `"il_step"` mode or at terminal |
| `done` | `bool` | `True` at the terminal step of the episode |
| `info` | `dict` | `{"path": str, "turn": int, "player": int}` — for tracing and debugging |

---

## Transforms Available

### State transforms

| Class | Import | What it returns |
|---|---|---|
| `RawStateTransform` | `dataset.transforms.state` | `{"planets": ndarray (n, 7), "fleets": ndarray (m, 7)}` |

### Action transforms

| Class | Import | What it returns |
|---|---|---|
| `RawActionTransform` | `dataset.transforms.action` | `ndarray (n_actions, 3)` — `[from_planet_id, angle, ships]`; shape `(0, 3)` if no actions |

### Reward transforms

| Class | Import | What it returns |
|---|---|---|
| `BinaryOutcomeReward` | `dataset.transforms.reward` | `0.0` each non-terminal turn; `+1.0` win, `-1.0` loss, `0.0` draw at terminal |

### Step filters

| Class | Import | Constructor args | Keeps turn when |
|---|---|---|---|
| `HasActionFilter` | `dataset.transforms.filters` | — | Player took ≥ 1 action |
| `EarlyGameFilter` | `dataset.transforms.filters` | `max_turn: int` | `step.turn <= max_turn` |
| `CompositeFilter` | `dataset.transforms.filters` | `*filters` | All sub-filters return `True` (logical AND) |

```python
from dataset.transforms.filters import HasActionFilter, EarlyGameFilter, CompositeFilter

f = CompositeFilter(HasActionFilter(), EarlyGameFilter(max_turn=150))
```

---

## Adding a Custom Transform

Transforms are plain callables — no base class required.

### Custom StateTransform

```python
class NormalisedStateTransform:
    """Normalise planet positions to [0, 1]."""
    def __call__(self, step, player):
        planets = step.planets.copy()
        planets[:, 2] /= 100.0   # x
        planets[:, 3] /= 100.0   # y
        return {"planets": planets, "fleets": step.fleets}
```

### Custom StepFilter

```python
class LateGameFilter:
    """Only include turns in the second half of the episode."""
    def __init__(self, total_steps):
        self.half = total_steps // 2

    def __call__(self, step, player):
        return step.turn >= self.half
```

Duck typing means any callable `(StepRecord, int) -> bool` works as a filter and
any callable `(StepRecord, int) -> Any` works as a transform.

---

## OrbitDataset and DataLoader

`OrbitDataset` wraps a pre-built `list[TrainingSample]` as a PyTorch `Dataset`.

```python
import torch
from torch.utils.data import DataLoader
from dataset.torch_adapter import OrbitDataset

def state_to_tensor(s):
    # s is whatever state_transform returned
    return torch.from_numpy(s["planets"])   # example: use planet matrix

def action_to_tensor(a):
    return torch.from_numpy(a)              # (n_actions, 3)

dataset = OrbitDataset(samples, state_to_tensor, action_to_tensor)
loader  = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

for batch in loader:
    states  = batch["state"]     # tensor
    actions = batch["action"]    # tensor
    rewards = batch["reward"]    # float32 tensor
    dones   = batch["done"]      # bool tensor
```

### `LazyOrbitDataset`

**Not yet implemented.** Raises `NotImplementedError` on `__len__` and `__getitem__`.
Reserved for a future cycle that streams samples on-demand without pre-building the list.

---

## Reutilización en RL Online

You can construct a `StepRecord` manually from a live environment observation to
feed it into transforms during online training:

```python
import numpy as np
from dataset.episode import StepRecord

def obs_to_step_record(obs, turn):
    # obs["planets"] and obs["fleets"] come directly from the kaggle_environments obs
    planets = np.array(obs["planets"], dtype=np.float32)   # (n, 7)
    fleets  = np.array(obs["fleets"],  dtype=np.float32)   # (m, 7)

    return StepRecord(
        turn=turn,
        planets=planets,
        fleets=fleets,
        actions_p0=np.empty((0, 3), dtype=np.float32),  # fill in after acting
        actions_p1=np.empty((0, 3), dtype=np.float32),
        comet_planet_ids=np.empty((0,), dtype=np.int32),
        is_terminal=False,  # set True on terminal obs
    )

# Then apply any registered state transform:
state_transform = RawStateTransform()
state = state_transform(obs_to_step_record(obs, t), player=0)
```

---

## Config-Based Usage

Instead of constructing `DataCatalog` and `SampleBuilder` manually, use
`PipelineConfig.from_json()` to drive the pipeline from a JSON file:

```python
from dataset.config import PipelineConfig

cfg     = PipelineConfig.from_json("training/pipeline.json")
catalog = cfg.build_catalog()
builder = cfg.build_builder()
samples = list(builder.build_from_catalog(catalog))
```

See [docs/pipeline_config.md](pipeline_config.md) for the full JSON schema,
valid values, and example configs.
