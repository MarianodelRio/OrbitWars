# Data Pipeline

## Overview

Training data is stored as HDF5 files, one file per episode. Files are discovered by `DataCatalog`, loaded turn-by-turn via `EpisodeReader`, and converted to model inputs by `StateBuilder` + `ActionCodec`. For fast training, a one-time pre-computation pass builds a single `PrecomputedILDataset` cache.

## Generating Episodes

Set `"save_data": true` in the appropriate config file before running matches:

- `scripts/matches/config.json` — saves data from single/batch matches
- `scripts/tournament/config.json` — saves data from tournament runs

Data is written automatically to:

```
data/
  matches/
    <bot1>_vs_<bot2>/
      <timestamp>_match_<nnnn>.h5
  tournaments/
    <timestamp>/
      <bot_a>_vs_<bot_b>/
        match_<nnnn>.h5
```

## HDF5 Schema

All datasets are padded to 500 steps along axis 0 (the turn axis). Use `f.attrs["total_steps"]` to determine real data length.

### Datasets

| Dataset | Shape | Dtype | Notes |
|---|---|---|---|
| `planets` | (500, 50, 7) | float32 | Raw planet vectors per turn, zero-padded beyond real count |
| `fleets` | (500, 200, 7) | float32 | Raw fleet vectors per turn, zero-padded |
| `actions_p0` | (500, 50, 3) | float32 | Player 0 moves `[from_planet_id, angle, ships]`, zero-padded |
| `actions_p1` | (500, 50, 3) | float32 | Player 1 moves, zero-padded |
| `n_planets` | (500,) | int32 | Real planet count each turn |
| `n_fleets` | (500,) | int32 | Real fleet count each turn |
| `n_actions_p0` | (500,) | int32 | Real action count each turn, player 0 |
| `n_actions_p1` | (500,) | int32 | Real action count each turn, player 1 |
| `comet_planet_ids` | (500, 16) | int32 | Comet planet IDs, -1 for empty slots |
| `comets_json` | (500,) | utf-8 string | Full comets list serialized as JSON string |
| `terminals` | (500,) | bool | True only at the final real turn index |

### Root Attributes

| Attribute | Type | Notes |
|---|---|---|
| `total_steps` | int | Number of real turns played |
| `winner` | int | 0 or 1 for winner index, -1 for draw |
| `done_reason` | str | `"step_limit"` or `"elimination"` |
| `final_ships_p0` | float | Final ship count for player 0 |
| `final_ships_p1` | float | Final ship count for player 1 |
| `bot0` | str | Name of player 0 bot |
| `bot1` | str | Name of player 1 bot |

Rows beyond `total_steps` are zero-padded. `terminals[total_steps - 1]` is the only `True` value.

## DataCatalog

`dataset/catalog.py` provides episode discovery and filtering.

### `DataCatalog.scan(roots=None) -> DataCatalog`

Scans all `.h5` files under `roots` (default: `data/matches/` and `data/tournaments/`), reading only file attributes. Returns a `DataCatalog` instance.

### `DataCatalog.filter(...) -> DataCatalog`

Returns a filtered sub-catalog. Parameters:

| Parameter | Type | Description |
|---|---|---|
| `bot` | str / None | Include episodes where bot0 or bot1 contains this string |
| `opponent` | str / None | Further filter to episodes where the other side contains this string |
| `winner_only` | bool | Only include episodes where `bot` won |
| `done_reason` | str / None | `"step_limit"` or `"elimination"` |
| `min_steps` | int / None | Minimum `total_steps` |
| `max_steps` | int / None | Maximum `total_steps` |

### `EpisodeMeta` fields

| Field | Type | Description |
|---|---|---|
| `path` | Path | Absolute path to the `.h5` file |
| `bot0` | str | Name of player 0 |
| `bot1` | str | Name of player 1 |
| `winner` | int | 0, 1, or -1 (draw) |
| `done_reason` | str | `"step_limit"` or `"elimination"` |
| `total_steps` | int | Number of real turns |
| `final_ships_p0` | float | Final ship count for player 0 |
| `final_ships_p1` | float | Final ship count for player 1 |

## EpisodeReader and StepRecord

`EpisodeReader` reads a single `.h5` file turn by turn. Use as a context manager:

```python
from dataset.catalog import DataCatalog
from dataset.episode import EpisodeReader

catalog = DataCatalog.scan()
meta = catalog.episodes[0]

with EpisodeReader(meta, cache=True) as reader:
    for t in range(reader.total_steps):
        step = reader.step(t)
        print(step.planets.shape)   # (n_planets, 7)
```

When `cache=True`, all arrays are loaded into memory at `__enter__` time; subsequent `step()` calls are pure numpy slices. When `cache=False`, every `step()` call issues HDF5 reads.

### `StepRecord` fields

| Field | Type | Description |
|---|---|---|
| `turn` | int | Turn index (0-based) |
| `planets` | ndarray (n, 7) | Real planets only, no padding |
| `fleets` | ndarray (n, 7) | Real fleets only, no padding |
| `actions_p0` | ndarray (n, 3) | Player 0 moves for this turn |
| `actions_p1` | ndarray (n, 3) | Player 1 moves for this turn |
| `comet_planet_ids` | ndarray (k,) | Real comet IDs only, no -1 padding |
| `is_terminal` | bool | True on the last turn |

## NeuralILDataset (Lazy)

`NeuralILDataset` in `bots/neural/training.py` stores a lightweight index of `(episode, step, player)` tuples and builds each sample on demand in `__getitem__`. An LRU cache of open `EpisodeReader` instances (up to `reader_cache_size=64`) amortises HDF5 I/O across all steps of the same episode.

Build it with `build_il_dataset(catalog, state_builder, codec, perspective="winner")`.

`perspective` controls which player's actions are included:
- `"winner"`: only the winning player's steps (skips draws)
- `"loser"`: only the losing player's steps
- `"both"`: both players from every episode

Each item returned by `__getitem__` is a dict with keys: `planet_features`, `fleet_features`, `fleet_mask`, `global_features`, `planet_mask`, `action_types`, `target_idxs`, `amount_bins`, `value_target`, `score_diff`, `ownership_10`, `opponent_launch`.

## PrecomputedILDataset (Fast)

`PrecomputedILDataset` reads pre-computed tensors from an HDF5 cache built by `build_il_cache()`. It uses a chunk-buffered read strategy: loads 512 samples at a time via contiguous HDF5 slices (approximately 0.02 ms/sample vs 5 ms for on-the-fly computation).

Build the cache once:
```python
from bots.neural.training import build_il_cache
from dataset.catalog import DataCatalog
from bots.neural.state_builder import StateBuilder
from bots.neural.action_codec import ActionCodec

catalog = DataCatalog.scan()
state_builder = StateBuilder(max_planets=50, max_fleets=200)
codec = ActionCodec(n_amount_bins=8)
build_il_cache(catalog, state_builder, codec, cache_path="data/il_cache.h5")
```

Load for training:
```python
from bots.neural.training import PrecomputedILDataset
dataset = PrecomputedILDataset("data/il_cache.h5")
```

The cache HDF5 schema (`schema_version=3`) stores: `planet_features`, `fleet_features`, `fleet_mask`, `global_features`, `planet_mask`, `action_types`, `target_idxs`, `amount_bins`, `value_target`, `score_diff`, `ownership_10`, `opponent_launch`, `episode_idx`, `episode_paths`. Root attributes include `at_counts`, `amt_counts` (for class weight computation), `perspective`, `n_episodes`, `n_samples`, `Dp`, `Df`, `Dg`.

## Cache vs Lazy Tradeoff

| | NeuralILDataset (Lazy) | PrecomputedILDataset (Fast) |
|---|---|---|
| First-run cost | None | One-time cache build (minutes) |
| Per-sample cost | ~5 ms (StateBuilder + ActionCodec) | ~0.02 ms (HDF5 slice) |
| Memory | O(index) + LRU cache (bounded) | O(buffer) ~4.5 MB |
| Disk | Only raw episode files | Raw files + cache HDF5 |
| Use when | Few episodes, or first experiments | Large dataset, repeated training |

## Using in Training

Configure in `il_config.json` under `data_pipeline.builder`:

```json
{
  "data_pipeline": {
    "catalog": {
      "roots": null,
      "filter": {
        "bot": "heuristic.sniper",
        "winner_only": true,
        "min_steps": 50
      }
    },
    "builder": {
      "perspective": "winner",
      "cache_path": "data/il_cache.h5"
    }
  }
}
```

When `cache_path` is set and the file does not exist, the trainer builds the cache automatically before training starts. Subsequent runs load it directly.
