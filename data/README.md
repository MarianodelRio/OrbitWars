# data/

## Purpose

Master dataset for Imitation Learning, offline RL, and neural network training.
All data is raw and non-derived — nothing that can be reconstructed from the stored
state is included.

## How to Enable

Set `"save_data": true` in either config file:

- `scripts/matches/config.json` — enables saving for single/batch matches
- `scripts/tournament/config.json` — enables saving for tournament runs

## Directory Layout

```
data/
  matches/
    <bot1>_vs_<bot2>/
      <timestamp>_match_<nnnn>.h5     # written by scripts/matches/run.py
  tournaments/
    <timestamp>/
      <bot_a>_vs_<bot_b>/
        match_<nnnn>.h5               # written by scripts/tournament/run.py
```

## HDF5 Schema

All datasets are padded to 500 steps along axis 0 (the turn axis).

### Datasets

| Dataset | Shape | Dtype | Notes |
|---|---|---|---|
| planets | (500, 50, 7) | float32 | Raw planet vectors per turn, zero-padded beyond real count |
| fleets | (500, 200, 7) | float32 | Raw fleet vectors per turn, zero-padded |
| actions_p0 | (500, 50, 3) | float32 | Player 0 moves [from_planet_id, angle, ships], zero-padded |
| actions_p1 | (500, 50, 3) | float32 | Player 1 moves, zero-padded |
| n_planets | (500,) | int32 | Real planet count each turn |
| n_fleets | (500,) | int32 | Real fleet count each turn |
| n_actions_p0 | (500,) | int32 | Real action count each turn, player 0 |
| n_actions_p1 | (500,) | int32 | Real action count each turn, player 1 |
| comet_planet_ids | (500, 16) | int32 | Comet planet IDs, -1 for empty slots |
| comets_json | (500,) | utf-8 string | Full comets list serialized as JSON string |
| terminals | (500,) | bool | True only at the final real turn index |

### Root Attributes

| Attribute | Type | Notes |
|---|---|---|
| total_steps | int | Number of real turns played |
| winner | int | 0 or 1 for winner index, -1 for draw |
| done_reason | str | "step_limit" or "elimination" |
| final_ships_p0 | float | Final ship count for player 0 |
| final_ships_p1 | float | Final ship count for player 1 |
| bot0 | str | Name of player 0 bot |
| bot1 | str | Name of player 1 bot |

## Padding

All rows beyond `total_steps` are zero-padded. Use `f.attrs["total_steps"]` to
determine the real data length. `terminals[total_steps - 1]` is the only `True`
value.

## Quick Load Example

```python
import h5py
import numpy as np

f = h5py.File("path/to/match.h5", "r")
T = f.attrs["total_steps"]
planets = f["planets"][:T]       # (T, 50, 7)
actions = f["actions_p0"][:T]    # (T, 50, 3)
n_act   = f["n_actions_p0"][:T]  # (T,) — real action count per turn
```
