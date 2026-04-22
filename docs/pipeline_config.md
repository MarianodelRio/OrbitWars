# Dataset Pipeline Config — `training/pipeline.json`

Controls what episodes are loaded and how training samples are built.

## Smoke test

Run this after any config change to verify the pipeline end-to-end:

```bash
python scripts/probe_pipeline.py
```

---

## `catalog` — what episodes to load

### `roots`
Directories to scan for `.h5` episode files.

- `null` → uses `data/matches/` and `data/tournaments/` (repo defaults)
- `["path/to/dir1", "path/to/dir2"]` → custom directories

### `filter` — episode-level filters (all optional, `null` = no filter)

| Field | Type | What it does |
|---|---|---|
| `bot` | string or null | Keep only episodes where this bot played (as p0 or p1) |
| `opponent` | string or null | Keep only episodes where this bot was the rival |
| `winner_only` | bool | `true` = keep only episodes where `bot` won. Requires `bot` to be set |
| `done_reason` | string or null | Keep only episodes ending for this reason. Valid values: `"step_limit"`, `"elimination"` |
| `min_steps` | int or null | Discard episodes shorter than N turns |
| `max_steps` | int or null | Discard episodes longer than N turns |

---

## `builder` — how to build training samples

### `perspective`
Which player's actions to extract.

- `"winner"` — only the winning player's steps (draw episodes are skipped entirely)
- `"both"` — both players' steps from every episode
- `0` — always player 0
- `1` — always player 1

### `mode`
Learning paradigm.

- `"il_step"` — imitation learning: each sample is (state, action). `reward` is `None`.
- `"rl_transition"` — reinforcement learning: each sample is (state, action, reward, done). Requires `reward_transform` to be set.

### `state_transform`
How to process the game state before it becomes a sample.

- `"raw"` — numpy arrays as-is: `state["planets"]` shape `(n_planets, 7)`, `state["fleets"]` shape `(n_fleets, 7)`

### `action_transform`
How to process the action.

- `"raw"` — numpy array as-is: shape `(n_actions, 3)` → `[from_planet_id, angle, ships]`

### `reward_transform`
Reward signal. Only used in `"rl_transition"` mode.

- `null` — no reward (required for `"il_step"`)
- `"binary_outcome"` — `0.0` each non-terminal turn; `+1.0` won, `-1.0` lost, `0.0` draw at terminal

### `step_filter`
Which turns to include in the sample set.

- `"has_action"` — only turns where the player launched at least one fleet
- `"early_game:N"` — only turns 0..N (e.g. `"early_game:200"`)
- `null` — all turns

---

## Common configs

**Imitation learning on winner (default):**
```json
{
  "catalog": {
    "roots": null,
    "filter": {
      "bot": null,
      "opponent": null,
      "winner_only": false,
      "done_reason": null,
      "min_steps": null,
      "max_steps": null
    }
  },
  "builder": {
    "perspective": "winner",
    "mode": "il_step",
    "state_transform": "raw",
    "action_transform": "raw",
    "reward_transform": null,
    "step_filter": "has_action"
  }
}
```

**RL transitions, only wins by a specific bot, early game only:**
```json
{
  "catalog": {
    "roots": null,
    "filter": {
      "bot": "my_bot",
      "opponent": null,
      "winner_only": true,
      "done_reason": null,
      "min_steps": null,
      "max_steps": null
    }
  },
  "builder": {
    "perspective": "winner",
    "mode": "rl_transition",
    "state_transform": "raw",
    "action_transform": "raw",
    "reward_transform": "binary_outcome",
    "step_filter": "early_game:150"
  }
}
```

**Self-play / both players, IL, all turns:**
```json
{
  "catalog": {
    "roots": null,
    "filter": {
      "bot": null,
      "opponent": null,
      "winner_only": false,
      "done_reason": null,
      "min_steps": null,
      "max_steps": null
    }
  },
  "builder": {
    "perspective": "both",
    "mode": "il_step",
    "state_transform": "raw",
    "action_transform": "raw",
    "reward_transform": null,
    "step_filter": null
  }
}
```
