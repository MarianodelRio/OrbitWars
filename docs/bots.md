# Bot Catalog

## Registered Bots

| Registry Name | File | Strategy | When to Use |
|---|---|---|---|
| `heuristic.baseline` | `bots/heuristic/baseline.py` | Randomly picks one of your planets and one non-owned target; sends half its garrison | Weakest baseline; use as sanity check or easy opponent for data collection |
| `heuristic.sniper` | `bots/heuristic/sniper.py` | Every owned planet attacks the nearest non-owned planet with exactly enough ships to capture | Fast, aggressive expander; good IL data source; strong early opponent |
| `heuristic.oracle_sniper` | `bots/heuristic/oracle_sniper.py` | Scores targets by `production / (eta * garrison)`; avoids re-targeting already-targeted planets; skips planets with <= 5 ships | Best heuristic; uses fleet speed formula for ETA; tracks targets across turns |
| `scoring.bot` | `bots/scoring/bot.py` | (see file) | Scoring-based heuristic |

## The agent_fn Contract

```python
def agent_fn(obs: dict, config=None) -> list:
    ...
```

- `obs` is a dict with keys: `player` (int), `planets` (list of 7-elem lists), `fleets` (list of 7-elem lists), `angular_velocity` (float), `initial_planets` (list), `comets` (list), `comet_planet_ids` (list of int), `step` (int), `remainingOverageTime` (float).
- Returns a list of moves. Each move is `[from_planet_id, direction_angle_radians, num_ships]`.
- Return `[]` to take no action.
- Only launch from planets you own; `num_ships` must not exceed the planet's current garrison.

## Bot Registry

`bots/registry.py` maps short names to fully qualified `module.path:function_name` import paths.

```python
REGISTRY = {
    "heuristic.baseline":      "bots.heuristic.baseline:agent_fn",
    "heuristic.sniper":        "bots.heuristic.sniper:agent_fn",
    "heuristic.oracle_sniper": "bots.heuristic.oracle_sniper:agent_fn",
    "scoring.bot":             "bots.scoring.bot:agent_fn",
}
```

### `resolve(name: str)`

Returns a callable `agent_fn` for a registered bot name, or `None` if not found. Internally calls `game.env.evaluator.load_agent(path)`.

### `resolve_checkpoint(path: str)`

Loads a `NeuralBot` from a checkpoint file and returns `make_agent(bot)` — a plain `agent_fn(obs, config)` callable.

### `list_bots() -> list[str]`

Returns a sorted list of all registered bot names.

## NeuralBot

`NeuralBot` in `bots/neural/bot.py` wraps `PlanetPolicyModel` for inference.

### Constructor

```python
NeuralBot(
    model: PlanetPolicyModel,
    state_builder: StateBuilder,
    codec: ActionCodec,
    device: str = "cpu",
)
```

The model is moved to `device` and set to `eval()` mode at construction. `_hidden` (LSTM state) is initialized to `None`.

### `act(obs, config=None) -> list`

Accepts either a raw `obs` dict (from `kaggle_environments`) or a `StepRecord` (from `EpisodeReader`). Builds a `StructuredState`, runs a forward pass with `torch.no_grad()`, updates `_hidden`, then calls `codec.decode_per_planet()` to produce the action list.

### `reset() -> None`

Clears the LSTM hidden state by setting `_hidden = None`. Must be called between episodes.

### `NeuralBot.load(path, device="cpu") -> NeuralBot`

Class method. Loads a checkpoint `.pt` file. Reconstructs `PlanetPolicyConfig`, `PlanetPolicyModel`, `StateBuilder`, and `ActionCodec` from the saved `config` dict and `state_dict`. Only `model_type = "planet_policy"` is supported.

### Module-level `agent_fn`

```python
# bots/neural/bot.py bottom
def agent_fn(obs, config=None):
    ...
```

A module-level singleton backed by a default (untrained) `NeuralBot`. Used for `kaggle_environments` compatibility when no checkpoint is specified. Replace with `NeuralBot.load()` for trained inference.

## Creating a New Bot

Minimum template:

```python
# bots/heuristic/my_bot.py
from bots.interface import Bot

class MyBot(Bot):
    @property
    def name(self) -> str:
        return "heuristic.my_bot"

    def act(self, obs, config=None) -> list:
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        # ... strategy ...
        return []  # list of [planet_id, angle_radians, n_ships]

agent_fn = MyBot()
```

Checklist:
- Extend `Bot` from `bots/interface.py`.
- Implement `name` property and `act(obs, config)`.
- Expose `agent_fn` at module level (instance of your bot or a plain function).
- Keep the file self-contained: no cross-bot imports.
- Handle both `obs` as `dict` (live game) and `obs` as `StepRecord` (dataset replay) if the bot will be used in training.

To register the bot, add an entry to `REGISTRY` in `bots/registry.py`:

```python
"heuristic.my_bot": "bots.heuristic.my_bot:agent_fn",
```
