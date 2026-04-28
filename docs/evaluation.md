# Evaluation

## Single Match

Edit `scripts/matches/config.json` then run:

```bash
python scripts/matches/run.py
```

Config fields:

| Field | Type | Description |
|---|---|---|
| `bot1` | str | Module path: `module.path:attr` |
| `bot2` | str | Module path: `module.path:attr` |
| `mode` | str | `"single"` (one or more matches with summary) or `"evaluate"` (batch with win-rate table) |
| `steps` | int | Max steps per match (default 500) |
| `n_matches` | int | Number of matches to run |
| `save_log` | bool | Write result JSON to `experiments/matches/` |
| `save_data` | bool | Write episode HDF5 files to `data/matches/<bot1>_vs_<bot2>/` |

In `"single"` mode, each match result is printed with winner name, rewards, and step count, then a summary line. In `"evaluate"` mode, a win-rate table is printed.

## Tournament

Edit `scripts/tournament/config.json` then run:

```bash
python scripts/tournament/run.py
```

Config fields:

| Field | Type | Description |
|---|---|---|
| `bots` | dict | Map of `short_name: "module.path:attr"` |
| `n_matches` | int | Matches per pair |
| `steps` | int | Max steps per match |
| `save_log` | bool | Write JSON leaderboard to `experiments/tournaments/` |
| `self_play` | bool | Also run each bot against itself |
| `save_data` | bool | Write episode HDF5 files to `data/tournaments/<timestamp>/` |

The tournament runs a round-robin over all bot pairs. After all matches, a leaderboard sorted by Elo is printed:

```
Rank   Bot                   Wins  Draws     ELO
--------------------------------------------------
1      heuristic.sniper        12      2    1068.4
2      heuristic.baseline       3      2     931.6
```

Results are saved to `experiments/tournaments/<timestamp>.json`.

## Elo Rating

`tournament/elo.py` provides the standard Elo update function:

```python
from tournament.elo import update_elo

elo = {"bot_a": 1000.0, "bot_b": 1000.0}
elo = update_elo(elo, winner="bot_a", loser="bot_b", k=32)
```

Formula: `new_rating = old_rating + k * (actual - expected)` where `expected = 1 / (1 + 10^((opponent_rating - rating)/400))`.

For draws, the tournament script applies a half-point update by averaging both directions with `score=0.5`.

All bots start at Elo 1000. Elo is computed fresh from match results each tournament run — it is not persisted between runs.

## Experiments Directory

Match and tournament logs are saved automatically (when `save_log: true`) to:

```
experiments/
  matches/
    <label>_<timestamp>.json   # written by scripts/matches/run.py
  tournaments/
    <timestamp>.json           # written by scripts/tournament/run.py
  submissions/
    <label>_<timestamp>.json   # written by scripts/submission/run.py
```

Each JSON contains the full config, per-match results, and aggregate summary. These files are for human review; they are not read by any training code.

## Evaluator (Training)

`training/evaluation/evaluator.py` provides `Evaluator`, which is used by `ILTrainer` and `RLTrainer` during training to measure win rates against registered opponents.

```python
from pathlib import Path
from training.evaluation.evaluator import Evaluator
from bots.neural.bot import NeuralBot

bot = NeuralBot.load("runs/my_run/run_001/checkpoints/best.pt")
evaluator = Evaluator(
    bot=bot,
    opponents=["heuristic.baseline", "heuristic.sniper"],
    n_matches=20,
    run_dir=Path("runs/my_run/run_001"),  # or None to skip file output
)
results = evaluator.run(epoch=10)
# results["heuristic.baseline"]["win_rate"] -> float
```

Each call to `run(epoch)` plays `n_matches` matches per opponent and returns a dict keyed by opponent name. Each value contains: `epoch`, `vs`, `n_matches`, `win_rate`, `draw_rate`, `loss_rate`, `avg_score_neural`, `avg_score_opponent`, `timestamp`. When `run_dir` is set, results are also written to `run_dir/eval/epoch_NNN_vs_<opponent>.json`.

Opponents must be registered in `bots/registry.py`. Unregistered names are skipped with a warning.

Alternatively, load from a checkpoint directly:

```python
evaluator = Evaluator.from_checkpoint(
    checkpoint_path=Path("runs/.../checkpoints/best.pt"),
    opponents=["heuristic.baseline"],
    n_matches=20,
)
```
