"""Evaluator: runs match evaluation between a NeuralBot and registered opponents."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from bots.neural.bot import NeuralBot
from bots.interface import make_agent
from game.env.evaluator import evaluate
from bots.registry import resolve


class Evaluator:
    def __init__(
        self,
        bot: NeuralBot,
        opponents: list,
        n_matches: int = 10,
        run_dir: "Path | None" = None,
    ) -> None:
        self._bot = bot
        self._opponents = opponents
        self._n_matches = n_matches
        self._run_dir = run_dir

    def run(self, epoch: "int | None" = None) -> dict:
        neural_fn = make_agent(self._bot)
        results = {}

        for opp_name in self._opponents:
            opponent_fn = resolve(opp_name)
            if opponent_fn is None:
                print(f"[Evaluator] Warning: opponent '{opp_name}' not in registry, skipping.")
                continue

            raw = evaluate(neural_fn, opponent_fn, n_matches=self._n_matches)

            n = self._n_matches
            wins_neural = raw["wins"][0]
            wins_opp = raw["wins"][1]
            draws = raw["draws"]

            result = {
                "epoch": epoch,
                "checkpoint": None,
                "vs": opp_name,
                "n_matches": n,
                "win_rate": wins_neural / n,
                "draw_rate": draws / n,
                "loss_rate": wins_opp / n,
                "avg_score_neural": raw["avg_ships"][0],
                "avg_score_opponent": raw["avg_ships"][1],
                "avg_game_length": raw.get("avg_game_length", 0.0),
                "timestamp": datetime.utcnow().isoformat(),
            }

            if self._run_dir is not None and epoch is not None:
                eval_dir = self._run_dir / "eval"
                eval_dir.mkdir(parents=True, exist_ok=True)
                safe_opp = opp_name.replace(".", "_")
                out_path = eval_dir / f"epoch_{epoch:03d}_vs_{safe_opp}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

            results[opp_name] = result

        return results

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        opponents: list,
        n_matches: int = 10,
    ) -> "Evaluator":
        bot = NeuralBot.load(str(checkpoint_path))
        return cls(bot=bot, opponents=opponents, n_matches=n_matches)
