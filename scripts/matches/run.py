"""
Run one or multiple matches between two bots.

Config: scripts/matches/config.json

Modes:
  "single"   — one match, prints winner and rewards, optionally renders HTML
  "evaluate" — N matches, prints win rate, avg ships, avg game length
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from game.env.evaluator import evaluate, load_agent
from game.env.runner import run_match
from experiments.logger import save as log_experiment

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def main():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    bot1 = load_agent(cfg["bot1"])
    bot2 = load_agent(cfg["bot2"])
    mode = cfg.get("mode", "single")
    steps = cfg.get("steps", 500)
    n_matches = cfg.get("n_matches", 1)
    save_log = cfg.get("save_log", True)

    if mode == "single":
        b1 = cfg["bot1"].split(":")[0].split(".")[-1]
        b2 = cfg["bot2"].split(":")[0].split(".")[-1]

        match_logs = []
        wins_b1 = 0
        wins_b2 = 0
        draws = 0
        for i in range(1, n_matches + 1):
            result = run_match(bot1, bot2, steps=steps)

            if result["winner"] == 0:
                winner_name = b1
                wins_b1 += 1
            elif result["winner"] == 1:
                winner_name = b2
                wins_b2 += 1
            else:
                winner_name = "Draw"
                draws += 1

            print(
                f"Match {i}/{n_matches}  ->  {winner_name}"
                f"  (P0={result['rewards'][0]:.0f}"
                f"  P1={result['rewards'][1]:.0f}"
                f"  steps={result['steps']})"
            )
            match_logs.append({
                "match": i,
                "winner": winner_name,
                "rewards": [float(r) for r in result["rewards"]],
                "steps": result["steps"],
            })

        print("-" * 48)
        print(f"{b1} {wins_b1} \u2013 {wins_b2} {b2}  (draws: {draws})")

        if save_log:
            path = log_experiment("matches", {
                "mode": "single",
                "bot1": cfg["bot1"],
                "bot2": cfg["bot2"],
                "n_matches": n_matches,
                "matches": match_logs,
                "summary": {
                    "wins_bot1": wins_b1,
                    "wins_bot2": wins_b2,
                    "draws": draws,
                },
            }, label=f"{b1}_vs_{b2}")
            print(f"Log:     {path}")

    elif mode == "evaluate":
        print(f"Running {n_matches} matches ({steps} steps each)...")
        results = evaluate(bot1, bot2, n_matches=n_matches, steps=steps)
        b1 = cfg["bot1"].split(":")[0].split(".")[-1]
        b2 = cfg["bot2"].split(":")[0].split(".")[-1]
        print(f"\n{'Bot':<20} {'Win rate':>9} {'Avg ships':>10}")
        print("-" * 42)
        print(f"{b1:<20} {results['win_rate'][0]:>8.1%} {results['avg_ships'][0]:>10.1f}")
        print(f"{b2:<20} {results['win_rate'][1]:>8.1%} {results['avg_ships'][1]:>10.1f}")
        print(f"\nDraws: {results['draws']}/{n_matches}")
        print(f"Avg game length: {results['avg_game_length']:.1f} steps")

        if save_log:
            path = log_experiment("matches", {
                "mode": "evaluate",
                "bot1": cfg["bot1"], "bot2": cfg["bot2"],
                "n_matches": n_matches,
                "wins": results["wins"],
                "draws": results["draws"],
                "win_rate": results["win_rate"],
                "avg_ships": results["avg_ships"],
                "avg_game_length": results["avg_game_length"],
            }, label=f"{b1}_vs_{b2}")
            print(f"Log:     {path}")

    else:
        print(f"Unknown mode '{mode}'. Use 'single' or 'evaluate'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
