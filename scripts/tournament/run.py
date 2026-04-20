"""
Round-robin tournament between all configured bots.
Prints a leaderboard and optionally saves results to tournament/results/.

Config: scripts/tournament/config.json
"""
import json
import os
import sys
from datetime import datetime
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from game.env.evaluator import evaluate, load_agent
from tournament.elo import update_elo as _update_elo


def _apply_elo(elo: dict, winner: str, loser: str) -> dict:
    return _update_elo(elo, winner, loser)


def _apply_elo_draw(elo: dict, name_a: str, name_b: str) -> dict:
    # Draw: apply half-point to each side by averaging both directions
    ra, rb = elo[name_a], elo[name_b]
    k = 32
    expected_a = 1 / (1 + 10 ** ((rb - ra) / 400))
    result = dict(elo)
    result[name_a] = ra + k * (0.5 - expected_a)
    result[name_b] = rb + k * (0.5 - (1 - expected_a))
    return result

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
EXPERIMENTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments", "tournaments"
)


def main():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    n_matches = cfg.get("n_matches", 10)
    steps = cfg.get("steps", 500)
    save_log = cfg.get("save_log", True)
    bot_registry = cfg["bots"]

    agents = {name: load_agent(path) for name, path in bot_registry.items()}
    names = list(agents.keys())

    wins = {n: 0 for n in names}
    draws = {n: 0 for n in names}
    elo = {n: 1000.0 for n in names}
    matchups = []

    print(f"Tournament: {len(names)} bots, {n_matches} matches per pair, {steps} steps\n")

    for name_a, name_b in combinations(names, 2):
        print(f"  {name_a} vs {name_b} ...", end=" ", flush=True)
        result = evaluate(agents[name_a], agents[name_b], n_matches=n_matches, steps=steps)
        w_a, w_b = result["wins"]
        d = result["draws"]
        wins[name_a] += w_a
        wins[name_b] += w_b
        draws[name_a] += d
        draws[name_b] += d

        # Update ELO for each match result
        for _ in range(w_a):
            elo = _apply_elo(elo, name_a, name_b)
        for _ in range(w_b):
            elo = _apply_elo(elo, name_b, name_a)
        for _ in range(d):
            elo = _apply_elo_draw(elo, name_a, name_b)

        print(f"{w_a}-{w_b} (draws: {d})")
        matchups.append({
            "bot1": name_a, "bot2": name_b,
            "wins_bot1": w_a, "wins_bot2": w_b, "draws": d,
            "avg_ships_bot1": result["avg_ships"][0],
            "avg_ships_bot2": result["avg_ships"][1],
        })

    # Leaderboard sorted by ELO
    print(f"\n{'Rank':<6} {'Bot':<20} {'Wins':>6} {'Draws':>6} {'ELO':>7}")
    print("-" * 48)
    ranking = sorted(names, key=lambda n: -elo[n])
    for i, name in enumerate(ranking, 1):
        print(f"{i:<6} {name:<20} {wins[name]:>6} {draws[name]:>6} {elo[name]:>7.1f}")

    if save_log:
        os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(EXPERIMENTS_DIR, f"{timestamp}.json")
        output = {
            "timestamp": timestamp,
            "config": cfg,
            "leaderboard": [
                {"rank": i + 1, "bot": name, "wins": wins[name],
                 "draws": draws[name], "elo": round(elo[name], 1)}
                for i, name in enumerate(ranking)
            ],
            "matchups": matchups,
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
