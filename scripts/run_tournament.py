"""CLI wrapper for the tournament runner."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tournament.runner import run_tournament
from game.env.evaluator import load_agent

REGISTERED_BOTS = {
    "sniper":   "bots.heuristic.sniper:agent_fn",
    "random":   "bots.heuristic.random_bot:agent_fn",
    "baseline": "bots.heuristic.baseline:agent_fn",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a round-robin tournament.")
    parser.add_argument("--n", type=int, default=5, help="Matches per pair")
    parser.add_argument("--steps", type=int, default=200, help="Steps per match")
    args = parser.parse_args()

    bots = {name: load_agent(path) for name, path in REGISTERED_BOTS.items()}
    wins = run_tournament(bots, n_matches=args.n, steps=args.steps)

    print(f"\n{'Bot':<15} {'Wins':>6}")
    print("-" * 23)
    for name, w in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"{name:<15} {w:>6}")
