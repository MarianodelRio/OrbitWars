import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.env.evaluator import evaluate, load_agent

def main():
    parser = argparse.ArgumentParser(description="Evaluate two bots over N matches")
    parser.add_argument("--bot1", default="bots.heuristic.sniper:agent_fn")
    parser.add_argument("--bot2", default="bots.heuristic.baseline:agent_fn")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    bot1 = load_agent(args.bot1)
    bot2 = load_agent(args.bot2)
    results = evaluate(bot1, bot2, n_matches=args.n, steps=args.steps)

    print(f"Results over {args.n} matches ({args.steps} steps each):")
    print(f"  Bot1 win rate: {results['win_rate'][0]:.1%}")
    print(f"  Bot2 win rate: {results['win_rate'][1]:.1%}")
    print(f"  Draws: {results['draws']}")
    print(f"  Avg ships — Bot1: {results['avg_ships'][0]:.1f}, Bot2: {results['avg_ships'][1]:.1f}")
    print(f"  Avg game length: {results['avg_game_length']:.1f} steps")

if __name__ == "__main__":
    main()
