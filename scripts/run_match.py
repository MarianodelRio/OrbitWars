import sys
import os
import argparse
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.env.runner import run_match

def load_agent(path_str):
    module_path, attr = path_str.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)

def main():
    parser = argparse.ArgumentParser(description="Run an Orbit Wars match")
    parser.add_argument("--bot1", default="bots.heuristic.sniper:agent_fn")
    parser.add_argument("--bot2", default="bots.heuristic.baseline:agent_fn")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", default="game.html")
    args = parser.parse_args()

    bot1 = load_agent(args.bot1)
    bot2 = load_agent(args.bot2)

    result = run_match(bot1, bot2, steps=args.steps, render=args.render, output_file=args.output)
    print(f"Winner: {result['winner']}")
    print(f"Rewards: {result['rewards']}")
    print(f"Steps: {result['steps']}")
    if args.render:
        print(f"HTML saved to: {args.output}")

if __name__ == "__main__":
    main()
