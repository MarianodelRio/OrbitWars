import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.env.evaluator import evaluate, load_agent

REGISTERED_BOTS = {
    "sniper": "bots.heuristic.sniper:agent_fn",
    "random": "bots.heuristic.random_bot:agent_fn",
    "baseline": "bots.heuristic.baseline:agent_fn",
}

def main():
    names = list(REGISTERED_BOTS.keys())
    agents = {name: load_agent(path) for name, path in REGISTERED_BOTS.items()}
    wins = {name: 0 for name in names}
    print("Round-robin tournament")
    print("-" * 40)
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            results = evaluate(agents[name1], agents[name2], n_matches=5, steps=200)
            w1, w2 = results["wins"]
            wins[name1] += w1
            wins[name2] += w2
            print(f"{name1} vs {name2}: {w1}-{w2} (draws: {results['draws']})")
    print("\nStandings:")
    for name, w in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"  {name}: {w} wins")

if __name__ == "__main__":
    main()
