"""Round-robin tournament runner."""
from itertools import combinations
from game.env.evaluator import evaluate


def run_tournament(bot_registry: dict, n_matches: int = 5, steps: int = 200) -> dict:
    """Run a round-robin tournament. bot_registry maps name -> agent callable."""
    wins = {name: 0 for name in bot_registry}
    names = list(bot_registry.keys())

    for name_a, name_b in combinations(names, 2):
        result = evaluate(bot_registry[name_a], bot_registry[name_b], n_matches=n_matches, steps=steps)
        wins[name_a] += result["wins"][0]
        wins[name_b] += result["wins"][1]

    return wins
