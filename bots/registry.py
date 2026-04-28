"""Bot registry: maps short names to importable agent paths."""

from __future__ import annotations

from game.env.evaluator import load_agent

REGISTRY: dict[str, str] = {
    "heuristic.baseline": "bots.heuristic.baseline:agent_fn",
    "heuristic.sniper": "bots.heuristic.sniper:agent_fn",
    "heuristic.oracle_sniper": "bots.heuristic.oracle_sniper:agent_fn",
    "scoring.bot": "bots.scoring.bot:agent_fn",
}


def resolve(name: str):
    """Resolve a short registry name to a callable agent_fn, or None if not found."""
    path = REGISTRY.get(name)
    if path is None:
        return None
    return load_agent(path)


def resolve_checkpoint(path: str):
    """Load a NeuralBot from a checkpoint file and return its act method as agent_fn."""
    from bots.neural.bot import NeuralBot
    from bots.interface import make_agent
    bot = NeuralBot.load(path)
    return make_agent(bot)


def list_bots() -> list[str]:
    """Return sorted list of registered bot names."""
    return sorted(REGISTRY.keys())
