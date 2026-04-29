"""OpponentPool: manages a pool of opponents for self-play and league training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PoolEntry:
    name: str
    loader: object  # callable: () -> agent_fn

    def __post_init__(self):
        self._agent = None

    def get_agent(self):
        if self._agent is None:
            self._agent = self.loader()
        return self._agent


class OpponentPool:
    def __init__(self, max_snapshots: int = 5) -> None:
        self.max_snapshots = max_snapshots
        self._entries: list[PoolEntry] = []
        self._snapshot_entries: list[PoolEntry] = []

    def add_heuristic(self, name: str, agent_fn_path: str) -> None:
        def loader():
            from game.env.evaluator import load_agent
            return load_agent(agent_fn_path)

        entry = PoolEntry(name=name, loader=loader)
        self._entries.append(entry)

    def add_frozen_checkpoint(self, path: str) -> None:
        def loader():
            from bots.neural.bot import NeuralBot
            from bots.interface import make_agent
            bot = NeuralBot.load(path)
            return make_agent(bot)

        entry = PoolEntry(name=f"frozen:{path}", loader=loader)
        self._entries.append(entry)

    def add_snapshot(self, path: Path, iteration: int) -> None:
        path_str = str(path)

        def loader():
            from bots.neural.bot import NeuralBot
            from bots.interface import make_agent
            bot = NeuralBot.load(path_str)
            return make_agent(bot)

        entry = PoolEntry(name=f"snapshot:{iteration}", loader=loader)
        self._entries.append(entry)
        self._snapshot_entries.append(entry)

        if len(self._snapshot_entries) > self.max_snapshots:
            oldest = self._snapshot_entries[0]
            self._snapshot_entries = self._snapshot_entries[1:]
            # Remove from _entries by identity
            self._entries = [e for e in self._entries if e is not oldest]

    def sample(self, rng: random.Random | None = None, self_play_prob: float = 0.0, current_model_fn=None, return_name: bool = False):
        if current_model_fn is not None and random.random() < self_play_prob:
            fn = current_model_fn
            return (fn, "self") if return_name else fn
        if not self._entries:
            fn = lambda obs, config=None: []
            return (fn, "empty") if return_name else fn
        if rng is not None:
            entry = rng.choice(self._entries)
        else:
            entry = random.choice(self._entries)
        fn = entry.get_agent()
        return (fn, entry.name) if return_name else fn

    def size(self) -> int:
        return len(self._entries)
