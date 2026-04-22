"""SampleBuilder and TrainingSample — convert episodes into training samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from dataset.catalog import DataCatalog
    from dataset.episode import EpisodeReader


@dataclass
class TrainingSample:
    state: Any
    action: Any
    reward: float | None
    next_state: Any | None
    done: bool
    info: dict  # {"path": str, "turn": int, "player": int}


class SampleBuilder:
    def __init__(
        self,
        state_transform,
        action_transform,
        reward_transform=None,
        step_filter=None,
        perspective="winner",  # int | "winner" | "both"
        mode="il_step",        # "il_step" | "rl_transition"
    ) -> None:
        if mode == "rl_transition" and reward_transform is None:
            raise ValueError("reward_transform is required when mode='rl_transition'")
        self.state_transform = state_transform
        self.action_transform = action_transform
        self.reward_transform = reward_transform
        self.step_filter = step_filter
        self.perspective = perspective
        self.mode = mode

    def build_episode(self, reader: "EpisodeReader") -> list[TrainingSample]:
        meta = reader.meta

        # Resolve players from perspective
        if self.perspective == "winner":
            if meta.winner == -1:
                return []
            players = [meta.winner]
        elif self.perspective == "both":
            players = [0, 1]
        elif self.perspective in (0, 1):
            players = [self.perspective]
        else:
            raise ValueError(f"Invalid perspective: {self.perspective!r}")

        samples: list[TrainingSample] = []
        for step in reader.steps():
            for player in players:
                if self.step_filter is not None and not self.step_filter(step, player):
                    continue

                state = self.state_transform(step, player)
                action = self.action_transform(step, player)

                if self.mode == "il_step":
                    reward = None
                    next_state = None
                    done = step.is_terminal
                else:  # rl_transition
                    done = step.is_terminal
                    if done:
                        next_step = None
                        next_state = None
                    else:
                        next_step = reader.step(step.turn + 1)
                        next_state = self.state_transform(next_step, player)
                    reward = self.reward_transform(step, next_step if not done else None, meta, player)

                samples.append(TrainingSample(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    info={"path": str(meta.path), "turn": step.turn, "player": player},
                ))

        return samples

    def build_from_catalog(self, catalog: "DataCatalog") -> Iterator[TrainingSample]:
        from dataset.episode import EpisodeReader
        for meta in catalog.episodes:
            with EpisodeReader(meta) as reader:
                yield from self.build_episode(reader)
