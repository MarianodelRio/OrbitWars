"""RewardTransform Protocol and concrete implementations."""

from __future__ import annotations

from typing import Protocol

from dataset.catalog import EpisodeMeta
from dataset.episode import StepRecord


class RewardTransform(Protocol):
    def __call__(
        self,
        step: StepRecord,
        next_step: StepRecord | None,
        meta: EpisodeMeta,
        player: int,
    ) -> float: ...


class BinaryOutcomeReward:
    """0.0 each turn. +1.0 on terminal if player won, -1.0 if player lost, 0.0 on draw."""

    def __call__(
        self,
        step: StepRecord,
        next_step: StepRecord | None,
        meta: EpisodeMeta,
        player: int,
    ) -> float:
        if not step.is_terminal:
            return 0.0
        if meta.winner == player:
            return 1.0
        if meta.winner == -1:
            return 0.0
        return -1.0
