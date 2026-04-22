"""StepFilter Protocol and concrete implementations."""

from __future__ import annotations

from typing import Protocol

from dataset.episode import StepRecord


class StepFilter(Protocol):
    def __call__(self, step: StepRecord, player: int) -> bool: ...


class HasActionFilter:
    """True only if the player took at least one action this turn."""

    def __call__(self, step: StepRecord, player: int) -> bool:
        arr = step.actions_p0 if player == 0 else step.actions_p1
        return len(arr) > 0


class EarlyGameFilter:
    """True only if step.turn <= max_turn."""

    def __init__(self, max_turn: int) -> None:
        self.max_turn = max_turn

    def __call__(self, step: StepRecord, player: int) -> bool:
        return step.turn <= self.max_turn


class CompositeFilter:
    """Logical AND of multiple StepFilters."""

    def __init__(self, *filters: StepFilter) -> None:
        self.filters = filters

    def __call__(self, step: StepRecord, player: int) -> bool:
        return all(f(step, player) for f in self.filters)
