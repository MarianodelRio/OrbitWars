"""StateTransform Protocol and concrete implementations."""

from __future__ import annotations

from typing import Any, Protocol

from dataset.episode import StepRecord


class StateTransform(Protocol):
    def __call__(self, step: StepRecord, player: int) -> Any: ...


class RawStateTransform:
    """Returns planets and fleets as a dict of unpadded numpy arrays."""

    def __call__(self, step: StepRecord, player: int) -> dict:
        return {
            "planets": step.planets,  # (n_planets, 7)
            "fleets": step.fleets,    # (n_fleets, 7)
        }
