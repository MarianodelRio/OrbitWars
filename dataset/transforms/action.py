"""ActionTransform Protocol and concrete implementations."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from dataset.episode import StepRecord


class ActionTransform(Protocol):
    def __call__(self, step: StepRecord, player: int) -> Any: ...


class RawActionTransform:
    """Returns the action array for the given player. Shape (n_actions, 3); (0, 3) if no actions."""

    def __call__(self, step: StepRecord, player: int) -> np.ndarray:
        return step.actions_p0 if player == 0 else step.actions_p1
