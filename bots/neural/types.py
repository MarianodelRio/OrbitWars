"""Pure-numpy intermediate structs for the neural bot. No PyTorch imports."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ActionContext:
    planet_ids: np.ndarray       # shape (n,) int32
    planet_positions: np.ndarray # shape (n, 2) float32 — raw x, y (not normalized)
    my_planet_mask: np.ndarray   # shape (n,) bool
    n_planets: int


@dataclass
class ModelInput:
    array: np.ndarray   # shape (D,) float32
    context: ActionContext


@dataclass
class ModelLabels:
    action_type: int    # 0 = NO_OP, 1 = LAUNCH
    source_idx: int     # -1 if NO_OP
    target_idx: int     # -1 if NO_OP
    amount_bin: int     # -1 if NO_OP
    value_target: float
