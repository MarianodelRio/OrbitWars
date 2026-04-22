"""ActionCodec: encodes raw dataset actions to ModelLabels."""

import math

import numpy as np

from .types import ActionContext, ModelLabels


class ActionCodec:
    NO_OP = 0
    LAUNCH = 1

    def __init__(self, n_amount_bins: int = 5) -> None:
        self.n_amount_bins = n_amount_bins
        self.BINS = [0.1, 0.25, 0.5, 0.75, 1.0]

    def encode(
        self,
        raw_actions: np.ndarray,
        context: ActionContext,
        planets: np.ndarray,
        value_target: float,
    ) -> ModelLabels:
        if raw_actions.shape[0] == 0:
            return ModelLabels(
                action_type=self.NO_OP,
                source_idx=-1,
                target_idx=-1,
                amount_bin=-1,
                value_target=value_target,
            )

        dominant = raw_actions[np.argmax(raw_actions[:, 2])]

        from_planet_id = int(dominant[0])
        matches = np.where(context.planet_ids == from_planet_id)[0]
        if len(matches) == 0:
            return ModelLabels(
                action_type=self.NO_OP,
                source_idx=-1,
                target_idx=-1,
                amount_bin=-1,
                value_target=value_target,
            )
        source_idx = int(matches[0])

        if context.n_planets <= 1:
            return ModelLabels(
                action_type=self.NO_OP,
                source_idx=-1,
                target_idx=-1,
                amount_bin=-1,
                value_target=value_target,
            )

        recorded_angle = float(dominant[1])
        source_pos = context.planet_positions[source_idx]
        source_x = float(source_pos[0])
        source_y = float(source_pos[1])

        best_idx = -1
        best_diff = float("inf")
        for j in range(context.n_planets):
            if j == source_idx:
                continue
            cand_pos = context.planet_positions[j]
            dx = float(cand_pos[0]) - source_x
            dy = float(cand_pos[1]) - source_y
            direction = math.atan2(dy, dx)
            diff_raw = direction - recorded_angle
            angular_diff = abs(((diff_raw + math.pi) % (2 * math.pi)) - math.pi)
            if angular_diff < best_diff:
                best_diff = angular_diff
                best_idx = j
        target_idx = best_idx

        n_ships = float(dominant[2])
        source_ships = float(planets[source_idx, 5])
        if source_ships <= 0:
            fraction = self.BINS[-1]
        else:
            fraction = float(np.clip(n_ships / source_ships, 0.0, 1.0))

        bins_arr = np.array(self.BINS, dtype=np.float32)
        amount_bin = int(np.argmin(np.abs(bins_arr - fraction)))

        return ModelLabels(
            action_type=self.LAUNCH,
            source_idx=source_idx,
            target_idx=target_idx,
            amount_bin=amount_bin,
            value_target=value_target,
        )

    def decode(self, output, context: ActionContext, planets: np.ndarray) -> list:
        """Decode a PolicyOutput into a list of game actions.

        output: PolicyOutput (from model.py) — may be batched (B=1 already squeezed by caller)
        context: ActionContext for the current turn
        planets: (n_planets, 7) float32 array — for available ships
        Returns: [] for NO_OP, or [[planet_id, angle, n_ships]] for LAUNCH
        """
        if context.n_planets == 0:
            return []

        # 1. Action type
        action_type_logits = output.action_type_logits
        if hasattr(action_type_logits, "cpu"):
            action_type_logits = action_type_logits.cpu().numpy()
        if int(np.argmax(action_type_logits)) == self.NO_OP:
            return []

        # 2. Source: argmax over source_logits masked to my_planet_mask
        source_logits = output.source_logits
        if hasattr(source_logits, "cpu"):
            source_logits = source_logits.cpu().numpy()
        source_logits = source_logits[:context.n_planets].copy()

        my_mask = context.my_planet_mask
        if not my_mask.any():
            return []
        source_logits[~my_mask] = -np.inf
        source_idx = int(np.argmax(source_logits))

        # 3. Target: argmax over target_logits excluding source_idx
        target_logits = output.target_logits
        if hasattr(target_logits, "cpu"):
            target_logits = target_logits.cpu().numpy()
        target_logits = target_logits[:context.n_planets].copy()
        target_logits[source_idx] = -np.inf
        if np.all(np.isneginf(target_logits)):
            return []
        target_idx = int(np.argmax(target_logits))

        # 4. Amount bin
        amount_logits = output.amount_logits
        if hasattr(amount_logits, "cpu"):
            amount_logits = amount_logits.cpu().numpy()
        amount_bin = int(np.argmax(amount_logits))
        amount_bin = int(np.clip(amount_bin, 0, len(self.BINS) - 1))

        # 5. Ships to send
        if planets.shape[0] > source_idx:
            source_ships = float(planets[source_idx, 5])
        else:
            return []
        n_ships = self.BINS[amount_bin] * source_ships
        if n_ships < 1.0:
            return []

        # 6. Angle from source to target
        source_pos = context.planet_positions[source_idx]
        target_pos = context.planet_positions[target_idx]
        angle = math.atan2(
            float(target_pos[1]) - float(source_pos[1]),
            float(target_pos[0]) - float(source_pos[0]),
        )

        source_planet_id = int(context.planet_ids[source_idx])
        return [[source_planet_id, angle, n_ships]]
