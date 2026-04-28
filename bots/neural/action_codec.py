"""ActionCodec: per-planet encode/decode for PlanetPolicyModel."""

import math

import numpy as np

from .types import ActionContext, PerPlanetLabels

BINS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]  # indices 0-7


class ActionCodec:
    NO_OP = 0
    LAUNCH = 1

    def __init__(self, n_amount_bins: int = 8, angular_diff_threshold: float = math.pi / 4) -> None:
        self.n_amount_bins = n_amount_bins
        self.BINS = BINS
        self.angular_diff_threshold = angular_diff_threshold

    def encode_per_planet(
        self,
        raw_actions: np.ndarray,
        context: ActionContext,
        planets: np.ndarray,
        value_target: float,
        max_planets: int,
    ) -> PerPlanetLabels:
        """Encode raw actions into per-planet labels.

        raw_actions: (n_actions, 3) float array — [from_planet_id, angle, ships]
        context: ActionContext for the current turn
        planets: (n_planets, 7) float32 — raw planet array (col 5 = ships)
        value_target: float game outcome for this player
        max_planets: padding target
        """
        planet_action_types = np.full(max_planets, -1, dtype=np.int32)
        planet_target_idxs = np.full(max_planets, -1, dtype=np.int32)
        planet_amount_bins = np.full(max_planets, -1, dtype=np.int32)
        my_planet_mask = np.zeros(max_planets, dtype=bool)

        # Build action map: from_planet_id -> action row (keep highest ship count)
        action_map: dict[int, np.ndarray] = {}
        if raw_actions.shape[0] > 0:
            for k in range(raw_actions.shape[0]):
                row = raw_actions[k]
                pid = int(row[0])
                if pid not in action_map or float(row[2]) > float(action_map[pid][2]):
                    action_map[pid] = row

        n = min(context.n_planets, max_planets)
        bins_arr = np.array(self.BINS, dtype=np.float32)

        for i in range(n):
            planet_id = int(context.planet_ids[i])
            if context.my_planet_mask[i]:
                my_planet_mask[i] = True
                if planet_id in action_map:
                    planet_action_types[i] = self.LAUNCH

                    # Angular-diff target inference (same logic as action_codec.py lines 58-76)
                    action_row = action_map[planet_id]
                    recorded_angle = float(action_row[1])
                    source_pos = context.planet_positions[i]
                    source_x = float(source_pos[0])
                    source_y = float(source_pos[1])

                    best_idx = -1
                    best_diff = float("inf")
                    for j in range(context.n_planets):
                        if j == i:
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

                    if best_diff > self.angular_diff_threshold:
                        planet_target_idxs[i] = -1
                    else:
                        planet_target_idxs[i] = best_idx

                    # Quantize ship fraction to amount bin
                    n_ships = float(action_row[2])
                    source_ships = float(planets[i, 5]) if planets.shape[0] > i else 0.0
                    if source_ships <= 0:
                        fraction = self.BINS[-1]
                    else:
                        fraction = float(np.clip(n_ships / source_ships, 0.0, 1.0))
                    planet_amount_bins[i] = int(np.argmin(np.abs(bins_arr - fraction)))
                else:
                    planet_action_types[i] = self.NO_OP
                    # target and amount stay at -1
            # else: my_planet_mask[i] is False, all three label arrays stay at -1 (PADDING)

        return PerPlanetLabels(
            planet_action_types=planet_action_types,
            planet_target_idxs=planet_target_idxs,
            planet_amount_bins=planet_amount_bins,
            my_planet_mask=my_planet_mask,
            value_target=value_target,
        )

    def decode_per_planet(
        self,
        output,
        context: ActionContext,
        planets: np.ndarray,
        max_planets: int,
        angular_velocity: float = 0.0,
        raw_ship_counts: np.ndarray | None = None,
    ) -> list:
        """Decode PlanetPolicyOutput into a list of game actions.

        output: PlanetPolicyOutput (batch dim already squeezed by caller)
        context: ActionContext for the current turn
        planets: (max_planets, 10) float32 — normalized planet_features (col 5 = ships/200)
        max_planets: size of the padded planet dimension
        Returns: list of [planet_id, angle, n_ships] actions
        """
        if context.n_planets == 0:
            return []

        # Convert tensors to numpy
        action_type_logits = output.action_type_logits
        if hasattr(action_type_logits, "cpu"):
            action_type_logits = action_type_logits.cpu().numpy()

        target_logits = output.target_logits
        if hasattr(target_logits, "cpu"):
            target_logits = target_logits.cpu().numpy()

        amount_logits = output.amount_logits
        if hasattr(amount_logits, "cpu"):
            amount_logits = amount_logits.cpu().numpy()

        # action_type_logits: (max_planets, 2)
        # target_logits: (max_planets, max_planets)
        # amount_logits: (max_planets, n_amount_bins)

        actions = []
        n = min(context.n_planets, max_planets)

        for i in range(n):
            if not context.my_planet_mask[i]:
                continue

            if int(np.argmax(action_type_logits[i])) == self.NO_OP:
                continue

            # LAUNCH
            target_logits_i = target_logits[i, :context.n_planets].copy()
            target_logits_i[i] = -np.inf  # mask self

            # Also mask positions where planet_ids match (same planet id as source)
            for j in range(context.n_planets):
                if context.planet_ids[j] == context.planet_ids[i]:
                    target_logits_i[j] = -np.inf

            if np.all(np.isneginf(target_logits_i)):
                continue

            target_idx = int(np.argmax(target_logits_i))

            amount_bin = int(np.argmax(amount_logits[i]))
            amount_bin = int(np.clip(amount_bin, 0, len(self.BINS) - 1))

            if raw_ship_counts is not None and i < len(raw_ship_counts):
                source_ships = float(raw_ship_counts[i])
            else:
                source_ships = 0.0

            fraction = self.BINS[amount_bin]
            n_ships = fraction * source_ships
            if n_ships < 1.0:
                continue

            source_pos = context.planet_positions[i]
            target_pos = context.planet_positions[target_idx]

            # Compute flight time estimate for orbital position adjustment
            dx0 = float(target_pos[0]) - float(source_pos[0])
            dy0 = float(target_pos[1]) - float(source_pos[1])
            distance = math.sqrt(dx0 * dx0 + dy0 * dy0)
            speed = 1.0  # fleet speed (normalised units; adjust if known)
            eta_turns = distance / speed if speed > 0 else 0.0

            # Advance orbital target position
            target_angle_base = math.atan2(float(target_pos[1]), float(target_pos[0]))
            adjusted_target_angle = target_angle_base + eta_turns * angular_velocity
            if angular_velocity != 0.0:
                orbit_radius = math.sqrt(float(target_pos[0]) ** 2 + float(target_pos[1]) ** 2)
                if orbit_radius > 0:
                    target_x = orbit_radius * math.cos(adjusted_target_angle)
                    target_y = orbit_radius * math.sin(adjusted_target_angle)
                else:
                    target_x = float(target_pos[0])
                    target_y = float(target_pos[1])
            else:
                target_x = float(target_pos[0])
                target_y = float(target_pos[1])

            angle = math.atan2(
                target_y - float(source_pos[1]),
                target_x - float(source_pos[0]),
            )

            actions.append([int(context.planet_ids[i]), angle, n_ships])

        return actions
