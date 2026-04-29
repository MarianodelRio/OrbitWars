"""StateBuilder: converts a StepRecord or live obs dict into StructuredState tensors."""

import math
from typing import TypedDict

import numpy as np

from dataset.episode import StepRecord
from .types import ActionContext


class StructuredState(TypedDict):
    planet_features: np.ndarray   # (max_planets, 24)  float32
    fleet_features: np.ndarray    # (max_fleets, 16)    float32
    fleet_mask: np.ndarray        # (max_fleets,)      bool
    planet_mask: np.ndarray       # (max_planets,)     bool
    global_features: np.ndarray   # (16,)               float32
    relational_tensor: np.ndarray  # (max_planets, max_planets, 4) float32
    context: ActionContext


class StateBuilder:
    def __init__(self, max_planets: int = 50, max_fleets: int = 200) -> None:
        self.max_planets = max_planets
        self.max_fleets = max_fleets

    @property
    def planet_feature_dim(self) -> int:
        return 24

    @property
    def fleet_feature_dim(self) -> int:
        return 16

    @property
    def global_feature_dim(self) -> int:
        return 16

    def from_obs(self, obs: dict, player: int) -> StructuredState:
        raw_planets = obs.get("planets", [])
        raw_fleets = obs.get("fleets", [])

        if len(raw_planets) == 0:
            planets = np.empty((0, 7), dtype=np.float32)
        else:
            planets = np.array(raw_planets, dtype=np.float32)

        if len(raw_fleets) == 0:
            fleets = np.empty((0, 7), dtype=np.float32)
        else:
            fleets = np.array(raw_fleets, dtype=np.float32)

        comet_ids = np.array(obs.get("comet_planet_ids", []), dtype=np.int32)
        turn = int(obs.get("step", 0))

        angular_velocity = float(obs.get("angular_velocity", 0.0))
        initial_planets_raw = obs.get("initial_planets", [])

        return self._build(
            planets, fleets, comet_ids, turn, player,
            angular_velocity=angular_velocity,
            initial_planets=initial_planets_raw,
        )

    def from_step(self, step: StepRecord, player: int, angular_velocity: float = 0.0, initial_planets=None) -> StructuredState:
        return self._build(
            step.planets, step.fleets, step.comet_planet_ids, step.turn, player,
            angular_velocity=angular_velocity,
            initial_planets=initial_planets,
        )

    def from_obs_structured(self, obs: dict, player: int) -> StructuredState:
        return self.from_obs(obs, player)

    def from_step_structured(self, step: StepRecord, player: int) -> StructuredState:
        return self.from_step(step, player)

    def __call__(self, step: StepRecord, player: int) -> StructuredState:
        return self.from_step(step, player)

    def _build_context(self, planets: np.ndarray, player: int) -> ActionContext:
        if planets.shape[0] == 0:
            return ActionContext(
                planet_ids=np.empty(0, dtype=np.int32),
                planet_positions=np.empty((0, 2), dtype=np.float32),
                my_planet_mask=np.empty(0, dtype=bool),
                n_planets=0,
            )

        planet_ids = planets[:, 0].astype(np.int32)
        planet_positions = planets[:, 2:4].astype(np.float32)
        my_planet_mask = (planets[:, 1] == player)
        n_planets = len(planet_ids)

        return ActionContext(
            planet_ids=planet_ids,
            planet_positions=planet_positions,
            my_planet_mask=my_planet_mask,
            n_planets=n_planets,
        )

    def _build_relational_tensor(self, planets: np.ndarray, n_planets: int) -> np.ndarray:
        tensor = np.zeros((self.max_planets, self.max_planets, 4), dtype=np.float32)
        if n_planets == 0:
            return tensor
        pts = planets[:n_planets, 2:4].astype(np.float64)
        owners = planets[:n_planets, 1].astype(np.int32)
        diff = pts[:, np.newaxis] - pts[np.newaxis, :]
        dists = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)
        angles_to_j = np.arctan2(diff[..., 1], diff[..., 0])
        angles_from_sun = np.arctan2(pts[:, 1] - 50.0, pts[:, 0] - 50.0)
        adiff = angles_to_j - angles_from_sun[:, np.newaxis]
        adiff = np.arctan2(np.sin(adiff), np.cos(adiff))
        same_owner = (owners[:, np.newaxis] == owners[np.newaxis, :]).astype(np.float32)
        idx = np.arange(n_planets)
        tensor[:n_planets, :n_planets, 0] = np.minimum(dists / 100.0, 1.0)
        tensor[:n_planets, :n_planets, 1] = np.abs(adiff).astype(np.float32) / np.pi
        tensor[:n_planets, :n_planets, 2] = same_owner
        tensor[:n_planets, :n_planets, 3] = (dists <= 300.0).astype(np.float32)
        tensor[idx, idx, :] = 0.0
        return tensor

    def _build(
        self,
        planets: np.ndarray,
        fleets: np.ndarray,
        comet_planet_ids: np.ndarray,
        turn: int,
        player: int,
        angular_velocity: float = 0.0,
        initial_planets=None,
    ) -> StructuredState:
        if initial_planets is None:
            initial_planets = []

        init_planet_map = {int(row[0]): row for row in initial_planets}

        planet_features = np.zeros((self.max_planets, 24), dtype=np.float32)
        planet_mask = np.zeros(self.max_planets, dtype=bool)
        fleet_features = np.zeros((self.max_fleets, 16), dtype=np.float32)
        fleet_mask = np.zeros(self.max_fleets, dtype=bool)

        comet_set = set(comet_planet_ids.tolist())

        n_planets = min(planets.shape[0], self.max_planets)
        n_fleets = min(fleets.shape[0], self.max_fleets)

        # ---------------------------------------------------------------
        # Threat feature pre-pass
        # ---------------------------------------------------------------
        enemy_ship_log_sum = np.zeros(self.max_planets, dtype=np.float32)
        friendly_ship_log_sum = np.zeros(self.max_planets, dtype=np.float32)
        enemy_ship_raw_sum = np.zeros(self.max_planets, dtype=np.float32)
        min_enemy_eta = np.full(self.max_planets, np.inf, dtype=np.float32)
        min_friendly_eta = np.full(self.max_planets, np.inf, dtype=np.float32)

        if n_fleets > 0 and n_planets > 0:
            nF = n_fleets
            nP = n_planets

            # Shared geometry block
            f_xy = fleets[:nF, 2:4].astype(np.float64)          # (nF, 2)
            p_xy = planets[:nP, 2:4].astype(np.float64)          # (nP, 2)
            f_angle = fleets[:nF, 4].astype(np.float64)          # (nF,)
            f_ships = fleets[:nF, 6].astype(np.float32)          # (nF,)
            f_ships_log = (np.log(1.0 + f_ships) / math.log(1001.0)).astype(np.float32)  # (nF,)
            f_speed = (1.0 + 5.0 * (np.log(np.maximum(f_ships, 1.0)) / math.log(1000.0)) ** 1.5).astype(np.float32)  # (nF,)
            p_owner = planets[:nP, 1].astype(np.int32)           # (nP,)
            is_enemy = (p_owner >= 0) & (p_owner != player)      # (nP,) bool
            is_friendly = (p_owner == player)                     # (nP,) bool
            f_owner_g         = fleets[:nF, 1].astype(np.int32)
            fleet_is_enemy    = (f_owner_g != player)
            fleet_is_friendly = (f_owner_g == player)

            dp = p_xy[None, :, :] - f_xy[:, None, :]             # (nF, nP, 2)
            angle_to_planet = np.arctan2(dp[..., 1], dp[..., 0]) # (nF, nP)
            raw_diff = f_angle[:, None] - angle_to_planet         # (nF, nP)
            angle_diff = np.arctan2(np.sin(raw_diff), np.cos(raw_diff))  # (nF, nP)
            in_cone = np.abs(angle_diff) < (math.pi / 4)         # (nF, nP) bool
            d = np.sqrt((dp ** 2).sum(axis=2)).astype(np.float32) # (nF, nP)
            eta = d / np.maximum(f_speed[:, None], 1e-6)          # (nF, nP)

            # Threat pre-pass (vectorized)
            ec = in_cone & fleet_is_enemy[:, None]
            fc = in_cone & fleet_is_friendly[:, None]
            enemy_ship_log_sum[:n_planets]    = (f_ships_log[:, None] * ec).sum(axis=0)
            enemy_ship_raw_sum[:n_planets]    = (f_ships[:, None]     * ec).sum(axis=0)
            friendly_ship_log_sum[:n_planets] = (f_ships_log[:, None] * fc).sum(axis=0)
            min_enemy_eta[:n_planets]         = np.where(ec, eta, np.inf).min(axis=0)
            min_friendly_eta[:n_planets]      = np.where(fc, eta, np.inf).min(axis=0)

        # ---------------------------------------------------------------
        # Strategic feature pre-pass
        # ---------------------------------------------------------------
        if n_planets > 0:
            pxy = planets[:n_planets, 2:4].astype(np.float32)  # (n_planets, 2)
            owners = planets[:n_planets, 1].astype(np.int32)

            diff_xy = pxy[:, np.newaxis, :] - pxy[np.newaxis, :, :]  # (n, n, 2)
            dists = np.sqrt((diff_xy ** 2).sum(axis=2))  # (n, n)
            np.fill_diagonal(dists, np.inf)  # exclude self

            enemy_mask = (owners != player) & (owners >= 0)  # (n,)
            friendly_mask = (owners == player)  # (n,)

            nearest_enemy_dist = np.zeros(n_planets, dtype=np.float32)
            nearest_friendly_dist = np.zeros(n_planets, dtype=np.float32)
            n_enemy_within_30 = np.zeros(n_planets, dtype=np.float32)
            is_frontline = np.zeros(n_planets, dtype=np.float32)

            enemy_dists_mat    = np.where(enemy_mask[None, :],    dists, np.inf)
            friendly_dists_mat = np.where(friendly_mask[None, :], dists, np.inf)
            nearest_enemy_dist[:n_planets]    = enemy_dists_mat.min(axis=1)
            nearest_friendly_dist[:n_planets] = friendly_dists_mat.min(axis=1)
            n_enemy_within_30[:n_planets]     = (enemy_dists_mat < 30.0).sum(axis=1).astype(np.float32)
            is_frontline[:n_planets]          = (nearest_enemy_dist[:n_planets] < 30.0).astype(np.float32)
        else:
            nearest_enemy_dist = np.zeros(self.max_planets, dtype=np.float32)
            nearest_friendly_dist = np.zeros(self.max_planets, dtype=np.float32)
            n_enemy_within_30 = np.zeros(self.max_planets, dtype=np.float32)
            is_frontline = np.zeros(self.max_planets, dtype=np.float32)

        # ---------------------------------------------------------------
        # Fill planet slots
        # ---------------------------------------------------------------
        for i in range(n_planets):
            row = planets[i]
            owner = row[1]
            x = float(row[2])
            y = float(row[3])
            radius = float(row[4])
            ships = float(row[5])
            production = float(row[6])
            planet_id = int(row[0])

            planet_features[i, 0] = float(owner == player)
            planet_features[i, 1] = float(owner not in (-1, player))
            planet_features[i, 2] = float(owner == -1)
            planet_features[i, 3] = x / 100.0
            planet_features[i, 4] = y / 100.0
            planet_features[i, 5] = math.log(1.0 + ships) / math.log(1001.0)
            planet_features[i, 6] = production / 5.0
            planet_features[i, 7] = radius / 3.0
            planet_features[i, 8] = float(planet_id in comet_set)
            planet_features[i, 9] = math.hypot(x - 50.0, y - 50.0) / 50.0

            # Orbital features (indices 10–14)
            dist_from_sun = math.hypot(x - 50.0, y - 50.0)
            is_orbital = int(dist_from_sun + radius < 50.0 and planet_id not in comet_set)
            if is_orbital:
                init_row = init_planet_map.get(planet_id)
                if init_row is not None:
                    theta = math.atan2(float(init_row[3]) - 50.0, float(init_row[2]) - 50.0)
                else:
                    theta = math.atan2(y - 50.0, x - 50.0)
            else:
                theta = 0.0
            planet_features[i, 10] = float(is_orbital)
            planet_features[i, 11] = math.sin(theta) if is_orbital else 0.0
            planet_features[i, 12] = math.cos(theta) if is_orbital else 0.0
            planet_features[i, 13] = (angular_velocity / 0.05) if is_orbital else 0.0
            planet_features[i, 14] = dist_from_sun / 50.0

            # Threat features (indices 15–19)
            planet_features[i, 15] = math.log(1.0 + enemy_ship_log_sum[i]) / math.log(1001.0)
            planet_features[i, 16] = math.log(1.0 + friendly_ship_log_sum[i]) / math.log(1001.0)
            planet_features[i, 17] = min(min_enemy_eta[i], 50.0) / 50.0 if min_enemy_eta[i] != np.inf else 0.0
            planet_features[i, 18] = min(min_friendly_eta[i], 50.0) / 50.0 if min_friendly_eta[i] != np.inf else 0.0
            garrison = float(planets[i, 5])
            planet_features[i, 19] = float(np.clip((garrison - enemy_ship_raw_sum[i]) / 200.0, -1.0, 1.0))

            # Strategic features (indices 20–23)
            planet_features[i, 20] = min(nearest_enemy_dist[i], 100.0) / 100.0
            planet_features[i, 21] = min(nearest_friendly_dist[i], 100.0) / 100.0
            planet_features[i, 22] = min(n_enemy_within_30[i], 10.0) / 10.0
            planet_features[i, 23] = is_frontline[i]

            planet_mask[i] = True

        # ---------------------------------------------------------------
        # Fill fleet slots
        # ---------------------------------------------------------------
        if n_fleets > 0:
            nF = n_fleets
            # Use precomputed arrays if geometry block ran, else compute now
            if n_planets == 0:
                _f_ships     = fleets[:nF, 6].astype(np.float32)
                _f_ships_log = (np.log(1.0 + _f_ships) / math.log(1001.0)).astype(np.float32)
                _f_speed     = (1.0 + 5.0 * (np.log(np.maximum(_f_ships, 1.0)) / math.log(1000.0)) ** 1.5).astype(np.float32)
                _f_angle     = fleets[:nF, 4].astype(np.float32)
            else:
                _f_ships     = f_ships
                _f_ships_log = f_ships_log
                _f_speed     = f_speed
                _f_angle     = f_angle.astype(np.float32)

            # Simple features (indices 0–8)
            fleet_features[:nF, 0] = (fleets[:nF, 1] == player).astype(np.float32)
            fleet_features[:nF, 1] = (fleets[:nF, 1] != player).astype(np.float32)
            fleet_features[:nF, 2] = fleets[:nF, 2] / 100.0
            fleet_features[:nF, 3] = fleets[:nF, 3] / 100.0
            fleet_features[:nF, 4] = np.sin(_f_angle)
            fleet_features[:nF, 5] = np.cos(_f_angle)
            fleet_features[:nF, 6] = _f_ships_log
            fleet_features[:nF, 7] = np.hypot(fleets[:nF, 2] - 50.0, fleets[:nF, 3] - 50.0) / 50.0
            fleet_features[:nF, 8] = np.minimum(_f_speed / 6.0, 1.0)
            fleet_mask[:nF] = True

            # Best-target features (indices 9–12)
            if n_planets > 0:
                d_in_cone  = np.where(in_cone, d, np.inf)
                best_dist_arr  = d_in_cone.min(axis=1)
                best_idx   = np.argmin(d_in_cone, axis=1)
                has_target = np.isfinite(best_dist_arr)
                best_owner = p_owner[best_idx]
                fleet_features[:nF, 9]  = np.where(has_target, np.minimum(best_dist_arr / 100.0, 1.0), 0.0)
                fleet_features[:nF, 10] = np.where(has_target, np.minimum(best_dist_arr / np.maximum(_f_speed, 1e-6) / 50.0, 1.0), 0.0)
                fleet_features[:nF, 11] = np.where(has_target, (best_owner == player).astype(np.float32), 0.0)
                fleet_features[:nF, 12] = np.where(has_target, ((best_owner >= 0) & (best_owner != player)).astype(np.float32), 0.0)
            # indices 13–15 remain 0.0 (reserved)

        # ---------------------------------------------------------------
        # Compute global features
        # ---------------------------------------------------------------
        global_features = np.zeros(16, dtype=np.float32)
        global_features[0] = turn / 500.0

        if planets.shape[0] > 0:
            total_ships = float(planets[:, 5].sum())
            if total_ships > 0:
                my_ships = float(planets[planets[:, 1] == player, 5].sum())
                global_features[1] = my_ships / total_ships
            else:
                global_features[1] = 0.5

            total_planets = float(planets.shape[0])
            my_planets = float((planets[:, 1] == player).sum())
            global_features[2] = my_planets / total_planets
        else:
            global_features[1] = 0.5
            global_features[2] = 0.0

        global_features[3] = min(fleets.shape[0], self.max_fleets) / 200.0

        # index 4: centered turn
        global_features[4] = (turn - 250.0) / 250.0

        # index 5–8: vectorized planet-based global features
        if n_planets > 0:
            ships_g        = planets[:n_planets, 5].astype(np.float32)
            owners_g       = planets[:n_planets, 1].astype(np.int32)
            prod_g         = planets[:n_planets, 6].astype(np.float32)
            my_mask_g      = (owners_g == player)
            enemy_mask_g   = (owners_g >= 0) & (owners_g != player)
            neutral_mask_g = (owners_g == -1)
            total_ships_g  = max(float(ships_g.sum()), 1.0)
        else:
            ships_g        = np.empty(0, dtype=np.float32)
            owners_g       = np.empty(0, dtype=np.int32)
            prod_g         = np.empty(0, dtype=np.float32)
            my_mask_g      = np.empty(0, dtype=bool)
            enemy_mask_g   = np.empty(0, dtype=bool)
            neutral_mask_g = np.empty(0, dtype=bool)
            total_ships_g  = 1.0

        # index 5: enemy_max_ship_share
        enemy_unique = np.unique(owners_g[enemy_mask_g]) if enemy_mask_g.any() else np.empty(0, dtype=np.int32)
        if len(enemy_unique) > 0:
            global_features[5] = max(
                float(ships_g[owners_g == eo].sum()) / total_ships_g
                for eo in enemy_unique
            )

        # index 6: neutral_share
        global_features[6] = float(ships_g[neutral_mask_g].sum()) / total_ships_g

        # index 7: my_production_share
        total_prod_g = max(float(prod_g.sum()), 1.0)
        global_features[7] = float(prod_g[my_mask_g].sum()) / total_prod_g

        # index 8: enemy_max_planet_share
        total_planets_g = max(n_planets, 1)
        if len(enemy_unique) > 0:
            global_features[8] = max(
                float((owners_g == eo).sum()) / total_planets_g
                for eo in enemy_unique
            )

        # index 9: n_my_fleets / 200
        if n_fleets > 0:
            n_my_fleets = int((fleets[:n_fleets, 1] == player).sum())
        else:
            n_my_fleets = 0
        global_features[9] = min(n_my_fleets / 200.0, 1.0)

        # index 10: n_enemy_fleets / 200
        n_enemy_fleets = n_fleets - n_my_fleets
        global_features[10] = min(n_enemy_fleets / 200.0, 1.0)

        # index 11: next_comet_spawn_turns / 500
        comet_spawns = [50, 150, 250, 350, 450]
        future_spawns = [t for t in comet_spawns if t > turn]
        next_spawn_in = (min(future_spawns) - turn) if future_spawns else (500 - turn)
        global_features[11] = max(next_spawn_in, 0) / 500.0

        # index 12-14: phase one-hot
        global_features[12] = 1.0 if turn < 100 else 0.0
        global_features[13] = 1.0 if 100 <= turn < 350 else 0.0
        global_features[14] = 1.0 if turn >= 350 else 0.0

        # index 15: my_eliminated
        my_planet_count = int(my_mask_g.sum()) if n_planets > 0 else 0
        my_fleet_count = n_my_fleets
        global_features[15] = 1.0 if (my_planet_count == 0 and my_fleet_count == 0) else 0.0

        # ---------------------------------------------------------------
        # Relational tensor
        # ---------------------------------------------------------------
        relational_tensor = self._build_relational_tensor(planets, n_planets)

        context = self._build_context(planets, player)

        return StructuredState(
            planet_features=planet_features,
            fleet_features=fleet_features,
            fleet_mask=fleet_mask,
            planet_mask=planet_mask,
            global_features=global_features,
            relational_tensor=relational_tensor,
            context=context,
        )
