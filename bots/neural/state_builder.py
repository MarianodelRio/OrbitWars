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
        sun_x, sun_y = 50.0, 50.0
        for i in range(n_planets):
            xi, yi = float(planets[i, 2]), float(planets[i, 3])
            oi = int(planets[i, 1])
            for j in range(n_planets):
                if i == j:
                    continue
                xj, yj = float(planets[j, 2]), float(planets[j, 3])
                oj = int(planets[j, 1])
                d = math.hypot(xj - xi, yj - yi)
                # channel 0: distance_norm
                tensor[i, j, 0] = min(d / 100.0, 1.0)
                # channel 1: angle_diff/pi
                angle_to_j = math.atan2(yj - yi, xj - xi)
                angle_from_sun = math.atan2(yi - sun_y, xi - sun_x)
                adiff = angle_to_j - angle_from_sun
                adiff = math.atan2(math.sin(adiff), math.cos(adiff))
                tensor[i, j, 1] = abs(adiff) / math.pi
                # channel 2: same_owner
                tensor[i, j, 2] = 1.0 if oi == oj else 0.0
                # channel 3: is_reachable_in_50_turns (max speed ~6.0)
                tensor[i, j, 3] = 1.0 if d <= 6.0 * 50.0 else 0.0
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

        for fi in range(n_fleets):
            fleet = fleets[fi]
            fleet_owner = int(fleet[1])
            fx, fy = float(fleet[2]), float(fleet[3])
            fangle = float(fleet[4])
            fships = float(fleet[6])
            fships_log = math.log(1.0 + fships) / math.log(1001.0)
            speed = 1.0 + 5.0 * (math.log(max(fships, 1.0)) / math.log(1000.0)) ** 1.5

            for pi in range(n_planets):
                px, py = float(planets[pi, 2]), float(planets[pi, 3])
                dpx, dpy = px - fx, py - fy
                angle_to_planet = math.atan2(dpy, dpx)
                diff = fangle - angle_to_planet
                angle_diff = abs(math.atan2(math.sin(diff), math.cos(diff)))
                if angle_diff < math.pi / 4:
                    d = math.hypot(dpx, dpy)
                    eta = d / max(speed, 1e-6)
                    if fleet_owner == player:
                        friendly_ship_log_sum[pi] += fships_log
                        min_friendly_eta[pi] = min(min_friendly_eta[pi], eta)
                    else:
                        enemy_ship_log_sum[pi] += fships_log
                        enemy_ship_raw_sum[pi] += fships
                        min_enemy_eta[pi] = min(min_enemy_eta[pi], eta)

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

            for pi in range(n_planets):
                enemy_dists = dists[pi, enemy_mask]
                fm = friendly_mask.copy()
                fm[pi] = False
                friendly_dists = dists[pi, fm]

                nearest_enemy_dist[pi] = enemy_dists.min() if len(enemy_dists) > 0 else 100.0
                nearest_friendly_dist[pi] = friendly_dists.min() if len(friendly_dists) > 0 else 100.0
                n_enemy_within_30[pi] = float((enemy_dists < 30.0).sum()) if len(enemy_dists) > 0 else 0.0
                is_frontline[pi] = 1.0 if nearest_enemy_dist[pi] < 30.0 else 0.0
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
        planet_pos_map = {
            int(planets[pi, 0]): (float(planets[pi, 2]), float(planets[pi, 3]), int(planets[pi, 1]))
            for pi in range(n_planets)
        }

        for i in range(n_fleets):
            row = fleets[i]
            owner = row[1]
            x = float(row[2])
            y = float(row[3])
            angle = float(row[4])
            ships = float(row[6])

            fleet_features[i, 0] = float(owner == player)
            fleet_features[i, 1] = float(owner != player)
            fleet_features[i, 2] = x / 100.0
            fleet_features[i, 3] = y / 100.0
            fleet_features[i, 4] = math.sin(angle)
            fleet_features[i, 5] = math.cos(angle)
            fleet_features[i, 6] = math.log(1.0 + ships) / math.log(1001.0)
            fleet_features[i, 7] = math.hypot(x - 50.0, y - 50.0) / 50.0

            # Fleet extended features (indices 8–12)
            ships_f = ships
            speed = 1.0 + 5.0 * (math.log(max(ships_f, 1.0)) / math.log(1000.0)) ** 1.5
            fleet_features[i, 8] = min(speed / 6.0, 1.0)

            fx_f, fy_f = x, y
            fangle_f = angle
            best_dist = None
            best_owner = -1
            for pid, (tx, ty, towner) in planet_pos_map.items():
                ddx, ddy = tx - fx_f, ty - fy_f
                ang_to = math.atan2(ddy, ddx)
                diff = fangle_f - ang_to
                adiff = abs(math.atan2(math.sin(diff), math.cos(diff)))
                if adiff < math.pi / 4:
                    d = math.hypot(ddx, ddy)
                    if best_dist is None or d < best_dist:
                        best_dist = d
                        best_owner = towner

            if best_dist is not None:
                fleet_features[i, 9] = min(best_dist / 100.0, 1.0)
                fleet_features[i, 10] = min((best_dist / max(speed, 1e-6)) / 50.0, 1.0)
                fleet_features[i, 11] = float(best_owner == player)
                fleet_features[i, 12] = float(best_owner >= 0 and best_owner != player)
            # indices 13–15 remain 0.0 (reserved)

            fleet_mask[i] = True

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

        # index 5: enemy_max_ship_share
        total_ships_g = max(sum(float(p[5]) for p in planets[:n_planets]), 1.0)
        enemy_owners = set(int(p[1]) for p in planets[:n_planets] if int(p[1]) >= 0 and int(p[1]) != player)
        if enemy_owners:
            global_features[5] = max(
                sum(float(p[5]) for p in planets[:n_planets] if int(p[1]) == eo) / total_ships_g
                for eo in enemy_owners
            )

        # index 6: neutral_share
        neutral_ships = sum(float(p[5]) for p in planets[:n_planets] if int(p[1]) == -1)
        global_features[6] = neutral_ships / total_ships_g

        # index 7: my_production_share
        total_prod_g = max(sum(float(p[6]) for p in planets[:n_planets]), 1.0)
        my_prod_g = sum(float(p[6]) for p in planets[:n_planets] if int(p[1]) == player)
        global_features[7] = my_prod_g / total_prod_g

        # index 8: enemy_max_planet_share
        total_planets_g = max(n_planets, 1)
        if enemy_owners:
            global_features[8] = max(
                sum(1 for p in planets[:n_planets] if int(p[1]) == eo) / total_planets_g
                for eo in enemy_owners
            )

        # index 9: n_my_fleets / 200
        n_my_fleets = sum(1 for fi in range(n_fleets) if int(fleets[fi, 1]) == player)
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
        my_planet_count = sum(1 for p in planets[:n_planets] if int(p[1]) == player)
        my_fleet_count = sum(1 for fi in range(n_fleets) if int(fleets[fi, 1]) == player)
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
