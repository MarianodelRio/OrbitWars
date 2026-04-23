"""StateBuilderV2: converts a StepRecord or live obs dict into StructuredStateV2 tensors."""

import math
from typing import TypedDict

import numpy as np

from dataset.episode import StepRecord
from .types import ActionContext


class StructuredStateV2(TypedDict):
    planet_features: np.ndarray   # (max_planets, 10)  float32
    fleet_features: np.ndarray    # (max_fleets, 8)    float32
    fleet_mask: np.ndarray        # (max_fleets,)      bool
    planet_mask: np.ndarray       # (max_planets,)     bool
    global_features: np.ndarray   # (4,)               float32
    context: ActionContext


class StateBuilderV2:
    def __init__(self, max_planets: int = 50, max_fleets: int = 200) -> None:
        self.max_planets = max_planets
        self.max_fleets = max_fleets

    def from_obs(self, obs: dict, player: int) -> StructuredStateV2:
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

        return self._build_v2(planets, fleets, comet_ids, turn, player)

    def from_step(self, step: StepRecord, player: int) -> StructuredStateV2:
        return self._build_v2(step.planets, step.fleets, step.comet_planet_ids, step.turn, player)

    def from_obs_structured(self, obs: dict, player: int) -> StructuredStateV2:
        return self.from_obs(obs, player)

    def from_step_structured(self, step: StepRecord, player: int) -> StructuredStateV2:
        return self.from_step(step, player)

    def __call__(self, step: StepRecord, player: int) -> StructuredStateV2:
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

    def _build_v2(
        self,
        planets: np.ndarray,
        fleets: np.ndarray,
        comet_planet_ids: np.ndarray,
        turn: int,
        player: int,
    ) -> StructuredStateV2:
        planet_features = np.zeros((self.max_planets, 10), dtype=np.float32)
        planet_mask = np.zeros(self.max_planets, dtype=bool)
        fleet_features = np.zeros((self.max_fleets, 8), dtype=np.float32)
        fleet_mask = np.zeros(self.max_fleets, dtype=bool)

        comet_set = set(comet_planet_ids.tolist())

        # Fill planet slots
        n_planets = min(planets.shape[0], self.max_planets)
        for i in range(n_planets):
            row = planets[i]
            owner = row[1]
            x = row[2]
            y = row[3]
            radius = row[4]
            ships = row[5]
            production = row[6]
            planet_id = int(row[0])

            planet_features[i, 0] = float(owner == player)
            planet_features[i, 1] = float(owner not in (-1, player))
            planet_features[i, 2] = float(owner == -1)
            planet_features[i, 3] = x / 100.0
            planet_features[i, 4] = y / 100.0
            planet_features[i, 5] = float(np.clip(ships / 200.0, 0.0, 1.0))
            planet_features[i, 6] = production / 5.0
            planet_features[i, 7] = radius / 3.0
            planet_features[i, 8] = float(planet_id in comet_set)
            planet_features[i, 9] = math.hypot(x - 50.0, y - 50.0) / 50.0
            planet_mask[i] = True

        # Fill fleet slots
        n_fleets = min(fleets.shape[0], self.max_fleets)
        for i in range(n_fleets):
            row = fleets[i]
            owner = row[1]
            x = row[2]
            y = row[3]
            angle = row[4]
            ships = row[6]

            fleet_features[i, 0] = float(owner == player)
            fleet_features[i, 1] = float(owner != player)
            fleet_features[i, 2] = x / 100.0
            fleet_features[i, 3] = y / 100.0
            fleet_features[i, 4] = math.sin(angle)
            fleet_features[i, 5] = math.cos(angle)
            fleet_features[i, 6] = float(np.clip(ships / 200.0, 0.0, 1.0))
            fleet_features[i, 7] = math.hypot(x - 50.0, y - 50.0) / 50.0
            fleet_mask[i] = True

        # Compute global features
        global_features = np.zeros(4, dtype=np.float32)
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

        context = self._build_context(planets, player)

        return StructuredStateV2(
            planet_features=planet_features,
            fleet_features=fleet_features,
            fleet_mask=fleet_mask,
            planet_mask=planet_mask,
            global_features=global_features,
            context=context,
        )
