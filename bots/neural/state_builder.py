"""StateBuilder: converts a StepRecord or live obs dict into a padded, normalized ModelInput."""

import math

import numpy as np

from dataset.episode import StepRecord
from .types import ActionContext, ModelInput


class StateBuilder:
    def __init__(self, max_planets: int = 50, max_fleets: int = 100) -> None:
        self.max_planets = max_planets
        self.max_fleets = max_fleets

    @property
    def input_dim(self) -> int:
        return self.max_planets * 7 + self.max_fleets * 7

    def from_obs(self, obs: dict, player: int) -> ModelInput:
        raw_planets = obs["planets"]
        raw_fleets = obs["fleets"]

        if len(raw_planets) == 0:
            planets = np.empty((0, 7), dtype=np.float32)
        else:
            planets = np.array(raw_planets, dtype=np.float32)

        if len(raw_fleets) == 0:
            fleets = np.empty((0, 7), dtype=np.float32)
        else:
            fleets = np.array(raw_fleets, dtype=np.float32)

        return self._build(planets, fleets, player)

    def from_step(self, step: StepRecord, player: int) -> ModelInput:
        return self._build(step.planets, step.fleets, player)

    def __call__(self, step: StepRecord, player: int) -> ModelInput:
        return self.from_step(step, player)

    def _build(self, planets: np.ndarray, fleets: np.ndarray, player: int) -> ModelInput:
        context = self._build_context(planets, player)
        array = self._build_array(planets, fleets, player)
        return ModelInput(array=array, context=context)

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

    def _build_array(self, planets: np.ndarray, fleets: np.ndarray, player: int) -> np.ndarray:
        out = np.zeros(self.input_dim, dtype=np.float32)

        n_planets = min(planets.shape[0], self.max_planets)
        for i in range(n_planets):
            row = planets[i]
            owner = row[1]
            x = row[2]
            y = row[3]
            ships = row[5]
            production = row[6]
            offset = i * 7
            out[offset + 0] = float(owner == player)
            out[offset + 1] = float(owner not in (-1, player))
            out[offset + 2] = float(owner == -1)
            out[offset + 3] = x / 100.0
            out[offset + 4] = y / 100.0
            out[offset + 5] = float(np.clip(ships / 200.0, 0.0, 1.0))
            out[offset + 6] = production / 5.0

        n_fleets = min(fleets.shape[0], self.max_fleets)
        fleet_base = self.max_planets * 7
        for i in range(n_fleets):
            row = fleets[i]
            owner = row[1]
            x = row[2]
            y = row[3]
            angle = row[4]
            ships = row[6]
            offset = fleet_base + i * 7
            out[offset + 0] = float(owner == player)
            out[offset + 1] = float(owner != player)
            out[offset + 2] = x / 100.0
            out[offset + 3] = y / 100.0
            out[offset + 4] = math.sin(angle)
            out[offset + 5] = math.cos(angle)
            out[offset + 6] = float(np.clip(ships / 200.0, 0.0, 1.0))

        return out
