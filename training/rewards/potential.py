"""PotentialReward: potential-based reward shaping for Orbit Wars RL training."""

from __future__ import annotations


class PotentialReward:
    def __init__(
        self,
        w_planets: float = 1.0,
        w_production: float = 0.5,
        w_ships: float = 0.1,
        gamma: float = 0.99,
        lam: float = 0.05,
        clip_abs: float = 0.2,
    ) -> None:
        self.w_planets = w_planets
        self.w_production = w_production
        self.w_ships = w_ships
        self.gamma = gamma
        self.lam = lam
        self.clip_abs = clip_abs

    def _potential(self, obs: dict, player: int) -> float:
        planets = obs.get("planets", [])
        if not planets:
            return 0.0

        my_planets = sum(1 for p in planets if p[1] == player)
        total_planets = max(len(planets), 1)

        my_production = sum(p[6] for p in planets if p[1] == player)
        total_production = max(sum(p[6] for p in planets), 1)

        my_ships = sum(p[5] for p in planets if p[1] == player)
        total_ships = max(sum(p[5] for p in planets), 1)

        return (
            self.w_planets * (my_planets / total_planets)
            + self.w_production * (my_production / total_production)
            + self.w_ships * (my_ships / total_ships)
        )

    def compute(self, prev_obs: dict, curr_obs: dict, player: int) -> float:
        phi_prev = self._potential(prev_obs, player)
        phi_curr = self._potential(curr_obs, player)
        shaped = self.gamma * phi_curr - phi_prev
        clipped = max(-self.clip_abs, min(self.clip_abs, shaped))
        return self.lam * clipped
