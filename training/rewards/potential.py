"""PotentialReward: potential-based reward shaping for Orbit Wars RL training."""

from __future__ import annotations

import math


class PotentialReward:
    def __init__(
        self,
        w_production: float = 1.0,
        w_planets: float = 0.5,
        w_ships: float = 0.1,
        gamma: float = 0.99,
        lam: float = 0.1,
        clip_abs: float = 0.2,  # deprecated, ignored
        r_terminal_win: float = 10.0,
        r_terminal_loss: float = -10.0,
        r_terminal_margin_coef: float = 5.0,
        r_event_capture_enemy: float = 0.5,
        r_event_capture_comet: float = 0.2,
        r_event_eliminate_opponent: float = 1.0,
        r_event_lose_planet: float = -0.3,
        r_event_ships_wasted_coef: float = 0.0,
        r_explore: float = 0.01,
        explore_iterations: int = 200,
    ) -> None:
        self.w_production = w_production
        self.w_planets = w_planets
        self.w_ships = w_ships
        self.gamma = gamma
        self.lam = lam
        self.r_terminal_win = r_terminal_win
        self.r_terminal_loss = r_terminal_loss
        self.r_terminal_margin_coef = r_terminal_margin_coef
        self.r_event_capture_enemy = r_event_capture_enemy
        self.r_event_capture_comet = r_event_capture_comet
        self.r_event_eliminate_opponent = r_event_eliminate_opponent
        self.r_event_lose_planet = r_event_lose_planet
        self.r_event_ships_wasted_coef = r_event_ships_wasted_coef
        self.r_explore = r_explore
        self.explore_iterations = explore_iterations

        self._combat_flags: dict = {}
        self._explore_active: bool = True
        self._call_count: int = 0

    def notify_iteration(self, n: int) -> None:
        self._explore_active = (n <= self.explore_iterations)

    def reset_episode(self) -> None:
        self._combat_flags.clear()

    def _potential(self, obs: dict, player: int) -> float:
        planets = obs.get("planets", [])
        if not planets:
            return 0.0

        my_planets = sum(1 for p in planets if p[1] == player)
        total_planets = max(len(planets), 1)

        my_production = sum(p[6] for p in planets if p[1] == player)
        total_production = max(sum(p[6] for p in planets), 1)

        my_ships = sum(p[5] for p in planets if p[1] == player)
        log_ships_share = math.log(1 + my_ships) / math.log(1001)  # fixed max 1000 ships

        return (
            self.w_production * (my_production / total_production)
            + self.w_planets * (my_planets / total_planets)
            + self.w_ships * log_ships_share
        )

    def _compute_shaping(self, prev_obs: dict, curr_obs: dict, player: int) -> float:
        return self.lam * (self.gamma * self._potential(curr_obs, player) - self._potential(prev_obs, player))

    def _compute_terminal(self, prev_obs: dict, curr_obs: dict, player: int) -> float:
        curr_planets = curr_obs.get("planets", [])
        prev_planets = prev_obs.get("planets", [])

        my_curr = sum(1 for p in curr_planets if p[1] == player)
        my_prev = sum(1 for p in prev_planets if p[1] == player)

        opp_curr = [p for p in curr_planets if p[1] >= 0 and p[1] != player]
        opp_prev = [p for p in prev_planets if p[1] >= 0 and p[1] != player]

        my_score = sum(p[5] for p in curr_planets if p[1] == player)
        opp_ids = set(p[1] for p in opp_curr)
        max_opp_score = max(
            (sum(p[5] for p in curr_planets if p[1] == opp_id) for opp_id in opp_ids),
            default=0,
        )
        margin = (my_score - max_opp_score) / max(my_score + max_opp_score, 1)

        reward = 0.0

        # Loss: player loses all planets
        if my_curr == 0 and my_prev > 0:
            self._combat_flags.clear()
            reward += self.r_terminal_loss + self.r_terminal_margin_coef * margin
            return reward  # early return — no other rewards apply on loss

        # Eliminate-opponent event (additive, does NOT return early)
        if len(opp_curr) < len(opp_prev) and my_curr > 0:
            reward += self.r_event_eliminate_opponent

        # Win: all opponents gone
        if my_curr > 0 and len(opp_curr) == 0 and len(opp_prev) > 0:
            self._combat_flags.clear()
            reward += self.r_terminal_win + self.r_terminal_margin_coef * margin

        return reward

    def _compute_events(self, prev_obs: dict, curr_obs: dict, player: int) -> float:
        curr_planets = curr_obs.get("planets", [])
        prev_planets = prev_obs.get("planets", [])

        prev_owner_map = {p[0]: p[1] for p in prev_planets}
        total = 0.0

        for p in curr_planets:
            prev_owner = prev_owner_map.get(p[0], -1)
            curr_owner = p[1]
            if prev_owner >= 0 and prev_owner != player and curr_owner == player:
                total += self.r_event_capture_enemy
            if prev_owner == player and curr_owner >= 0 and curr_owner != player:
                total += self.r_event_lose_planet

        return total

    def _compute_explore(self, prev_obs: dict, curr_obs: dict, player: int) -> float:
        if not self._explore_active:
            return 0.0

        curr_planets = curr_obs.get("planets", [])
        prev_planets = prev_obs.get("planets", [])

        prev_owner_map = {p[0]: p[1] for p in prev_planets}
        total = 0.0

        for p in curr_planets:
            if p[1] != prev_owner_map.get(p[0], p[1]) and p[0] not in self._combat_flags:
                self._combat_flags[p[0]] = True
                total += self.r_explore

        return total

    def compute(self, prev_obs: dict, curr_obs: dict, player: int) -> float:
        self._call_count += 1
        return (
            self._compute_shaping(prev_obs, curr_obs, player)
            + self._compute_terminal(prev_obs, curr_obs, player)
            + self._compute_events(prev_obs, curr_obs, player)
            + self._compute_explore(prev_obs, curr_obs, player)
        )
