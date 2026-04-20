import sys
import os
from math import atan2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from bots.interface import Bot
from .scoring import select_target


class OrbitalKineticistBot(Bot):
    def act(self, obs, config=None):
        if isinstance(obs, dict):
            obs_dict = obs
        else:
            obs_dict = obs.__dict__ if hasattr(obs, '__dict__') else dict(obs)

        planets = obs_dict.get("planets", [])
        player = obs_dict.get("player", 1)
        angular_velocity = obs_dict.get("angular_velocity", 0.035)

        my_planets = [p for p in planets if p[1] == player]
        candidate_targets = [p for p in planets if p[1] != player]

        if not my_planets or not candidate_targets:
            return []

        actions = []
        for source in my_planets:
            if source[5] <= 10:
                continue

            result = select_target(
                source, candidate_targets, player, angular_velocity
            )
            if result is None:
                continue

            ix, iy = result["intercept"]
            angle = atan2(iy - source[3], ix - source[2])
            actions.append([source[0], angle, result["ships_to_send"]])

        return actions


agent_fn = OrbitalKineticistBot()
