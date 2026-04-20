import math
import random
from bots.interface import Bot

class BaselineBot(Bot):
    def act(self, obs, config=None):
        planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        my_planets = [p for p in planets if p[1] == player]
        targets = [p for p in planets if p[1] != player]
        if not my_planets or not targets:
            return []
        source = random.choice(my_planets)
        target = random.choice(targets)
        num_ships = max(1, int(source[5] // 2))
        angle = math.atan2(target[3] - source[3], target[2] - source[2])
        return [[source[0], angle, num_ships]]

agent_fn = BaselineBot()
