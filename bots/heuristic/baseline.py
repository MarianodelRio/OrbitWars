import math
from bots.interface import Bot


class BaselineBot(Bot):
    """Simple expand bot: sends half its ships to the nearest non-owned planet."""

    def act(self, obs, config=None):
        moves = []
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        my_planets = [p for p in raw_planets if p[1] == player]
        targets = [p for p in raw_planets if p[1] != player]
        if not my_planets or not targets:
            return moves
        for mine in my_planets:
            if mine[5] <= 1:
                continue
            nearest = min(targets, key=lambda t: math.hypot(mine[2] - t[2], mine[3] - t[3]))
            ships = mine[5] // 2
            angle = math.atan2(nearest[3] - mine[3], nearest[2] - mine[2])
            moves.append([mine[0], angle, ships])
        return moves


agent_fn = BaselineBot()
