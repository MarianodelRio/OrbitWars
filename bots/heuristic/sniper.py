import math
from bots.interface import Bot

class SniperBot(Bot):
    def act(self, obs, config=None):
        moves = []
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        my_planets = [p for p in raw_planets if p[1] == player]
        targets = [p for p in raw_planets if p[1] != player]
        if not targets:
            return moves
        for mine in my_planets:
            nearest = min(targets, key=lambda t: math.hypot(mine[2] - t[2], mine[3] - t[3]))
            ships_needed = nearest[5] + 1
            if mine[5] >= ships_needed:
                angle = math.atan2(nearest[3] - mine[3], nearest[2] - mine[2])
                moves.append([mine[0], angle, ships_needed])
        return moves

agent_fn = SniperBot()
