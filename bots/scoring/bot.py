import math
from bots.interface import Bot
from .scoring import score_target, compute_required_ships


class ScoringBot(Bot):
    def act(self, obs, config=None):
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets

        my_planets = [p for p in planets if p[1] == player]
        targets = [p for p in planets if p[1] != player]

        if not my_planets or not targets:
            return []

        actions = []
        for source in my_planets:
            if source[5] <= 10:
                continue

            viable = []
            for target in targets:
                required = compute_required_ships(source, target)
                if required < source[5]:
                    viable.append((score_target(source, target), required, target))

            if not viable:
                continue

            best_score, best_required, best_target = max(viable, key=lambda x: x[0])

            if best_score <= 0:
                continue

            ships_to_send = best_required

            if ships_to_send < 5 or ships_to_send >= source[5] * 0.8:
                continue

            angle = math.atan2(best_target[3] - source[3], best_target[2] - source[2])
            actions.append([source[0], angle, ships_to_send])

        return actions


agent_fn = ScoringBot()
