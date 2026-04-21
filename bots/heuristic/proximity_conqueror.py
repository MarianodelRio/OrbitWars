import math
from bots.interface import Bot


class ProximityConquerorBot(Bot):
    @property
    def name(self) -> str:
        return "heuristic.proximity_conqueror"

    def act(self, obs, config=None):
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets

        my_planets = [p for p in planets if p[1] == player]
        targets = [p for p in planets if p[1] != player]

        if not my_planets or not targets:
            return []

        def is_static(p):
            return math.hypot(p[2] - 50, p[3] - 50) + p[4] >= 50

        static_targets = [p for p in targets if is_static(p)]
        other_targets = [p for p in targets if not is_static(p)]

        def dist(a, b):
            return math.hypot(a[2] - b[2], a[3] - b[3])

        # Pass 1: each source independently picks its best conquerable target
        # ranked list: static by distance first, then others by distance
        proposals = {}  # source_id -> (target_tuple, distance)
        source_by_id = {p[0]: p for p in my_planets}

        for mine in my_planets:
            if mine[5] <= 1:
                continue
            ranked = (
                sorted(static_targets, key=lambda t: dist(mine, t))
                + sorted(other_targets, key=lambda t: dist(mine, t))
            )
            for candidate in ranked:
                if mine[5] > candidate[5] + 1:
                    proposals[mine[0]] = (candidate, dist(mine, candidate))
                    break

        # Pass 2: resolve conflicts — per target, keep only the closest claimant
        # claimed[target_id] = (source_id, distance)
        claimed = {}
        displaced = []

        for src_id, (tgt, d) in proposals.items():
            tgt_id = tgt[0]
            if tgt_id not in claimed:
                claimed[tgt_id] = (src_id, d)
            else:
                existing_src_id, existing_d = claimed[tgt_id]
                if d < existing_d:
                    displaced.append(existing_src_id)
                    claimed[tgt_id] = (src_id, d)
                else:
                    displaced.append(src_id)

        # Pass 3: displaced sources pick next available target excluding claimed ones
        taken_targets = set(claimed.keys())

        for src_id in displaced:
            mine = source_by_id[src_id]
            if mine[5] <= 1:
                continue
            ranked = (
                sorted(static_targets, key=lambda t: dist(mine, t))
                + sorted(other_targets, key=lambda t: dist(mine, t))
            )
            for candidate in ranked:
                if candidate[0] in taken_targets:
                    continue
                if mine[5] > candidate[5] + 1:
                    claimed[candidate[0]] = (src_id, dist(mine, candidate))
                    taken_targets.add(candidate[0])
                    break

        # Build target lookup
        target_by_id = {p[0]: p for p in targets}

        # Build moves
        moves = []
        # Invert claimed: source_id -> target_id
        source_to_target = {src_id: tgt_id for tgt_id, (src_id, _) in claimed.items()}

        for src_id, tgt_id in source_to_target.items():
            source = source_by_id[src_id]
            target = target_by_id[tgt_id]
            angle = math.atan2(target[3] - source[3], target[2] - source[2])
            ships_to_send = target[5] + 1
            moves.append([source[0], angle, ships_to_send])

        return moves


agent_fn = ProximityConquerorBot()
