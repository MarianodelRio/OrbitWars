from .predictor import get_intercept, blocks_sun


def select_target(source, candidates, player, angular_velocity):
    origin_pos = (source[2], source[3])
    source_ships = source[5]
    source_production = source[6]

    best_score = -1.0
    best_target = None
    best_intercept = None
    best_ships_to_send = 0

    for target in candidates:
        if target[0] == source[0]:
            continue

        intercept_point, eta = get_intercept(
            origin_pos, target, source_ships, angular_velocity
        )

        owner = target[1]
        ships = target[5]
        production = target[6]

        if owner == -1:
            future_ships = ships + 1
        else:
            future_ships = ships + production * eta

        ships_to_send = int(future_ships * 1.2) + 1

        if ships_to_send < 10:
            continue
        if source_ships <= ships_to_send:
            continue

        score = (production ** 2) / (max(eta, 0.1) * (future_ships + 1))

        dynamic_threshold = source_production / 20.0
        if score < dynamic_threshold:
            continue
        if score < 0.01:
            continue

        if blocks_sun(origin_pos, intercept_point):
            continue

        if score > best_score:
            best_score = score
            best_target = target
            best_intercept = intercept_point
            best_ships_to_send = ships_to_send

    if best_target is None:
        return None

    return {
        "target": best_target,
        "intercept": best_intercept,
        "ships_to_send": best_ships_to_send,
    }
