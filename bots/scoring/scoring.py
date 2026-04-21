import math


def distance(a, b) -> float:
    return math.hypot(b[2] - a[2], b[3] - a[3])


def fleet_speed(ships) -> float:
    ships = max(1, ships)
    return 1.0 + 5.0 * (math.log(ships) / math.log(1000)) ** 1.5


def compute_required_ships(source, target) -> int:
    travel_time = distance(source, target) / fleet_speed(source[5])
    if target[1] == -1:
        required = target[5]
    else:
        required = target[5] + target[6] * travel_time
    required *= 1.2
    return int(required)


def score_target(source, target) -> float:
    dist = distance(source, target)
    score = (target[6] * 5) - (target[5] * 1) - (dist * 0.2)
    if target[1] == -1:
        score += 2
    return score
