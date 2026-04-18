import math


def dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def nearest_planet(source, candidates):
    if not candidates:
        return None
    return min(candidates, key=lambda p: dist(source, p))
