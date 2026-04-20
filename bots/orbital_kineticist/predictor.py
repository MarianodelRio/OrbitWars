import math
from math import log, hypot, atan2, cos, sin, pi


def fleet_speed(ships):
    return 1.0 + 5.0 * (log(max(ships, 1)) / log(1000)) ** 1.5


def _seg_point_dist(ax, ay, bx, by, px, py):
    """Minimum distance from point (px, py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return hypot(px - proj_x, py - proj_y)


def blocks_sun(origin, intercept_point):
    return _seg_point_dist(
        origin[0], origin[1],
        intercept_point[0], intercept_point[1],
        50.0, 50.0
    ) < 10.5


def _planet_is_static(planet):
    return hypot(planet[2] - 50, planet[3] - 50) + planet[4] >= 50


def get_intercept(origin_pos, target_planet, num_ships, angular_velocity):
    speed = fleet_speed(num_ships)
    ox, oy = origin_pos

    if _planet_is_static(target_planet):
        tx, ty = target_planet[2], target_planet[3]
        dist = hypot(tx - ox, ty - oy)
        eta = dist / speed if speed > 0 else 999
        return (tx, ty), eta

    current_angle = math.atan2(target_planet[3] - 50, target_planet[2] - 50)
    orbit_radius = math.hypot(target_planet[2] - 50, target_planet[3] - 50)
    planet_radius = target_planet[4]

    for t in range(1, 151):
        angle_t = current_angle + angular_velocity * t
        tx = 50 + orbit_radius * cos(angle_t)
        ty = 50 + orbit_radius * sin(angle_t)
        dist = hypot(tx - ox, ty - oy)
        eta = dist / speed if speed > 0 else 999
        if abs(eta - t) < 0.5:
            return (tx, ty), float(t)

    # Fallback to current position
    tx, ty = target_planet[2], target_planet[3]
    dist = hypot(tx - ox, ty - oy)
    eta = dist / speed if speed > 0 else 999
    return (tx, ty), eta
