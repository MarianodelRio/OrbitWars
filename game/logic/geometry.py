import math


def dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def nearest_planet(source, candidates):
    if not candidates:
        return None
    return min(candidates, key=lambda p: dist(source, p))


def fleet_speed(ships, max_speed=6.0):
    if ships <= 1:
        return 1.0
    return 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def angle_to(source, target):
    return math.atan2(target.y - source.y, target.x - source.x)


def eta(source, target, ships):
    d = dist(source, target)
    speed = fleet_speed(ships)
    return math.ceil(d / speed)


def orbit_predict(planet, angular_velocity, steps):
    cx, cy = 50.0, 50.0
    dx, dy = planet.x - cx, planet.y - cy
    r = math.hypot(dx, dy)
    current_angle = math.atan2(dy, dx)
    new_angle = current_angle + angular_velocity * steps
    return cx + r * math.cos(new_angle), cy + r * math.sin(new_angle)


def path_crosses_sun(x1, y1, x2, y2, sun_x=50.0, sun_y=50.0, sun_r=10.0):
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - sun_x, y1 - sun_y
    a = dx*dx + dy*dy
    if a == 0:
        return math.hypot(fx, fy) < sun_r
    b = 2 * (fx*dx + fy*dy)
    c = fx*fx + fy*fy - sun_r*sun_r
    discriminant = b*b - 4*a*c
    if discriminant <= 0:
        return False
    sq = math.sqrt(discriminant)
    t1 = (-b - sq) / (2*a)
    t2 = (-b + sq) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)
