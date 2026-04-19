import math
from game.logic.geometry import angle_to, fleet_speed

def incoming_fleets(planet, fleets):
    result = []
    for f in fleets:
        dx, dy = planet.x - f.x, planet.y - f.y
        angle_diff = abs(math.atan2(dy, dx) - f.angle)
        while angle_diff > math.pi:
            angle_diff = abs(angle_diff - 2 * math.pi)
        if angle_diff < 0.3:
            result.append(f)
    return result

def enemy_fleets_arriving(planet, fleets, player):
    return [f for f in incoming_fleets(planet, fleets) if f.owner != player]

def is_under_attack(planet, fleets, player):
    return len(enemy_fleets_arriving(planet, fleets, player)) > 0
