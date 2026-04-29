"""Unit tests for game/logic/threat.py."""

import math
import types

from game.logic.threat import incoming_fleets, enemy_fleets_arriving, is_under_attack


def test_threat_no_fleets_returns_zeros():
    planet = types.SimpleNamespace(x=50, y=50)
    assert incoming_fleets(planet, []) == []
    assert enemy_fleets_arriving(planet, [], player=0) == []
    assert is_under_attack(planet, [], player=0) is False


def test_threat_enemy_fleet_increases_threat():
    planet = types.SimpleNamespace(x=50, y=50)
    # Fleet at (40, 50) pointing directly at planet at (50, 50):
    # dx = 50-40 = 10, dy = 50-50 = 0 => angle = atan2(0, 10) = 0.0
    fleet = types.SimpleNamespace(x=40, y=50, angle=0.0, owner=1)

    # Enemy fleet (owner=1) is heading toward planet; player=0 is under attack
    assert is_under_attack(planet, [fleet], player=0) is True

    # From owner=1's perspective, own fleet does not constitute an attack on player 1
    assert is_under_attack(planet, [fleet], player=1) is False


def test_incoming_fleet_at_boundary_not_included():
    # angle_diff == 0.3 exactly → NOT incoming (strict < 0.3)
    planet = types.SimpleNamespace(x=50, y=50)
    # fleet at (40, 50), angle to planet = atan2(0, 10) = 0.0
    # set fleet.angle = 0.3 so angle_diff = abs(0.0 - 0.3) = 0.3 → not < 0.3
    fleet = types.SimpleNamespace(x=40, y=50, angle=0.3, owner=1)
    assert incoming_fleets(planet, [fleet]) == []


def test_fleet_pointing_away_not_incoming():
    planet = types.SimpleNamespace(x=50, y=50)
    fleet = types.SimpleNamespace(x=40, y=50, angle=math.pi, owner=1)
    assert incoming_fleets(planet, [fleet]) == []


def test_allied_fleet_not_counted_as_under_attack():
    planet = types.SimpleNamespace(x=50, y=50)
    # fleet pointing directly at planet (angle=0, from left)
    fleet = types.SimpleNamespace(x=40, y=50, angle=0.0, owner=0)
    assert not is_under_attack(planet, [fleet], player=0)


def test_multiple_enemy_fleets_attack():
    planet = types.SimpleNamespace(x=50, y=50)
    fleet1 = types.SimpleNamespace(x=40, y=50, angle=0.0, owner=1)
    fleet2 = types.SimpleNamespace(x=60, y=50, angle=math.pi, owner=1)
    assert is_under_attack(planet, [fleet1, fleet2], player=0)


def test_fleet_pointing_opposite_direction():
    planet = types.SimpleNamespace(x=50, y=50)
    # fleet at (40,50): angle to planet = 0.0; fleet.angle = pi (opposite)
    fleet = types.SimpleNamespace(x=40, y=50, angle=math.pi, owner=1)
    assert incoming_fleets(planet, [fleet]) == []
