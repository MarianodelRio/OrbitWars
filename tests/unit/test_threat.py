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
