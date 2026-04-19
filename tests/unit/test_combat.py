import pytest
from game.logic.combat import simulate_combat

def test_defender_wins_equal():
    ships, owner = simulate_combat(10, 0, {1: 10})
    assert owner == 0
    assert ships == 0

def test_attacker_wins():
    ships, owner = simulate_combat(5, 0, {1: 10})
    assert owner == 1
    assert ships == 5

def test_allied_reinforcement():
    ships, owner = simulate_combat(5, 0, {0: 5})
    assert owner == 0
    assert ships == 10

def test_no_attackers():
    ships, owner = simulate_combat(7, 0, {})
    assert owner == 0
    assert ships == 7
