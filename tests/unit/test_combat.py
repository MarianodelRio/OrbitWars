import pytest
from game.logic.combat import simulate_combat

# --- 2-player cases ---

def test_defender_wins_equal():
    # Attacker ties garrison: defender holds, 0 ships remain
    ships, owner = simulate_combat(10, 0, {1: 10})
    assert owner == 0
    assert ships == 0

def test_attacker_wins():
    ships, owner = simulate_combat(5, 0, {1: 10})
    assert owner == 1
    assert ships == 5

def test_defender_holds():
    ships, owner = simulate_combat(10, 0, {1: 6})
    assert owner == 0
    assert ships == 4

def test_allied_reinforcement():
    ships, owner = simulate_combat(5, 0, {0: 5})
    assert owner == 0
    assert ships == 10

def test_no_attackers():
    ships, owner = simulate_combat(7, 0, {})
    assert owner == 0
    assert ships == 7

# --- Multi-attacker cases ---

def test_two_attackers_tie_all_destroyed():
    # Two enemy attackers tie each other: both destroyed, garrison unchanged
    ships, owner = simulate_combat(5, 0, {1: 8, 2: 8})
    assert owner == 0
    assert ships == 5

def test_two_attackers_largest_captures():
    # Attacker 1 (10) beats attacker 2 (5), survivor (5) beats garrison (3)
    ships, owner = simulate_combat(3, 0, {1: 10, 2: 5})
    assert owner == 1
    assert ships == 2  # 10-5=5, then 5-3=2

def test_two_attackers_largest_loses_to_garrison():
    # Attacker 1 (8) beats attacker 2 (5), survivor (3) loses to garrison (10)
    ships, owner = simulate_combat(10, 0, {1: 8, 2: 5})
    assert owner == 0
    assert ships == 7  # 8-5=3, then 10-3=7

def test_three_attackers_iterative():
    # 15 vs 10 → survivor 5; 5 vs 3 → survivor 2; 2 vs garrison 1 → capture
    ships, owner = simulate_combat(1, 0, {1: 15, 2: 10, 3: 3})
    assert owner == 1
    assert ships == 1  # (15-10)=5, (5-3)=2, (2-1)=1
