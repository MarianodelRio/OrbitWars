"""Unit tests for PotentialReward."""

import pytest
from training.rewards.potential import PotentialReward


def make_obs(planets):
    """planets: list of (id, owner, x, y, radius, ships, production)"""
    return {"planets": planets}


def test_empty_obs_returns_zero():
    pr = PotentialReward()
    result = pr.compute({}, {}, player=0)
    assert result == 0.0


def test_empty_planets_list_returns_zero():
    pr = PotentialReward()
    result = pr.compute({"planets": []}, {"planets": []}, player=0)
    assert result == 0.0


def test_planet_gain_positive_reward():
    pr = PotentialReward(gamma=1.0, lam=1.0, clip_abs=1.0)
    # Before: player 0 owns 1/3 planets
    prev = make_obs([
        [0, 0, 10, 10, 1, 10, 2],
        [1, 1, 20, 20, 1, 10, 2],
        [2, 1, 30, 30, 1, 10, 2],
    ])
    # After: player 0 owns 2/3 planets
    curr = make_obs([
        [0, 0, 10, 10, 1, 10, 2],
        [1, 0, 20, 20, 1, 10, 2],
        [2, 1, 30, 30, 1, 10, 2],
    ])
    reward = pr.compute(prev, curr, player=0)
    assert reward > 0.0


def test_planet_loss_negative_reward():
    pr = PotentialReward(gamma=1.0, lam=1.0, clip_abs=1.0)
    # Before: player 0 owns 2/3 planets
    prev = make_obs([
        [0, 0, 10, 10, 1, 10, 2],
        [1, 0, 20, 20, 1, 10, 2],
        [2, 1, 30, 30, 1, 10, 2],
    ])
    # After: player 0 owns 1/3 planets
    curr = make_obs([
        [0, 0, 10, 10, 1, 10, 2],
        [1, 1, 20, 20, 1, 10, 2],
        [2, 1, 30, 30, 1, 10, 2],
    ])
    reward = pr.compute(prev, curr, player=0)
    assert reward < 0.0


def test_large_weights_produce_large_reward():
    # Large weights on a planet-capture event should produce a reward substantially greater than 1
    pr = PotentialReward(w_planets=10.0, w_production=10.0, w_ships=10.0, gamma=1.0, lam=1.0)
    prev = make_obs([[0, 1, 10, 10, 1, 10, 2]])
    curr = make_obs([[0, 0, 10, 10, 1, 10, 2]])
    reward = pr.compute(prev, curr, player=0)
    assert reward > 1.0


def test_no_change_near_zero():
    pr = PotentialReward(gamma=1.0, lam=0.05, clip_abs=0.2)
    planets = [
        [0, 0, 10, 10, 1, 10, 2],
        [1, 1, 20, 20, 1, 10, 2],
    ]
    obs = make_obs(planets)
    reward = pr.compute(obs, obs, player=0)
    # With gamma=1.0 and same state, shaped = 1*phi - phi = 0
    assert reward == 0.0


def test_terminal_win_fires_with_eliminate_opponent():
    pr = PotentialReward()
    # prev: player 0 owns planet 0, opponent player 1 owns planets 1 and 2
    prev_obs = make_obs([
        [0, 0, 10, 10, 3, 100, 2],
        [1, 1, 50, 50, 3, 50, 1],
        [2, 1, 80, 80, 3, 50, 1],
    ])
    # curr: player 0 owns all three planets, opponent has none
    curr_obs = make_obs([
        [0, 0, 10, 10, 3, 100, 2],
        [1, 0, 50, 50, 3, 50, 1],
        [2, 0, 80, 80, 3, 50, 1],
    ])
    result = pr._compute_terminal(prev_obs, curr_obs, 0)
    assert result >= pr.r_terminal_win + pr.r_event_eliminate_opponent


def test_log_ships_share_bounded():
    # Use w_production=0.0, w_planets=0.0, w_ships=1.0 so _potential returns only ships component
    pr = PotentialReward(w_production=0.0, w_planets=0.0, w_ships=1.0)
    # Player 0 owns one planet with 1000 ships (the max)
    obs = make_obs([
        [0, 0, 10, 10, 3, 1000, 2],
    ])
    result = pr._potential(obs, 0)
    # log(1+1000)/log(1001) == 1.0, so result should be w_ships * 1.0 = 1.0
    assert result <= 1.0 + 1e-9


def test_explore_disabled_after_notify_iteration():
    pr = PotentialReward(r_explore=0.01, explore_iterations=5)
    pr.notify_iteration(6)
    # Build obs where a planet changes owner
    prev_obs = make_obs([
        [0, 1, 10, 10, 3, 50, 1],
    ])
    curr_obs = make_obs([
        [0, 0, 10, 10, 3, 50, 1],
    ])
    result = pr._compute_explore(prev_obs, curr_obs, 0)
    assert result == 0.0


def test_reset_episode_clears_combat_flags():
    pr = PotentialReward()
    pr._combat_flags = {0: True, 1: True}
    pr.reset_episode()
    assert len(pr._combat_flags) == 0


# ---------------------------------------------------------------------------
# Fleet ships in potential (Bug 2 regression)
# ---------------------------------------------------------------------------

def test_fleet_ships_added_to_planet_ships():
    pr = PotentialReward(w_production=0.0, w_planets=0.0, w_ships=1.0)
    # fleet: [fleet_id, player, x, y, angle, eta, ships]
    obs_without_fleet = {"planets": [[0, 0, 10, 10, 1, 100, 2]], "fleets": []}
    obs_with_fleet = {"planets": [[0, 0, 10, 10, 1, 100, 2]], "fleets": [[1, 0, 20, 20, 0.0, 3, 50]]}
    assert pr._potential(obs_with_fleet, 0) > pr._potential(obs_without_fleet, 0)


def test_opponent_fleet_ships_not_counted():
    pr = PotentialReward(w_production=0.0, w_planets=0.0, w_ships=1.0)
    obs_without_fleet = {"planets": [[0, 0, 10, 10, 1, 100, 2]], "fleets": []}
    obs_with_enemy_fleet = {"planets": [[0, 0, 10, 10, 1, 100, 2]], "fleets": [[1, 1, 20, 20, 0.0, 3, 50]]}
    assert pr._potential(obs_with_enemy_fleet, 0) == pytest.approx(pr._potential(obs_without_fleet, 0))


def test_fleet_ships_zero_for_empty_fleet_list():
    pr = PotentialReward(w_ships=1.0, w_planets=0.0, w_production=0.0)
    obs = {"planets": [[0, 0, 10, 10, 1, 100, 2]], "fleets": []}
    import math as _math
    expected = _math.log(1 + 100) / _math.log(1001)
    assert pr._potential(obs, 0) == pytest.approx(expected)


def test_compute_events_captures_enemy_and_lose_planet():
    pr = PotentialReward()
    # prev: planet 0 owned by player 1, planet 1 owned by player 0
    prev = make_obs([[0, 1, 10, 10, 1, 10, 2], [1, 0, 20, 20, 1, 10, 2]])
    # curr: planet 0 owned by player 0, planet 1 owned by player 1
    curr = make_obs([[0, 0, 10, 10, 1, 10, 2], [1, 1, 20, 20, 1, 10, 2]])
    result = pr._compute_events(prev, curr, player=0)
    assert result == pytest.approx(pr.r_event_capture_enemy + pr.r_event_lose_planet)
