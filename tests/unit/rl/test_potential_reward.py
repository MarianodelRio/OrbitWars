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


def test_clipping():
    # Use large weights so shaped reward would exceed clip_abs without clipping
    pr = PotentialReward(w_planets=10.0, w_production=10.0, w_ships=10.0, gamma=1.0, lam=1.0, clip_abs=0.2)
    prev = make_obs([[0, 1, 10, 10, 1, 10, 2]])
    curr = make_obs([[0, 0, 10, 10, 1, 10, 2]])
    reward = pr.compute(prev, curr, player=0)
    assert abs(reward) <= 0.2 + 1e-9


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
