"""Unit tests for StateBuilder output shapes, dtypes, masks, and feature values."""

import math

import numpy as np
import pytest

from bots.neural.state_builder import StateBuilder


MAX_PLANETS = 10
MAX_FLEETS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _builder():
    return StateBuilder(max_planets=MAX_PLANETS, max_fleets=MAX_FLEETS)


def _obs(n_planets=5, n_fleets=8, player=0, step=100, comet_ids=None):
    """Minimal obs dict with planets on a line and fleets at center."""
    planets = [
        [float(i), float(0 if i < n_planets // 2 + 1 else 1), float(i * 10), 50.0, 1.5, 50.0 + i * 5, 2.0]
        for i in range(n_planets)
    ]
    fleets = [
        [float(i), float(i % 2), 50.0, 50.0, 0.5, 0.0, 10.0]
        for i in range(n_fleets)
    ]
    return {
        "planets": planets,
        "fleets": fleets,
        "comet_planet_ids": comet_ids or [],
        "step": step,
        "player": player,
    }


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_planet_features_shape():
    state = _builder().from_obs(_obs(), player=0)
    assert state["planet_features"].shape == (MAX_PLANETS, 10)


def test_fleet_features_shape():
    state = _builder().from_obs(_obs(), player=0)
    assert state["fleet_features"].shape == (MAX_FLEETS, 8)


def test_global_features_shape():
    state = _builder().from_obs(_obs(), player=0)
    assert state["global_features"].shape == (4,)


def test_planet_mask_shape():
    state = _builder().from_obs(_obs(n_planets=5), player=0)
    assert state["planet_mask"].shape == (MAX_PLANETS,)


def test_fleet_mask_shape():
    state = _builder().from_obs(_obs(n_fleets=8), player=0)
    assert state["fleet_mask"].shape == (MAX_FLEETS,)


# ---------------------------------------------------------------------------
# Dtype tests
# ---------------------------------------------------------------------------

def test_planet_features_dtype():
    state = _builder().from_obs(_obs(), player=0)
    assert state["planet_features"].dtype == np.float32


def test_fleet_features_dtype():
    state = _builder().from_obs(_obs(), player=0)
    assert state["fleet_features"].dtype == np.float32


def test_global_features_dtype():
    state = _builder().from_obs(_obs(), player=0)
    assert state["global_features"].dtype == np.float32


def test_planet_mask_dtype():
    state = _builder().from_obs(_obs(), player=0)
    assert state["planet_mask"].dtype == bool


def test_fleet_mask_dtype():
    state = _builder().from_obs(_obs(), player=0)
    assert state["fleet_mask"].dtype == bool


# ---------------------------------------------------------------------------
# Mask count tests
# ---------------------------------------------------------------------------

def test_planet_mask_count_matches_n_planets():
    n = 7
    state = _builder().from_obs(_obs(n_planets=n), player=0)
    assert int(state["planet_mask"].sum()) == n


def test_fleet_mask_count_matches_n_fleets():
    n = 12
    state = _builder().from_obs(_obs(n_fleets=n), player=0)
    assert int(state["fleet_mask"].sum()) == n


def test_planet_mask_clipped_to_max_planets():
    """More planets than max_planets — mask saturates at max_planets."""
    n = MAX_PLANETS + 5
    state = _builder().from_obs(_obs(n_planets=n), player=0)
    assert int(state["planet_mask"].sum()) == MAX_PLANETS


def test_fleet_mask_clipped_to_max_fleets():
    n = MAX_FLEETS + 5
    state = _builder().from_obs(_obs(n_fleets=n), player=0)
    assert int(state["fleet_mask"].sum()) == MAX_FLEETS


# ---------------------------------------------------------------------------
# Feature value tests
# ---------------------------------------------------------------------------

def test_planet_ownership_flags_my_planet():
    """First planet (owner=player 0) should have is_mine=1, is_enemy=0, is_neutral=0."""
    state = _builder().from_obs(_obs(n_planets=3, player=0), player=0)
    pf = state["planet_features"]
    assert pf[0, 0] == pytest.approx(1.0)  # is_mine
    assert pf[0, 1] == pytest.approx(0.0)  # is_enemy
    assert pf[0, 2] == pytest.approx(0.0)  # is_neutral


def test_planet_ownership_flags_enemy_planet():
    """An enemy planet should have is_mine=0, is_enemy=1, is_neutral=0."""
    obs = _obs(n_planets=4, player=0)
    # In _obs, planets with i >= n_planets//2+1 are player 1 (enemy for player 0)
    # For n=4: mine=[0,1,2], enemy=[3]
    state = _builder().from_obs(obs, player=0)
    pf = state["planet_features"]
    assert pf[3, 0] == pytest.approx(0.0)  # is_mine
    assert pf[3, 1] == pytest.approx(1.0)  # is_enemy
    assert pf[3, 2] == pytest.approx(0.0)  # is_neutral


def test_ships_normalized_by_200():
    obs = _obs(n_planets=2, player=0)
    obs["planets"][0][5] = 100.0  # set ships to 100
    state = _builder().from_obs(obs, player=0)
    assert state["planet_features"][0, 5] == pytest.approx(0.5)


def test_ships_clipped_at_one():
    obs = _obs(n_planets=2, player=0)
    obs["planets"][0][5] = 500.0  # > 200 → clipped
    state = _builder().from_obs(obs, player=0)
    assert state["planet_features"][0, 5] == pytest.approx(1.0)


def test_position_normalized_by_100():
    obs = _obs(n_planets=2, player=0)
    obs["planets"][0][2] = 50.0   # x=50
    obs["planets"][0][3] = 75.0   # y=75
    state = _builder().from_obs(obs, player=0)
    assert state["planet_features"][0, 3] == pytest.approx(0.5)   # x/100
    assert state["planet_features"][0, 4] == pytest.approx(0.75)  # y/100


def test_turn_progress_feature():
    state = _builder().from_obs(_obs(step=250), player=0)
    assert state["global_features"][0] == pytest.approx(250 / 500.0)


def test_turn_zero_gives_zero_progress():
    state = _builder().from_obs(_obs(step=0), player=0)
    assert state["global_features"][0] == pytest.approx(0.0)


def test_comet_flag_set_for_comet_planet():
    obs = _obs(n_planets=3, comet_ids=[0])  # planet with id=0 is a comet
    state = _builder().from_obs(obs, player=0)
    assert state["planet_features"][0, 8] == pytest.approx(1.0)
    assert state["planet_features"][1, 8] == pytest.approx(0.0)
    assert state["planet_features"][2, 8] == pytest.approx(0.0)


def test_dist_to_center_at_center():
    """Planet at (50, 50) has dist_to_center = 0."""
    obs = _obs(n_planets=1, player=0)
    obs["planets"][0][2] = 50.0  # x = 50
    obs["planets"][0][3] = 50.0  # y = 50
    state = _builder().from_obs(obs, player=0)
    assert state["planet_features"][0, 9] == pytest.approx(0.0, abs=1e-5)


def test_fleet_sin_cos_of_angle():
    obs = _obs(n_planets=2, n_fleets=1, player=0)
    obs["fleets"][0][4] = math.pi / 2  # angle = π/2
    state = _builder().from_obs(obs, player=0)
    ff = state["fleet_features"]
    assert ff[0, 4] == pytest.approx(math.sin(math.pi / 2))  # sin(π/2) = 1
    assert abs(ff[0, 5]) < 1e-5                               # cos(π/2) ≈ 0


# ---------------------------------------------------------------------------
# Padding tests
# ---------------------------------------------------------------------------

def test_padding_planet_slots_are_zero():
    state = _builder().from_obs(_obs(n_planets=3), player=0)
    assert np.allclose(state["planet_features"][3:], 0.0)
    assert not state["planet_mask"][3:].any()


def test_padding_fleet_slots_are_zero():
    state = _builder().from_obs(_obs(n_fleets=5), player=0)
    assert np.allclose(state["fleet_features"][5:], 0.0)
    assert not state["fleet_mask"][5:].any()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_planets_does_not_crash():
    obs = {"planets": [], "fleets": [], "comet_planet_ids": [], "step": 0, "player": 0}
    state = _builder().from_obs(obs, player=0)
    assert state["planet_features"].shape == (MAX_PLANETS, 10)
    assert int(state["planet_mask"].sum()) == 0


def test_empty_fleets_does_not_crash():
    state = _builder().from_obs(_obs(n_fleets=0), player=0)
    assert state["fleet_features"].shape == (MAX_FLEETS, 8)
    assert int(state["fleet_mask"].sum()) == 0


def test_context_n_planets_matches():
    state = _builder().from_obs(_obs(n_planets=7), player=0)
    assert state["context"].n_planets == 7


def test_context_planet_ids_are_integers():
    state = _builder().from_obs(_obs(n_planets=4), player=0)
    assert state["context"].planet_ids.dtype == np.int32


def test_context_my_planet_mask_is_bool():
    state = _builder().from_obs(_obs(n_planets=4), player=0)
    assert state["context"].my_planet_mask.dtype == bool


# ---------------------------------------------------------------------------
# from_obs_structured is alias
# ---------------------------------------------------------------------------

def test_from_obs_structured_same_as_from_obs():
    builder = _builder()
    obs = _obs()
    s1 = builder.from_obs(obs, player=0)
    s2 = builder.from_obs_structured(obs, player=0)
    assert np.array_equal(s1["planet_features"], s2["planet_features"])
    assert np.array_equal(s1["planet_mask"], s2["planet_mask"])
