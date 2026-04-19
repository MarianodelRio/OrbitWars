import pytest
from game.state.models import parse_obs, GameState, Planet, Fleet

SAMPLE_OBS = {
    "player": 0,
    "step": 5,
    "angular_velocity": 0.03,
    "planets": [
        [0, 0, 20.0, 20.0, 2.0, 10.0, 2.0],
        [1, 1, 80.0, 80.0, 2.0, 8.0, 2.0],
        [2, -1, 50.0, 70.0, 1.0, 5.0, 1.0],
    ],
    "fleets": [
        [0, 1, 75.0, 75.0, 3.14, 1, 3.0],
    ],
    "comet_planet_ids": [2],
    "initial_planets": [],
}

def test_parse_obs_basic():
    state = parse_obs(SAMPLE_OBS)
    assert state.step == 5
    assert state.player == 0
    assert state.angular_velocity == pytest.approx(0.03)
    assert len(state.planets) == 3
    assert len(state.fleets) == 1

def test_parse_obs_my_planets():
    state = parse_obs(SAMPLE_OBS)
    assert len(state.my_planets) == 1
    assert state.my_planets[0].owner == 0

def test_parse_obs_enemy_planets():
    state = parse_obs(SAMPLE_OBS)
    assert len(state.enemy_planets) == 1
    assert state.enemy_planets[0].owner == 1

def test_parse_obs_neutral_planets():
    state = parse_obs(SAMPLE_OBS)
    assert len(state.neutral_planets) == 1
    assert state.neutral_planets[0].owner == -1

def test_parse_obs_comet_ids():
    state = parse_obs(SAMPLE_OBS)
    assert state.comet_planet_ids == [2]

def test_parse_obs_initial_planets_empty():
    state = parse_obs(SAMPLE_OBS)
    assert state.initial_planets == []
